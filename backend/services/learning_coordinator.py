"""Learning coordinator service.

Coordinates ingestion of operational feedback, generation of learning signals,
and invoking lightweight training / model update steps. This is a small glue
layer so the engine, event orchestrator and state manager can call a single
service to advance learning after route outcomes are recorded.
"""

from typing import Optional, Dict, Any
import logging

from ..core.learning.feedback_processor import FeedbackProcessor
from ..core.learning.td_learning import (
    TemporalDifferenceLearner,
    NeuralNetworkTDLearner,
)
from ..core.models.domain import OperationalOutcome
from ..core.powell.engine import PowellEngine

logger = logging.getLogger(__name__)


class LearningCoordinator:
    """Coordinate feedback processing and lightweight model updates.

    This component keeps responsibilities intentionally small so it is easy to
    unit test and to replace with more advanced orchestration later.
    """

    def __init__(
        self,
        engine: Optional[PowellEngine] = None,
        feedback_processor: Optional[FeedbackProcessor] = None,
        td_learner: Optional[TemporalDifferenceLearner] = None,
        nn_td_learner: Optional[NeuralNetworkTDLearner] = None,
    ):
        self.engine = engine
        self.processor = feedback_processor or FeedbackProcessor()
        self.td_learner = td_learner or TemporalDifferenceLearner()
        self.nn_td_learner = nn_td_learner or NeuralNetworkTDLearner()
        # Telemetry: track VFA training progress
        self.vfa_telemetry = {
            "last_training_loss": None,
            "last_training_samples": 0,
            "total_training_steps": 0,
            "last_training_timestamp": None,
        }

    def process_outcome(
        self, outcome: OperationalOutcome, state: Optional[object] = None
    ) -> Dict[str, Any]:
        """Process a single OperationalOutcome.

        - Compute learning signals
        - Update engine-level models (CFA/PFA/VFA) via engine.learn_from_feedback
        - Optionally perform a small TD update step when neural components exist

        Returns the computed signals for observability.
        """

        logger.info(
            f"LearningCoordinator: processing outcome for route {outcome.route_id}"
        )

        # Compute signals
        signals = self.processor.process_outcome(outcome)

        # If engine provided, call its generic learn hook with normalized dict
        if self.engine:
            try:
                engine_payload = {
                    "route_id": outcome.route_id,
                    "vehicle_id": outcome.vehicle_id,
                    "predicted_fuel_cost": outcome.predicted_fuel_cost,
                    "actual_fuel_cost": outcome.actual_fuel_cost,
                    "predicted_duration_minutes": outcome.predicted_duration_minutes,
                    "actual_duration_minutes": outcome.actual_duration_minutes,
                    "predicted_distance_km": outcome.predicted_distance_km,
                    "actual_distance_km": outcome.actual_distance_km,
                    "success": outcome.on_time,
                    "customer_satisfaction_score": outcome.customer_satisfaction_score,
                    "notes": outcome.notes,
                }

                # Allow engine to update its internal models
                self.engine.learn_from_feedback(engine_payload)
            except Exception as e:
                logger.error(f"Engine learning hook failed: {e}")

        # If engine has VFA, attempt to complete a previously recorded pending
        # experience (recorded at commit time) for this route. If none exists,
        # fall back to adding a terminal experience immediately.
        try:
            if self.engine and hasattr(self.engine, "vfa") and state is not None:
                try:
                    route_id = getattr(outcome, "route_id", None)
                    reward = self._estimate_reward_from_outcome(outcome)

                    if (
                        route_id
                        and getattr(self.engine.vfa, "pending_by_route", None)
                        and route_id in self.engine.vfa.pending_by_route
                    ):
                        completed = self.engine.vfa.complete_pending_experience(
                            route_id, reward, done=not outcome.on_time
                        )
                        if completed:
                            logger.info(
                                f"Completed pending VFA experience for {route_id}"
                            )
                    else:
                        # No pending experience recorded; add a terminal experience now
                        try:
                            s_feats = self.engine.vfa.extract_state_features_from_state(
                                state
                            )
                            action = getattr(outcome, "route_id", "route")
                            self.engine.vfa.add_experience(
                                s_feats, action, reward, None, True
                            )
                        except Exception:
                            logger.debug("Failed to add immediate VFA experience")

                    # Optionally trigger a small training run when aggregates indicate
                    if self.should_retrain_vfa():
                        try:
                            batch = 32
                            epochs = 1
                            if hasattr(self.engine, "config"):
                                vfa_conf = self.engine.config.get("vfa", {})
                                batch = int(vfa_conf.get("train_batch_size", batch))
                                epochs = int(vfa_conf.get("train_epochs", epochs))

                            updates = self.engine.vfa.train_from_buffer(
                                batch_size=batch, epochs=epochs
                            )
                            logger.info(
                                f"VFA trained from buffer: {updates} update steps"
                            )
                        except Exception as e:
                            logger.debug(f"VFA training attempt failed: {e}")
                except Exception as e:
                    logger.debug(f"Failed to complete VFA pending experience: {e}")
        except Exception:
            logger.debug("Engine VFA completion hook unavailable or failed")

        # If engine and state available, ask PFA to mine rules from recent outcomes
        try:
            if self.engine and state is not None and hasattr(self.engine, "pfa"):
                try:
                    self.engine.pfa.mine_rules_from_state(state)
                    # Export learned rules for persistence
                    exported = self.engine.pfa.export_rules_for_learning_state()
                    if exported:
                        # Return exported rules to caller so orchestrator can persist
                        signals["pfa_rules"] = exported
                except Exception as e:
                    logger.debug(f"PFA mining skipped/failed: {e}")
        except Exception:
            logger.debug("Engine or state not available for PFA mining")

        # Optionally perform TD step using lightweight TD learner (scalar example)
        try:
            reward = self._estimate_reward_from_outcome(outcome)
            # For scalar TD learner we don't have state values here; perform bookkeeping
            # This keeps the TD learner metrics updated for monitoring.
            _ = self.td_learner.td_learning_step(0.0, reward, 0.0, terminal=True)
        except Exception:
            logger.debug("TD learner step skipped or failed (non-critical)")

        # If neural TD learner available and engine has VFA network, attempt a safe update
        try:
            if (
                hasattr(self.nn_td_learner, "network")
                and self.nn_td_learner.network is not None
                and self.engine
                and hasattr(self.engine, "vfa")
            ):
                # We do not attempt a full TD update here because constructing proper
                # state/next_state feature tensors requires more context. Instead we
                # record that a learning signal arrived so orchestration can schedule
                # a proper batch training job.
                logger.debug(
                    "Neural TD learner available — recommend scheduling batch training"
                )
        except Exception:
            logger.debug("Neural TD learner check failed — skipping")

        return signals

    def _estimate_reward_from_outcome(self, outcome: OperationalOutcome) -> float:
        """Estimate a simple scalar reward from an outcome (placeholder)."""

        # Reward = revenue_estimate - actual_fuel_cost (placeholder)
        revenue_estimate = outcome.successful_deliveries * 1000.0
        reward = revenue_estimate - outcome.actual_fuel_cost
        return reward

    def should_retrain_vfa(self) -> bool:
        return self.processor.should_retrain_vfa()

    def should_update_cfa(self) -> bool:
        return self.processor.should_update_cfa_parameters()

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {
            "aggregate_metrics": self.processor.get_aggregate_metrics(),
            "model_accuracies": self.processor.get_model_accuracies(),
            "vfa_telemetry": self.vfa_telemetry,
        }
        if self.engine and hasattr(self.engine, "vfa"):
            metrics["vfa_stats"] = {
                "trained_samples": getattr(self.engine.vfa, "trained_samples", 0),
                "total_loss": getattr(self.engine.vfa, "total_loss", 0.0),
                "pending_experiences": len(
                    getattr(self.engine.vfa, "pending_by_route", {})
                ),
                "buffer_size": len(getattr(self.engine.vfa, "experience_buffer", [])),
            }
        return metrics

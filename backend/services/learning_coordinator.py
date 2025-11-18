"""Learning coordinator service with world-class enhancements.

Coordinates ingestion of operational feedback, generation of learning signals,
and invoking world-class training procedures across all policy approximations.

Enhanced with:
- Prioritized experience replay for VFA
- Adam optimization for CFA parameters
- Apriori pattern mining for PFA
- Comprehensive telemetry and monitoring
"""

from typing import Optional, Dict, Any
import logging
from datetime import datetime

from ..core.learning.feedback_processor import FeedbackProcessor
from ..core.learning.td_learning import (
    TemporalDifferenceLearner,
    NeuralNetworkTDLearner,
)
from ..core.models.domain import OperationalOutcome
from ..core.powell.engine import PowellEngine

logger = logging.getLogger(__name__)


class LearningCoordinator:
    """Coordinate feedback processing and world-class model updates.

    Enhanced with comprehensive telemetry for all learning components:
    - VFA: Prioritized replay, regularization, LR scheduling
    - CFA: Adam optimization, convergence detection
    - PFA: Apriori pattern mining, exploration
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

        # Comprehensive telemetry tracking
        self.telemetry = {
            # VFA telemetry
            "vfa": {
                "last_training_loss": None,
                "last_training_samples": 0,
                "total_training_steps": 0,
                "last_training_timestamp": None,
                "prioritized_replay_size": 0,
                "current_learning_rate": None,
                "early_stopping_triggered": False,
            },
            # CFA telemetry
            "cfa": {
                "fuel_cost_per_km": None,
                "driver_cost_per_hour": None,
                "fuel_accuracy_mape": None,
                "time_accuracy_mape": None,
                "fuel_converged": False,
                "time_converged": False,
                "total_updates": 0,
            },
            # PFA telemetry
            "pfa": {
                "total_rules": 0,
                "active_rules": 0,
                "patterns_mined": 0,
                "last_mining_timestamp": None,
                "avg_rule_confidence": 0.0,
                "avg_rule_lift": 0.0,
                "exploration_rate": None,
            },
            # General telemetry
            "general": {
                "total_outcomes_processed": 0,
                "last_outcome_timestamp": None,
                "coordinator_initialized": datetime.now(),
            },
        }

    def process_outcome(
        self, outcome: OperationalOutcome, state: Optional[object] = None
    ) -> Dict[str, Any]:
        """Process a single OperationalOutcome with world-class learning.

        Enhanced workflow:
        1. Compute learning signals (FeedbackProcessor)
        2. Update CFA parameters (Adam optimization)
        3. Add VFA experience (prioritized replay)
        4. Mine PFA patterns (Apriori algorithm)
        5. Train VFA (with regularization & LR scheduling)
        6. Update comprehensive telemetry

        Returns the computed signals and telemetry for observability.
        """

        logger.info(
            f"LearningCoordinator: processing outcome for route {outcome.route_id}"
        )

        # Compute learning signals
        signals = self.processor.process_outcome(outcome)

        # Update general telemetry
        self.telemetry["general"]["total_outcomes_processed"] += 1
        self.telemetry["general"]["last_outcome_timestamp"] = datetime.now()

        # ========================================
        # 1. CFA Parameter Update (Adam Optimizer)
        # ========================================
        if self.engine and hasattr(self.engine, "cfa"):
            try:
                cfa_payload = {
                    "predicted_fuel_cost": outcome.predicted_fuel_cost,
                    "actual_fuel_cost": outcome.actual_fuel_cost,
                    "predicted_duration_minutes": outcome.predicted_duration_minutes,
                    "actual_duration_minutes": outcome.actual_duration_minutes,
                    "actual_distance_km": outcome.actual_distance_km,
                }

                # Update CFA using Adam optimizer
                self.engine.cfa.update_from_feedback(cfa_payload)

                # Update CFA telemetry
                if hasattr(self.engine.cfa, "parameter_manager"):
                    params = self.engine.cfa.parameter_manager.get_cost_parameters()
                    accuracies = self.engine.cfa.parameter_manager.get_prediction_accuracy()
                    is_converged = self.engine.cfa.parameter_manager.is_converged()

                    self.telemetry["cfa"]["fuel_cost_per_km"] = params["fuel_cost_per_km"]
                    self.telemetry["cfa"]["driver_cost_per_hour"] = params["driver_cost_per_hour"]
                    self.telemetry["cfa"]["fuel_accuracy_mape"] = accuracies.get("fuel_mape", 0.0)
                    self.telemetry["cfa"]["time_accuracy_mape"] = accuracies.get("time_mape", 0.0)
                    self.telemetry["cfa"]["fuel_converged"] = is_converged
                    self.telemetry["cfa"]["time_converged"] = is_converged
                    self.telemetry["cfa"]["total_updates"] += 1

                    logger.debug(
                        f"CFA updated: fuel={params['fuel_cost_per_km']:.4f} KES/km, "
                        f"time={params['driver_cost_per_hour']:.2f} KES/hr, "
                        f"fuel_mape={accuracies.get('fuel_mape', 0.0):.2%}"
                    )

            except Exception as e:
                logger.error(f"CFA parameter update failed: {e}")

        # ========================================
        # 2. Legacy engine learning hook (for compatibility)
        # ========================================
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

                # Allow engine to update its internal models (legacy)
                if hasattr(self.engine, "learn_from_feedback"):
                    self.engine.learn_from_feedback(engine_payload)
            except Exception as e:
                logger.debug(f"Engine learning hook failed: {e}")

        # ========================================
        # 3. VFA Experience Replay (Prioritized)
        # ========================================
        try:
            if self.engine and hasattr(self.engine, "vfa") and state is not None:
                try:
                    route_id = getattr(outcome, "route_id", None)
                    reward = self._estimate_reward_from_outcome(outcome)

                    # Complete pending experience or add terminal experience
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
                        # No pending experience; add terminal experience with priority
                        try:
                            s_feats = self.engine.vfa.extract_state_features_from_state(state)
                            action = getattr(outcome, "route_id", "route")

                            # Compute TD error as priority
                            priority = abs(reward)  # Simple priority (can be enhanced)

                            self.engine.vfa.add_experience(
                                s_feats, action, reward, None, True, priority=priority
                            )
                        except Exception:
                            logger.debug("Failed to add immediate VFA experience")

                    # Update VFA telemetry
                    if hasattr(self.engine.vfa, "experience_coordinator"):
                        self.telemetry["vfa"]["prioritized_replay_size"] = len(
                            self.engine.vfa.experience_coordinator
                        )

                    if hasattr(self.engine.vfa, "lr_scheduler"):
                        self.telemetry["vfa"]["current_learning_rate"] = (
                            self.engine.vfa.lr_scheduler.get_lr()
                        )

                    # Trigger training with world-class enhancements
                    if self.should_retrain_vfa():
                        try:
                            batch = 32
                            epochs = 10  # Increased for better learning
                            if hasattr(self.engine, "config"):
                                vfa_conf = self.engine.config.get("vfa", {})
                                batch = int(vfa_conf.get("train_batch_size", batch))
                                epochs = int(vfa_conf.get("train_epochs", epochs))

                            # Train with all enhancements (prioritized replay, regularization, LR scheduling)
                            updates = self.engine.vfa.train_from_buffer(
                                batch_size=batch, epochs=epochs
                            )

                            # Update telemetry
                            self.telemetry["vfa"]["total_training_steps"] += updates
                            self.telemetry["vfa"]["last_training_samples"] = batch * updates
                            self.telemetry["vfa"]["last_training_timestamp"] = datetime.now()

                            if hasattr(self.engine.vfa, "regularization"):
                                early_stop_stats = self.engine.vfa.regularization.get_statistics()
                                self.telemetry["vfa"]["early_stopping_triggered"] = (
                                    early_stop_stats.get("early_stopped", False)
                                )
                                if early_stop_stats.get("current_val_loss"):
                                    self.telemetry["vfa"]["last_training_loss"] = (
                                        early_stop_stats["current_val_loss"]
                                    )

                            logger.info(
                                f"VFA trained: {updates} updates, batch={batch}, epochs={epochs}, "
                                f"buffer_size={len(self.engine.vfa.experience_coordinator)}, "
                                f"lr={self.telemetry['vfa']['current_learning_rate']:.6f}"
                            )

                        except Exception as e:
                            logger.error(f"VFA training failed: {e}")

                except Exception as e:
                    logger.debug(f"VFA experience processing failed: {e}")
        except Exception:
            logger.debug("Engine VFA unavailable")

        # ========================================
        # 4. PFA Pattern Mining (Apriori)
        # ========================================
        try:
            if self.engine and state is not None and hasattr(self.engine, "pfa"):
                try:
                    # Mine rules using enhanced Apriori algorithm
                    self.engine.pfa.mine_rules_from_state(state)

                    # Update PFA telemetry
                    if hasattr(self.engine.pfa, "pattern_coordinator"):
                        stats = self.engine.pfa.pattern_coordinator.get_rule_statistics()
                        self.telemetry["pfa"]["total_rules"] = stats.get("total_rules", 0)
                        self.telemetry["pfa"]["active_rules"] = stats.get("active_rules", 0)
                        self.telemetry["pfa"]["avg_rule_confidence"] = stats.get("avg_confidence", 0.0)
                        self.telemetry["pfa"]["avg_rule_lift"] = stats.get("avg_lift", 0.0)
                        self.telemetry["pfa"]["patterns_mined"] = len(
                            getattr(self.engine.pfa.pattern_coordinator, "rules", [])
                        )
                        self.telemetry["pfa"]["last_mining_timestamp"] = datetime.now()

                    if hasattr(self.engine.pfa, "rule_exploration"):
                        exploration_stats = self.engine.pfa.rule_exploration.get_statistics()
                        if exploration_stats and "exploration_rate" in exploration_stats:
                            self.telemetry["pfa"]["exploration_rate"] = exploration_stats["exploration_rate"]

                    # Export learned rules for persistence
                    exported = self.engine.pfa.export_rules_for_learning_state()
                    if exported:
                        # Return exported rules to caller so orchestrator can persist
                        signals["pfa_rules"] = exported
                        logger.info(
                            f"PFA mined patterns: {self.telemetry['pfa']['patterns_mined']} rules, "
                            f"avg_confidence={self.telemetry['pfa']['avg_rule_confidence']:.2f}, "
                            f"avg_lift={self.telemetry['pfa']['avg_rule_lift']:.2f}"
                        )

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
        """Get comprehensive metrics from all learning components.

        Returns a unified metrics dictionary containing:
        - Aggregate metrics from FeedbackProcessor
        - Model accuracies
        - Comprehensive telemetry (VFA, CFA, PFA, general)
        - Real-time component statistics
        """
        metrics = {
            # Legacy metrics from FeedbackProcessor
            "aggregate_metrics": self.processor.get_aggregate_metrics(),
            "model_accuracies": self.processor.get_model_accuracies(),

            # World-class comprehensive telemetry
            "telemetry": self.telemetry.copy(),
        }

        # Add real-time VFA statistics
        if self.engine and hasattr(self.engine, "vfa"):
            metrics["vfa_realtime"] = {
                "trained_samples": getattr(self.engine.vfa, "trained_samples", 0),
                "total_loss": getattr(self.engine.vfa, "total_loss", 0.0),
                "pending_experiences": len(
                    getattr(self.engine.vfa, "pending_by_route", {})
                ),
                "buffer_size": len(getattr(self.engine.vfa, "experience_buffer", [])),
            }

            # Add prioritized replay stats
            if hasattr(self.engine.vfa, "experience_coordinator"):
                metrics["vfa_realtime"]["prioritized_buffer_size"] = len(
                    self.engine.vfa.experience_coordinator
                )

        # Add real-time CFA statistics
        if self.engine and hasattr(self.engine, "cfa"):
            if hasattr(self.engine.cfa, "parameter_manager"):
                params = self.engine.cfa.parameter_manager.get_cost_parameters()
                accuracies = self.engine.cfa.parameter_manager.get_prediction_accuracy()
                is_converged = self.engine.cfa.parameter_manager.is_converged()

                metrics["cfa_realtime"] = {
                    "parameters": params,
                    "accuracies": accuracies,
                    "converged": is_converged,
                }

        # Add real-time PFA statistics
        if self.engine and hasattr(self.engine, "pfa"):
            pfa_stats = {
                "total_rules": len(getattr(self.engine.pfa, "rules_by_id", {})),
            }

            if hasattr(self.engine.pfa, "pattern_coordinator"):
                pfa_stats["pattern_stats"] = self.engine.pfa.pattern_coordinator.get_rule_statistics()

            if hasattr(self.engine.pfa, "rule_exploration"):
                pfa_stats["exploration_stats"] = self.engine.pfa.rule_exploration.get_statistics()

            metrics["pfa_realtime"] = pfa_stats

        return metrics

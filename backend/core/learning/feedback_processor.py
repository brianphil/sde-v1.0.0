"""Process operational feedback for learning updates."""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import statistics

from ..models.domain import OperationalOutcome
from ..models.state import SystemState, LearningState

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """Process route outcomes and generate learning signals.
    
    Responsibilities:
    - Ingest OperationalOutcome data
    - Compute prediction errors
    - Generate learning signals for each policy class
    - Aggregate metrics for model evaluation
    """

    def __init__(self, window_size: int = 100):
        """Initialize processor.
        
        Args:
            window_size: How many recent outcomes to keep for aggregate metrics
        """
        self.window_size = window_size
        self.recent_outcomes: List[OperationalOutcome] = []
        self.metric_history: List[Dict[str, float]] = []

    def process_outcome(self, outcome: OperationalOutcome) -> Dict[str, Any]:
        """Process single operational outcome.
        
        Returns:
            Learning signals for CFA, VFA, PFA, DLA
        """

        # Track outcome
        self.recent_outcomes.append(outcome)
        if len(self.recent_outcomes) > self.window_size:
            self.recent_outcomes.pop(0)

        logger.info(f"Processing outcome: route={outcome.route_id}, on_time={outcome.on_time}")

        # Compute learning signals
        signals = {
            "cfa_signals": self._compute_cfa_signals(outcome),
            "vfa_signals": self._compute_vfa_signals(outcome),
            "pfa_signals": self._compute_pfa_signals(outcome),
            "dla_signals": self._compute_dla_signals(outcome),
            "metrics": self._compute_metrics(outcome),
        }

        # Record metrics
        self.metric_history.append(signals["metrics"])
        if len(self.metric_history) > self.window_size:
            self.metric_history.pop(0)

        return signals

    def _compute_cfa_signals(self, outcome: OperationalOutcome) -> Dict[str, float]:
        """Compute learning signals for Cost Function Approximation.
        
        CFA learns: predict fuel_cost, time_cost, delay_penalty accurately
        """

        fuel_error = outcome.actual_fuel_cost - outcome.predicted_fuel_cost
        time_error = outcome.actual_duration_minutes - outcome.predicted_duration_minutes

        # Accuracy metrics
        fuel_accuracy = outcome.get_accuracy_fuel()
        time_accuracy = outcome.get_accuracy_duration()

        return {
            "fuel_error": fuel_error,
            "time_error": time_error,
            "fuel_accuracy": fuel_accuracy,
            "time_accuracy": time_accuracy,
            "prediction_signal": "increase_params" if fuel_error > 0 else "decrease_params",
        }

    def _compute_vfa_signals(self, outcome: OperationalOutcome) -> Dict[str, float]:
        """Compute learning signals for Value Function Approximation.
        
        VFA learns: predict long-term value (profit, fleet utilization, etc.)
        TD-learning update: V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
        """

        # Estimate immediate reward (profit from route)
        # In production: use actual order prices and costs
        estimated_profit = 1500 - outcome.actual_fuel_cost  # Placeholder

        # Value update magnitude depends on error
        value_error = abs(estimated_profit) * (1.0 if outcome.on_time else 0.7)

        return {
            "immediate_reward": estimated_profit,
            "reward_error": value_error,
            "on_time_bonus": 1.0 if outcome.on_time else 0.0,
            "td_learning_magnitude": value_error,
        }

    def _compute_pfa_signals(self, outcome: OperationalOutcome) -> Dict[str, float]:
        """Compute learning signals for Policy Function Approximation.
        
        PFA learns: which rules work well in which conditions
        """

        # Rule success: was the rule's recommendation good?
        rule_success = 1.0 if outcome.on_time else 0.0

        # Support increase if rule was used successfully
        support_adjustment = 0.05 if rule_success > 0.5 else -0.02

        return {
            "rule_success": rule_success,
            "support_adjustment": support_adjustment,
            "satisfaction_signal": outcome.customer_satisfaction_score or 0.5,
        }

    def _compute_dla_signals(self, outcome: OperationalOutcome) -> Dict[str, float]:
        """Compute learning signals for Direct Lookahead Approximation.
        
        DLA learns: demand forecast accuracy, consolidation effectiveness
        """

        # Multi-period planning quality
        planning_quality = (
            (outcome.get_accuracy_fuel() + outcome.get_accuracy_duration()) / 2.0
        )

        return {
            "forecast_accuracy": planning_quality,
            "consolidation_achieved": 1.0,  # Placeholder
            "multi_period_efficiency": planning_quality,
        }

    def _compute_metrics(self, outcome: OperationalOutcome) -> Dict[str, float]:
        """Compute aggregate metrics from outcome."""

        return {
            "on_time": 1.0 if outcome.on_time else 0.0,
            "fuel_efficiency": outcome.actual_distance_km / (outcome.actual_fuel_cost / 150.0 + 1.0),
            "time_efficiency": outcome.successful_deliveries / (outcome.actual_duration_minutes / 60.0 + 1.0),
            "success_rate": outcome.successful_deliveries / (outcome.successful_deliveries + outcome.failed_deliveries + 1.0),
            "customer_satisfaction": outcome.customer_satisfaction_score or 0.5,
        }

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics from recent window."""

        if not self.metric_history:
            return {}

        metrics = {}
        for key in self.metric_history[0].keys():
            values = [m[key] for m in self.metric_history]
            metrics[f"{key}_mean"] = statistics.mean(values)
            if len(values) > 1:
                metrics[f"{key}_stdev"] = statistics.stdev(values)

        return metrics

    def get_model_accuracies(self) -> Dict[str, float]:
        """Get current model prediction accuracies."""

        if not self.recent_outcomes:
            return {
                "cfa_fuel_accuracy": 0.0,
                "cfa_time_accuracy": 0.0,
                "vfa_accuracy": 0.0,
                "pfa_coverage": 0.0,
            }

        cfa_fuel_accs = [o.get_accuracy_fuel() for o in self.recent_outcomes]
        cfa_time_accs = [o.get_accuracy_duration() for o in self.recent_outcomes]

        return {
            "cfa_fuel_accuracy": statistics.mean(cfa_fuel_accs),
            "cfa_time_accuracy": statistics.mean(cfa_time_accs),
            "vfa_accuracy": 0.5,  # Would compute from VFA network
            "pfa_coverage": 0.5,  # Would compute from rules
        }

    def update_learning_state(self, current_learning: LearningState) -> LearningState:
        """Update learning state based on recent feedback."""

        from ..models.state import LearningState
        from dataclasses import replace

        accuracies = self.get_model_accuracies()

        updated_learning = replace(
            current_learning,
            cfa_accuracy_fuel=accuracies["cfa_fuel_accuracy"],
            cfa_accuracy_time=accuracies["cfa_time_accuracy"],
            vfa_accuracy=accuracies["vfa_accuracy"],
            pfa_coverage=accuracies["pfa_coverage"],
            last_updated=datetime.now(),
        )

        return updated_learning

    def should_retrain_vfa(self) -> bool:
        """Check if VFA should be retrained based on recent performance."""
        if len(self.metric_history) < 10:
            return False

        recent_accuracy = statistics.mean(m.get("on_time", 0.0) for m in self.metric_history[-10:])
        return recent_accuracy < 0.7  # Retrain if on-time rate drops below 70%

    def should_update_cfa_parameters(self) -> bool:
        """Check if CFA parameters should be updated."""
        if len(self.recent_outcomes) < 5:
            return False

        fuel_accs = [o.get_accuracy_fuel() for o in self.recent_outcomes[-5:]]
        avg_accuracy = statistics.mean(fuel_accs)
        return avg_accuracy < 0.8  # Update if accuracy drops below 80%

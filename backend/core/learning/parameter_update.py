"""Adaptive parameter update algorithms for Cost Function Approximation (CFA).

This module implements sde parameter learning for cost models, using:
- Gradient-based optimization
- Adaptive learning rates (Adam-style)
- Momentum and bias correction
- Constraint handling
- Convergence detection
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import statistics
import math

logger = logging.getLogger(__name__)


@dataclass
class ParameterState:
    """Tracks state of a single learnable parameter."""

    value: float
    gradient_sum: float = 0.0
    gradient_sq_sum: float = 0.0
    update_count: int = 0
    momentum: float = 0.0

    # Constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Convergence tracking
    recent_values: List[float] = field(default_factory=list)
    converged: bool = False


class AdaptiveParameterUpdater:
    """Adaptive parameter updates using Adam-style optimization.

    Implements:
    - Adaptive learning rates per parameter
    - Momentum for smooth updates
    - Bias correction for early iterations
    - Gradient clipping for stability
    - Convergence detection
    """

    def __init__(
        self,
        base_learning_rate: float = 0.01,
        beta1: float = 0.9,  # Momentum decay
        beta2: float = 0.999,  # RMSprop decay
        epsilon: float = 1e-8,  # Numerical stability
        max_gradient: float = 1.0,  # Gradient clipping
        convergence_window: int = 20,
        convergence_threshold: float = 0.001,
    ):
        """Initialize adaptive updater.

        Args:
            base_learning_rate: Initial learning rate (α)
            beta1: Exponential decay for first moment (momentum)
            beta2: Exponential decay for second moment (RMSprop)
            epsilon: Small constant for numerical stability
            max_gradient: Maximum allowed gradient (clipping threshold)
            convergence_window: Window size for convergence detection
            convergence_threshold: Std dev threshold for convergence
        """
        self.base_learning_rate = base_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_gradient = max_gradient
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

        self.parameters: Dict[str, ParameterState] = {}
        self.global_step = 0

    def register_parameter(
        self,
        name: str,
        initial_value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """Register a learnable parameter.

        Args:
            name: Parameter identifier
            initial_value: Starting value
            min_value: Optional lower bound
            max_value: Optional upper bound
        """
        self.parameters[name] = ParameterState(
            value=initial_value,
            min_value=min_value,
            max_value=max_value,
        )
        logger.info(
            f"Registered parameter '{name}': {initial_value} (bounds: [{min_value}, {max_value}])"
        )

    def compute_gradient(
        self,
        parameter_name: str,
        prediction_error: float,
        feature_value: float = 1.0,
    ) -> float:
        """Compute gradient for parameter update.

        For cost parameters: gradient = error * feature_value
        Error > 0: predicted too low → increase parameter
        Error < 0: predicted too high → decrease parameter

        Args:
            parameter_name: Which parameter to update
            prediction_error: actual - predicted
            feature_value: Feature magnitude (for scaling)

        Returns:
            Computed gradient
        """
        # Gradient is proportional to error and feature
        gradient = prediction_error * feature_value

        # Clip gradient for stability
        if abs(gradient) > self.max_gradient:
            gradient = self.max_gradient * (1 if gradient > 0 else -1)
            logger.debug(f"Gradient clipped for '{parameter_name}': {gradient}")

        return gradient

    def update_parameter(
        self,
        parameter_name: str,
        gradient: float,
    ) -> Tuple[float, float]:
        """Update parameter using Adam optimizer.

        Adam combines:
        - Momentum (first moment): smooths updates
        - RMSprop (second moment): adapts learning rate
        - Bias correction: accounts for initialization bias

        Update rule:
            m_t = β1 * m_(t-1) + (1 - β1) * g_t
            v_t = β2 * v_(t-1) + (1 - β2) * g_t^2
            m̂_t = m_t / (1 - β1^t)
            v̂_t = v_t / (1 - β2^t)
            θ_t = θ_(t-1) - α * m̂_t / (√v̂_t + ε)

        Args:
            parameter_name: Parameter to update
            gradient: Computed gradient

        Returns:
            (new_value, effective_learning_rate)
        """
        if parameter_name not in self.parameters:
            raise ValueError(f"Parameter '{parameter_name}' not registered")

        param = self.parameters[parameter_name]
        self.global_step += 1
        param.update_count += 1

        # Update first moment (momentum)
        param.momentum = self.beta1 * param.momentum + (1 - self.beta1) * gradient

        # Update second moment (RMSprop)
        param.gradient_sq_sum = self.beta2 * param.gradient_sq_sum + (
            1 - self.beta2
        ) * (gradient**2)

        # Bias correction
        momentum_corrected = param.momentum / (1 - self.beta1**param.update_count)
        variance_corrected = param.gradient_sq_sum / (
            1 - self.beta2**param.update_count
        )

        # Compute adaptive learning rate
        adaptive_lr = self.base_learning_rate / (
            math.sqrt(variance_corrected) + self.epsilon
        )

        # Update parameter value
        update = adaptive_lr * momentum_corrected
        new_value = param.value + update

        # Apply constraints
        if param.min_value is not None:
            new_value = max(new_value, param.min_value)
        if param.max_value is not None:
            new_value = min(new_value, param.max_value)

        # Store old value and update
        old_value = param.value
        param.value = new_value

        # Track convergence
        param.recent_values.append(new_value)
        if len(param.recent_values) > self.convergence_window:
            param.recent_values.pop(0)

        # Check convergence
        if len(param.recent_values) == self.convergence_window:
            value_std = statistics.stdev(param.recent_values)
            param.converged = value_std < self.convergence_threshold

        logger.debug(
            f"Parameter '{parameter_name}': {old_value:.4f} → {new_value:.4f} "
            f"(gradient: {gradient:.4f}, lr: {adaptive_lr:.6f})"
        )

        return new_value, adaptive_lr

    def update_from_error(
        self,
        parameter_name: str,
        prediction_error: float,
        feature_value: float = 1.0,
    ) -> float:
        """Convenience method: compute gradient and update in one call.

        Args:
            parameter_name: Parameter to update
            prediction_error: actual - predicted
            feature_value: Feature magnitude

        Returns:
            New parameter value
        """
        gradient = self.compute_gradient(
            parameter_name, prediction_error, feature_value
        )
        new_value, _ = self.update_parameter(parameter_name, gradient)
        return new_value

    def get_parameter_value(self, parameter_name: str) -> float:
        """Get current parameter value."""
        if parameter_name not in self.parameters:
            raise ValueError(f"Parameter '{parameter_name}' not registered")
        return self.parameters[parameter_name].value

    def get_all_parameters(self) -> Dict[str, float]:
        """Get all parameter values as dict."""
        return {name: param.value for name, param in self.parameters.items()}

    def is_converged(self, parameter_name: Optional[str] = None) -> bool:
        """Check if parameter(s) have converged.

        Args:
            parameter_name: Specific parameter, or None for all

        Returns:
            True if converged
        """
        if parameter_name:
            return self.parameters[parameter_name].converged
        else:
            return all(param.converged for param in self.parameters.values())

    def get_convergence_status(self) -> Dict[str, bool]:
        """Get convergence status for all parameters."""
        return {name: param.converged for name, param in self.parameters.items()}

    def reset_parameter(self, parameter_name: str, new_value: Optional[float] = None):
        """Reset parameter to initial value or specified value."""
        if parameter_name not in self.parameters:
            return

        param = self.parameters[parameter_name]
        if new_value is not None:
            param.value = new_value
        param.gradient_sum = 0.0
        param.gradient_sq_sum = 0.0
        param.momentum = 0.0
        param.update_count = 0
        param.recent_values = []
        param.converged = False

    def get_learning_metrics(self) -> Dict[str, any]:
        """Get comprehensive learning statistics."""
        metrics = {
            "global_step": self.global_step,
            "base_learning_rate": self.base_learning_rate,
            "parameters": {},
        }

        for name, param in self.parameters.items():
            metrics["parameters"][name] = {
                "value": param.value,
                "update_count": param.update_count,
                "momentum": param.momentum,
                "converged": param.converged,
                "recent_std": (
                    statistics.stdev(param.recent_values)
                    if len(param.recent_values) > 1
                    else 0.0
                ),
            }

        return metrics


class CFAParameterManager:
    """Manages CFA cost parameters with domain-specific logic.

    Parameters managed:
    - fuel_cost_per_km: Cost of fuel per kilometer
    - base_time_cost: Base hourly driver cost
    - delay_penalty_per_min: Cost of being late
    - vehicle_type_multipliers: Per-vehicle-type cost adjustments
    """

    def __init__(
        self,
        initial_fuel_cost: float = 9.0,  # KES per km
        initial_time_cost: float = 500.0,  # KES per hour
        initial_delay_penalty: float = 50.0,  # KES per minute late
    ):
        """Initialize CFA parameter manager."""
        self.updater = AdaptiveParameterUpdater(
            base_learning_rate=0.01,
            convergence_window=20,
            convergence_threshold=0.01,
        )

        # Register cost parameters with reasonable bounds
        self.updater.register_parameter(
            "fuel_cost_per_km",
            initial_value=initial_fuel_cost,
            min_value=5.0,  # Never below 5 KES/km
            max_value=20.0,  # Never above 20 KES/km
        )

        self.updater.register_parameter(
            "driver_cost_per_hour",
            initial_value=initial_time_cost,
            min_value=300.0,  # Minimum wage
            max_value=1000.0,  # Maximum realistic
        )

        self.updater.register_parameter(
            "delay_penalty_per_min",
            initial_value=initial_delay_penalty,
            min_value=10.0,
            max_value=200.0,
        )

        # Track prediction accuracy
        self.fuel_predictions: List[Tuple[float, float]] = []  # (predicted, actual)
        self.time_predictions: List[Tuple[float, float]] = []

    def update_from_outcome(
        self,
        predicted_fuel_cost: float,
        actual_fuel_cost: float,
        predicted_duration_min: float,
        actual_duration_min: float,
        distance_km: float,
    ):
        """Update parameters based on operational outcome.

        Args:
            predicted_fuel_cost: What we predicted
            actual_fuel_cost: What actually happened
            predicted_duration_min: Predicted time
            actual_duration_min: Actual time
            distance_km: Route distance
        """
        # Track predictions
        self.fuel_predictions.append((predicted_fuel_cost, actual_fuel_cost))
        self.time_predictions.append((predicted_duration_min, actual_duration_min))

        # Limit history
        if len(self.fuel_predictions) > 100:
            self.fuel_predictions.pop(0)
        if len(self.time_predictions) > 100:
            self.time_predictions.pop(0)

        # Update fuel cost parameter
        fuel_error = actual_fuel_cost - predicted_fuel_cost
        if distance_km > 0:
            # Error per km
            fuel_error_per_km = fuel_error / distance_km
            self.updater.update_from_error(
                "fuel_cost_per_km",
                prediction_error=fuel_error_per_km,
                feature_value=1.0,
            )

        # Update time cost parameter
        time_error = actual_duration_min - predicted_duration_min
        if predicted_duration_min > 0:
            # Convert to hourly cost error
            time_hours = actual_duration_min / 60.0
            time_error_per_hour = time_error / (time_hours + 0.01)
            self.updater.update_from_error(
                "driver_cost_per_hour",
                prediction_error=time_error_per_hour,
                feature_value=0.5,  # Slower update for time
            )

    def update_delay_penalty(self, delay_minutes: float, customer_satisfaction: float):
        """Update delay penalty based on customer feedback.

        Args:
            delay_minutes: How late was delivery
            customer_satisfaction: 0-1 score
        """
        if delay_minutes > 0:
            # If customer very unhappy, increase penalty
            # If customer okay despite delay, decrease penalty
            satisfaction_error = (0.8 - customer_satisfaction) * delay_minutes

            self.updater.update_from_error(
                "delay_penalty_per_min",
                prediction_error=satisfaction_error,
                feature_value=0.1,  # Very slow updates
            )

    def get_cost_parameters(self) -> Dict[str, float]:
        """Get current cost parameters."""
        return self.updater.get_all_parameters()

    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Compute prediction accuracy metrics."""
        if not self.fuel_predictions:
            return {
                "fuel_mape": 0.0,
                "time_mape": 0.0,
                "fuel_rmse": 0.0,
                "time_rmse": 0.0,
            }

        # Fuel accuracy (MAPE)
        fuel_mapes = [
            abs(actual - predicted) / (actual + 0.01)
            for predicted, actual in self.fuel_predictions
        ]
        fuel_mape = statistics.mean(fuel_mapes)

        # Time accuracy (MAPE)
        time_mapes = [
            abs(actual - predicted) / (actual + 0.01)
            for predicted, actual in self.time_predictions
        ]
        time_mape = statistics.mean(time_mapes)

        # RMSE
        fuel_rmse = math.sqrt(
            statistics.mean(
                [
                    (actual - predicted) ** 2
                    for predicted, actual in self.fuel_predictions
                ]
            )
        )

        time_rmse = math.sqrt(
            statistics.mean(
                [
                    (actual - predicted) ** 2
                    for predicted, actual in self.time_predictions
                ]
            )
        )

        return {
            "fuel_mape": fuel_mape,
            "time_mape": time_mape,
            "fuel_rmse": fuel_rmse,
            "time_rmse": time_rmse,
            "fuel_accuracy": 1.0 - fuel_mape,
            "time_accuracy": 1.0 - time_mape,
        }

    def is_converged(self) -> bool:
        """Check if all parameters have converged."""
        return self.updater.is_converged()

    def get_learning_metrics(self) -> Dict[str, any]:
        """Get comprehensive learning metrics."""
        base_metrics = self.updater.get_learning_metrics()
        accuracy_metrics = self.get_prediction_accuracy()

        return {
            **base_metrics,
            "accuracy": accuracy_metrics,
            "convergence_status": self.updater.get_convergence_status(),
        }

"""Regularization and validation for preventing overfitting in learning.

This module implements sde regularization techniques and validation strategies:

- L1/L2 regularization (Lasso/Ridge)
- Dropout for neural networks
- Early stopping based on validation performance
- K-fold cross-validation
- Train/validation/test splitting
- Gradient clipping (already implemented, consolidated here)
- Learning curve analysis
- Overfitting detection
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
import statistics
from datetime import datetime
import math
import random

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Tracks validation metrics over training."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def overfitting_score(self) -> float:
        """Calculate overfitting score.

        Returns:
            Positive value indicates overfitting (val > train loss)
        """
        if self.train_loss == 0:
            return 0.0
        return (self.val_loss - self.train_loss) / self.train_loss


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors validation loss and stops training when it stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore weights at best validation loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float("inf")
        self.best_weights: Optional[Dict[str, Any]] = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def __call__(
        self,
        val_loss: float,
        epoch: int,
        model_weights: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
            model_weights: Current model weights (for restoration)

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.wait = 0
            if model_weights is not None:
                self.best_weights = model_weights.copy()
        else:
            # No improvement
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best validation loss: {self.best_loss:.6f}"
                )

        return self.should_stop

    def get_best_weights(self) -> Optional[Dict[str, Any]]:
        """Get best model weights."""
        if self.restore_best_weights:
            return self.best_weights
        return None


class L2Regularizer:
    """L2 regularization (Ridge regression, weight decay).

    Adds penalty: λ * Σ(w²) to loss function.
    Encourages smaller weights, preventing overfitting.
    """

    def __init__(self, lambda_: float = 0.01):
        """Initialize L2 regularizer.

        Args:
            lambda_: Regularization strength
        """
        self.lambda_ = lambda_

    def compute_penalty(self, weights: List[float]) -> float:
        """Compute L2 penalty.

        Args:
            weights: Model weights

        Returns:
            Regularization penalty
        """
        return self.lambda_ * sum(w**2 for w in weights)

    def compute_gradient(self, weights: List[float]) -> List[float]:
        """Compute gradient of L2 penalty.

        Args:
            weights: Model weights

        Returns:
            Gradient of regularization term
        """
        return [2 * self.lambda_ * w for w in weights]


class L1Regularizer:
    """L1 regularization (Lasso regression).

    Adds penalty: λ * Σ|w| to loss function.
    Encourages sparsity (many weights become exactly zero).
    """

    def __init__(self, lambda_: float = 0.01):
        """Initialize L1 regularizer.

        Args:
            lambda_: Regularization strength
        """
        self.lambda_ = lambda_

    def compute_penalty(self, weights: List[float]) -> float:
        """Compute L1 penalty.

        Args:
            weights: Model weights

        Returns:
            Regularization penalty
        """
        return self.lambda_ * sum(abs(w) for w in weights)

    def compute_gradient(self, weights: List[float]) -> List[float]:
        """Compute sub-gradient of L1 penalty.

        Args:
            weights: Model weights

        Returns:
            Sub-gradient of regularization term
        """
        return [self.lambda_ * (1 if w > 0 else -1 if w < 0 else 0) for w in weights]


class ElasticNetRegularizer:
    """Elastic Net regularization (combination of L1 and L2).

    Penalty: λ₁ * Σ|w| + λ₂ * Σ(w²)
    Combines benefits of both L1 and L2.
    """

    def __init__(self, lambda1: float = 0.01, lambda2: float = 0.01):
        """Initialize Elastic Net regularizer.

        Args:
            lambda1: L1 regularization strength
            lambda2: L2 regularization strength
        """
        self.l1 = L1Regularizer(lambda1)
        self.l2 = L2Regularizer(lambda2)

    def compute_penalty(self, weights: List[float]) -> float:
        """Compute Elastic Net penalty."""
        return self.l1.compute_penalty(weights) + self.l2.compute_penalty(weights)

    def compute_gradient(self, weights: List[float]) -> List[float]:
        """Compute gradient of Elastic Net penalty."""
        l1_grad = self.l1.compute_gradient(weights)
        l2_grad = self.l2.compute_gradient(weights)
        return [g1 + g2 for g1, g2 in zip(l1_grad, l2_grad)]


class DropoutRegularizer:
    """Dropout regularization for neural networks.

    Randomly drops units during training to prevent co-adaptation.
    """

    def __init__(self, dropout_rate: float = 0.5):
        """Initialize dropout.

        Args:
            dropout_rate: Probability of dropping a unit (0-1)
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")

        self.dropout_rate = dropout_rate
        self.training = True

    def apply(self, values: List[float]) -> List[float]:
        """Apply dropout to values.

        Args:
            values: Input values

        Returns:
            Values with dropout applied (if training)
        """
        if not self.training:
            # During inference, no dropout
            return values

        # Apply dropout mask
        dropped = []
        for value in values:
            if random.random() < self.dropout_rate:
                # Drop unit
                dropped.append(0.0)
            else:
                # Scale by 1/(1-p) to maintain expected value
                dropped.append(value / (1 - self.dropout_rate))

        return dropped

    def set_training(self, training: bool):
        """Set training mode.

        Args:
            training: Whether in training mode
        """
        self.training = training


class GradientClipper:
    """Gradient clipping to prevent exploding gradients.

    Clips gradients to a maximum norm or value.
    """

    def __init__(
        self,
        clip_type: str = "norm",  # "norm" or "value"
        clip_value: float = 1.0,
    ):
        """Initialize gradient clipper.

        Args:
            clip_type: Type of clipping ("norm" or "value")
            clip_value: Maximum gradient norm or value
        """
        self.clip_type = clip_type
        self.clip_value = clip_value

    def clip(self, gradients: List[float]) -> List[float]:
        """Clip gradients.

        Args:
            gradients: Gradient values

        Returns:
            Clipped gradients
        """
        if self.clip_type == "norm":
            # Clip by global norm
            norm = math.sqrt(sum(g**2 for g in gradients))
            if norm > self.clip_value:
                scale = self.clip_value / (norm + 1e-8)
                return [g * scale for g in gradients]
            return gradients

        elif self.clip_type == "value":
            # Clip by value
            return [max(-self.clip_value, min(self.clip_value, g)) for g in gradients]

        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")


class ValidationSplitter:
    """Utility for splitting data into train/validation/test sets."""

    @staticmethod
    def train_val_split(
        data: List[Any],
        val_ratio: float = 0.2,
        shuffle: bool = True,
    ) -> Tuple[List[Any], List[Any]]:
        """Split data into train and validation sets.

        Args:
            data: Data to split
            val_ratio: Fraction for validation
            shuffle: Whether to shuffle before splitting

        Returns:
            (train_data, val_data)
        """
        if shuffle:
            data = data.copy()
            random.shuffle(data)

        split_idx = int(len(data) * (1 - val_ratio))
        return data[:split_idx], data[split_idx:]

    @staticmethod
    def train_val_test_split(
        data: List[Any],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Split data into train, validation, and test sets.

        Args:
            data: Data to split
            val_ratio: Fraction for validation
            test_ratio: Fraction for test
            shuffle: Whether to shuffle before splitting

        Returns:
            (train_data, val_data, test_data)
        """
        if shuffle:
            data = data.copy()
            random.shuffle(data)

        n = len(data)
        test_idx = int(n * (1 - test_ratio))
        val_idx = int(test_idx * (1 - val_ratio / (1 - test_ratio)))

        return data[:val_idx], data[val_idx:test_idx], data[test_idx:]


class KFoldValidator:
    """K-fold cross-validation for robust model evaluation."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True):
        """Initialize K-fold validator.

        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
        """
        self.n_splits = n_splits
        self.shuffle = shuffle

    def split(self, data: List[Any]) -> List[Tuple[List[Any], List[Any]]]:
        """Generate K-fold splits.

        Args:
            data: Data to split

        Returns:
            List of (train, val) splits
        """
        if self.shuffle:
            data = data.copy()
            random.shuffle(data)

        n = len(data)
        fold_size = n // self.n_splits
        splits = []

        for i in range(self.n_splits):
            # Validation fold
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            val_data = data[val_start:val_end]

            # Training folds
            train_data = data[:val_start] + data[val_end:]

            splits.append((train_data, val_data))

        return splits


class OverfittingDetector:
    """Detect overfitting by monitoring train/val metrics.

    Overfitting indicators:
    - Validation loss increases while train loss decreases
    - Large gap between train and validation accuracy
    - Increasing variance in validation metrics
    """

    def __init__(
        self,
        window_size: int = 10,
        overfitting_threshold: float = 0.1,  # 10% gap
    ):
        """Initialize overfitting detector.

        Args:
            window_size: Number of recent epochs to consider
            overfitting_threshold: Threshold for overfitting detection
        """
        self.window_size = window_size
        self.overfitting_threshold = overfitting_threshold
        self.metrics_history: List[ValidationMetrics] = []

    def add_metrics(self, metrics: ValidationMetrics):
        """Add validation metrics.

        Args:
            metrics: Validation metrics for current epoch
        """
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size * 2:
            self.metrics_history.pop(0)

    def is_overfitting(self) -> bool:
        """Check if model is overfitting.

        Returns:
            True if overfitting detected
        """
        if len(self.metrics_history) < self.window_size:
            return False

        recent_metrics = self.metrics_history[-self.window_size :]

        # Check 1: Validation loss increasing
        val_losses = [m.val_loss for m in recent_metrics]
        if len(val_losses) >= 3:
            # Check if validation loss trend is increasing
            recent_trend = val_losses[-3:]
            if recent_trend[-1] > recent_trend[0]:
                # Validation loss increased

                # Check if train loss decreased
                train_losses = [m.train_loss for m in recent_metrics[-3:]]
                if train_losses[-1] < train_losses[0]:
                    logger.warning(
                        "Overfitting detected: Val loss increasing, train loss decreasing"
                    )
                    return True

        # Check 2: Large train/val gap
        avg_overfit_score = statistics.mean(
            m.overfitting_score() for m in recent_metrics
        )
        if avg_overfit_score > self.overfitting_threshold:
            logger.warning(
                f"Overfitting detected: Train/val gap = {avg_overfit_score:.2%}"
            )
            return True

        # Check 3: High variance in validation loss
        if len(val_losses) >= 5:
            val_std = statistics.stdev(val_losses)
            val_mean = statistics.mean(val_losses)
            if val_mean > 0 and val_std / val_mean > 0.2:  # CV > 20%
                logger.warning(
                    f"High variance in validation loss: CV = {val_std/val_mean:.2%}"
                )

        return False

    def get_overfitting_score(self) -> float:
        """Get current overfitting score.

        Returns:
            Average overfitting score over recent epochs
        """
        if not self.metrics_history:
            return 0.0

        recent = self.metrics_history[-self.window_size :]
        return statistics.mean(m.overfitting_score() for m in recent)


class RegularizationCoordinator:
    """Coordinates regularization and validation for the learning system.

    Main interface for managing overfitting prevention.
    """

    def __init__(
        self,
        l2_lambda: float = 0.01,
        dropout_rate: float = 0.3,
        gradient_clip_value: float = 1.0,
        early_stopping_patience: int = 15,
        validation_split: float = 0.2,
    ):
        """Initialize regularization coordinator.

        Args:
            l2_lambda: L2 regularization strength
            dropout_rate: Dropout rate
            gradient_clip_value: Maximum gradient norm
            early_stopping_patience: Patience for early stopping
            validation_split: Fraction of data for validation
        """
        # Regularizers
        self.l2_regularizer = L2Regularizer(lambda_=l2_lambda)
        self.dropout = DropoutRegularizer(dropout_rate=dropout_rate)
        self.gradient_clipper = GradientClipper(
            clip_type="norm", clip_value=gradient_clip_value
        )

        # Validation
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.overfitting_detector = OverfittingDetector()
        self.validation_split = validation_split

        # Metrics tracking
        self.metrics_history: List[ValidationMetrics] = []

    def apply_regularization(
        self,
        weights: List[float],
        gradients: List[float],
        values: Optional[List[float]] = None,
        training: bool = True,
    ) -> Tuple[float, List[float], Optional[List[float]]]:
        """Apply all regularization techniques.

        Args:
            weights: Model weights
            gradients: Computed gradients
            values: Layer values (for dropout)
            training: Whether in training mode

        Returns:
            (penalty, regularized_gradients, dropout_values)
        """
        # L2 penalty
        penalty = self.l2_regularizer.compute_penalty(weights)
        reg_gradients = self.l2_regularizer.compute_gradient(weights)

        # Add regularization to gradients
        regularized_gradients = [g + rg for g, rg in zip(gradients, reg_gradients)]

        # Clip gradients
        clipped_gradients = self.gradient_clipper.clip(regularized_gradients)

        # Apply dropout if values provided
        dropout_values = None
        if values is not None:
            self.dropout.set_training(training)
            dropout_values = self.dropout.apply(values)

        return penalty, clipped_gradients, dropout_values

    def update_validation_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        model_weights: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update validation metrics and check for early stopping.

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            train_accuracy: Training accuracy
            val_accuracy: Validation accuracy
            model_weights: Current model weights

        Returns:
            True if should stop training
        """
        metrics = ValidationMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
        )

        self.metrics_history.append(metrics)
        self.overfitting_detector.add_metrics(metrics)

        # Check early stopping
        should_stop = self.early_stopping(val_loss, epoch, model_weights)

        # Log overfitting warnings
        if self.overfitting_detector.is_overfitting():
            logger.warning(
                f"Overfitting detected at epoch {epoch}. "
                f"Consider increasing regularization or stopping training."
            )

        return should_stop

    def get_statistics(self) -> Dict[str, Any]:
        """Get regularization and validation statistics.

        Returns:
            Dictionary with metrics
        """
        if not self.metrics_history:
            return {
                "total_epochs": 0,
                "best_val_loss": None,
                "overfitting_score": 0.0,
                "early_stopped": False,
            }

        recent_metrics = self.metrics_history[-10:]

        return {
            "total_epochs": len(self.metrics_history),
            "best_val_loss": self.early_stopping.best_loss,
            "early_stopped": self.early_stopping.should_stop,
            "stopped_epoch": self.early_stopping.stopped_epoch,
            "current_val_loss": self.metrics_history[-1].val_loss,
            "current_train_loss": self.metrics_history[-1].train_loss,
            "overfitting_score": self.overfitting_detector.get_overfitting_score(),
            "avg_recent_val_loss": statistics.mean(m.val_loss for m in recent_metrics),
            "l2_lambda": self.l2_regularizer.lambda_,
            "dropout_rate": self.dropout.dropout_rate,
            "gradient_clip_value": self.gradient_clipper.clip_value,
        }

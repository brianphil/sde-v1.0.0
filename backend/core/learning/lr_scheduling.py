"""Learning rate scheduling for adaptive optimization.

This module implements world-class learning rate scheduling strategies:

- Step decay: Reduce LR at fixed intervals
- Exponential decay: Smooth exponential reduction
- Cosine annealing: Cosine-based smooth decay
- Warm restarts (SGDR): Periodic LR restarts
- Reduce on plateau: Reduce when loss plateaus
- Cyclic learning rates: Oscillate between bounds
- 1-cycle policy: Single cycle with warmup and decay
- Warmup scheduling: Gradual increase then decay
"""

from typing import Optional, Callable, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import logging
from collections import deque

logger = logging.getLogger(__name__)


class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, initial_lr: float):
        """Initialize scheduler.

        Args:
            initial_lr: Initial learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.step_count = 0

    @abstractmethod
    def step(self, **kwargs) -> float:
        """Update learning rate and return new value.

        Args:
            **kwargs: Scheduler-specific parameters

        Returns:
            New learning rate
        """
        pass

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_lr = self.initial_lr
        self.step_count = 0


class StepLR(LRScheduler):
    """Step decay: reduce LR by factor every N steps.

    LR schedule:
        lr = initial_lr * gamma^(step // step_size)
    """

    def __init__(
        self,
        initial_lr: float,
        step_size: int = 100,
        gamma: float = 0.1,
    ):
        """Initialize step decay scheduler.

        Args:
            initial_lr: Initial learning rate
            step_size: Number of steps before decay
            gamma: Multiplicative decay factor
        """
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, **kwargs) -> float:
        """Update learning rate."""
        self.step_count += 1

        # Decay at step intervals
        if self.step_count % self.step_size == 0:
            self.current_lr *= self.gamma
            logger.info(
                f"StepLR: Reduced learning rate to {self.current_lr:.6f} "
                f"at step {self.step_count}"
            )

        return self.current_lr


class ExponentialLR(LRScheduler):
    """Exponential decay: smooth exponential reduction.

    LR schedule:
        lr = initial_lr * gamma^step
    """

    def __init__(
        self,
        initial_lr: float,
        gamma: float = 0.99,
        min_lr: float = 1e-6,
    ):
        """Initialize exponential decay scheduler.

        Args:
            initial_lr: Initial learning rate
            gamma: Decay rate (< 1.0)
            min_lr: Minimum learning rate
        """
        super().__init__(initial_lr)
        self.gamma = gamma
        self.min_lr = min_lr

    def step(self, **kwargs) -> float:
        """Update learning rate."""
        self.step_count += 1
        self.current_lr = max(self.min_lr, self.initial_lr * (self.gamma ** self.step_count))
        return self.current_lr


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing: smooth cosine-based decay.

    LR schedule:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * step / T))

    Provides smooth decay with slow start and end.
    """

    def __init__(
        self,
        initial_lr: float,
        T_max: int = 100,
        min_lr: float = 0.0,
    ):
        """Initialize cosine annealing scheduler.

        Args:
            initial_lr: Maximum learning rate
            T_max: Number of steps for one cosine cycle
            min_lr: Minimum learning rate
        """
        super().__init__(initial_lr)
        self.T_max = T_max
        self.min_lr = min_lr

    def step(self, **kwargs) -> float:
        """Update learning rate."""
        self.step_count += 1

        # Cosine annealing formula
        cosine_term = math.cos(math.pi * (self.step_count % self.T_max) / self.T_max)
        self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + cosine_term)

        return self.current_lr


class CosineAnnealingWarmRestarts(LRScheduler):
    """Cosine annealing with warm restarts (SGDR).

    Periodically resets learning rate to initial value.
    Helps escape local minima and find better solutions.

    Reference: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017)
    """

    def __init__(
        self,
        initial_lr: float,
        T_0: int = 100,
        T_mult: int = 2,
        min_lr: float = 0.0,
    ):
        """Initialize SGDR scheduler.

        Args:
            initial_lr: Maximum learning rate
            T_0: Number of steps until first restart
            T_mult: Factor to increase period after each restart
            min_lr: Minimum learning rate
        """
        super().__init__(initial_lr)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.min_lr = min_lr

        self.T_cur = 0  # Steps since last restart
        self.T_i = T_0  # Current period length
        self.restart_count = 0

    def step(self, **kwargs) -> float:
        """Update learning rate."""
        self.step_count += 1
        self.T_cur += 1

        # Check if restart needed
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
            self.restart_count += 1
            logger.info(f"SGDR: Warm restart #{self.restart_count} at step {self.step_count}")

        # Cosine annealing within current period
        cosine_term = math.cos(math.pi * self.T_cur / self.T_i)
        self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + cosine_term)

        return self.current_lr


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when metric plateaus.

    Monitors a metric (e.g., validation loss) and reduces LR
    when no improvement is observed for patience epochs.
    """

    def __init__(
        self,
        initial_lr: float,
        mode: str = "min",  # "min" or "max"
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 1e-6,
    ):
        """Initialize plateau scheduler.

        Args:
            initial_lr: Initial learning rate
            mode: "min" to minimize metric, "max" to maximize
            factor: Factor to reduce LR by
            patience: Number of steps with no improvement before reducing
            threshold: Minimum change to qualify as improvement
            min_lr: Minimum learning rate
        """
        super().__init__(initial_lr)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.num_bad_steps = 0

    def step(self, metric: float, **kwargs) -> float:
        """Update learning rate based on metric.

        Args:
            metric: Metric to monitor (e.g., validation loss)

        Returns:
            New learning rate
        """
        self.step_count += 1

        # Check if improvement
        is_improvement = False
        if self.mode == "min":
            if metric < self.best_metric - self.threshold:
                is_improvement = True
        else:
            if metric > self.best_metric + self.threshold:
                is_improvement = True

        if is_improvement:
            self.best_metric = metric
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1

        # Reduce LR if plateau detected
        if self.num_bad_steps >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.min_lr, self.current_lr * self.factor)

            if self.current_lr < old_lr:
                logger.info(
                    f"ReduceLROnPlateau: Reduced learning rate from {old_lr:.6f} to {self.current_lr:.6f} "
                    f"(metric={metric:.6f}, best={self.best_metric:.6f})"
                )

            self.num_bad_steps = 0

        return self.current_lr


class CyclicLR(LRScheduler):
    """Cyclic learning rates: oscillate between min and max LR.

    Cycles learning rate between lower and upper bounds with
    triangular, triangular2, or exponential policies.

    Reference: Smith, "Cyclical Learning Rates for Training Neural Networks" (2017)
    """

    def __init__(
        self,
        initial_lr: float,
        max_lr: float,
        step_size_up: int = 100,
        step_size_down: Optional[int] = None,
        mode: str = "triangular",  # "triangular", "triangular2", "exp_range"
        gamma: float = 0.99,  # For exp_range mode
    ):
        """Initialize cyclic LR scheduler.

        Args:
            initial_lr: Lower bound for learning rate
            max_lr: Upper bound for learning rate
            step_size_up: Steps in increasing half of cycle
            step_size_down: Steps in decreasing half (defaults to step_size_up)
            mode: Cyclic policy
            gamma: Decay factor for exp_range mode
        """
        super().__init__(initial_lr)
        self.base_lr = initial_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma

        self.cycle_count = 0

    def step(self, **kwargs) -> float:
        """Update learning rate."""
        self.step_count += 1

        # Calculate cycle position
        cycle = math.floor(1 + self.step_count / (self.step_size_up + self.step_size_down))
        x = abs(self.step_count / self.step_size_up - 2 * cycle + 1)

        # Calculate LR based on mode
        if self.mode == "triangular":
            scale_factor = 1.0
        elif self.mode == "triangular2":
            scale_factor = 1.0 / (2.0 ** (cycle - 1))
        elif self.mode == "exp_range":
            scale_factor = self.gamma ** self.step_count
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Linear interpolation between base_lr and max_lr
        self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale_factor

        # Track cycle changes
        if cycle != self.cycle_count:
            self.cycle_count = cycle
            logger.debug(f"CyclicLR: Starting cycle {cycle}")

        return self.current_lr


class OneCycleLR(LRScheduler):
    """1-cycle learning rate policy.

    Single cycle with warmup to max_lr then decay to min_lr.
    Effective for fast training convergence.

    Reference: Smith & Topin, "Super-Convergence" (2018)
    """

    def __init__(
        self,
        initial_lr: float,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,  # Percentage of cycle spent increasing LR
        div_factor: float = 25.0,  # initial_lr = max_lr / div_factor
        final_div_factor: float = 1e4,  # final_lr = initial_lr / final_div_factor
    ):
        """Initialize 1cycle scheduler.

        Args:
            initial_lr: Starting learning rate
            max_lr: Peak learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of cycle for warmup
            div_factor: How much smaller initial_lr is compared to max_lr
            final_div_factor: How much smaller final_lr is compared to initial_lr
        """
        super().__init__(initial_lr)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start

        self.start_lr = max_lr / div_factor
        self.final_lr = self.start_lr / final_div_factor

        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up

    def step(self, **kwargs) -> float:
        """Update learning rate."""
        self.step_count += 1

        if self.step_count <= self.step_size_up:
            # Warmup phase: increase from start_lr to max_lr
            pct = self.step_count / self.step_size_up
            self.current_lr = self.start_lr + (self.max_lr - self.start_lr) * pct

        else:
            # Decay phase: decrease from max_lr to final_lr using cosine
            pct = (self.step_count - self.step_size_up) / self.step_size_down
            cosine_term = math.cos(math.pi * pct)
            self.current_lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + cosine_term)

        return self.current_lr


class WarmupScheduler(LRScheduler):
    """Warmup scheduler: gradual warmup then apply another scheduler.

    Combines linear warmup with another scheduling strategy.
    """

    def __init__(
        self,
        base_scheduler: LRScheduler,
        warmup_steps: int = 1000,
        warmup_start_lr: float = 1e-6,
    ):
        """Initialize warmup scheduler.

        Args:
            base_scheduler: Scheduler to use after warmup
            warmup_steps: Number of warmup steps
            warmup_start_lr: Starting learning rate for warmup
        """
        super().__init__(base_scheduler.initial_lr)
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

    def step(self, **kwargs) -> float:
        """Update learning rate."""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Linear warmup
            pct = self.step_count / self.warmup_steps
            self.current_lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * pct

        else:
            # Use base scheduler
            self.current_lr = self.base_scheduler.step(**kwargs)

        return self.current_lr


class LRSchedulerCoordinator:
    """Coordinates learning rate scheduling for the learning system.

    Main interface for managing adaptive learning rates.
    """

    def __init__(
        self,
        initial_lr: float = 0.001,
        scheduler_type: str = "cosine_warmup",  # "step", "exponential", "cosine", "plateau", "cyclic", "1cycle", "cosine_warmup"
        **scheduler_kwargs,
    ):
        """Initialize LR scheduler coordinator.

        Args:
            initial_lr: Initial learning rate
            scheduler_type: Type of scheduler to use
            **scheduler_kwargs: Additional scheduler parameters
        """
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type

        # Create scheduler
        if scheduler_type == "step":
            self.scheduler = StepLR(initial_lr, **scheduler_kwargs)
        elif scheduler_type == "exponential":
            self.scheduler = ExponentialLR(initial_lr, **scheduler_kwargs)
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(initial_lr, **scheduler_kwargs)
        elif scheduler_type == "sgdr":
            self.scheduler = CosineAnnealingWarmRestarts(initial_lr, **scheduler_kwargs)
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(initial_lr, **scheduler_kwargs)
        elif scheduler_type == "cyclic":
            max_lr = scheduler_kwargs.pop("max_lr", initial_lr * 10)
            self.scheduler = CyclicLR(initial_lr, max_lr, **scheduler_kwargs)
        elif scheduler_type == "1cycle":
            max_lr = scheduler_kwargs.pop("max_lr", initial_lr * 10)
            total_steps = scheduler_kwargs.pop("total_steps", 1000)
            self.scheduler = OneCycleLR(initial_lr, max_lr, total_steps, **scheduler_kwargs)
        elif scheduler_type == "cosine_warmup":
            # Default: cosine annealing with warmup
            base = CosineAnnealingLR(initial_lr, **scheduler_kwargs)
            self.scheduler = WarmupScheduler(base, warmup_steps=100)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        # Tracking
        self.lr_history: List[float] = []

    def step(self, metric: Optional[float] = None) -> float:
        """Update learning rate.

        Args:
            metric: Optional metric for plateau-based scheduling

        Returns:
            New learning rate
        """
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires metric parameter")
            new_lr = self.scheduler.step(metric=metric)
        else:
            new_lr = self.scheduler.step()

        self.lr_history.append(new_lr)

        return new_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.scheduler.get_lr()

    def reset(self):
        """Reset scheduler."""
        self.scheduler.reset()
        self.lr_history.clear()

    def get_statistics(self) -> dict:
        """Get scheduler statistics."""
        return {
            "scheduler_type": self.scheduler_type,
            "initial_lr": self.initial_lr,
            "current_lr": self.get_lr(),
            "step_count": self.scheduler.step_count,
            "lr_history_length": len(self.lr_history),
            "min_lr_seen": min(self.lr_history) if self.lr_history else self.initial_lr,
            "max_lr_seen": max(self.lr_history) if self.lr_history else self.initial_lr,
        }

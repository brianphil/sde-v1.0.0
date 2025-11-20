"""Exploration strategies for balancing exploration-exploitation tradeoff.

This module implements sde exploration algorithms for reinforcement learning:

- Epsilon-greedy (ε-greedy): Simple random exploration
- Upper Confidence Bound (UCB): Optimistic exploration with confidence bounds
- Boltzmann exploration: Temperature-based probabilistic selection
- Thompson sampling: Bayesian posterior sampling
- Adaptive exploration: Decay schedules and state-dependent exploration
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import random
import math
import logging
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ActionStats:
    """Tracks statistics for an action to support exploration strategies."""

    action_id: str
    times_selected: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    reward_variance: float = 0.0
    last_selected: Optional[datetime] = None

    # For Thompson sampling
    success_count: int = 0
    failure_count: int = 0

    def update(self, reward: float, success: bool = True):
        """Update action statistics with new outcome.

        Args:
            reward: Reward received
            success: Whether action was successful
        """
        self.times_selected += 1
        self.last_selected = datetime.now()

        # Update reward statistics
        old_avg = self.avg_reward
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.times_selected

        # Update variance (Welford's online algorithm)
        if self.times_selected > 1:
            delta = reward - old_avg
            self.reward_variance = (
                (self.times_selected - 2) * self.reward_variance
                + delta * (reward - self.avg_reward)
            ) / (self.times_selected - 1)

        # Update success/failure for Thompson sampling
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    @abstractmethod
    def select_action(
        self,
        actions: List[str],
        action_values: Dict[str, float],
        action_stats: Dict[str, ActionStats],
        timestep: int,
    ) -> str:
        """Select an action using this exploration strategy.

        Args:
            actions: List of available actions
            action_values: Estimated value for each action
            action_stats: Statistics for each action
            timestep: Current timestep (for time-dependent strategies)

        Returns:
            Selected action ID
        """
        pass

    @abstractmethod
    def get_exploration_rate(self, timestep: int) -> float:
        """Get current exploration rate.

        Args:
            timestep: Current timestep

        Returns:
            Exploration rate (0-1)
        """
        pass


class EpsilonGreedy(ExplorationStrategy):
    """Epsilon-greedy exploration strategy.

    With probability ε, select a random action (explore).
    With probability 1-ε, select the best action (exploit).

    Supports adaptive epsilon decay over time.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        decay_type: str = "exponential",  # "exponential", "linear", "inverse"
    ):
        """Initialize epsilon-greedy strategy.

        Args:
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate (< 1.0)
            decay_type: Type of decay schedule
        """
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type

    def select_action(
        self,
        actions: List[str],
        action_values: Dict[str, float],
        action_stats: Dict[str, ActionStats],
        timestep: int,
    ) -> str:
        """Select action using ε-greedy policy."""
        if not actions:
            raise ValueError("No actions available")

        # Update epsilon based on timestep
        self._update_epsilon(timestep)

        # Explore with probability epsilon
        if random.random() < self.epsilon:
            action = random.choice(actions)
            logger.debug(
                f"Exploring: randomly selected action '{action}' (ε={self.epsilon:.3f})"
            )
            return action

        # Exploit: choose action with highest value
        best_action = max(actions, key=lambda a: action_values.get(a, 0.0))
        logger.debug(
            f"Exploiting: selected best action '{best_action}' (value={action_values.get(best_action, 0.0):.3f})"
        )
        return best_action

    def _update_epsilon(self, timestep: int):
        """Update epsilon based on decay schedule."""
        if self.decay_type == "exponential":
            # Exponential decay: ε_t = ε_0 * decay^t
            self.epsilon = max(
                self.epsilon_min, self.epsilon_init * (self.epsilon_decay**timestep)
            )
        elif self.decay_type == "linear":
            # Linear decay: ε_t = ε_0 - (ε_0 - ε_min) * t / T
            decay_steps = 10000  # Total steps to reach epsilon_min
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_init
                - (self.epsilon_init - self.epsilon_min) * timestep / decay_steps,
            )
        elif self.decay_type == "inverse":
            # Inverse decay: ε_t = ε_min + (ε_0 - ε_min) / (1 + t)
            self.epsilon = self.epsilon_min + (self.epsilon_init - self.epsilon_min) / (
                1 + timestep
            )

    def get_exploration_rate(self, timestep: int) -> float:
        """Get current exploration rate."""
        self._update_epsilon(timestep)
        return self.epsilon

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_init


class UpperConfidenceBound(ExplorationStrategy):
    """Upper Confidence Bound (UCB) exploration.

    Selects actions based on upper confidence bound:
        UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))

    where:
        - Q(a) is the estimated value of action a
        - c is the exploration constant (typically √2)
        - t is the total number of timesteps
        - N(a) is the number of times action a was selected

    This provides optimistic estimates in the face of uncertainty,
    encouraging exploration of less-tried actions.
    """

    def __init__(self, c: float = 1.414):  # √2
        """Initialize UCB strategy.

        Args:
            c: Exploration constant (higher = more exploration)
        """
        self.c = c

    def select_action(
        self,
        actions: List[str],
        action_values: Dict[str, float],
        action_stats: Dict[str, ActionStats],
        timestep: int,
    ) -> str:
        """Select action using UCB policy."""
        if not actions:
            raise ValueError("No actions available")

        # If any action hasn't been tried, try it
        untried_actions = [
            a
            for a in actions
            if action_stats.get(a, ActionStats(a)).times_selected == 0
        ]
        if untried_actions:
            action = random.choice(untried_actions)
            logger.debug(f"UCB: Trying untried action '{action}'")
            return action

        # Calculate UCB for each action
        ucb_values = {}
        total_timesteps = max(1, timestep)

        for action in actions:
            stats = action_stats.get(action, ActionStats(action))
            n_action = max(1, stats.times_selected)  # Avoid division by zero

            # UCB formula
            exploitation = action_values.get(action, 0.0)
            exploration_bonus = self.c * math.sqrt(math.log(total_timesteps) / n_action)

            ucb_values[action] = exploitation + exploration_bonus

        # Select action with highest UCB
        best_action = max(actions, key=lambda a: ucb_values[a])

        logger.debug(
            f"UCB: Selected '{best_action}' "
            f"(value={action_values.get(best_action, 0.0):.3f}, "
            f"UCB={ucb_values[best_action]:.3f})"
        )

        return best_action

    def get_exploration_rate(self, timestep: int) -> float:
        """UCB doesn't have a fixed exploration rate."""
        # Effective exploration rate decreases with timestep
        if timestep == 0:
            return 1.0
        return min(1.0, self.c * math.sqrt(math.log(timestep) / timestep))


class BoltzmannExploration(ExplorationStrategy):
    """Boltzmann (Softmax) exploration strategy.

    Selects actions probabilistically based on their values:
        P(a) = exp(Q(a) / τ) / Σ_b exp(Q(b) / τ)

    where τ is the temperature parameter:
        - High τ → more exploration (uniform selection)
        - Low τ → more exploitation (greedy selection)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        temperature_min: float = 0.01,
        temperature_decay: float = 0.99,
    ):
        """Initialize Boltzmann exploration.

        Args:
            temperature: Initial temperature
            temperature_min: Minimum temperature
            temperature_decay: Temperature decay rate
        """
        self.temperature_init = temperature
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay

    def select_action(
        self,
        actions: List[str],
        action_values: Dict[str, float],
        action_stats: Dict[str, ActionStats],
        timestep: int,
    ) -> str:
        """Select action using Boltzmann policy."""
        if not actions:
            raise ValueError("No actions available")

        # Update temperature
        self._update_temperature(timestep)

        # Calculate softmax probabilities
        values = [action_values.get(a, 0.0) for a in actions]

        # Scale by temperature
        scaled_values = [v / self.temperature for v in values]

        # Numerical stability: subtract max before exp
        max_value = max(scaled_values)
        exp_values = [math.exp(v - max_value) for v in scaled_values]

        # Normalize to get probabilities
        total = sum(exp_values)
        probabilities = [exp_v / total for exp_v in exp_values]

        # Sample action according to probabilities
        action = random.choices(actions, weights=probabilities)[0]

        logger.debug(
            f"Boltzmann: Selected '{action}' "
            f"(prob={probabilities[actions.index(action)]:.3f}, τ={self.temperature:.3f})"
        )

        return action

    def _update_temperature(self, timestep: int):
        """Decay temperature over time."""
        self.temperature = max(
            self.temperature_min,
            self.temperature_init * (self.temperature_decay**timestep),
        )

    def get_exploration_rate(self, timestep: int) -> float:
        """Get effective exploration rate based on temperature."""
        self._update_temperature(timestep)
        # Normalized temperature as exploration rate
        return min(1.0, self.temperature / self.temperature_init)

    def reset(self):
        """Reset temperature to initial value."""
        self.temperature = self.temperature_init


class ThompsonSampling(ExplorationStrategy):
    """Thompson Sampling (Posterior Sampling) exploration.

    Bayesian approach: maintain belief distribution over action values,
    sample from posterior, and select action with highest sampled value.

    Uses Beta distribution for binary rewards:
        Beta(α, β) where α = successes + 1, β = failures + 1
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Initialize Thompson Sampling.

        Args:
            prior_alpha: Prior success count (optimistic prior)
            prior_beta: Prior failure count
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select_action(
        self,
        actions: List[str],
        action_values: Dict[str, float],
        action_stats: Dict[str, ActionStats],
        timestep: int,
    ) -> str:
        """Select action using Thompson Sampling."""
        if not actions:
            raise ValueError("No actions available")

        # Sample from posterior for each action
        sampled_values = {}

        for action in actions:
            stats = action_stats.get(action, ActionStats(action))

            # Beta parameters
            alpha = self.prior_alpha + stats.success_count
            beta = self.prior_beta + stats.failure_count

            # Sample from Beta(alpha, beta)
            sampled_value = random.betavariate(alpha, beta)
            sampled_values[action] = sampled_value

        # Select action with highest sampled value
        best_action = max(actions, key=lambda a: sampled_values[a])

        logger.debug(
            f"Thompson: Selected '{best_action}' "
            f"(sampled={sampled_values[best_action]:.3f})"
        )

        return best_action

    def get_exploration_rate(self, timestep: int) -> float:
        """Thompson sampling doesn't have a fixed exploration rate."""
        # Exploration decreases as uncertainty decreases
        # This is a rough estimate
        return 1.0 / (1.0 + timestep / 100.0)


class AdaptiveExploration(ExplorationStrategy):
    """Adaptive exploration that switches between strategies based on performance.

    Combines multiple exploration strategies and adapts based on:
    - Learning progress
    - Reward variance
    - State novelty
    """

    def __init__(
        self,
        strategies: Optional[List[ExplorationStrategy]] = None,
        performance_window: int = 100,
    ):
        """Initialize adaptive exploration.

        Args:
            strategies: List of strategies to choose from
            performance_window: Window for tracking performance
        """
        if strategies is None:
            # Default strategies
            strategies = [
                EpsilonGreedy(epsilon=0.1),
                UpperConfidenceBound(c=1.414),
                BoltzmannExploration(temperature=1.0),
            ]

        self.strategies = strategies
        self.strategy_names = [type(s).__name__ for s in strategies]
        self.current_strategy_idx = 0
        self.performance_window = performance_window

        # Track performance of each strategy
        self.strategy_stats: Dict[str, ActionStats] = {
            name: ActionStats(name) for name in self.strategy_names
        }

        # Recent rewards for variance calculation
        self.recent_rewards: List[float] = []

    def select_action(
        self,
        actions: List[str],
        action_values: Dict[str, float],
        action_stats: Dict[str, ActionStats],
        timestep: int,
    ) -> str:
        """Select action using adaptive strategy selection."""
        # Select best-performing strategy periodically
        if timestep > 0 and timestep % 100 == 0:
            self._update_strategy_selection()

        # Use current strategy
        current_strategy = self.strategies[self.current_strategy_idx]
        action = current_strategy.select_action(
            actions, action_values, action_stats, timestep
        )

        logger.debug(
            f"Adaptive: Using {self.strategy_names[self.current_strategy_idx]} "
            f"to select '{action}'"
        )

        return action

    def update_strategy_performance(self, reward: float, success: bool = True):
        """Update performance tracking for current strategy.

        Args:
            reward: Reward received
            success: Whether action was successful
        """
        strategy_name = self.strategy_names[self.current_strategy_idx]
        self.strategy_stats[strategy_name].update(reward, success)

        # Track recent rewards
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.performance_window:
            self.recent_rewards.pop(0)

    def _update_strategy_selection(self):
        """Update which strategy to use based on performance."""
        # Choose strategy with highest average reward
        best_strategy_idx = 0
        best_avg_reward = -float("inf")

        for idx, name in enumerate(self.strategy_names):
            stats = self.strategy_stats[name]
            if stats.times_selected > 0:
                if stats.avg_reward > best_avg_reward:
                    best_avg_reward = stats.avg_reward
                    best_strategy_idx = idx

        self.current_strategy_idx = best_strategy_idx
        logger.info(
            f"Switched to {self.strategy_names[best_strategy_idx]} "
            f"(avg_reward={best_avg_reward:.3f})"
        )

    def get_exploration_rate(self, timestep: int) -> float:
        """Get exploration rate of current strategy."""
        return self.strategies[self.current_strategy_idx].get_exploration_rate(timestep)


class ExplorationCoordinator:
    """Coordinates exploration across the learning system.

    Main interface for managing exploration-exploitation tradeoff.
    """

    def __init__(
        self,
        strategy: Optional[ExplorationStrategy] = None,
        track_statistics: bool = True,
    ):
        """Initialize exploration coordinator.

        Args:
            strategy: Exploration strategy to use
            track_statistics: Whether to track action statistics
        """
        if strategy is None:
            # Default: adaptive exploration with multiple strategies
            strategy = AdaptiveExploration()

        self.strategy = strategy
        self.track_statistics = track_statistics

        # Action tracking
        self.action_stats: Dict[str, ActionStats] = {}
        self.timestep = 0

        # Metrics
        self.total_explorations = 0
        self.total_exploitations = 0

    def select_action(
        self,
        actions: List[str],
        action_values: Dict[str, float],
    ) -> str:
        """Select an action using the exploration strategy.

        Args:
            actions: Available actions
            action_values: Estimated value for each action

        Returns:
            Selected action
        """
        if not actions:
            raise ValueError("No actions available")

        # Ensure all actions have stats
        for action in actions:
            if action not in self.action_stats:
                self.action_stats[action] = ActionStats(action)

        # Select action using strategy
        selected_action = self.strategy.select_action(
            actions=actions,
            action_values=action_values,
            action_stats=self.action_stats,
            timestep=self.timestep,
        )

        # Track whether this was exploration or exploitation
        best_action = max(actions, key=lambda a: action_values.get(a, 0.0))
        if selected_action == best_action:
            self.total_exploitations += 1
        else:
            self.total_explorations += 1

        self.timestep += 1

        return selected_action

    def update_action_outcome(
        self,
        action: str,
        reward: float,
        success: bool = True,
    ):
        """Update action statistics with outcome.

        Args:
            action: Action that was taken
            reward: Reward received
            success: Whether action was successful
        """
        if action not in self.action_stats:
            self.action_stats[action] = ActionStats(action)

        self.action_stats[action].update(reward, success)

        # Update strategy performance if adaptive
        if isinstance(self.strategy, AdaptiveExploration):
            self.strategy.update_strategy_performance(reward, success)

    def get_exploration_rate(self) -> float:
        """Get current exploration rate."""
        return self.strategy.get_exploration_rate(self.timestep)

    def get_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics.

        Returns:
            Dictionary with exploration metrics
        """
        total_selections = self.total_explorations + self.total_exploitations

        return {
            "timestep": self.timestep,
            "exploration_rate": self.get_exploration_rate(),
            "total_explorations": self.total_explorations,
            "total_exploitations": self.total_exploitations,
            "exploration_ratio": (
                self.total_explorations / total_selections
                if total_selections > 0
                else 0.0
            ),
            "strategy": type(self.strategy).__name__,
            "action_count": len(self.action_stats),
            "actions": {
                action: {
                    "times_selected": stats.times_selected,
                    "avg_reward": stats.avg_reward,
                    "reward_variance": stats.reward_variance,
                }
                for action, stats in self.action_stats.items()
            },
        }

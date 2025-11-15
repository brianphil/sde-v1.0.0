"""Temporal Difference Learning implementation for VFA training."""

from typing import Optional, Tuple, List
import logging

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # Define torch as None for graceful fallback
    nn = None  # Define nn as None for graceful fallback

logger = logging.getLogger(__name__)


class TemporalDifferenceLearner:
    """Temporal Difference Learning for value function approximation.
    
    TD Learning enables the value function to learn from bootstrapped estimates
    of future value, rather than waiting for complete episode returns.
    
    Core Update Rule:
        V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
        
    where:
        α = learning rate (0.01 typical)
        r = immediate reward
        γ = discount factor (0.95 typical, weights future value)
        V(s) = current state value estimate
        V(s') = next state value estimate
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        gamma: float = 0.95,
        lambda_param: float = 0.0,
    ):
        """Initialize TD learner.
        
        Args:
            learning_rate: α, controls step size (0-1)
            gamma: γ, discount factor (0-1), weights future value
            lambda_param: λ, eligibility trace decay (0=TD(0), 1=Monte Carlo)
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_param = lambda_param

        self.update_count = 0
        self.total_td_error = 0.0

    def compute_td_target(
        self, reward: float, next_value: float, terminal: bool = False
    ) -> float:
        """Compute TD target for value update.
        
        TD target is the bootstrapped return estimate:
            target = r + γ * V(s') if not terminal
            target = r if terminal
        
        Args:
            reward: Immediate reward r
            next_value: Value estimate V(s')
            terminal: Whether next state is terminal
            
        Returns:
            TD target value
        """

        if terminal:
            return reward
        else:
            return reward + self.gamma * next_value

    def compute_td_error(self, current_value: float, td_target: float) -> float:
        """Compute TD error (temporal difference).
        
        TD error measures surprise: how much did prediction miss?
            TD error = r + γ * V(s') - V(s)
        
        Large error → need to update more
        Small error → close to correct
        
        Returns:
            TD error (can be positive or negative)
        """

        td_error = td_target - current_value
        return td_error

    def update_value(self, current_value: float, td_error: float) -> float:
        """Update value estimate using TD error.
        
        New value = old_value + α * TD_error
        
        This moves the value estimate toward the TD target,
        weighted by learning rate.
        """

        new_value = current_value + self.learning_rate * td_error
        return new_value

    def td_learning_step(
        self,
        current_value: float,
        reward: float,
        next_value: float,
        terminal: bool = False,
    ) -> Tuple[float, float]:
        """Execute single TD learning step.
        
        Args:
            current_value: Current V(s)
            reward: Immediate reward r
            next_value: Next state value V(s')
            terminal: Is next state terminal?
            
        Returns:
            (new_value, td_error)
        """

        # Step 1: Compute TD target
        td_target = self.compute_td_target(reward, next_value, terminal)

        # Step 2: Compute TD error
        td_error = self.compute_td_error(current_value, td_target)

        # Step 3: Update value
        new_value = self.update_value(current_value, td_error)

        # Track metrics
        self.update_count += 1
        self.total_td_error += abs(td_error)

        return new_value, td_error

    def td_lambda_eligibility_trace(
        self, eligibility_trace: dict, decay: float = 0.99
    ) -> dict:
        """Update eligibility traces for TD(λ).
        
        Eligibility traces track which states contributed to current TD error.
        Used for credit assignment across multiple steps.
        
        Args:
            eligibility_trace: Current trace values by state
            decay: Decay rate (typically 0.99)
            
        Returns:
            Updated eligibility traces
        """

        for state_id in eligibility_trace:
            eligibility_trace[state_id] *= decay * self.lambda_param

        return eligibility_trace

    def get_learning_metrics(self) -> dict:
        """Get learning statistics."""

        avg_td_error = (
            self.total_td_error / self.update_count if self.update_count > 0 else 0.0
        )

        return {
            "update_count": self.update_count,
            "total_td_error": self.total_td_error,
            "avg_td_error": avg_td_error,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
        }


class NeuralNetworkTDLearner:
    """TD Learning specifically for neural network value function.
    
    Bridges TemporalDifferenceLearner with PyTorch neural networks.
    Handles forward passes, loss computation, and parameter updates.
    """

    def __init__(self, network: Optional["nn.Module"] = None, learning_rate: float = 0.01):
        """Initialize NN TD learner.
        
        Args:
            network: PyTorch neural network
            learning_rate: Adam optimizer learning rate
        """

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - NeuralNetworkTDLearner disabled")
            return

        self.network = network
        self.learning_rate = learning_rate
        self.td_learner = TemporalDifferenceLearner(learning_rate=learning_rate)

        if network:
            self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()

    def td_learning_step_nn(
        self,
        state_features: "torch.Tensor",
        reward: float,
        next_state_features: "torch.Tensor",
        terminal: bool = False,
    ) -> Tuple[float, float]:
        """Execute TD learning step with neural network.
        
        Args:
            state_features: Current state features (1, feature_dim)
            reward: Immediate reward
            next_state_features: Next state features (1, feature_dim)
            terminal: Is episode terminal?
            
        Returns:
            (td_error, loss)
        """

        if not TORCH_AVAILABLE or not self.network:
            return 0.0, 0.0

        # Forward pass: current state
        current_value = self.network(state_features).squeeze()

        # Forward pass: next state (no gradient)
        with torch.no_grad():
            next_value = self.network(next_state_features).squeeze()

        # Compute TD target
        td_target = self.td_learner.compute_td_target(reward, next_value.item(), terminal)

        # Compute loss
        loss = self.criterion(
            current_value, torch.tensor(td_target, dtype=torch.float32)
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute TD error
        td_error = td_target - current_value.item()

        return td_error, loss.item()

    def batch_td_learning(
        self,
        states: "torch.Tensor",
        rewards: "torch.Tensor",
        next_states: "torch.Tensor",
        terminals: "torch.Tensor",
    ) -> float:
        """Batch TD learning (more efficient).
        
        Args:
            states: Batch of state features (batch_size, feature_dim)
            rewards: Batch of rewards (batch_size,)
            next_states: Batch of next states (batch_size, feature_dim)
            terminals: Batch of terminal flags (batch_size,)
            
        Returns:
            Average loss
        """

        if not TORCH_AVAILABLE or not self.network:
            return 0.0

        # Current values
        current_values = self.network(states)

        # Next values (no gradient)
        with torch.no_grad():
            next_values = self.network(next_states)

        # TD targets
        td_targets = rewards + self.td_learner.gamma * next_values.squeeze() * (1 - terminals)

        # Loss and update
        loss = self.criterion(current_values.squeeze(), td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_learning_metrics(self) -> dict:
        """Get combined learning metrics."""
        metrics = self.td_learner.get_learning_metrics()
        metrics["learning_framework"] = "NeuralNetworkTDLearner"
        return metrics

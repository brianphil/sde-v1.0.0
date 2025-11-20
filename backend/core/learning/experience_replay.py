"""Experience replay for efficient reinforcement learning.

This module implements sde experience replay mechanisms:

- Basic replay buffer with uniform sampling
- Prioritized experience replay (PER) with proportional prioritization
- Sum tree for efficient O(log n) sampling
- Importance sampling weights for bias correction
- Support for multi-step returns
- Hindsight experience replay (HER) for sparse rewards
"""

from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import random
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Experience(NamedTuple):
    """Represents a single experience (transition) in the MDP.

    Components of SARS' (State, Action, Reward, next State, done):
    - state: Current state features
    - action: Action taken
    - reward: Reward received
    - next_state: Resulting state features
    - done: Whether episode terminated
    - info: Additional metadata
    """

    state: Dict[str, float]
    action: str
    reward: float
    next_state: Dict[str, float]
    done: bool
    info: Dict[str, Any] = {}


@dataclass
class SumTreeNode:
    """Node in sum tree for efficient prioritized sampling."""

    priority: float = 0.0
    left: Optional["SumTreeNode"] = None
    right: Optional["SumTreeNode"] = None
    data_index: Optional[int] = None  # Only leaf nodes store data


class SumTree:
    """Sum tree data structure for O(log n) priority sampling.

    Tree properties:
    - Each node's value is the sum of its children
    - Leaf nodes store priorities of experiences
    - Root stores total priority
    - Enables efficient proportional sampling
    """

    def __init__(self, capacity: int):
        """Initialize sum tree.

        Args:
            capacity: Maximum number of experiences
        """
        self.capacity = capacity
        self.tree: List[float] = [0.0] * (2 * capacity - 1)
        self.data: List[Optional[Experience]] = [None] * capacity
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float, experience: Experience):
        """Add experience with priority to tree.

        Args:
            priority: Priority value (typically TD error)
            experience: Experience to store
        """
        tree_index = self.data_pointer + self.capacity - 1

        # Store experience
        self.data[self.data_pointer] = experience

        # Update tree priorities
        self.update(tree_index, priority)

        # Move pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        # Track actual size
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_index: int, priority: float):
        """Update priority of a specific experience.

        Args:
            tree_index: Index in tree array
            priority: New priority value
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagate change up the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value: float) -> Tuple[int, float, Optional[Experience]]:
        """Get leaf node corresponding to value.

        Args:
            value: Random value between 0 and total_priority

        Returns:
            (tree_index, priority, experience)
        """
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # Reached leaf node
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            # Traverse tree based on value
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self) -> float:
        """Get total priority (root of tree)."""
        return self.tree[0]

    def __len__(self) -> int:
        """Get number of experiences stored."""
        return self.size


class ReplayBuffer:
    """Basic experience replay buffer with uniform sampling.

    Stores experiences and samples mini-batches uniformly for training.
    """

    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.total_added = 0

    def add(self, experience: Experience):
        """Add experience to buffer.

        Args:
            experience: Experience to store
        """
        self.buffer.append(experience)
        self.total_added += 1

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences uniformly.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all experiences."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer.

    Samples experiences proportionally to their TD error (priority).
    Important experiences are replayed more frequently.

    Reference: Schaul et al., "Prioritized Experience Replay" (2015)
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,  # Importance sampling exponent
        beta_increment: float = 0.001,  # Anneal beta to 1.0
        epsilon: float = 0.01,  # Small constant to avoid zero priority
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta: Importance sampling correction (0 = no correction, 1 = full)
            beta_increment: Amount to increase beta per sample
            epsilon: Small constant added to priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Sum tree for efficient sampling
        self.tree = SumTree(capacity)

        # Track max priority for new experiences
        self.max_priority = 1.0

        # Metrics
        self.total_added = 0

    def add(self, experience: Experience, priority: Optional[float] = None):
        """Add experience with priority.

        Args:
            experience: Experience to store
            priority: Priority value (defaults to max priority)
        """
        if priority is None:
            priority = self.max_priority

        # Add epsilon to avoid zero priority
        priority = (abs(priority) + self.epsilon) ** self.alpha

        self.tree.add(priority, experience)
        self.total_added += 1

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch of experiences with importance sampling weights.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            (experiences, indices, is_weights)
            - experiences: List of sampled experiences
            - indices: Tree indices for updating priorities
            - is_weights: Importance sampling weights
        """
        if len(self.tree) == 0:
            return [], np.array([]), np.array([])

        batch_size = min(batch_size, len(self.tree))

        experiences = []
        indices = []
        priorities = []

        # Divide total priority into segments
        segment_size = self.tree.total_priority / batch_size

        # Sample one experience from each segment
        for i in range(batch_size):
            # Sample uniformly from segment
            a = segment_size * i
            b = segment_size * (i + 1)
            value = random.uniform(a, b)

            # Get experience with this cumulative priority
            tree_index, priority, experience = self.tree.get_leaf(value)

            if experience is not None:
                experiences.append(experience)
                indices.append(tree_index)
                priorities.append(priority)

        # Calculate importance sampling weights
        # w_i = (1 / N * 1 / P(i)) ^ beta
        if priorities:
            priorities = np.array(priorities)
            sampling_probabilities = priorities / self.tree.total_priority
            is_weights = np.power(len(self.tree) * sampling_probabilities, -self.beta)

            # Normalize by max weight for stability
            is_weights /= is_weights.max()

            # Anneal beta towards 1.0
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            is_weights = np.ones(len(experiences))

        return experiences, np.array(indices), is_weights

    def update_priorities(self, tree_indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled experiences.

        Args:
            tree_indices: Indices in sum tree
            priorities: New priority values (typically TD errors)
        """
        for tree_idx, priority in zip(tree_indices, priorities):
            # Update max priority
            self.max_priority = max(self.max_priority, abs(priority))

            # Update tree with new priority
            priority = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(int(tree_idx), priority)

    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return len(self.tree)


class HindsightExperienceReplay:
    """Hindsight Experience Replay (HER) for sparse reward problems.

    Stores failed experiences with alternative goals to create positive examples.
    Useful when rewards are sparse and hard to obtain.

    Reference: Andrychowicz et al., "Hindsight Experience Replay" (2017)
    """

    def __init__(
        self,
        base_buffer: ReplayBuffer,
        k_hindsight: int = 4,  # Additional HER experiences per real experience
    ):
        """Initialize HER.

        Args:
            base_buffer: Underlying replay buffer
            k_hindsight: Number of hindsight experiences to generate
        """
        self.base_buffer = base_buffer
        self.k_hindsight = k_hindsight
        self.episode_buffer: List[Experience] = []

    def add(self, experience: Experience):
        """Add experience and generate hindsight experiences.

        Args:
            experience: Real experience
        """
        # Add to episode buffer
        self.episode_buffer.append(experience)

        # If episode done, generate hindsight experiences
        if experience.done:
            self._generate_hindsight_experiences()
            self.episode_buffer.clear()

    def _generate_hindsight_experiences(self):
        """Generate hindsight experiences from episode."""
        # Add original experiences
        for exp in self.episode_buffer:
            self.base_buffer.add(exp)

        # Generate k hindsight experiences
        for _ in range(self.k_hindsight):
            if len(self.episode_buffer) < 2:
                continue

            # Sample random future state as alternative goal
            t = random.randint(0, len(self.episode_buffer) - 2)
            future_t = random.randint(t + 1, len(self.episode_buffer) - 1)

            original_exp = self.episode_buffer[t]
            future_state = self.episode_buffer[future_t].state

            # Create hindsight experience with alternative goal
            # In this context: pretend the future state was the goal
            hindsight_exp = Experience(
                state=original_exp.state,
                action=original_exp.action,
                reward=1.0 if t == future_t - 1 else 0.0,  # Reward for reaching goal
                next_state=original_exp.next_state,
                done=original_exp.done,
                info={**original_exp.info, "hindsight": True, "goal": future_state},
            )

            self.base_buffer.add(hindsight_exp)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample from underlying buffer."""
        return self.base_buffer.sample(batch_size)

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.base_buffer)


class ExperienceReplayCoordinator:
    """Coordinates experience replay for the learning system.

    Main interface for managing experience storage and sampling.
    """

    def __init__(
        self,
        buffer_type: str = "prioritized",  # "basic", "prioritized", "hindsight"
        capacity: int = 10000,
        batch_size: int = 32,
        prioritized_alpha: float = 0.6,
        prioritized_beta: float = 0.4,
    ):
        """Initialize experience replay coordinator.

        Args:
            buffer_type: Type of replay buffer to use
            capacity: Maximum buffer capacity
            batch_size: Default batch size for sampling
            prioritized_alpha: Prioritization exponent (for PER)
            prioritized_beta: Importance sampling exponent (for PER)
        """
        self.buffer_type = buffer_type
        self.batch_size = batch_size

        # Create replay buffer
        if buffer_type == "basic":
            self.buffer = ReplayBuffer(capacity=capacity)
        elif buffer_type == "prioritized":
            self.buffer = PrioritizedReplayBuffer(
                capacity=capacity,
                alpha=prioritized_alpha,
                beta=prioritized_beta,
            )
        elif buffer_type == "hindsight":
            base_buffer = ReplayBuffer(capacity=capacity)
            self.buffer = HindsightExperienceReplay(base_buffer=base_buffer)
        else:
            raise ValueError(f"Unknown buffer type: {buffer_type}")

        # Metrics
        self.total_samples = 0
        self.created_at = datetime.now()

    def add_experience(
        self,
        state: Dict[str, float],
        action: str,
        reward: float,
        next_state: Dict[str, float],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
    ):
        """Add experience to replay buffer.

        Args:
            state: Current state features
            action: Action taken
            reward: Reward received
            next_state: Next state features
            done: Whether episode terminated
            info: Additional metadata
            priority: Priority (for prioritized replay)
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
        )

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.add(experience, priority=priority)
        else:
            self.buffer.add(experience)

    def sample_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[List[Experience], Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample batch of experiences.

        Args:
            batch_size: Number of experiences (defaults to self.batch_size)

        Returns:
            (experiences, indices, is_weights)
            - For prioritized: includes indices and IS weights
            - For basic: indices and weights are None
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.buffer) == 0:
            return [], None, None

        self.total_samples += 1

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            experiences, indices, is_weights = self.buffer.sample(batch_size)
            return experiences, indices, is_weights
        else:
            experiences = self.buffer.sample(batch_size)
            return experiences, None, None

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for prioritized replay.

        Args:
            indices: Experience indices to update
            td_errors: New priority values (TD errors)
        """
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.update_priorities(indices, td_errors)

    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """Check if buffer has enough experiences to sample.

        Args:
            batch_size: Minimum batch size needed

        Returns:
            True if can sample
        """
        if batch_size is None:
            batch_size = self.batch_size

        return len(self.buffer) >= batch_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get replay buffer statistics.

        Returns:
            Dictionary with buffer metrics
        """
        buffer_size = len(self.buffer)

        stats = {
            "buffer_type": self.buffer_type,
            "buffer_size": buffer_size,
            "capacity": self.buffer.capacity if hasattr(self.buffer, "capacity") else 0,
            "utilization": (
                buffer_size / self.buffer.capacity
                if hasattr(self.buffer, "capacity") and self.buffer.capacity > 0
                else 0.0
            ),
            "total_added": (
                self.buffer.total_added if hasattr(self.buffer, "total_added") else 0
            ),
            "total_samples": self.total_samples,
            "batch_size": self.batch_size,
        }

        # Add prioritized replay specific stats
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            stats.update(
                {
                    "alpha": self.buffer.alpha,
                    "beta": self.buffer.beta,
                    "max_priority": self.buffer.max_priority,
                    "total_priority": self.buffer.tree.total_priority,
                }
            )

        return stats

    def clear(self):
        """Clear all experiences from buffer."""
        if hasattr(self.buffer, "clear"):
            self.buffer.clear()
        else:
            # For PrioritizedReplayBuffer, recreate tree
            capacity = self.buffer.capacity
            alpha = self.buffer.alpha
            beta = self.buffer.beta
            self.buffer = PrioritizedReplayBuffer(
                capacity=capacity,
                alpha=alpha,
                beta=beta,
            )

    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return len(self.buffer)

"""Advanced pattern mining for Policy Function Approximation (PFA).

This module implements world-class pattern mining and association rule learning for
extracting decision-making policies from operational data:

- Frequent pattern mining (Apriori-style)
- Association rule learning
- Sequential pattern mining
- Context-aware rule extraction
- Rule evaluation and pruning
- Confidence, support, and lift metrics
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import statistics
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Represents a frequent pattern (itemset)."""

    items: frozenset  # Immutable set of items
    support: float = 0.0  # Frequency in dataset
    count: int = 0  # Absolute count

    def __hash__(self):
        return hash(self.items)

    def __eq__(self, other):
        return isinstance(other, Pattern) and self.items == other.items


@dataclass
class AssociationRule:
    """Represents a learned policy rule: IF antecedent THEN consequent.

    Example: IF (same_region=True AND vehicle_available=True) THEN batch_orders
    """

    rule_id: str
    antecedent: frozenset  # Conditions (IF part)
    consequent: frozenset  # Actions (THEN part)

    # Quality metrics
    support: float = 0.0  # P(antecedent ∪ consequent)
    confidence: float = 0.0  # P(consequent | antecedent)
    lift: float = 0.0  # Confidence / P(consequent)
    conviction: float = 0.0  # How much better than random

    # Performance tracking
    times_applied: int = 0
    successes: int = 0  # Positive outcomes
    failures: int = 0  # Negative outcomes
    avg_reward: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    active: bool = True

    def success_rate(self) -> float:
        """Get empirical success rate."""
        if self.times_applied == 0:
            return 0.0
        return self.successes / self.times_applied

    def __hash__(self):
        return hash((self.antecedent, self.consequent))

    def __repr__(self):
        ant_str = " AND ".join(str(item) for item in sorted(self.antecedent))
        cons_str = " AND ".join(str(item) for item in sorted(self.consequent))
        return f"IF {ant_str} THEN {cons_str} (conf={self.confidence:.2f}, lift={self.lift:.2f})"


@dataclass
class Transaction:
    """Represents a decision transaction for pattern mining."""

    transaction_id: str
    items: Set[str]  # Features and actions
    context: Dict[str, Any]
    reward: float
    timestamp: datetime


class FrequentPatternMiner:
    """Apriori-style frequent pattern mining.

    Finds itemsets that occur frequently in the transaction database.
    Uses level-wise search with anti-monotone property:
    - If itemset is frequent, all subsets are frequent
    - If itemset is infrequent, all supersets are infrequent
    """

    def __init__(self, min_support: float = 0.1, max_itemset_size: int = 4):
        """Initialize frequent pattern miner.

        Args:
            min_support: Minimum support threshold (0-1)
            max_itemset_size: Maximum size of patterns to mine
        """
        self.min_support = min_support
        self.max_itemset_size = max_itemset_size

    def mine_patterns(self, transactions: List[Transaction]) -> List[Pattern]:
        """Mine frequent patterns using Apriori algorithm.

        Args:
            transactions: List of decision transactions

        Returns:
            List of frequent patterns sorted by support
        """
        if not transactions:
            return []

        n_transactions = len(transactions)
        min_count = int(self.min_support * n_transactions)

        # Start with 1-itemsets (individual items)
        item_counts = Counter()
        for trans in transactions:
            for item in trans.items:
                item_counts[item] += 1

        # Filter by minimum support
        frequent_1_itemsets = {
            frozenset([item]): count
            for item, count in item_counts.items()
            if count >= min_count
        }

        if not frequent_1_itemsets:
            return []

        # Store all frequent patterns
        all_patterns = []
        current_patterns = frequent_1_itemsets

        # Level-wise search
        for k in range(1, self.max_itemset_size + 1):
            if not current_patterns:
                break

            # Convert to Pattern objects
            for itemset, count in current_patterns.items():
                support = count / n_transactions
                all_patterns.append(Pattern(
                    items=itemset,
                    support=support,
                    count=count,
                ))

            # Generate candidate (k+1)-itemsets from k-itemsets
            if k < self.max_itemset_size:
                current_patterns = self._generate_candidates(
                    current_patterns,
                    transactions,
                    min_count
                )

        # Sort by support (descending)
        all_patterns.sort(key=lambda p: p.support, reverse=True)

        logger.info(f"Mined {len(all_patterns)} frequent patterns from {n_transactions} transactions")
        return all_patterns

    def _generate_candidates(
        self,
        frequent_k_itemsets: Dict[frozenset, int],
        transactions: List[Transaction],
        min_count: int,
    ) -> Dict[frozenset, int]:
        """Generate and count candidate (k+1)-itemsets.

        Uses self-join of frequent k-itemsets.
        """
        # Generate candidates by joining itemsets with k-1 common items
        candidates = set()
        itemsets = list(frequent_k_itemsets.keys())

        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                # Join if they share k-1 items
                union = itemsets[i] | itemsets[j]
                if len(union) == len(itemsets[i]) + 1:
                    # Check if all k-subsets are frequent (pruning)
                    if self._all_subsets_frequent(union, frequent_k_itemsets):
                        candidates.add(union)

        # Count candidate support
        candidate_counts = Counter()
        for trans in transactions:
            for candidate in candidates:
                if candidate.issubset(trans.items):
                    candidate_counts[candidate] += 1

        # Filter by minimum support
        return {
            itemset: count
            for itemset, count in candidate_counts.items()
            if count >= min_count
        }

    def _all_subsets_frequent(
        self,
        itemset: frozenset,
        frequent_k_itemsets: Dict[frozenset, int]
    ) -> bool:
        """Check if all k-subsets of (k+1)-itemset are frequent."""
        k = len(list(frequent_k_itemsets.keys())[0])

        # Generate all k-subsets
        items = list(itemset)
        for i in range(len(items)):
            subset = frozenset(items[:i] + items[i+1:])
            if len(subset) == k and subset not in frequent_k_itemsets:
                return False

        return True


class AssociationRuleLearner:
    """Learn association rules from frequent patterns.

    Generates rules: IF antecedent THEN consequent
    where antecedent ∪ consequent is a frequent pattern.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
        max_rules: int = 100,
    ):
        """Initialize rule learner.

        Args:
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold (1.0 = no correlation)
            max_rules: Maximum number of rules to keep
        """
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_rules = max_rules

    def learn_rules(
        self,
        patterns: List[Pattern],
        transactions: List[Transaction],
    ) -> List[AssociationRule]:
        """Generate association rules from frequent patterns.

        Args:
            patterns: Frequent patterns from mining
            transactions: Original transactions for metrics

        Returns:
            List of association rules sorted by lift
        """
        if not patterns or not transactions:
            return []

        # Build pattern lookup for support calculation
        pattern_support = {p.items: p.support for p in patterns}

        rules = []
        rule_id = 0

        # Generate rules from patterns with 2+ items
        for pattern in patterns:
            if len(pattern.items) < 2:
                continue

            # Try different antecedent/consequent splits
            items = list(pattern.items)

            # For each possible split
            for i in range(1, len(items)):
                # Generate all combinations of size i for antecedent
                for antecedent_items in self._combinations(items, i):
                    antecedent = frozenset(antecedent_items)
                    consequent = pattern.items - antecedent

                    if not consequent:
                        continue

                    # Calculate metrics
                    rule_support = pattern.support

                    # Get antecedent support
                    antecedent_support = pattern_support.get(antecedent, 0.0)
                    if antecedent_support == 0:
                        continue

                    # Confidence: P(consequent | antecedent)
                    confidence = rule_support / antecedent_support

                    if confidence < self.min_confidence:
                        continue

                    # Get consequent support
                    consequent_support = pattern_support.get(consequent, 0.0)
                    if consequent_support == 0:
                        continue

                    # Lift: confidence / P(consequent)
                    lift = confidence / consequent_support

                    if lift < self.min_lift:
                        continue

                    # Conviction: how much more often antecedent appears without consequent
                    conviction = (1 - consequent_support) / (1 - confidence) if confidence < 1 else float('inf')

                    # Create rule
                    rule = AssociationRule(
                        rule_id=f"RULE_{rule_id:04d}",
                        antecedent=antecedent,
                        consequent=consequent,
                        support=rule_support,
                        confidence=confidence,
                        lift=lift,
                        conviction=conviction,
                    )

                    rules.append(rule)
                    rule_id += 1

        # Sort by lift (descending) and limit
        rules.sort(key=lambda r: r.lift, reverse=True)
        rules = rules[:self.max_rules]

        logger.info(f"Learned {len(rules)} association rules from {len(patterns)} patterns")
        return rules

    def _combinations(self, items: List, r: int) -> List[List]:
        """Generate all combinations of r items."""
        if r == 0:
            return [[]]
        if not items:
            return []

        # Include first item
        with_first = [[items[0]] + rest for rest in self._combinations(items[1:], r - 1)]
        # Exclude first item
        without_first = self._combinations(items[1:], r)

        return with_first + without_first


class SequentialPatternMiner:
    """Mine sequential patterns from time-ordered transactions.

    Finds patterns like: "Order A THEN Order B within 1 hour"
    Useful for temporal decision policies.
    """

    def __init__(
        self,
        min_support: float = 0.1,
        max_time_gap: timedelta = timedelta(hours=4),
        max_sequence_length: int = 3,
    ):
        """Initialize sequential pattern miner.

        Args:
            min_support: Minimum support threshold
            max_time_gap: Maximum time between sequence items
            max_sequence_length: Maximum sequence length
        """
        self.min_support = min_support
        self.max_time_gap = max_time_gap
        self.max_sequence_length = max_sequence_length

    def mine_sequences(self, transactions: List[Transaction]) -> List[Tuple[List[str], float]]:
        """Mine frequent sequential patterns.

        Args:
            transactions: Time-ordered transactions

        Returns:
            List of (sequence, support) tuples
        """
        if not transactions:
            return []

        # Sort by timestamp
        sorted_trans = sorted(transactions, key=lambda t: t.timestamp)

        n_transactions = len(sorted_trans)
        min_count = int(self.min_support * n_transactions)

        # Find frequent 1-sequences
        item_counts = Counter()
        for trans in sorted_trans:
            for item in trans.items:
                item_counts[item] += 1

        frequent_1_seqs = {
            (item,): count
            for item, count in item_counts.items()
            if count >= min_count
        }

        all_sequences = []
        current_sequences = frequent_1_seqs

        # Grow sequences
        for k in range(1, self.max_sequence_length + 1):
            if not current_sequences:
                break

            # Add to results
            for seq, count in current_sequences.items():
                support = count / n_transactions
                all_sequences.append((list(seq), support))

            # Generate (k+1)-sequences
            if k < self.max_sequence_length:
                current_sequences = self._grow_sequences(
                    current_sequences,
                    sorted_trans,
                    min_count
                )

        # Sort by support
        all_sequences.sort(key=lambda s: s[1], reverse=True)

        logger.info(f"Mined {len(all_sequences)} sequential patterns")
        return all_sequences

    def _grow_sequences(
        self,
        frequent_k_seqs: Dict[Tuple, int],
        transactions: List[Transaction],
        min_count: int,
    ) -> Dict[Tuple, int]:
        """Grow k-sequences to (k+1)-sequences."""
        # Generate candidates
        candidates = set()
        sequences = list(frequent_k_seqs.keys())

        for seq in sequences:
            # Try appending each frequent 1-item
            for other_seq in sequences:
                if len(other_seq) == 1:
                    new_seq = seq + other_seq
                    candidates.add(new_seq)

        # Count candidate support
        candidate_counts = Counter()

        for candidate in candidates:
            count = self._count_sequence_support(candidate, transactions)
            if count >= min_count:
                candidate_counts[candidate] = count

        return candidate_counts

    def _count_sequence_support(self, sequence: Tuple, transactions: List[Transaction]) -> int:
        """Count how many transaction windows contain the sequence."""
        count = 0

        for i in range(len(transactions)):
            # Try to match sequence starting from this transaction
            if self._matches_sequence(sequence, transactions, i):
                count += 1

        return count

    def _matches_sequence(
        self,
        sequence: Tuple,
        transactions: List[Transaction],
        start_idx: int
    ) -> bool:
        """Check if sequence appears in transactions starting at start_idx."""
        if start_idx >= len(transactions):
            return False

        seq_idx = 0
        trans_idx = start_idx
        start_time = transactions[start_idx].timestamp

        while seq_idx < len(sequence) and trans_idx < len(transactions):
            trans = transactions[trans_idx]

            # Check time constraint
            if trans.timestamp - start_time > self.max_time_gap:
                return False

            # Check if current sequence item is in transaction
            if sequence[seq_idx] in trans.items:
                seq_idx += 1
                if seq_idx == len(sequence):
                    return True

            trans_idx += 1

        return False


class PatternMiningCoordinator:
    """Coordinates pattern mining for PFA rule learning.

    Main interface for extracting decision-making policies from operational data.
    """

    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        min_lift: float = 1.2,
        max_rules: int = 100,
    ):
        """Initialize pattern mining coordinator.

        Args:
            min_support: Minimum pattern frequency
            min_confidence: Minimum rule confidence
            min_lift: Minimum lift (correlation strength)
            max_rules: Maximum active rules
        """
        self.frequent_miner = FrequentPatternMiner(
            min_support=min_support,
            max_itemset_size=4,
        )

        self.rule_learner = AssociationRuleLearner(
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_rules=max_rules,
        )

        self.sequential_miner = SequentialPatternMiner(
            min_support=min_support,
            max_time_gap=timedelta(hours=4),
            max_sequence_length=3,
        )

        # Storage
        self.transactions: List[Transaction] = []
        self.active_rules: List[AssociationRule] = []
        self.rule_lookup: Dict[str, AssociationRule] = {}

        # Metrics
        self.mining_count = 0
        self.last_mining_time: Optional[datetime] = None

    def add_transaction(
        self,
        transaction_id: str,
        features: Set[str],
        actions: Set[str],
        context: Dict[str, Any],
        reward: float,
        timestamp: Optional[datetime] = None,
    ):
        """Add a decision transaction for mining.

        Args:
            transaction_id: Unique transaction identifier
            features: Context features (e.g., "high_demand", "vehicle_available")
            actions: Actions taken (e.g., "batch_orders", "use_fast_route")
            context: Additional context data
            reward: Outcome reward
            timestamp: Transaction time
        """
        items = features | actions

        transaction = Transaction(
            transaction_id=transaction_id,
            items=items,
            context=context,
            reward=reward,
            timestamp=timestamp or datetime.now(),
        )

        self.transactions.append(transaction)

        # Limit transaction history
        if len(self.transactions) > 1000:
            self.transactions.pop(0)

    def mine_and_update_rules(self, force: bool = False) -> int:
        """Mine patterns and update rules.

        Args:
            force: Force mining even if not enough transactions

        Returns:
            Number of new/updated rules
        """
        if not force and len(self.transactions) < 20:
            logger.info("Not enough transactions for mining (need 20+)")
            return 0

        logger.info(f"Mining patterns from {len(self.transactions)} transactions")

        # Mine frequent patterns
        patterns = self.frequent_miner.mine_patterns(self.transactions)

        if not patterns:
            logger.warning("No frequent patterns found")
            return 0

        # Learn association rules
        new_rules = self.rule_learner.learn_rules(patterns, self.transactions)

        # Update rule set
        self.active_rules = new_rules
        self.rule_lookup = {rule.rule_id: rule for rule in new_rules}

        self.mining_count += 1
        self.last_mining_time = datetime.now()

        logger.info(f"Updated rule set: {len(self.active_rules)} active rules")
        return len(self.active_rules)

    def get_matching_rules(self, context_features: Set[str]) -> List[AssociationRule]:
        """Find rules whose antecedent matches the context.

        Args:
            context_features: Current context features

        Returns:
            List of applicable rules sorted by confidence
        """
        matching_rules = []

        for rule in self.active_rules:
            if not rule.active:
                continue

            # Check if all antecedent items are in context
            if rule.antecedent.issubset(context_features):
                matching_rules.append(rule)

        # Sort by confidence * lift (quality metric)
        matching_rules.sort(key=lambda r: r.confidence * r.lift, reverse=True)

        return matching_rules

    def update_rule_performance(
        self,
        rule_id: str,
        success: bool,
        reward: float,
    ):
        """Update rule performance based on outcome.

        Args:
            rule_id: Rule that was applied
            success: Whether outcome was positive
            reward: Reward received
        """
        if rule_id not in self.rule_lookup:
            return

        rule = self.rule_lookup[rule_id]
        rule.times_applied += 1
        rule.last_used = datetime.now()

        if success:
            rule.successes += 1
        else:
            rule.failures += 1

        # Update running average reward
        alpha = 0.1  # Learning rate
        rule.avg_reward = (1 - alpha) * rule.avg_reward + alpha * reward

    def prune_low_performance_rules(self, min_success_rate: float = 0.3, min_applications: int = 10):
        """Remove rules with poor empirical performance.

        Args:
            min_success_rate: Minimum success rate to keep rule
            min_applications: Minimum applications before pruning
        """
        pruned = 0

        for rule in self.active_rules:
            if rule.times_applied >= min_applications:
                if rule.success_rate() < min_success_rate:
                    rule.active = False
                    pruned += 1

        if pruned > 0:
            logger.info(f"Pruned {pruned} low-performance rules")

        return pruned

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rule statistics.

        Returns:
            Dictionary with mining metrics
        """
        active_count = sum(1 for r in self.active_rules if r.active)

        if not self.active_rules:
            return {
                "total_rules": 0,
                "active_rules": 0,
                "avg_confidence": 0.0,
                "avg_lift": 0.0,
                "mining_count": self.mining_count,
            }

        active_rules = [r for r in self.active_rules if r.active]

        return {
            "total_rules": len(self.active_rules),
            "active_rules": active_count,
            "avg_confidence": statistics.mean(r.confidence for r in active_rules) if active_rules else 0.0,
            "avg_lift": statistics.mean(r.lift for r in active_rules) if active_rules else 0.0,
            "avg_success_rate": statistics.mean(r.success_rate() for r in active_rules if r.times_applied > 0) if active_rules else 0.0,
            "total_applications": sum(r.times_applied for r in active_rules),
            "mining_count": self.mining_count,
            "transaction_count": len(self.transactions),
            "last_mining": self.last_mining_time.isoformat() if self.last_mining_time else None,
        }

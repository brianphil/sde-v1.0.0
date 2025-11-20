"""Consolidation Pool - manages orders waiting for consolidation.

Workflow:
1. New order arrives
2. Classify as BULK or CONSOLIDATED
3. BULK → immediate dispatch
4. CONSOLIDATED → add to pool
5. Pool triggers consolidation decision
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime, timedelta
import logging

from ..models.domain import Order, Vehicle

logger = logging.getLogger(__name__)


class OrderClassification(Enum):
    """Order classification for consolidation."""

    BULK = "bulk"  # Meets min utilization → immediate dispatch
    CONSOLIDATED = "consolidated"  # Below min utilization → pool for batching
    URGENT = "urgent"  # High priority → special handling


@dataclass
class PoolConfiguration:
    """Configuration for consolidation pool behavior."""

    # Utilization thresholds for classification
    bulk_min_weight_utilization: float = 0.60  # 60% for bulk orders
    bulk_min_volume_utilization: float = 0.50  # 50% for bulk orders

    # Pool trigger conditions
    max_pool_size: int = 20  # Trigger consolidation at 20 orders
    max_pool_wait_time_minutes: int = 120  # 2 hours max wait
    min_batch_size: int = 2  # Minimum orders to consolidate

    # Trigger on geographic cluster size
    trigger_on_cluster_size: int = 3  # If 3+ orders to same cluster
    # Lower default to 2 to allow opportunistic consolidation for small pools
    trigger_on_cluster_size: int = 2  # If 2+ orders to same cluster

    # Time-based triggers
    scheduled_consolidation_times: List[str] = field(
        default_factory=lambda: ["09:00", "14:00", "17:00"]  # 3x daily
    )


@dataclass
class PooledOrder:
    """Order in consolidation pool with metadata."""

    order: Order
    added_at: datetime
    classification: OrderClassification
    geographic_cluster: Optional[str] = None
    compatibility_tags: Set[str] = field(default_factory=set)


class ConsolidationPool:
    """Manages orders waiting for consolidation."""

    def __init__(self, config: Optional[PoolConfiguration] = None):
        """Initialize consolidation pool."""
        self.config = config or PoolConfiguration()
        self.pool: Dict[str, PooledOrder] = {}  # order_id -> PooledOrder
        self.last_consolidation: Optional[datetime] = None

    def classify_order(
        self,
        order: Order,
        available_vehicles: List[Vehicle]
    ) -> OrderClassification:
        """Classify incoming order as BULK, CONSOLIDATED, or URGENT.

        Args:
            order: The incoming order
            available_vehicles: Available vehicles for dispatch

        Returns:
            OrderClassification
        """
        # Check for urgent priority
        if order.priority >= 2:
            logger.info(f"Order {order.order_id} classified as URGENT (priority={order.priority})")
            return OrderClassification.URGENT

        # Check if order meets bulk utilization threshold for any vehicle
        for vehicle in available_vehicles:
            weight_util = order.weight_tonnes / vehicle.capacity_weight_tonnes
            volume_util = order.volume_m3 / vehicle.capacity_volume_m3

            if (weight_util >= self.config.bulk_min_weight_utilization or
                volume_util >= self.config.bulk_min_volume_utilization):

                logger.info(
                    f"Order {order.order_id} classified as BULK "
                    f"({weight_util:.1%} weight, {volume_util:.1%} volume on {vehicle.vehicle_type})"
                )
                return OrderClassification.BULK

        # Below threshold - needs consolidation
        logger.info(
            f"Order {order.order_id} classified as CONSOLIDATED "
            f"({order.weight_tonnes}T, {order.volume_m3}m³ - below bulk threshold)"
        )
        return OrderClassification.CONSOLIDATED

    def add_order(
        self,
        order: Order,
        classification: OrderClassification,
        geographic_cluster: Optional[str] = None
    ) -> bool:
        """Add order to consolidation pool.

        Returns:
            True if order added, False if rejected
        """
        if classification == OrderClassification.BULK:
            logger.warning(f"Attempted to add BULK order {order.order_id} to pool - should be dispatched immediately")
            return False

        pooled_order = PooledOrder(
            order=order,
            added_at=datetime.now(),
            classification=classification,
            geographic_cluster=geographic_cluster,
        )

        self.pool[order.order_id] = pooled_order
        logger.info(
            f"Added order {order.order_id} to consolidation pool "
            f"(pool size: {len(self.pool)}, cluster: {geographic_cluster})"
        )

        return True

    def should_trigger_consolidation(self) -> bool:
        """Check if pool should trigger consolidation decision.

        Triggers on:
        - Pool size threshold
        - Max wait time exceeded
        - Geographic cluster size threshold
        - Scheduled time reached
        """
        if not self.pool:
            return False

        now = datetime.now()

        # Trigger 1: Pool size threshold
        if len(self.pool) >= self.config.max_pool_size:
            logger.info(f"Consolidation triggered: pool size ({len(self.pool)}) >= threshold ({self.config.max_pool_size})")
            return True

        # Trigger 2: Max wait time exceeded
        oldest_order = min(self.pool.values(), key=lambda x: x.added_at)
        wait_time = (now - oldest_order.added_at).total_seconds() / 60

        if wait_time >= self.config.max_pool_wait_time_minutes:
            logger.info(f"Consolidation triggered: max wait time ({wait_time:.0f} min) exceeded")
            return True

        # Trigger 3: Geographic cluster size
        cluster_sizes = self._get_cluster_sizes()
        max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 0

        if max_cluster_size >= self.config.trigger_on_cluster_size:
            logger.info(f"Consolidation triggered: cluster size ({max_cluster_size}) >= threshold ({self.config.trigger_on_cluster_size})")
            return True

        # Trigger 4: Scheduled consolidation time
        current_time_str = now.strftime("%H:%M")
        if current_time_str in self.config.scheduled_consolidation_times:
            # Check if we haven't consolidated in the last 30 minutes
            if not self.last_consolidation or (now - self.last_consolidation) > timedelta(minutes=30):
                logger.info(f"Consolidation triggered: scheduled time ({current_time_str})")
                return True

        return False

    def _get_cluster_sizes(self) -> Dict[str, int]:
        """Get size of each geographic cluster in pool."""
        cluster_sizes: Dict[str, int] = {}

        for pooled_order in self.pool.values():
            if pooled_order.geographic_cluster:
                cluster_sizes[pooled_order.geographic_cluster] = \
                    cluster_sizes.get(pooled_order.geographic_cluster, 0) + 1

        return cluster_sizes

    def get_orders_for_consolidation(
        self,
        max_orders: Optional[int] = None
    ) -> Dict[str, Order]:
        """Get orders from pool for consolidation.

        Args:
            max_orders: Maximum number of orders to retrieve

        Returns:
            Dict of order_id -> Order
        """
        # Prioritize by wait time (oldest first)
        sorted_orders = sorted(
            self.pool.values(),
            key=lambda x: x.added_at
        )

        if max_orders:
            sorted_orders = sorted_orders[:max_orders]

        return {
            po.order.order_id: po.order
            for po in sorted_orders
        }

    def remove_orders(self, order_ids: List[str]) -> None:
        """Remove orders from pool (after consolidation)."""
        for order_id in order_ids:
            if order_id in self.pool:
                del self.pool[order_id]
                logger.debug(f"Removed order {order_id} from consolidation pool")

        self.last_consolidation = datetime.now()
        logger.info(f"Removed {len(order_ids)} orders from pool (remaining: {len(self.pool)})")

    def get_pool_status(self) -> Dict:
        """Get current pool status for monitoring."""
        if not self.pool:
            return {
                "size": 0,
                "clusters": {},
                "oldest_order_wait_minutes": 0,
            }

        now = datetime.now()
        oldest_order = min(self.pool.values(), key=lambda x: x.added_at)
        wait_time = (now - oldest_order.added_at).total_seconds() / 60

        cluster_sizes = self._get_cluster_sizes()

        return {
            "size": len(self.pool),
            "clusters": cluster_sizes,
            "oldest_order_wait_minutes": round(wait_time, 1),
            "should_trigger": self.should_trigger_consolidation(),
        }

    def get_cluster_orders(self, cluster_id: str) -> Dict[str, Order]:
        """Get all orders in a specific geographic cluster."""
        return {
            po.order.order_id: po.order
            for po in self.pool.values()
            if po.geographic_cluster == cluster_id
        }

    def update_geographic_clusters(self, clusters: Dict[str, List[str]]) -> None:
        """Update geographic cluster assignments for pooled orders.

        Args:
            clusters: Dict mapping cluster_id to list of order_ids
        """
        # Clear existing cluster assignments
        for pooled_order in self.pool.values():
            pooled_order.geographic_cluster = None

        # Apply new cluster assignments
        for cluster_id, order_ids in clusters.items():
            for order_id in order_ids:
                if order_id in self.pool:
                    self.pool[order_id].geographic_cluster = cluster_id

        logger.info(f"Updated geographic clusters for {len(self.pool)} pooled orders")

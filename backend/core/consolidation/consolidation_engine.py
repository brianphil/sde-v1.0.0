"""Complete Consolidation Engine.

Orchestrates the consolidation workflow and prepares decision context for Powell SDE:
1. Order arrival → classification (bulk vs consolidated)
2. Consolidation pool management
3. Staged filtering (geographic → service → time)
4. Prepare compatible order groups for Powell SDE decision-making
5. Powell SDE (CFA/VFA/PFA/DLA) makes final routing decisions

IMPORTANT: This engine does NOT make routing decisions on its own.
It prepares filtered, compatible order groups and lets Powell SDE decide optimal routes.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from ..models.domain import Order
from ..models.state import SystemState
from .geographic_clustering import GeographicClusteringEngine
from .consolidation_pool import ConsolidationPool, OrderClassification, PoolConfiguration
from .compatibility_filters import (
    CompatibilityFilterChain,
    ServiceLevelConfig,
    TimeWindowConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of consolidation process."""

    bulk_order_ids: List[str]  # Orders classified as bulk (for Powell SDE to route immediately)
    urgent_order_ids: List[str]  # Orders classified as urgent (for Powell SDE to route immediately)
    pooled_order_ids: List[str]  # Orders added to consolidation pool
    orders_classified: Dict[str, OrderClassification]
    pool_status: Dict
    should_trigger_consolidation: bool


@dataclass
class ConsolidationOpportunity:
    """Prepared consolidation opportunity for Powell SDE decision-making."""

    cluster_id: str
    order_ids: List[str]
    geographic_compatible: bool
    service_level_compatible: bool
    time_window_compatible: bool
    estimated_total_weight: float
    estimated_total_volume: float
    compatibility_score: float  # 0.0 - 1.0


class ConsolidationEngine:
    """Consolidation engine that prepares order groups for Powell SDE.

    This engine does NOT make routing decisions. Instead, it:
    1. Classifies orders (bulk/consolidated/urgent)
    2. Manages consolidation pool
    3. Filters compatible order groups
    4. Provides structured input to Powell SDE
    5. Powell SDE (CFA/VFA/PFA/DLA) makes final routing decisions

    Workflow:
    1. Order arrives → classify (bulk/consolidated/urgent)
    2. BULK/URGENT → flag for immediate Powell SDE decision
    3. CONSOLIDATED → add to pool
    4. Pool triggers → prepare consolidation opportunities
    5. Powell SDE evaluates opportunities and creates optimal routes
    """

    def __init__(
        self,
        pool_config: Optional[PoolConfiguration] = None,
        service_config: Optional[ServiceLevelConfig] = None,
        time_config: Optional[TimeWindowConfig] = None,
    ):
        """Initialize consolidation engine."""
        self.geo_clustering = GeographicClusteringEngine()
        self.pool = ConsolidationPool(pool_config)
        self.filter_chain = CompatibilityFilterChain(service_config, time_config)

        logger.info("Consolidation engine initialized (works with Powell SDE)")

    def process_new_order(
        self,
        order: Order,
        state: SystemState
    ) -> ConsolidationResult:
        """Process new order arrival and classify for Powell SDE routing.

        Args:
            order: Newly arrived order
            state: Current system state

        Returns:
            ConsolidationResult indicating how Powell SDE should handle this order
        """
        logger.info(f"Processing new order: {order.order_id}")

        # Get available vehicles
        available_vehicles = [
            v for v in state.fleet.values()
            if v.is_available_at(state.environment.current_time)
        ]

        # Classify order
        classification = self.pool.classify_order(order, available_vehicles)

        orders_classified = {order.order_id: classification}
        bulk_order_ids = []
        urgent_order_ids = []
        pooled_order_ids = []

        if classification == OrderClassification.BULK:
            # Flag for immediate Powell SDE routing
            logger.info(f"Order {order.order_id} is BULK - flagging for immediate Powell SDE routing")
            bulk_order_ids.append(order.order_id)

        elif classification == OrderClassification.URGENT:
            # Flag for urgent Powell SDE routing
            logger.info(f"Order {order.order_id} is URGENT - flagging for immediate Powell SDE routing")
            urgent_order_ids.append(order.order_id)

        else:
            # Add to consolidation pool
            logger.info(f"Order {order.order_id} is CONSOLIDATED - adding to pool")

            # Get geographic cluster for order
            clusters = self.geo_clustering.cluster_orders({order.order_id: order})
            cluster_id = list(clusters.keys())[0] if clusters else None

            self.pool.add_order(order, classification, cluster_id)
            pooled_order_ids.append(order.order_id)

        return ConsolidationResult(
            bulk_order_ids=bulk_order_ids,
            urgent_order_ids=urgent_order_ids,
            pooled_order_ids=pooled_order_ids,
            orders_classified=orders_classified,
            pool_status=self.pool.get_pool_status(),
            should_trigger_consolidation=self.pool.should_trigger_consolidation()
        )

    def prepare_consolidation_opportunities(
        self,
        state: SystemState
    ) -> List[ConsolidationOpportunity]:
        """Prepare consolidation opportunities for Powell SDE decision-making.

        This method prepares compatible order groups but does NOT create routes.
        Powell SDE (CFA/VFA/PFA/DLA) will evaluate these opportunities and decide
        which groups to route and how to route them optimally.

        Returns:
            List of ConsolidationOpportunity objects for Powell SDE evaluation
        """
        if not self.pool.should_trigger_consolidation():
            logger.debug("Consolidation not triggered")
            return []

        logger.info(f"Preparing consolidation opportunities from pool (size: {len(self.pool.pool)})")

        # Get orders from pool
        pool_orders = self.pool.get_orders_for_consolidation()

        if not pool_orders:
            return []

        # Stage 1: Geographic clustering
        logger.info("Stage 1: Geographic clustering")
        geo_clusters = self.geo_clustering.cluster_orders(pool_orders)
        logger.info(f"  → {len(geo_clusters)} geographic clusters")

        # Update pool with cluster assignments
        self.pool.update_geographic_clusters(geo_clusters)

        # Stage 2 & 3: Service-level and time window filtering
        logger.info("Stage 2 & 3: Service-level and time window filtering")
        compatible_groups = self.filter_chain.apply_filters(pool_orders, geo_clusters)
        logger.info(f"  → {len(compatible_groups)} compatible groups")

        # Stage 4: Convert to ConsolidationOpportunity objects for Powell SDE
        opportunities = []
        for cluster_id, order_ids in compatible_groups.items():
            if len(order_ids) < self.pool.config.min_batch_size:
                logger.debug(f"Cluster {cluster_id} has only {len(order_ids)} orders, skipping")
                continue

            group_orders = [pool_orders[oid] for oid in order_ids if oid in pool_orders]

            if not group_orders:
                continue

            # Calculate total load
            total_weight = sum(o.weight_tonnes for o in group_orders)
            total_volume = sum(o.volume_m3 for o in group_orders)

            # Calculate compatibility score (based on geographic clustering quality)
            compatibility_score = self._calculate_group_compatibility(group_orders)

            opportunity = ConsolidationOpportunity(
                cluster_id=cluster_id,
                order_ids=order_ids,
                geographic_compatible=True,  # Already filtered
                service_level_compatible=True,  # Already filtered
                time_window_compatible=True,  # Already filtered
                estimated_total_weight=total_weight,
                estimated_total_volume=total_volume,
                compatibility_score=compatibility_score
            )

            opportunities.append(opportunity)
            logger.info(
                f"Prepared opportunity {cluster_id}: {len(order_ids)} orders, "
                f"{total_weight:.1f}T, {total_volume:.1f}m³, score={compatibility_score:.2f}"
            )

        logger.info(f"Prepared {len(opportunities)} consolidation opportunities for Powell SDE")

        return opportunities

    def remove_routed_orders(self, order_ids: List[str]) -> None:
        """Remove orders from pool after Powell SDE has routed them.

        Args:
            order_ids: List of order IDs that have been successfully routed
        """
        self.pool.remove_orders(order_ids)
        logger.info(f"Removed {len(order_ids)} routed orders from consolidation pool")

    def _calculate_group_compatibility(self, orders: List[Order]) -> float:
        """Calculate compatibility score for a group of orders.

        Score based on:
        - Geographic clustering quality (bearing similarity)
        - Distance similarity
        - Priority similarity

        Returns:
            Score 0.0 - 1.0 (higher = more compatible)
        """
        if len(orders) <= 1:
            return 1.0

        # Calculate average pairwise compatibility
        total_score = 0.0
        pairs = 0

        for i in range(len(orders)):
            for j in range(i + 1, len(orders)):
                score = self.geo_clustering.calculate_route_compatibility_score(
                    orders[i],
                    orders[j]
                )
                total_score += score
                pairs += 1

        return total_score / pairs if pairs > 0 else 0.0

    def get_pool_status(self) -> Dict:
        """Get current consolidation pool status."""
        return self.pool.get_pool_status()

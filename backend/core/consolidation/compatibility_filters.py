"""Compatibility Filters for Consolidation.

Staged filtering pipeline:
1. Geographic clustering (done first - reduces search space)
2. Service-level compatibility (priority, special handling)
3. Time window compatibility
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

from ..models.domain import Order

logger = logging.getLogger(__name__)


class CompatibilityFilter(ABC):
    """Base class for compatibility filters."""

    @abstractmethod
    def filter(
        self,
        orders: Dict[str, Order],
        anchor_order: Optional[Order] = None
    ) -> Dict[str, List[str]]:
        """Filter orders into compatible groups.

        Args:
            orders: Orders to filter
            anchor_order: Optional anchor order to filter against

        Returns:
            Dict mapping group_id to list of compatible order_ids
        """
        pass


@dataclass
class ServiceLevelConfig:
    """Configuration for service-level filtering."""

    # Priority handling
    allow_priority_mixing: bool = True
    max_priority_difference: int = 1  # 0=normal, 1=high, 2=urgent

    # Special handling rules
    fresh_food_dedicated: bool = True
    hazardous_dedicated: bool = True
    fragile_can_mix: bool = True
    fragile_max_weight_partner: float = 2.0  # Max 2T if fragile

    # Customer preferences
    respect_customer_preferences: bool = True


class ServiceLevelFilter(CompatibilityFilter):
    """Filter orders by service-level compatibility.

    Checks:
    - Priority levels
    - Special handling requirements
    - Customer preferences
    """

    def __init__(self, config: Optional[ServiceLevelConfig] = None):
        """Initialize service-level filter."""
        self.config = config or ServiceLevelConfig()

    def filter(
        self,
        orders: Dict[str, Order],
        anchor_order: Optional[Order] = None
    ) -> Dict[str, List[str]]:
        """Filter orders by service-level compatibility."""
        if not orders:
            return {}

        if anchor_order:
            # Filter against specific anchor order
            return self._filter_against_anchor(orders, anchor_order)

        # Group all orders by compatibility
        return self._group_by_service_level(orders)

    def _filter_against_anchor(
        self,
        orders: Dict[str, Order],
        anchor_order: Order
    ) -> Dict[str, List[str]]:
        """Filter orders compatible with anchor order."""
        compatible_ids = [anchor_order.order_id] if anchor_order.order_id in orders else []

        for order_id, order in orders.items():
            if order_id == anchor_order.order_id:
                continue

            if self.are_compatible(anchor_order, order):
                compatible_ids.append(order_id)

        return {"compatible": compatible_ids} if compatible_ids else {}

    def _group_by_service_level(self, orders: Dict[str, Order]) -> Dict[str, List[str]]:
        """Group orders by service-level characteristics."""
        groups: Dict[str, List[str]] = {}

        for order_id, order in orders.items():
            group_key = self._get_service_level_key(order)
            groups.setdefault(group_key, []).append(order_id)

        return groups

    def _get_service_level_key(self, order: Order) -> str:
        """Get service-level grouping key for order."""
        key_parts = []

        # Priority
        if not self.config.allow_priority_mixing:
            key_parts.append(f"p{order.priority}")

        # Special handling
        special = order.special_handling or []

        if "fresh_food" in special and self.config.fresh_food_dedicated:
            key_parts.append("fresh")

        if "hazardous" in special and self.config.hazardous_dedicated:
            key_parts.append("hazardous")

        if "fragile" in special:
            key_parts.append("fragile")

        return "_".join(key_parts) if key_parts else "standard"

    def are_compatible(self, order1: Order, order2: Order) -> bool:
        """Check if two orders are service-level compatible."""
        # Check priority
        if not self._check_priority_compatible(order1, order2):
            return False

        # Check special handling
        if not self._check_special_handling_compatible(order1, order2):
            return False

        return True

    def _check_priority_compatible(self, order1: Order, order2: Order) -> bool:
        """Check priority compatibility."""
        if not self.config.allow_priority_mixing:
            return order1.priority == order2.priority

        priority_diff = abs(order1.priority - order2.priority)
        return priority_diff <= self.config.max_priority_difference

    def _check_special_handling_compatible(self, order1: Order, order2: Order) -> bool:
        """Check special handling compatibility."""
        special1 = set(order1.special_handling or [])
        special2 = set(order2.special_handling or [])

        # Fresh food must be alone
        if self.config.fresh_food_dedicated:
            if "fresh_food" in special1 or "fresh_food" in special2:
                return special1 == special2

        # Hazardous must be alone
        if self.config.hazardous_dedicated:
            if "hazardous" in special1 or "hazardous" in special2:
                return special1 == special2

        # Fragile with weight restrictions
        if not self.config.fragile_can_mix:
            if "fragile" in special1 and "fragile" not in special2:
                if order2.weight_tonnes > self.config.fragile_max_weight_partner:
                    return False
            if "fragile" in special2 and "fragile" not in special1:
                if order1.weight_tonnes > self.config.fragile_max_weight_partner:
                    return False

        return True


@dataclass
class TimeWindowConfig:
    """Configuration for time window filtering."""

    # Time window overlap requirements
    min_overlap_minutes: int = 60  # Minimum 1 hour overlap
    buffer_minutes: int = 15  # Safety buffer

    # Delivery sequencing
    allow_sequential_windows: bool = True  # Can deliver in sequence
    max_route_duration_hours: float = 8.0  # Max 8 hours total route


class TimeWindowFilter(CompatibilityFilter):
    """Filter orders by time window compatibility.

    Checks:
    - Overlapping time windows
    - Sequential delivery feasibility
    - Total route duration
    """

    def __init__(self, config: Optional[TimeWindowConfig] = None):
        """Initialize time window filter."""
        self.config = config or TimeWindowConfig()

    def filter(
        self,
        orders: Dict[str, Order],
        anchor_order: Optional[Order] = None
    ) -> Dict[str, List[str]]:
        """Filter orders by time window compatibility."""
        if not orders:
            return {}

        if anchor_order:
            return self._filter_against_anchor(orders, anchor_order)

        return self._group_by_time_windows(orders)

    def _filter_against_anchor(
        self,
        orders: Dict[str, Order],
        anchor_order: Order
    ) -> Dict[str, List[str]]:
        """Filter orders compatible with anchor time window."""
        compatible_ids = [anchor_order.order_id] if anchor_order.order_id in orders else []

        for order_id, order in orders.items():
            if order_id == anchor_order.order_id:
                continue

            if self.are_time_compatible(anchor_order, order):
                compatible_ids.append(order_id)

        return {"compatible": compatible_ids} if compatible_ids else {}

    def _group_by_time_windows(self, orders: Dict[str, Order]) -> Dict[str, List[str]]:
        """Group orders by overlapping time windows."""
        # For simplicity, group by time window start hour
        groups: Dict[str, List[str]] = {}

        for order_id, order in orders.items():
            if order.time_window:
                hour = order.time_window.start_time.hour
                key = f"window_{hour:02d}"
                groups.setdefault(key, []).append(order_id)
            else:
                groups.setdefault("anytime", []).append(order_id)

        return groups

    def are_time_compatible(self, order1: Order, order2: Order) -> bool:
        """Check if two orders have compatible time windows."""
        # If either has no time window, compatible
        if not order1.time_window or not order2.time_window:
            return True

        # Check for overlap
        overlap = self._calculate_overlap_minutes(order1, order2)

        if overlap >= self.config.min_overlap_minutes:
            return True

        # Check if can be served sequentially
        if self.config.allow_sequential_windows:
            return self._check_sequential_feasibility(order1, order2)

        return False

    def _calculate_overlap_minutes(self, order1: Order, order2: Order) -> int:
        """Calculate time window overlap in minutes."""
        if not order1.time_window or not order2.time_window:
            return 0

        start1 = order1.time_window.start_time
        end1 = order1.time_window.end_time
        start2 = order2.time_window.start_time
        end2 = order2.time_window.end_time

        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_end > overlap_start:
            overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60
            return int(overlap_minutes)

        return 0

    def _check_sequential_feasibility(self, order1: Order, order2: Order) -> bool:
        """Check if orders can be served sequentially."""
        if not order1.time_window or not order2.time_window:
            return True

        # Check if one ends before the other starts (with buffer)
        buffer = timedelta(minutes=self.config.buffer_minutes)

        end1 = order1.time_window.end_time
        start2 = order2.time_window.start_time

        end2 = order2.time_window.end_time
        start1 = order1.time_window.start_time

        # Can serve order1 then order2?
        if end1 + buffer <= start2:
            return True

        # Can serve order2 then order1?
        if end2 + buffer <= start1:
            return True

        return False


class CompatibilityFilterChain:
    """Chain of compatibility filters applied in sequence.

    Pipeline:
    1. Service-level filter (reduces incompatible combinations)
    2. Time window filter (validates scheduling)
    """

    def __init__(
        self,
        service_config: Optional[ServiceLevelConfig] = None,
        time_config: Optional[TimeWindowConfig] = None
    ):
        """Initialize filter chain."""
        self.service_filter = ServiceLevelFilter(service_config)
        self.time_filter = TimeWindowFilter(time_config)

    def apply_filters(
        self,
        orders: Dict[str, Order],
        geographic_clusters: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Apply filter chain to geographic clusters.

        Args:
            orders: All orders
            geographic_clusters: Initial geographic clustering

        Returns:
            Final compatible groups after all filters
        """
        final_groups = {}

        # For each geographic cluster, apply service-level and time filters
        for geo_cluster_id, order_ids in geographic_clusters.items():
            cluster_orders = {oid: orders[oid] for oid in order_ids if oid in orders}

            if not cluster_orders:
                continue

            # Apply service-level filter
            service_groups = self.service_filter.filter(cluster_orders)

            # For each service-level group, apply time window filter
            for service_key, service_order_ids in service_groups.items():
                service_orders = {oid: orders[oid] for oid in service_order_ids if oid in orders}

                time_groups = self.time_filter.filter(service_orders)

                # Combine keys
                for time_key, final_order_ids in time_groups.items():
                    combined_key = f"{geo_cluster_id}_{service_key}_{time_key}"
                    final_groups[combined_key] = final_order_ids

        logger.info(
            f"Filter chain: {len(geographic_clusters)} geo clusters → "
            f"{len(final_groups)} final compatible groups"
        )

        return final_groups

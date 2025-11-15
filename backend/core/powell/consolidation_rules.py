"""Consolidation Rules and Business Logic Constraints.

This module defines intelligent consolidation rules to ensure:
- Vehicle utilization efficiency
- Cargo compatibility
- Operational feasibility
- Cost-effectiveness
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import logging

from ..models.domain import Order, Vehicle, Route

logger = logging.getLogger(__name__)


class ConsolidationViolation(Enum):
    """Types of consolidation constraint violations."""

    LOW_UTILIZATION = "low_utilization"  # Vehicle under-utilized
    INCOMPATIBLE_CARGO = "incompatible_cargo"  # Cargo types conflict
    CAPACITY_EXCEEDED = "capacity_exceeded"  # Exceeds vehicle capacity
    TIME_WINDOW_CONFLICT = "time_window_conflict"  # Cannot meet all deadlines
    EXCESSIVE_DETOUR = "excessive_detour"  # Route too inefficient
    MIXED_PRIORITIES = "mixed_priorities"  # High priority with low priority


@dataclass
class ConsolidationConstraints:
    """Configurable consolidation constraints."""

    # Minimum vehicle utilization (by weight)
    min_weight_utilization: float = 0.40  # 40% minimum

    # Minimum vehicle utilization (by volume)
    min_volume_utilization: float = 0.30  # 30% minimum

    # Maximum acceptable utilization (safety margin)
    max_weight_utilization: float = 0.95  # 95% maximum
    max_volume_utilization: float = 0.95  # 95% maximum

    # Consolidation preferences
    prefer_same_destination: bool = True  # Prefer single-destination routes
    max_destinations_per_route: int = 3  # Maximum cities in one route

    # Cargo compatibility rules
    allow_fresh_food_mixing: bool = False  # Fresh food requires dedicated vehicle
    allow_fragile_mixing: bool = True  # Fragile can mix with compatible cargo
    allow_hazardous_mixing: bool = False  # Hazardous requires dedicated vehicle

    # Priority handling
    allow_priority_mixing: bool = True  # Can mix priorities
    max_priority_difference: int = 1  # Maximum priority gap (0=normal, 1=high, 2=urgent)

    # Route efficiency
    max_detour_ratio: float = 1.3  # Route can be max 30% longer than direct
    min_orders_for_consolidation: int = 2  # Minimum orders to justify consolidation

    # Time window constraints
    time_window_buffer_minutes: int = 15  # Safety buffer for time windows

    # Cost efficiency
    min_cost_savings_for_consolidation: float = 500.0  # KES - minimum savings to consolidate


class ConsolidationValidator:
    """Validates and enforces consolidation business rules."""

    def __init__(self, constraints: Optional[ConsolidationConstraints] = None):
        """Initialize validator with constraints."""
        self.constraints = constraints or ConsolidationConstraints()

    def validate_route(
        self,
        route: Route,
        vehicle: Vehicle,
        orders: Dict[str, Order],
    ) -> Tuple[bool, List[ConsolidationViolation], str]:
        """Validate a route against all consolidation rules.

        Returns:
            (is_valid, violations, reasoning)
        """
        violations = []
        reasoning_parts = []

        # Get orders for this route
        route_orders = [orders[oid] for oid in route.order_ids if oid in orders]

        if not route_orders:
            return False, [ConsolidationViolation.CAPACITY_EXCEEDED], "No valid orders"

        # 1. Check utilization constraints
        utilization_valid, util_violations, util_reason = self._check_utilization(
            route, vehicle, route_orders
        )
        if not utilization_valid:
            violations.extend(util_violations)
            reasoning_parts.append(util_reason)

        # 2. Check cargo compatibility
        compat_valid, compat_violations, compat_reason = self._check_cargo_compatibility(
            route_orders
        )
        if not compat_valid:
            violations.extend(compat_violations)
            reasoning_parts.append(compat_reason)

        # 3. Check priority mixing
        priority_valid, priority_violations, priority_reason = self._check_priority_mixing(
            route_orders
        )
        if not priority_valid:
            violations.extend(priority_violations)
            reasoning_parts.append(priority_reason)

        # 4. Check destination constraints
        dest_valid, dest_violations, dest_reason = self._check_destination_constraints(
            route
        )
        if not dest_valid:
            violations.extend(dest_violations)
            reasoning_parts.append(dest_reason)

        is_valid = len(violations) == 0
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "All constraints satisfied"

        return is_valid, violations, reasoning

    def _check_utilization(
        self,
        route: Route,
        vehicle: Vehicle,
        orders: List[Order],
    ) -> Tuple[bool, List[ConsolidationViolation], str]:
        """Check if vehicle utilization is within acceptable bounds."""
        violations = []

        # Calculate utilization
        total_weight = sum(o.weight_tonnes for o in orders)
        total_volume = sum(o.volume_m3 for o in orders)

        weight_util = total_weight / vehicle.capacity_weight_tonnes
        volume_util = total_volume / vehicle.capacity_volume_m3

        # Check minimum utilization
        if weight_util < self.constraints.min_weight_utilization:
            violations.append(ConsolidationViolation.LOW_UTILIZATION)
            return (
                False,
                violations,
                f"Weight utilization {weight_util:.1%} below minimum {self.constraints.min_weight_utilization:.1%} "
                f"({total_weight:.1f}T / {vehicle.capacity_weight_tonnes:.1f}T on {vehicle.vehicle_type} truck)"
            )

        if volume_util < self.constraints.min_volume_utilization:
            violations.append(ConsolidationViolation.LOW_UTILIZATION)
            return (
                False,
                violations,
                f"Volume utilization {volume_util:.1%} below minimum {self.constraints.min_volume_utilization:.1%} "
                f"({total_volume:.1f}m³ / {vehicle.capacity_volume_m3:.1f}m³ on {vehicle.vehicle_type} truck)"
            )

        # Check maximum utilization (safety)
        if weight_util > self.constraints.max_weight_utilization:
            violations.append(ConsolidationViolation.CAPACITY_EXCEEDED)
            return (
                False,
                violations,
                f"Weight utilization {weight_util:.1%} exceeds safe maximum {self.constraints.max_weight_utilization:.1%}"
            )

        if volume_util > self.constraints.max_volume_utilization:
            violations.append(ConsolidationViolation.CAPACITY_EXCEEDED)
            return (
                False,
                violations,
                f"Volume utilization {volume_util:.1%} exceeds safe maximum {self.constraints.max_volume_utilization:.1%}"
            )

        return True, [], ""

    def _check_cargo_compatibility(
        self, orders: List[Order]
    ) -> Tuple[bool, List[ConsolidationViolation], str]:
        """Check if cargo types are compatible."""
        violations = []

        # Extract special handling requirements
        has_fresh_food = any("fresh_food" in (o.special_handling or []) for o in orders)
        has_fragile = any("fragile" in (o.special_handling or []) for o in orders)
        has_hazardous = any("hazardous" in (o.special_handling or []) for o in orders)

        # Fresh food mixing rules
        if has_fresh_food and len(orders) > 1:
            if not self.constraints.allow_fresh_food_mixing:
                violations.append(ConsolidationViolation.INCOMPATIBLE_CARGO)
                return (
                    False,
                    violations,
                    "Fresh food requires dedicated vehicle (cannot mix with other cargo)"
                )

        # Hazardous mixing rules
        if has_hazardous and len(orders) > 1:
            if not self.constraints.allow_hazardous_mixing:
                violations.append(ConsolidationViolation.INCOMPATIBLE_CARGO)
                return (
                    False,
                    violations,
                    "Hazardous cargo requires dedicated vehicle"
                )

        # Fragile + heavy cargo check
        if has_fragile and not self.constraints.allow_fragile_mixing:
            other_heavy = [o for o in orders if o.weight_tonnes > 2.0]
            if other_heavy:
                violations.append(ConsolidationViolation.INCOMPATIBLE_CARGO)
                return (
                    False,
                    violations,
                    "Fragile cargo cannot be mixed with heavy loads"
                )

        return True, [], ""

    def _check_priority_mixing(
        self, orders: List[Order]
    ) -> Tuple[bool, List[ConsolidationViolation], str]:
        """Check if priority levels can be mixed."""
        if len(orders) <= 1:
            return True, [], ""

        violations = []
        priorities = [o.priority for o in orders]
        min_priority = min(priorities)
        max_priority = max(priorities)
        priority_diff = max_priority - min_priority

        if priority_diff > self.constraints.max_priority_difference:
            violations.append(ConsolidationViolation.MIXED_PRIORITIES)
            return (
                False,
                violations,
                f"Priority difference ({priority_diff}) exceeds maximum ({self.constraints.max_priority_difference}). "
                f"Cannot mix urgent/high priority with normal priority orders."
            )

        return True, [], ""

    def _check_destination_constraints(
        self, route: Route
    ) -> Tuple[bool, List[ConsolidationViolation], str]:
        """Check destination-related constraints."""
        violations = []

        num_destinations = len(route.destination_cities)

        if num_destinations > self.constraints.max_destinations_per_route:
            violations.append(ConsolidationViolation.EXCESSIVE_DETOUR)
            return (
                False,
                violations,
                f"Route has {num_destinations} destinations, exceeds maximum {self.constraints.max_destinations_per_route}"
            )

        return True, [], ""

    def should_consolidate(
        self,
        orders: List[Order],
        estimated_single_cost: float,
        estimated_consolidated_cost: float,
    ) -> Tuple[bool, str]:
        """Determine if consolidation makes business sense.

        Returns:
            (should_consolidate, reasoning)
        """
        # Need minimum orders
        if len(orders) < self.constraints.min_orders_for_consolidation:
            return False, f"Only {len(orders)} order(s), need minimum {self.constraints.min_orders_for_consolidation} for consolidation"

        # Check cost savings
        cost_savings = estimated_single_cost - estimated_consolidated_cost

        if cost_savings < self.constraints.min_cost_savings_for_consolidation:
            return (
                False,
                f"Cost savings ({cost_savings:.0f} KES) below minimum threshold ({self.constraints.min_cost_savings_for_consolidation:.0f} KES)"
            )

        return True, f"Consolidation saves {cost_savings:.0f} KES with {len(orders)} orders"

    def get_optimal_vehicle_for_load(
        self,
        total_weight: float,
        total_volume: float,
        available_vehicles: List[Vehicle],
    ) -> Optional[Vehicle]:
        """Find the most appropriately-sized vehicle for a load.

        Prefers smallest vehicle that meets minimum utilization thresholds.
        """
        candidates = []

        for vehicle in available_vehicles:
            # Check if load fits
            if not vehicle.has_capacity_for(total_weight, total_volume):
                continue

            # Calculate utilization
            weight_util = total_weight / vehicle.capacity_weight_tonnes
            volume_util = total_volume / vehicle.capacity_volume_m3

            # Check minimum utilization
            if (
                weight_util >= self.constraints.min_weight_utilization
                and volume_util >= self.constraints.min_volume_utilization
            ):
                # Score: prefer higher utilization (more efficient)
                avg_util = (weight_util + volume_util) / 2
                candidates.append((avg_util, vehicle))

        if not candidates:
            return None

        # Return vehicle with highest utilization (best fit)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


def get_consolidation_opportunities(
    orders: Dict[str, Order],
    constraints: Optional[ConsolidationConstraints] = None,
) -> Dict[str, List[str]]:
    """Identify which orders can be consolidated together.

    Returns:
        Dictionary mapping consolidation group key to list of order IDs
    """
    constraints = constraints or ConsolidationConstraints()
    groups = {}

    for order_id, order in orders.items():
        # Group by destination city (primary consolidation criteria)
        group_key = f"{order.destination_city.value}"

        # Add special handling to group key if strict separation required
        if order.special_handling:
            if "fresh_food" in order.special_handling and not constraints.allow_fresh_food_mixing:
                group_key += "_fresh"
            if "hazardous" in order.special_handling and not constraints.allow_hazardous_mixing:
                group_key += "_hazardous"

        # Add priority to group key if strict separation required
        if not constraints.allow_priority_mixing:
            group_key += f"_p{order.priority}"

        if group_key not in groups:
            groups[group_key] = []

        groups[group_key].append(order_id)

    # Filter out single-order groups if consolidation requires minimum
    if constraints.min_orders_for_consolidation > 1:
        groups = {
            k: v for k, v in groups.items()
            if len(v) >= constraints.min_orders_for_consolidation
        }

    return groups

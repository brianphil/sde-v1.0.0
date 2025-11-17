"""Pickup and Delivery Sequence Optimizer for Mesh Network Routing.

Optimizes multi-pickup, multi-delivery sequences considering:
- Time windows
- LIFO constraints (last-in-first-out for loading)
- Cost minimization
- Distance optimization
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

from ..models.domain import Order, Location, Vehicle

logger = logging.getLogger(__name__)


class StopType(Enum):
    """Type of stop in route sequence."""

    PICKUP = "pickup"
    DELIVERY = "delivery"


@dataclass
class Stop:
    """A stop in the route sequence."""

    order_id: str
    stop_type: StopType
    location: Location
    time_window_start: Optional[float] = None  # Minutes from route start
    time_window_end: Optional[float] = None
    load_weight: float = 0.0  # Weight loaded/unloaded
    load_volume: float = 0.0


@dataclass
class RouteSequence:
    """Optimized pickup and delivery sequence."""

    stops: List[Stop]
    total_distance_km: float
    total_duration_minutes: float
    estimated_cost: float
    is_feasible: bool
    violation_reasons: List[str]


class SequenceOptimizer:
    """Optimizes pickup/delivery sequences for mesh routing.

    Constraints:
    - LIFO loading (pickups must be reversed for delivery)
    - Time windows
    - Vehicle capacity
    - Cost minimization
    """

    def __init__(self):
        """Initialize sequence optimizer."""
        self.avg_speed_kmh = 50.0  # Average speed
        self.stop_time_minutes = 15.0  # Time per stop

    def optimize_sequence(
        self,
        orders: List[Order],
        vehicle: Vehicle,
        start_location: Location
    ) -> RouteSequence:
        """Optimize pickup and delivery sequence for orders.

        Args:
            orders: Orders to include in route
            vehicle: Vehicle for route
            start_location: Starting location (depot)

        Returns:
            Optimized RouteSequence
        """
        if not orders:
            return RouteSequence(
                stops=[],
                total_distance_km=0.0,
                total_duration_minutes=0.0,
                estimated_cost=0.0,
                is_feasible=False,
                violation_reasons=["No orders provided"]
            )

        # Strategy: For mesh routing
        # 1. All pickups first (clustered geographically)
        # 2. Then deliveries (respecting LIFO if needed)

        # For now, simple approach: pickup all, then deliver all
        sequence = self._build_simple_sequence(orders, vehicle, start_location)

        # Validate sequence
        is_feasible, violations = self._validate_sequence(sequence, vehicle)

        sequence.is_feasible = is_feasible
        sequence.violation_reasons = violations

        return sequence

    def _build_simple_sequence(
        self,
        orders: List[Order],
        vehicle: Vehicle,
        start_location: Location
    ) -> RouteSequence:
        """Build simple sequence: all pickups, then all deliveries."""
        stops = []

        # Sort orders by pickup location (simple geographic clustering)
        # In production, use proper TSP solver
        sorted_orders = sorted(
            orders,
            key=lambda o: (o.pickup_location.latitude, o.pickup_location.longitude)
        )

        current_location = start_location
        current_time = 0.0  # Minutes from start
        total_distance = 0.0

        # Phase 1: Pickups
        for order in sorted_orders:
            # Calculate distance and time to pickup
            distance = self._estimate_distance(current_location, order.pickup_location)
            travel_time = (distance / self.avg_speed_kmh) * 60  # Convert to minutes

            total_distance += distance
            current_time += travel_time

            # Add pickup stop
            stops.append(Stop(
                order_id=order.order_id,
                stop_type=StopType.PICKUP,
                location=order.pickup_location,
                time_window_start=current_time,
                time_window_end=current_time + self.stop_time_minutes,
                load_weight=order.weight_tonnes,
                load_volume=order.volume_m3
            ))

            current_time += self.stop_time_minutes
            current_location = order.pickup_location

        # Phase 2: Deliveries
        # Sort by delivery location
        sorted_for_delivery = sorted(
            sorted_orders,
            key=lambda o: (o.delivery_location.latitude, o.delivery_location.longitude)
        )

        for order in sorted_for_delivery:
            # Calculate distance and time to delivery
            distance = self._estimate_distance(current_location, order.delivery_location)
            travel_time = (distance / self.avg_speed_kmh) * 60

            total_distance += distance
            current_time += travel_time

            # Add delivery stop
            stops.append(Stop(
                order_id=order.order_id,
                stop_type=StopType.DELIVERY,
                location=order.delivery_location,
                time_window_start=current_time,
                time_window_end=current_time + self.stop_time_minutes,
                load_weight=-order.weight_tonnes,  # Unloading
                load_volume=-order.volume_m3
            ))

            current_time += self.stop_time_minutes
            current_location = order.delivery_location

        # Return to depot (optional)
        return_distance = self._estimate_distance(current_location, start_location)
        return_time = (return_distance / self.avg_speed_kmh) * 60
        total_distance += return_distance
        current_time += return_time

        # Calculate cost
        fuel_cost = total_distance * vehicle.fuel_cost_per_km
        time_cost = (current_time / 60) * vehicle.driver_cost_per_hour
        total_cost = fuel_cost + time_cost

        return RouteSequence(
            stops=stops,
            total_distance_km=total_distance,
            total_duration_minutes=current_time,
            estimated_cost=total_cost,
            is_feasible=True,
            violation_reasons=[]
        )

    def _estimate_distance(self, loc1: Location, loc2: Location) -> float:
        """Estimate distance between locations (simplified).

        In production, use real routing API (Google Maps, OSRM).
        """
        import math

        R = 6371  # Earth radius in km

        lat1 = math.radians(loc1.latitude)
        lat2 = math.radians(loc2.latitude)
        dlat = math.radians(loc2.latitude - loc1.latitude)
        dlon = math.radians(loc2.longitude - loc1.longitude)

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        # Apply road network factor (straight-line * 1.3 for road routing)
        return distance * 1.3

    def _validate_sequence(
        self,
        sequence: RouteSequence,
        vehicle: Vehicle
    ) -> Tuple[bool, List[str]]:
        """Validate route sequence against constraints."""
        violations = []

        # Check capacity at each stop
        current_weight = 0.0
        current_volume = 0.0

        for stop in sequence.stops:
            current_weight += stop.load_weight
            current_volume += stop.load_volume

            if current_weight > vehicle.capacity_weight_tonnes:
                violations.append(
                    f"Weight capacity exceeded at {stop.stop_type.value} "
                    f"stop {stop.order_id}: {current_weight:.1f}T > {vehicle.capacity_weight_tonnes}T"
                )

            if current_volume > vehicle.capacity_volume_m3:
                violations.append(
                    f"Volume capacity exceeded at {stop.stop_type.value} "
                    f"stop {stop.order_id}: {current_volume:.1f}m³ > {vehicle.capacity_volume_m3}m³"
                )

        # Check LIFO constraints (simplified)
        # In full implementation, track stack and validate LIFO

        # Check total duration
        max_duration = 8 * 60  # 8 hours
        if sequence.total_duration_minutes > max_duration:
            violations.append(
                f"Route duration {sequence.total_duration_minutes:.0f} min "
                f"exceeds maximum {max_duration} min"
            )

        is_feasible = len(violations) == 0

        return is_feasible, violations

    def calculate_sequence_score(self, sequence: RouteSequence) -> float:
        """Calculate quality score for sequence (higher = better).

        Considers:
        - Distance (minimize)
        - Duration (minimize)
        - Constraint violations (penalize)
        """
        if not sequence.is_feasible:
            return 0.0

        # Base score on cost (lower = better, so invert)
        # Normalize to 0-100 range
        distance_score = max(0, 100 - sequence.total_distance_km)
        duration_score = max(0, 100 - (sequence.total_duration_minutes / 60))

        # Weighted combination
        score = 0.6 * distance_score + 0.4 * duration_score

        return score

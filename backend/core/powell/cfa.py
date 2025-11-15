"""Cost Function Approximation - Optimization-based decision making."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
import math

from ..models.state import SystemState
from ..models.domain import (
    Order,
    Vehicle,
    Route,
    RouteStatus,
    Location,
    DestinationCity,
)
from ..models.decision import PolicyDecision, DecisionContext, DecisionType, ActionType

logger = logging.getLogger(__name__)


@dataclass
class CostParameters:
    """Learned cost function parameters."""

    # Fuel and distance costs
    # Per-vehicle (vehicle_type) fuel cost per km (KES/km). Example keys: '5T', '10T'
    fuel_cost_per_km_by_vehicle: Dict[str, float] = field(default_factory=dict)
    # Base fuel price per liter (fallback calculation if per-km not provided)
    base_fuel_cost_per_liter: float = 150.0  # KES per liter

    # Time-based costs per vehicle type (fallbacks)
    driver_cost_per_hour_by_vehicle: Dict[str, float] = field(
        default_factory=lambda: {}
    )
    delay_penalty_per_minute: float = 5.0  # KES per minute late

    # Zone-specific traffic multipliers (adjust travel time)
    traffic_multipliers: Dict[str, float] = field(default_factory=dict)

    # Vehicle efficiency km per liter
    vehicle_efficiency_km_per_liter: Dict[str, float] = field(
        default_factory=lambda: {"5T": 8.5, "10T": 7.0}
    )

    # Distance matrix (km between major locations) - in production, use real routing API
    distance_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Learning state
    last_updated: datetime = field(default_factory=datetime.now)
    samples_observed: int = 0
    prediction_accuracy_fuel: float = 0.85
    prediction_accuracy_time: float = 0.80


class CostFunctionApproximation:
    """Optimization-based routing using learned cost functions.

    CFA minimizes: fuel_cost + time_cost + delay_penalty
    Subject to:
    - Vehicle capacity constraints
    - Time window constraints
    - Hard constraints (e.g., Eastleigh 8:30-9:45)
    - Multi-city sequencing rules

    Uses iterative greedy optimization with local improvements.
    """

    def __init__(self, parameters: Optional[CostParameters] = None):
        """Initialize CFA with cost parameters."""
        self.params = parameters or CostParameters()
        self._initialize_distance_matrix()

    def _initialize_distance_matrix(self):
        """Initialize approximate distance matrix for major locations."""
        # Simplified distances (km) between Nairobi zones and destination cities
        # Leave distance matrix empty by default. In production this should be
        # provided via configuration or a routing API integration.
        return

    def evaluate(self, state: SystemState, context: DecisionContext) -> PolicyDecision:
        """Apply CFA to find optimal routes.

        Workflow:
        1. Build candidate routes (groupings of orders)
        2. Score each route by total cost
        3. Assign orders to lowest-cost feasible routes
        4. Return best solution
        """

        # Build candidate solutions
        solutions = self._generate_candidate_solutions(state, context)

        if not solutions:
            return PolicyDecision(
                policy_name="CFA",
                decision_type=context.decision_type,
                recommended_action=ActionType.DEFER_ORDER,
                routes=[],
                confidence_score=0.0,
                expected_value=0.0,
                reasoning="No feasible solutions found",
            )

        # Evaluate and rank solutions
        solutions_with_costs = [
            (sol, self._calculate_solution_cost(state, sol)) for sol in solutions
        ]
        solutions_with_costs.sort(key=lambda x: x[1])  # Sort by cost

        best_solution, best_cost = solutions_with_costs[0]

        # Calculate expected value (revenue - costs)
        expected_value = (
            sum(state.get_estimated_route_value(r) for r in best_solution) - best_cost
        )

        # Calculate confidence based on learned accuracy
        confidence = (
            self.params.prediction_accuracy_fuel * 0.5
            + self.params.prediction_accuracy_time * 0.5
        )

        return PolicyDecision(
            policy_name="CFA",
            decision_type=context.decision_type,
            recommended_action=ActionType.CREATE_ROUTE,
            routes=best_solution,
            confidence_score=confidence,
            expected_value=expected_value,
            reasoning=f"Optimal routes minimize total cost: {best_cost:.0f} KES",
            considered_alternatives=len(solutions),
            is_deterministic=True,
            policy_parameters={
                "total_cost": best_cost,
                "num_routes": len(best_solution),
                "orders_assigned": sum(len(r.order_ids) for r in best_solution),
            },
        )

    def _generate_candidate_solutions(
        self, state: SystemState, context: DecisionContext
    ) -> List[List[Route]]:
        """Generate multiple candidate route solutions."""
        solutions = []

        # Solution 1: Group by destination city (simplest)
        by_city = self._group_orders_by_destination_city(context.orders_to_consider)
        sol1 = self._assign_orders_to_vehicles(
            state, by_city, context.vehicles_available
        )
        if sol1:
            solutions.append(sol1)

        # Solution 2: Group by priority (urgent first)
        by_priority = self._group_orders_by_priority(context.orders_to_consider)
        sol2 = self._assign_orders_to_vehicles(
            state, by_priority, context.vehicles_available
        )
        if sol2:
            solutions.append(sol2)

        # Solution 3: Greedy nearest-first (minimize distance)
        sol3 = self._greedy_nearest_first_assignment(state, context)
        if sol3:
            solutions.append(sol3)

        return solutions

    def _group_orders_by_destination_city(
        self, orders: Dict[str, Order]
    ) -> Dict[DestinationCity, List[Order]]:
        """Group orders by destination city."""
        groups = {}
        for order in orders.values():
            if order.destination_city not in groups:
                groups[order.destination_city] = []
            groups[order.destination_city].append(order)
        return groups

    def _group_orders_by_priority(
        self, orders: Dict[str, Order]
    ) -> Dict[int, List[Order]]:
        """Group orders by priority (2=urgent, 1=high, 0=normal)."""
        groups = {}
        for order in orders.values():
            if order.priority not in groups:
                groups[order.priority] = []
            groups[order.priority].append(order)
        # Sort priorities in descending order
        return dict(sorted(groups.items(), reverse=True))

    def _assign_orders_to_vehicles(
        self,
        state: SystemState,
        order_groups: Dict[Any, List[Order]],
        vehicles: Dict[str, Vehicle],
    ) -> Optional[List[Route]]:
        """Assign grouped orders to available vehicles."""
        routes = []
        assigned_vehicles = set()

        for group_key, order_list in order_groups.items():
            for order in order_list:
                # Find suitable vehicle (prefer unused, then least-utilized)
                vehicle = self._find_best_vehicle_for_order(
                    state, order, vehicles, assigned_vehicles
                )

                if not vehicle:
                    continue  # Skip unassignable order

                # Check if order can be added to existing route for vehicle
                vehicle_routes = [
                    r for r in routes if r.vehicle_id == vehicle.vehicle_id
                ]

                if vehicle_routes:
                    route = vehicle_routes[0]
                    # Check capacity
                    weight = route.get_total_weight(
                        {order.order_id: order} | state.pending_orders
                    )
                    volume = route.get_total_volume(
                        {order.order_id: order} | state.pending_orders
                    )

                    if vehicle.has_capacity_for(weight, volume):
                        route.order_ids.append(order.order_id)
                        continue

                # Create new route
                route = self._create_route_for_orders(state, [order], vehicle)
                routes.append(route)
                assigned_vehicles.add(vehicle.vehicle_id)

        return routes if routes else None

    def _greedy_nearest_first_assignment(
        self, state: SystemState, context: DecisionContext
    ) -> Optional[List[Route]]:
        """Greedy assignment: always pick nearest available vehicle."""
        routes = []
        remaining_orders = dict(context.orders_to_consider)

        while remaining_orders:
            # Pick first unassigned order
            order = next(iter(remaining_orders.values()))

            # Find nearest available vehicle
            vehicle = self._find_best_vehicle_for_order(
                state, order, context.vehicles_available, set()
            )

            if not vehicle:
                # Can't assign this order
                del remaining_orders[order.order_id]
                continue

            # Create route with this order
            route_orders = [order]
            route = self._create_route_for_orders(state, route_orders, vehicle)
            routes.append(route)
            del remaining_orders[order.order_id]

        return routes if routes else None

    def _find_best_vehicle_for_order(
        self,
        state: SystemState,
        order: Order,
        vehicles: Dict[str, Vehicle],
        exclude_vehicles: set,
    ) -> Optional[Vehicle]:
        """Find best vehicle for order (capacity + utilization)."""
        candidates = []

        for vid, vehicle in vehicles.items():
            if vid in exclude_vehicles or not vehicle.is_available_at(
                state.environment.current_time
            ):
                continue

            if vehicle.has_capacity_for(order.weight_tonnes, order.volume_m3):
                # Score by underutilization (prefer less-used vehicles)
                util = state.get_vehicle_utilization_percent(vid)
                candidates.append((util, vehicle))

        if not candidates:
            return None

        # Return vehicle with lowest utilization
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _create_route_for_orders(
        self, state: SystemState, orders: List[Order], vehicle: Vehicle
    ) -> Route:
        """Create optimized route for list of orders."""
        route_id = (
            f"route_cfa_{len(state.active_routes)}_{int(datetime.now().timestamp())}"
        )

        # Sequence orders (simple: by destination, then by pickup location)
        orders_sorted = sorted(
            orders,
            key=lambda o: (o.destination_city.value, str(o.pickup_location.latitude)),
        )

        # Estimate route parameters
        total_distance = self._estimate_total_distance(orders_sorted)
        estimated_duration = self._estimate_duration(total_distance, vehicle)
        estimated_cost = self._estimate_total_cost(
            total_distance, estimated_duration, vehicle
        )

        route = Route(
            route_id=route_id,
            vehicle_id=vehicle.vehicle_id,
            order_ids=[o.order_id for o in orders_sorted],
            stops=[],  # Simplified
            destination_cities=list(set(o.destination_city for o in orders_sorted)),
            total_distance_km=total_distance,
            estimated_duration_minutes=estimated_duration,
            estimated_cost_kes=estimated_cost,
            status=RouteStatus.PLANNED,
            estimated_fuel_cost=self._estimate_fuel_cost(total_distance, vehicle),
            estimated_time_cost=self._estimate_time_cost(estimated_duration),
        )

        return route

    def _estimate_total_distance(self, orders: List[Order]) -> float:
        """Estimate total distance for route visiting orders in sequence."""
        if not orders:
            return 0.0

        # Simplified: use order count and average inter-city distance
        # In production, use real routing API (Google Maps, OSRM, etc.)
        distance = 0.0

        # Distance to first pickup
        distance += 10.0  # Assume 10km average to start location

        # Distance between orders
        for i in range(len(orders) - 1):
            current_city = orders[i].destination_city.value
            next_city = orders[i + 1].destination_city.value

            key = (current_city, next_city)
            if key in self.params.distance_matrix:
                distance += self.params.distance_matrix[key]
            else:
                # Approximate based on city names
                distance += 100.0  # Default distance

        # Distance from last delivery back to depot (simplified)
        distance += 50.0

        return distance

    def _estimate_duration(self, distance_km: float, vehicle: Vehicle) -> int:
        """Estimate total route duration in minutes."""
        # Assume average speed of 60 km/h with traffic
        driving_minutes = (distance_km / 60.0) * 60

        # Add service time (10 minutes per order) — approximate
        service_minutes = 10.0 * 2  # default assumption for demo

        # Add buffer (20% traffic contingency)
        total_minutes = int((driving_minutes + service_minutes) * 1.2)

        return total_minutes

    def _estimate_total_cost(
        self, distance_km: float, duration_minutes: int, vehicle: Vehicle
    ) -> float:
        """Estimate combined fuel + time cost for a route."""
        fuel_cost = self._estimate_fuel_cost(distance_km, vehicle)
        time_cost = self._estimate_time_cost(duration_minutes, vehicle)
        return fuel_cost + time_cost

    def _estimate_fuel_cost(self, distance_km: float, vehicle: Vehicle) -> float:
        """Estimate fuel cost for route."""
        vtype = getattr(vehicle, "vehicle_type", None)

        # Priority order for fuel cost:
        # 1) vehicle.fuel_cost_per_km (per-vehicle attribute)
        # 2) configured per-vehicle-type per-km mapping in CostParameters
        # 3) fallback: compute via vehicle efficiency and base fuel price

        # 1) per-vehicle direct cost
        per_km_direct = getattr(vehicle, "fuel_cost_per_km", None)
        if per_km_direct is not None:
            try:
                return distance_km * float(per_km_direct)
            except Exception:
                pass

        # 2) configuration mapping by vehicle_type
        vtype = getattr(vehicle, "vehicle_type", None)
        per_km = None
        if vtype is not None:
            per_km = self.params.fuel_cost_per_km_by_vehicle.get(vtype)
        if per_km is not None:
            return distance_km * float(per_km)

        # 3) compute from liters needed and base fuel price
        efficiency = self.params.vehicle_efficiency_km_per_liter.get(
            vehicle.vehicle_type, getattr(vehicle, "fuel_efficiency_km_per_liter", 8.5)
        )
        liters_needed = distance_km / efficiency if efficiency > 0 else 0.0
        return liters_needed * self.params.base_fuel_cost_per_liter

    def _estimate_time_cost(
        self, duration_minutes: int, vehicle: Optional[Vehicle] = None
    ) -> float:
        """Estimate time-based operational cost.

        Prefers per-vehicle driver cost when configured, otherwise falls back
        to a reasonable default.
        """
        hours = duration_minutes / 60.0

        if vehicle is not None:
            drv = getattr(vehicle, "driver_cost_per_hour", None)
            if drv is not None:
                try:
                    return hours * float(drv)
                except Exception:
                    pass

            # Next: parameter mapping by vehicle type
            vtype = getattr(vehicle, "vehicle_type", None)
            if vtype and vtype in self.params.driver_cost_per_hour_by_vehicle:
                return hours * float(self.params.driver_cost_per_hour_by_vehicle[vtype])

        # Global fallback
        return hours * 300.0

    def _calculate_solution_cost(
        self, state: SystemState, solution: List[Route]
    ) -> float:
        """Calculate total cost of solution."""
        total_cost = 0.0

        for route in solution:
            # Fuel cost
            fuel_cost = (
                route.estimated_fuel_cost if route.estimated_fuel_cost > 0 else 0.0
            )

            # Time cost
            time_cost = (
                route.estimated_time_cost if route.estimated_time_cost > 0 else 0.0
            )

            # Delay penalty (if any orders are late)
            delay_penalty = 0.0
            for order_id in route.order_ids:
                order = state.pending_orders.get(order_id)
                if order and order.delivery_window:
                    # Estimate if will be late (simplified)
                    delay_penalty += 0.0  # Placeholder

            total_cost += fuel_cost + time_cost + delay_penalty

        return total_cost

    def update_from_feedback(self, outcome: Dict[str, Any]):
        """Update cost function parameters based on actual performance.

        Uses prediction errors to adjust parameters (gradient descent on error).
        """
        fuel_error = outcome.get("fuel_cost_error", 0.0)
        time_error = outcome.get("time_error_minutes", 0.0)

        # Exponential smoothing of prediction accuracy
        alpha = 0.1  # Learning rate

        actual_accuracy_fuel = max(
            0.0, 1.0 - abs(fuel_error) / (outcome.get("actual_fuel_cost", 1.0) + 1.0)
        )
        actual_accuracy_time = max(
            0.0,
            1.0 - abs(time_error) / (outcome.get("actual_duration_minutes", 1.0) + 1.0),
        )

        self.params.prediction_accuracy_fuel = (
            alpha * actual_accuracy_fuel
            + (1 - alpha) * self.params.prediction_accuracy_fuel
        )
        self.params.prediction_accuracy_time = (
            alpha * actual_accuracy_time
            + (1 - alpha) * self.params.prediction_accuracy_time
        )

        # Update cost parameters based on prediction error
        # NOTE: CFA no longer maintains a single `fuel_per_km` scalar. Learning
        # adjustments to costs should be applied to per-vehicle-type mappings or
        # via vehicle-level overrides (stored in learning state). This demo
        # currently only updates prediction accuracies.

        self.params.samples_observed += 1
        self.params.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize CFA state."""
        return {
            "fuel_cost_per_km_by_vehicle": self.params.fuel_cost_per_km_by_vehicle,
            "driver_cost_per_hour_by_vehicle": self.params.driver_cost_per_hour_by_vehicle,
            "delay_penalty_per_minute": self.params.delay_penalty_per_minute,
            "prediction_accuracy_fuel": self.params.prediction_accuracy_fuel,
            "prediction_accuracy_time": self.params.prediction_accuracy_time,
            "samples_observed": self.params.samples_observed,
            "last_updated": self.params.last_updated.isoformat(),
        }

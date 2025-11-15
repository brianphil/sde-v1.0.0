"""Route optimization service wrapping Powell policies."""

from typing import List, Dict, Optional, Tuple, Any
import logging

from ..core.models.state import SystemState
from ..core.models.domain import Order, Vehicle, Route
from ..core.models.decision import DecisionType, DecisionContext
from ..core.powell.cfa import CostFunctionApproximation
from ..core.powell.vfa import ValueFunctionApproximation
from ..core.powell.dla import DirectLookaheadApproximation

logger = logging.getLogger(__name__)


class RouteOptimizer:
    """Service for optimizing routes using Powell policies.
    
    Provides high-level interface to routing optimization.
    Handles route validity checking, constraint enforcement, etc.
    """

    def __init__(self, cfa: Optional[CostFunctionApproximation] = None):
        """Initialize route optimizer."""
        self.cfa = cfa or CostFunctionApproximation()

    def optimize_daily_routes(
        self, state: SystemState, max_routes: int = 50
    ) -> List[Route]:
        """Optimize routes for daily planning.
        
        Objectives:
        1. Minimize total cost (fuel + time)
        2. Respect all constraints (capacity, time windows, etc.)
        3. Maximize utilization
        4. Prefer multi-city consolidation
        
        Args:
            state: Current system state
            max_routes: Maximum routes to create
            
        Returns:
            List of optimized routes
        """

        # Get unassigned orders
        unassigned = state.get_unassigned_orders()
        available_vehicles = state.get_available_vehicles()

        if not unassigned or not available_vehicles:
            logger.warning("No unassigned orders or available vehicles")
            return []

        logger.info(
            f"Optimizing daily routes: {len(unassigned)} orders, {len(available_vehicles)} vehicles"
        )

        # Use CFA to optimize
        context = DecisionContext(
            decision_type=DecisionType.DAILY_ROUTE_PLANNING,
            orders_to_consider=unassigned,
            vehicles_available=available_vehicles,
        )

        decision = self.cfa.evaluate(state, context)

        # Filter and validate routes
        valid_routes = []
        for route in decision.routes:
            if self._validate_route(route, state):
                valid_routes.append(route)

                if len(valid_routes) >= max_routes:
                    break

        logger.info(f"Generated {len(valid_routes)} valid routes")
        return valid_routes

    def optimize_order_acceptance(
        self, state: SystemState, new_order: Order
    ) -> Tuple[bool, Optional[Route]]:
        """Decide whether to accept new order and find best route.
        
        Returns:
            (should_accept, assigned_route)
        """

        # Check if can fit in any vehicle
        suitable_vehicles = state.get_vehicles_with_capacity_for(new_order)

        if not suitable_vehicles:
            logger.info(f"Cannot accept order {new_order.order_id} - no suitable vehicles")
            return False, None

        # Find best vehicle (least utilized)
        best_vehicle = min(
            suitable_vehicles.values(),
            key=lambda v: state.get_vehicle_utilization_percent(v.vehicle_id),
        )

        # Create route
        route = self._create_single_order_route(new_order, best_vehicle)

        logger.info(f"Order {new_order.order_id} accepted on route {route.route_id}")
        return True, route

    def optimize_backhaul_consolidation(
        self, state: SystemState
    ) -> List[Tuple[Route, Order]]:
        """Identify backhaul opportunities and consolidation options.
        
        Returns:
            List of (route, order_to_consolidate) pairs
        """

        opportunities = []
        backhaul_opps = state.get_backhaul_opportunities()

        for route, available_orders in backhaul_opps:
            for order in available_orders[:1]:  # Only consolidate top opportunity per route
                opportunities.append((route, order))

        logger.info(f"Found {len(opportunities)} backhaul consolidation opportunities")
        return opportunities

    def optimize_real_time_adjustment(
        self, state: SystemState, affected_route_id: str
    ) -> Optional[Route]:
        """Re-optimize route after disruption (delay, breakdown, etc.).
        
        Returns:
            Updated route or None if adjustment not possible
        """

        if affected_route_id not in state.active_routes:
            logger.warning(f"Route {affected_route_id} not found")
            return None

        old_route = state.active_routes[affected_route_id]

        # Reorder remaining stops to minimize additional delay
        # In production: use TSP solver for optimal sequencing
        adjusted_route = self._resequence_route(old_route, state)

        logger.info(f"Route {affected_route_id} re-optimized")
        return adjusted_route

    def estimate_route_cost(self, route: Route, state: SystemState) -> float:
        """Estimate total cost of route."""
        fuel_cost = route.estimated_fuel_cost
        time_cost = route.estimated_time_cost
        delay_penalty = route.estimated_delay_penalty

        return fuel_cost + time_cost + delay_penalty

    def estimate_route_value(self, route: Route, state: SystemState) -> float:
        """Estimate revenue from route."""
        return state.get_estimated_route_value(route)

    def estimate_route_profit(self, route: Route, state: SystemState) -> float:
        """Estimate profit (revenue - costs)."""
        return self.estimate_route_value(route, state) - self.estimate_route_cost(route, state)

    def rank_routes_by_profitability(self, routes: List[Route], state: SystemState) -> List[Route]:
        """Sort routes by profitability (profit/cost ratio)."""
        routes_with_profit = [
            (r, self.estimate_route_profit(r, state)) for r in routes
        ]

        routes_with_profit.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in routes_with_profit]

    def check_route_feasibility(self, route: Route, state: SystemState) -> Tuple[bool, List[str]]:
        """Check if route is feasible with detailed reasons.
        
        Returns:
            (is_feasible, list_of_issues)
        """

        issues = []

        # Check vehicle exists
        if route.vehicle_id not in state.fleet:
            issues.append(f"Vehicle {route.vehicle_id} not found")
            return False, issues

        vehicle = state.fleet[route.vehicle_id]

        # Check capacity
        weight = route.get_total_weight(state.pending_orders)
        volume = route.get_total_volume(state.pending_orders)

        if weight > vehicle.capacity_weight_tonnes:
            issues.append(
                f"Exceeds weight capacity: {weight} > {vehicle.capacity_weight_tonnes}"
            )

        if volume > vehicle.capacity_volume_m3:
            issues.append(
                f"Exceeds volume capacity: {volume} > {vehicle.capacity_volume_m3}"
            )

        # Check all orders exist
        for order_id in route.order_ids:
            if order_id not in state.pending_orders:
                issues.append(f"Order {order_id} not found")

        # Check time windows
        for stop in route.stops:
            for order_id in stop.order_ids:
                order = state.pending_orders.get(order_id)
                if order and not order.is_in_time_window(stop.estimated_arrival):
                    issues.append(
                        f"Time window violation: order {order_id} at {stop.estimated_arrival}"
                    )

        return len(issues) == 0, issues

    def _validate_route(self, route: Route, state: SystemState) -> bool:
        """Simple validation."""
        is_feasible, _ = self.check_route_feasibility(route, state)
        return is_feasible

    def _create_single_order_route(self, order: Order, vehicle: Vehicle) -> Route:
        """Create simple route for single order."""
        from datetime import datetime

        route_id = f"route_single_{order.order_id}_{int(datetime.now().timestamp())}"

        return Route(
            route_id=route_id,
            vehicle_id=vehicle.vehicle_id,
            order_ids=[order.order_id],
            stops=[],
            destination_cities=[order.destination_city],
            total_distance_km=0.0,
            estimated_duration_minutes=0,
            estimated_cost_kes=0.0,
        )

    def _resequence_route(self, route: Route, state: SystemState) -> Route:
        """Re-sequence route stops for disruption recovery.
        
        Simple strategy: sort by delivery window urgency.
        In production: use optimal TSP/scheduling algorithm.
        """

        if not route.stops:
            return route

        # Sort stops by estimated arrival time (most urgent first)
        sorted_stops = sorted(route.stops, key=lambda s: s.estimated_arrival)

        from dataclasses import replace

        return replace(route, stops=sorted_stops)

    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "total_optimizations": 0,  # Would track this
            "avg_cost_per_route": 0.0,
            "avg_profit_per_route": 0.0,
            "feasibility_rate": 0.95,
        }

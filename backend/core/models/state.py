"""Immutable system state for decision making."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, List, Optional
import copy

from .domain import Order, Vehicle, Route, Customer, Location, OperationalOutcome


@dataclass(frozen=True)
class EnvironmentState:
    """Current environmental conditions affecting routing."""

    current_time: datetime

    # Traffic conditions by zone (0.0-1.0, where 1.0 = completely congested)
    traffic_conditions: Dict[str, float] = field(default_factory=dict)

    # Weather
    weather: str = "clear"  # "clear", "rain", "heavy_rain"

    # Day of week patterns
    day_of_week: str = ""

    # Historical patterns (average travel time multiplier)
    time_multipliers: Dict[str, float] = field(default_factory=dict)

    # Named time windows used by policies (e.g., pickup windows, rush hours)
    # Format: {"window_name": {"start_minute": 510, "end_minute": 585, "zones": ["CBD","Industrial"]}}
    time_windows: Dict[str, Dict[str, any]] = field(default_factory=dict)

    def get_traffic_for_zone(self, zone: Optional[str]) -> float:
        """Get traffic congestion for zone (0.0-1.0)."""
        if zone is None:
            return 0.0
        return self.traffic_conditions.get(zone, 0.0)

    def is_time_window_active(self, window_name: str) -> bool:
        """Check whether a named time window is active in this environment.

        The window config uses minutes-since-midnight for start/end. If the
        window is not found, returns False.
        """
        cfg = self.time_windows.get(window_name)
        if not cfg:
            return False

        current = self.current_time
        minutes = current.hour * 60 + current.minute
        start = int(cfg.get("start_minute", -1))
        end = int(cfg.get("end_minute", -1))
        if start < 0 or end < 0:
            return False

        return start <= minutes <= end


@dataclass(frozen=True)
class LearningState:
    """Learned parameters and models state."""

    # Cost Function Approximation parameters
    cfa_parameters: Dict[str, float] = field(default_factory=dict)
    # {"fuel_per_km": 0.025, "traffic_buffer_multiplier": 1.2, "delay_penalty_weight": 0.5}

    # Value Function Approximation neural network state
    vfa_model_weights: Optional[Dict[str, any]] = None
    vfa_learning_rate: float = 0.01
    vfa_gamma: float = 0.95  # Discount factor
    vfa_trained_samples: int = 0

    # Policy Function Approximation rules
    pfa_rules: List[Dict] = field(default_factory=list)
    # [{"condition": "order.priority==2", "action": "prioritize", "confidence": 0.95}]

    # DLA forecast parameters
    dla_demand_forecast: Dict[str, float] = field(default_factory=dict)
    dla_consolidation_threshold: float = 0.8

    # Prediction accuracy metrics for model confidence
    cfa_accuracy_fuel: float = 0.0  # 0.0-1.0
    cfa_accuracy_time: float = 0.0
    vfa_accuracy: float = 0.0
    pfa_coverage: float = 0.0  # Fraction of decisions matching learned rules

    # Last update timestamp
    last_updated: datetime = field(default_factory=datetime.now)

    def get_model_confidence(self, model_name: str) -> float:
        """Get confidence score for learned model."""
        if model_name == "cfa":
            return (self.cfa_accuracy_fuel + self.cfa_accuracy_time) / 2.0
        elif model_name == "vfa":
            return self.vfa_accuracy
        elif model_name == "pfa":
            return self.pfa_coverage
        return 0.0


@dataclass(frozen=True)
class SystemState:
    """Immutable complete system state for decision making.

    This is the single source of truth for all policy classes and the engine.
    Always treat as immutable; create new states via clone_with_updates() method.
    """

    # Core entities
    pending_orders: Dict[str, Order] = field(default_factory=dict)
    active_routes: Dict[str, Route] = field(default_factory=dict)
    fleet: Dict[str, Vehicle] = field(default_factory=dict)
    customers: Dict[str, Customer] = field(default_factory=dict)
    completed_routes: Dict[str, Route] = field(default_factory=dict)

    # Learning and environment
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    learning: LearningState = field(default_factory=LearningState)

    # Operational outcomes for learning
    recent_outcomes: List[OperationalOutcome] = field(default_factory=list)

    # Timestamp of this state
    timestamp: datetime = field(default_factory=datetime.now)

    # Metadata
    state_id: str = ""
    decision_context: Dict[str, any] = field(default_factory=dict)

    def clone_with_updates(self, **updates) -> "SystemState":
        """Create new SystemState with specified updates (immutable pattern)."""
        return replace(self, **updates)

    # Aggregation methods

    def get_total_pending_weight(self) -> float:
        """Total weight of all pending orders."""
        return sum(o.weight_tonnes for o in self.pending_orders.values())

    def get_total_pending_volume(self) -> float:
        """Total volume of all pending orders."""
        return sum(o.volume_m3 for o in self.pending_orders.values())

    def get_available_vehicles(self) -> Dict[str, Vehicle]:
        """Get all currently available vehicles."""
        from .domain import VehicleStatus

        return {
            vid: v
            for vid, v in self.fleet.items()
            if v.status == VehicleStatus.AVAILABLE
            and v.available_at <= self.environment.current_time
        }

    def get_used_capacity_by_vehicle(self) -> Dict[str, tuple[float, float]]:
        """Get (weight, volume) used by each vehicle in active routes."""
        used = {}
        for route in self.active_routes.values():
            weight = route.get_total_weight(self.pending_orders)
            volume = route.get_total_volume(self.pending_orders)
            used[route.vehicle_id] = (weight, volume)
        return used

    def get_available_capacity_by_vehicle(self) -> Dict[str, tuple[float, float]]:
        """Get (weight, volume) available in each vehicle."""
        used = self.get_used_capacity_by_vehicle()
        available = {}
        for vid, vehicle in self.fleet.items():
            used_w, used_v = used.get(vid, (0.0, 0.0))
            avail_w = vehicle.capacity_weight_tonnes - used_w
            avail_v = vehicle.capacity_volume_m3 - used_v
            available[vid] = (avail_w, avail_v)
        return available

    def get_unassigned_orders(self) -> Dict[str, Order]:
        """Get orders not yet assigned to routes."""
        from .domain import OrderStatus

        return {
            oid: o
            for oid, o in self.pending_orders.items()
            if o.status in (OrderStatus.PENDING,)
        }

    def get_orders_for_city(self, city_name: str) -> Dict[str, Order]:
        """Get all orders with destination in specific city."""
        return {
            oid: o
            for oid, o in self.pending_orders.items()
            if o.destination_city.value == city_name
        }

    def get_active_routes_for_vehicle(self, vehicle_id: str) -> List[Route]:
        """Get all active routes assigned to vehicle."""
        return [r for r in self.active_routes.values() if r.vehicle_id == vehicle_id]

    def get_vehicles_with_capacity_for(
        self, order: Order, min_capacity_utilization: float = 0.0
    ) -> Dict[str, Vehicle]:
        """Get vehicles that can accommodate order."""
        available_capacity = self.get_available_capacity_by_vehicle()
        result = {}

        for vid, vehicle in self.fleet.items():
            if not vehicle.is_available_at(self.environment.current_time):
                continue

            avail_w, avail_v = available_capacity.get(
                vid, (vehicle.capacity_weight_tonnes, vehicle.capacity_volume_m3)
            )

            if order.can_fit_in_vehicle(avail_w, avail_v):
                result[vid] = vehicle

        return result

    def get_vehicle_utilization_percent(self, vehicle_id: str) -> float:
        """Get current capacity utilization of vehicle (0.0-100.0)."""
        vehicle = self.fleet.get(vehicle_id)
        if not vehicle:
            return 0.0

        used_w, used_v = self.get_used_capacity_by_vehicle().get(vehicle_id, (0.0, 0.0))

        weight_util = (
            (used_w / vehicle.capacity_weight_tonnes * 100)
            if vehicle.capacity_weight_tonnes > 0
            else 0
        )
        volume_util = (
            (used_v / vehicle.capacity_volume_m3 * 100)
            if vehicle.capacity_volume_m3 > 0
            else 0
        )

        return max(weight_util, volume_util)  # Bottleneck is limiting factor

    def get_orders_for_customer_blocked_times(self, customer_id: str) -> List[Order]:
        """Get orders from customers who have blocked delivery times."""
        customer = self.customers.get(customer_id)
        if not customer:
            return []

        blocked_orders = []
        for order in self.pending_orders.values():
            if order.customer_id == customer_id:
                if not customer.can_deliver_at(self.environment.current_time):
                    blocked_orders.append(order)

        return blocked_orders

    def get_eastern_africa_mesh_routes(self) -> List[str]:
        """Get routes serving multi-city mesh (Nairobi → Nakuru → Eldoret → Kitale)."""
        multi_city_routes = []
        for route in self.active_routes.values():
            if len(route.destination_cities) > 1:
                multi_city_routes.append(route.route_id)
        return multi_city_routes

    def get_learning_confidence_vector(self) -> Dict[str, float]:
        """Get confidence scores for all learned models."""
        return {
            "cfa": self.learning.get_model_confidence("cfa"),
            "vfa": self.learning.get_model_confidence("vfa"),
            "pfa": self.learning.get_model_confidence("pfa"),
            "dla": (
                self.learning.get_model_confidence("dla")
                if self.learning.dla_demand_forecast
                else 0.0
            ),
        }

    def is_eastleigh_window_active(self) -> bool:
        """Deprecated: legacy helper removed.

        Use `environment.is_time_window_active(window_name)` or
        `get_learning_confidence_vector()` instead. This method remains for
        backward-compatibility but always returns False.
        """
        return False

    def filter_orders_by_priority(
        self, orders: Dict[str, Order], priority_level: int
    ) -> Dict[str, Order]:
        """Filter orders by priority level."""
        return {oid: o for oid, o in orders.items() if o.priority >= priority_level}

    def filter_orders_by_special_handling(
        self, orders: Dict[str, Order], handling_type: str
    ) -> Dict[str, Order]:
        """Filter orders requiring special handling."""
        return {
            oid: o for oid, o in orders.items() if handling_type in o.special_handling
        }

    def get_fresh_food_orders(self) -> Dict[str, Order]:
        """Get orders marked as fresh food (high priority)."""
        return self.filter_orders_by_special_handling(self.pending_orders, "fresh_food")

    def get_estimated_route_value(self, route: Route) -> float:
        """Estimate business value of route (based on order prices)."""
        total_value = 0.0
        for order_id in route.order_ids:
            order = self.pending_orders.get(order_id)
            if order:
                total_value += order.price_kes
        return total_value

    def get_route_profitability(self, route: Route) -> float:
        """Estimate route profitability (revenue - estimated costs)."""
        value = self.get_estimated_route_value(route)
        costs = route.estimated_cost_kes
        return value - costs

    def get_backhaul_opportunities(self) -> List[tuple[Route, List[Order]]]:
        """Identify active routes with backhaul opportunities (return capacity).

        Returns list of (route, available_orders) tuples for consolidation.
        """
        opportunities = []

        for route in self.active_routes.values():
            used_w, used_v = self.get_used_capacity_by_vehicle().get(
                route.vehicle_id, (0.0, 0.0)
            )
            vehicle = self.fleet.get(route.vehicle_id)

            if not vehicle:
                continue

            avail_w, avail_v = vehicle.get_remaining_capacity(used_w, used_v)

            # Find unassigned orders that fit
            fitting_orders = []
            for order in self.get_unassigned_orders().values():
                if order.can_fit_in_vehicle(avail_w, avail_v):
                    fitting_orders.append(order)

            if fitting_orders:
                opportunities.append((route, fitting_orders))

        return opportunities

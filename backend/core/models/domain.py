"""Domain models for Senga Sequential Decision Engine."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum


class OrderStatus(str, Enum):
    """Order lifecycle statuses."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    FAILED = "failed"


class VehicleStatus(str, Enum):
    """Vehicle lifecycle statuses."""

    AVAILABLE = "available"
    IN_TRANSIT = "in_transit"
    LOADING = "loading"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"


class RouteStatus(str, Enum):
    """Route lifecycle statuses."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DestinationCity(str, Enum):
    """Supported destination cities in mesh network."""

    NAKURU = "Nakuru"
    ELDORET = "Eldoret"
    KITALE = "Kitale"
    KISUMU = "kisumu"


@dataclass
class Location:
    """Geographic location with coordinates and metadata."""

    latitude: float
    longitude: float
    address: str
    place_id: Optional[str] = None
    zone: Optional[str] = None  # e.g., "Depot", "CBD"

    def __hash__(self):
        return hash((self.latitude, self.longitude))


@dataclass
class TimeWindow:
    """Time window constraint for orders or vehicles."""

    start_time: datetime
    end_time: datetime

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within window."""
        return self.start_time <= timestamp <= self.end_time

    def is_valid(self) -> bool:
        """Check if window is valid (start before end)."""
        return self.start_time < self.end_time


@dataclass
class Order:
    """Delivery order with locations, capacity, and constraints."""

    order_id: str
    customer_id: str
    customer_name: str

    # Locations
    pickup_location: Location
    destination_city: DestinationCity
    destination_location: Optional[Location] = None
    # Capacity requirements
    weight_tonnes: float = 0.0
    volume_m3: float = 0.0

    # Timing
    time_window: TimeWindow = field(
        default_factory=lambda: TimeWindow(datetime.now(), datetime.now())
    )  # When order can be picked up
    delivery_window: Optional[TimeWindow] = None  # When delivery is acceptable

    # Attributes
    priority: int = 0  # 0=normal, 1=high, 2=urgent
    special_handling: List[str] = field(
        default_factory=list
    )  # ["fresh_food", "fragile"]
    customer_constraints: Dict[str, any] = field(default_factory=dict)

    # Financial
    price_kes: float = 0.0

    # Status
    status: OrderStatus = OrderStatus.PENDING
    assigned_route_id: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def can_fit_in_vehicle(
        self, vehicle_capacity_weight: float, vehicle_capacity_volume: float
    ) -> bool:
        """Check if order fits in vehicle capacity."""
        return (
            self.weight_tonnes <= vehicle_capacity_weight
            and self.volume_m3 <= vehicle_capacity_volume
        )

    def is_in_time_window(self, timestamp: datetime) -> bool:
        """Check if timestamp is in order's time window."""
        return self.time_window.contains(timestamp)


@dataclass
class Vehicle:
    """Vehicle with capacity, availability, and routing."""

    vehicle_id: str
    vehicle_type: str  # "5T", "10T"

    # Capacity
    capacity_weight_tonnes: float
    capacity_volume_m3: float

    # Location and availability
    current_location: Location
    available_at: datetime
    status: VehicleStatus = VehicleStatus.AVAILABLE

    # Routing
    assigned_route_id: Optional[str] = None

    # Efficiency
    # Fuel efficiency (km per liter) - kept for backward compatibility
    fuel_efficiency_km_per_liter: float = 8.5

    # Optional per-vehicle economic parameters (preferred source for CFA)
    # These should be set when constructing vehicles so the system scales without
    # requiring central config maps keyed by vehicle type.
    fuel_cost_per_km: Optional[float] = None  # Direct cost per km (KES/km)
    driver_cost_per_hour: Optional[float] = None  # Driver cost (KES/hour)

    # Metadata
    driver_id: Optional[str] = None
    maintenance_due: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def is_available_at(self, timestamp: datetime) -> bool:
        """Check if vehicle is available at given time."""
        return self.status == VehicleStatus.AVAILABLE and self.available_at <= timestamp

    def has_capacity_for(self, weight: float, volume: float) -> bool:
        """Check if vehicle has capacity for cargo."""
        return (
            weight <= self.capacity_weight_tonnes and volume <= self.capacity_volume_m3
        )

    def get_remaining_capacity(
        self, used_weight: float, used_volume: float
    ) -> Tuple[float, float]:
        """Get remaining capacity after usage."""
        return (
            self.capacity_weight_tonnes - used_weight,
            self.capacity_volume_m3 - used_volume,
        )


@dataclass
class RouteStop:
    """Single stop on a route (pickup or delivery)."""

    stop_id: str
    order_ids: List[str]  # Orders at this stop
    location: Location
    stop_type: str  # "pickup" or "delivery"
    sequence_order: int  # Position in route
    estimated_arrival: datetime
    estimated_duration_minutes: int
    status: str = "planned"  # "planned", "in_progress", "completed"
    actual_arrival: Optional[datetime] = None
    actual_duration_minutes: Optional[int] = None


@dataclass
class Route:
    """Optimized route for vehicle with sequence of stops."""

    route_id: str
    vehicle_id: str

    # Orders on route
    order_ids: List[str]
    stops: List[RouteStop]  # Ordered sequence of stops

    # Routing details
    destination_cities: List[DestinationCity]  # Cities served on route
    total_distance_km: float
    estimated_duration_minutes: int
    estimated_cost_kes: float

    # Status
    status: RouteStatus = RouteStatus.PLANNED

    # Performance
    estimated_fuel_cost: float = 0.0
    estimated_time_cost: float = 0.0
    estimated_delay_penalty: float = 0.0

    actual_distance_km: Optional[float] = None
    actual_duration_minutes: Optional[int] = None
    actual_cost_kes: Optional[float] = None
    actual_fuel_cost: Optional[float] = None

    # Metadata
    decision_id: Optional[str] = None  # Decision that created this route
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def get_total_weight(self, orders: Dict[str, Order]) -> float:
        """Calculate total weight of all orders."""
        return sum(orders[oid].weight_tonnes for oid in self.order_ids if oid in orders)

    def get_total_volume(self, orders: Dict[str, Order]) -> float:
        """Calculate total volume of all orders."""
        return sum(orders[oid].volume_m3 for oid in self.order_ids if oid in orders)

    def is_feasible(self, vehicle: Vehicle, orders: Dict[str, Order]) -> bool:
        """Check if route is feasible for vehicle."""
        weight = self.get_total_weight(orders)
        volume = self.get_total_volume(orders)
        return vehicle.has_capacity_for(weight, volume)


@dataclass
class Customer:
    """Customer profile with constraints and preferences."""

    customer_id: str
    name: str

    # Locations
    locations: List[Location]  # Multiple locations (pickups, deliveries)

    # Constraints
    delivery_blocked_times: List[Dict] = field(default_factory=list)
    # [{"day": "Wednesday", "time_start": "14:00", "time_end": "15:30"}]

    preferred_windows: List[Dict] = field(default_factory=list)
    # [{"days": ["Tuesday", "Thursday"], "time_start": "13:30", "time_end": "14:00"}]

    # Attributes
    priority_level: int = 0  # 0=standard, 1=high, 2=vip
    fresh_food_customer: bool = False
    special_notes: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def can_deliver_at(self, timestamp: datetime) -> bool:
        """Check if customer is available for delivery at timestamp."""
        day_name = timestamp.strftime("%A")
        time_str = timestamp.strftime("%H:%M")

        for blocked in self.delivery_blocked_times:
            if blocked["day"] == day_name:
                if blocked["time_start"] <= time_str <= blocked["time_end"]:
                    return False
        return True

    def has_preference_for(self, timestamp: datetime) -> bool:
        """Check if timestamp matches customer's preferred windows."""
        day_name = timestamp.strftime("%A")
        time_str = timestamp.strftime("%H:%M")

        if not self.preferred_windows:
            return True

        for pref in self.preferred_windows:
            if day_name in pref["days"]:
                if pref["time_start"] <= time_str <= pref["time_end"]:
                    return True
        return False


@dataclass
class OperationalOutcome:
    """Recorded outcome of executed route for learning."""

    outcome_id: str
    route_id: str
    vehicle_id: str

    # Predictions vs actuals
    predicted_fuel_cost: float
    actual_fuel_cost: float

    predicted_duration_minutes: int
    actual_duration_minutes: int

    predicted_distance_km: float
    actual_distance_km: float

    # Performance
    on_time: bool
    delay_minutes: int = 0

    # Delivery results
    successful_deliveries: int = 0
    failed_deliveries: int = 0

    # Context
    traffic_conditions: Dict[str, float] = field(
        default_factory=dict
    )  # zone -> congestion
    weather: str = ""
    day_of_week: str = ""

    # Satisfaction
    customer_satisfaction_score: Optional[float] = None
    notes: str = ""

    # Metadata
    recorded_at: datetime = field(default_factory=datetime.now)

    def get_prediction_error_fuel(self) -> float:
        """Calculate fuel cost prediction error."""
        return self.actual_fuel_cost - self.predicted_fuel_cost

    def get_prediction_error_duration(self) -> int:
        """Calculate duration prediction error in minutes."""
        return self.actual_duration_minutes - self.predicted_duration_minutes

    def get_accuracy_fuel(self) -> float:
        """Calculate fuel prediction accuracy (1.0 is perfect)."""
        if self.predicted_fuel_cost == 0:
            return 0.0
        error = abs(self.get_prediction_error_fuel()) / self.predicted_fuel_cost
        return max(0.0, 1.0 - error)

    def get_accuracy_duration(self) -> float:
        """Calculate duration prediction accuracy (1.0 is perfect)."""
        if self.predicted_duration_minutes == 0:
            return 0.0
        error = (
            abs(self.get_prediction_error_duration()) / self.predicted_duration_minutes
        )
        return max(0.0, 1.0 - error)

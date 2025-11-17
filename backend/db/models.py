"""ORM Models for Database Persistence.

SQLAlchemy ORM models mapping to domain models:
- UserModel: Authentication and authorization
- CustomerModel: Customer information and constraints
- OrderModel: Delivery orders
- VehicleModel: Fleet vehicles
- RouteModel: Optimized routes
- RouteStopModel: Individual stops on routes
- DecisionModel: Decision audit trail
- OperationalOutcomeModel: Learning feedback

All models use async SQLAlchemy 2.0 style.
"""

from datetime import datetime
from typing import Optional
import json

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    JSON,
)
from sqlalchemy.orm import relationship

from .database import Base
from ..core.models.domain import OrderStatus, VehicleStatus, RouteStatus, DestinationCity


# ===========================
# User & Authentication
# ===========================

class UserModel(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)

    # Authorization
    role = Column(String(50), nullable=False, default="user")  # user, admin, manager
    is_active = Column(Boolean, nullable=False, default=True)
    is_superuser = Column(Boolean, nullable=False, default=False)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    last_login = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_users_active", "is_active"),
        Index("idx_users_role", "role"),
    )


# ===========================
# Customer
# ===========================

class CustomerModel(Base):
    """Customer model with constraints and preferences."""

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), unique=True, nullable=False, index=True)
    customer_name = Column(String(255), nullable=False)

    # Contact information
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    address = Column(Text, nullable=True)

    # Constraints (stored as JSON)
    constraints = Column(JSON, nullable=True)  # {"max_wait_hours": 24, "preferred_time": "morning"}
    preferences = Column(JSON, nullable=True)  # Customer preferences

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    # Relationships
    orders = relationship("OrderModel", back_populates="customer", lazy="selectin")


# ===========================
# Order
# ===========================

class OrderModel(Base):
    """Order model for delivery requests."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id"), nullable=False, index=True)

    # Locations (stored as JSON)
    pickup_location = Column(JSON, nullable=False)  # {lat, lon, address, zone}
    destination_city = Column(SQLEnum(DestinationCity), nullable=False, index=True)
    destination_location = Column(JSON, nullable=True)

    # Capacity requirements
    weight_tonnes = Column(Float, nullable=False, default=0.0)
    volume_m3 = Column(Float, nullable=False, default=0.0)

    # Timing (stored as JSON)
    time_window_start = Column(DateTime, nullable=False)
    time_window_end = Column(DateTime, nullable=False)
    delivery_window_start = Column(DateTime, nullable=True)
    delivery_window_end = Column(DateTime, nullable=True)

    # Attributes
    priority = Column(Integer, nullable=False, default=0)  # 0=normal, 1=high, 2=urgent
    special_handling = Column(JSON, nullable=True)  # ["fresh_food", "fragile"]
    customer_constraints = Column(JSON, nullable=True)

    # Financial
    price_kes = Column(Float, nullable=False, default=0.0)

    # Status
    status = Column(SQLEnum(OrderStatus), nullable=False, default=OrderStatus.PENDING, index=True)
    assigned_route_id = Column(String(50), ForeignKey("routes.route_id"), nullable=True, index=True)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    # Relationships
    customer = relationship("CustomerModel", back_populates="orders")
    route = relationship("RouteModel", back_populates="orders", foreign_keys=[assigned_route_id])

    __table_args__ = (
        Index("idx_orders_status_created", "status", "created_at"),
        Index("idx_orders_destination_status", "destination_city", "status"),
        Index("idx_orders_priority", "priority"),
    )


# ===========================
# Vehicle
# ===========================

class VehicleModel(Base):
    """Vehicle model for fleet management."""

    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), unique=True, nullable=False, index=True)
    vehicle_type = Column(String(50), nullable=False, index=True)

    # Capacity
    capacity_weight_tonnes = Column(Float, nullable=False)
    capacity_volume_m3 = Column(Float, nullable=False)

    # Location and availability
    current_location = Column(JSON, nullable=False)  # {lat, lon, address, zone}
    available_at = Column(DateTime, nullable=False, index=True)
    status = Column(SQLEnum(VehicleStatus), nullable=False, default=VehicleStatus.AVAILABLE, index=True)

    # Routing
    assigned_route_id = Column(String(50), ForeignKey("routes.route_id"), nullable=True, index=True)

    # Efficiency
    fuel_efficiency_km_per_liter = Column(Float, nullable=False, default=8.5)
    fuel_cost_per_km = Column(Float, nullable=True)
    driver_cost_per_hour = Column(Float, nullable=True)

    # Metadata
    driver_id = Column(String(50), nullable=True)
    maintenance_due = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    # Relationships
    routes = relationship("RouteModel", back_populates="vehicle", foreign_keys="[RouteModel.vehicle_id]")

    __table_args__ = (
        Index("idx_vehicles_status_available", "status", "available_at"),
        Index("idx_vehicles_type", "vehicle_type"),
    )


# ===========================
# Route & RouteStop
# ===========================

class RouteModel(Base):
    """Route model for optimized delivery routes."""

    __tablename__ = "routes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    route_id = Column(String(50), unique=True, nullable=False, index=True)
    vehicle_id = Column(String(50), ForeignKey("vehicles.vehicle_id"), nullable=False, index=True)

    # Orders on route (stored as JSON array)
    order_ids = Column(JSON, nullable=False)  # ["ORD_001", "ORD_002"]

    # Routing details
    destination_cities = Column(JSON, nullable=False)  # ["NAKURU", "ELDORET"]
    total_distance_km = Column(Float, nullable=False)
    estimated_duration_minutes = Column(Integer, nullable=False)
    estimated_cost_kes = Column(Float, nullable=False)

    # Status
    status = Column(SQLEnum(RouteStatus), nullable=False, default=RouteStatus.PLANNED, index=True)

    # Performance (estimated)
    estimated_fuel_cost = Column(Float, nullable=False, default=0.0)
    estimated_time_cost = Column(Float, nullable=False, default=0.0)
    estimated_delay_penalty = Column(Float, nullable=False, default=0.0)

    # Performance (actual)
    actual_distance_km = Column(Float, nullable=True)
    actual_duration_minutes = Column(Integer, nullable=True)
    actual_cost_kes = Column(Float, nullable=True)
    actual_fuel_cost = Column(Float, nullable=True)

    # Metadata
    decision_id = Column(String(50), ForeignKey("decisions.decision_id"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    vehicle = relationship("VehicleModel", back_populates="routes", foreign_keys=[vehicle_id])
    orders = relationship("OrderModel", back_populates="route", foreign_keys="[OrderModel.assigned_route_id]")
    stops = relationship("RouteStopModel", back_populates="route", cascade="all, delete-orphan")
    decision = relationship("DecisionModel", back_populates="routes")

    __table_args__ = (
        Index("idx_routes_status_created", "status", "created_at"),
        Index("idx_routes_vehicle_status", "vehicle_id", "status"),
    )


class RouteStopModel(Base):
    """Route stop model for individual stops on routes."""

    __tablename__ = "route_stops"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stop_id = Column(String(50), unique=True, nullable=False, index=True)
    route_id = Column(String(50), ForeignKey("routes.route_id"), nullable=False, index=True)

    # Stop details
    order_ids = Column(JSON, nullable=False)  # Orders at this stop
    location = Column(JSON, nullable=False)  # {lat, lon, address, zone}
    stop_type = Column(String(20), nullable=False)  # "pickup" or "delivery"
    sequence_order = Column(Integer, nullable=False)

    # Timing (estimated)
    estimated_arrival = Column(DateTime, nullable=False)
    estimated_duration_minutes = Column(Integer, nullable=False)

    # Status and actuals
    status = Column(String(20), nullable=False, default="planned")  # planned, in_progress, completed
    actual_arrival = Column(DateTime, nullable=True)
    actual_duration_minutes = Column(Integer, nullable=True)

    # Relationships
    route = relationship("RouteModel", back_populates="stops")

    __table_args__ = (
        Index("idx_route_stops_route_sequence", "route_id", "sequence_order"),
        Index("idx_route_stops_status", "status"),
    )


# ===========================
# Decision (Audit Trail)
# ===========================

class DecisionModel(Base):
    """Decision model for audit trail of routing decisions."""

    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String(50), unique=True, nullable=False, index=True)

    # Decision context
    decision_type = Column(String(50), nullable=False, index=True)  # ORDER_ARRIVAL, DAILY_ROUTE_PLANNING, etc.
    policy_used = Column(String(50), nullable=False)  # PFA, CFA, VFA, DLA, CFA_VFA, etc.

    # State snapshot (stored as JSON)
    state_snapshot = Column(JSON, nullable=True)  # Simplified state at decision time

    # Decision outcome
    routes_created = Column(JSON, nullable=False)  # List of route_ids created
    orders_routed = Column(JSON, nullable=False)  # List of order_ids routed
    total_cost_estimate = Column(Float, nullable=False, default=0.0)

    # Metrics
    decision_confidence = Column(Float, nullable=True)  # 0.0 - 1.0
    computation_time_ms = Column(Integer, nullable=True)

    # Status
    committed = Column(Boolean, nullable=False, default=False)
    executed = Column(Boolean, nullable=False, default=False)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    committed_at = Column(DateTime, nullable=True)

    # Relationships
    routes = relationship("RouteModel", back_populates="decision", foreign_keys="[RouteModel.decision_id]")
    outcomes = relationship("OperationalOutcomeModel", back_populates="decision")

    __table_args__ = (
        Index("idx_decisions_type_created", "decision_type", "created_at"),
        Index("idx_decisions_policy", "policy_used"),
        Index("idx_decisions_committed", "committed"),
    )


# ===========================
# Operational Outcome (Learning)
# ===========================

class OperationalOutcomeModel(Base):
    """Operational outcome model for learning feedback."""

    __tablename__ = "operational_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    outcome_id = Column(String(50), unique=True, nullable=False, index=True)
    route_id = Column(String(50), ForeignKey("routes.route_id"), nullable=False, index=True)
    decision_id = Column(String(50), ForeignKey("decisions.decision_id"), nullable=True, index=True)

    # Performance metrics
    actual_cost = Column(Float, nullable=False)
    actual_duration_minutes = Column(Integer, nullable=False)
    actual_distance_km = Column(Float, nullable=False)

    # Prediction errors
    cost_prediction_error = Column(Float, nullable=True)
    duration_prediction_error = Column(Integer, nullable=True)
    distance_prediction_error = Column(Float, nullable=True)

    # Revenue and profit
    total_revenue = Column(Float, nullable=False, default=0.0)
    net_profit = Column(Float, nullable=True)

    # Delivery performance
    on_time_deliveries = Column(Integer, nullable=False, default=0)
    late_deliveries = Column(Integer, nullable=False, default=0)
    failed_deliveries = Column(Integer, nullable=False, default=0)

    # Additional metrics (stored as JSON)
    additional_metrics = Column(JSON, nullable=True)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)

    # Relationships
    decision = relationship("DecisionModel", back_populates="outcomes")

    __table_args__ = (
        Index("idx_outcomes_route", "route_id"),
        Index("idx_outcomes_created", "created_at"),
    )


# ===========================
# Helper Functions
# ===========================

def location_to_dict(location) -> dict:
    """Convert Location dataclass to dict for JSON storage."""
    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "address": location.address,
        "place_id": getattr(location, "place_id", None),
        "zone": getattr(location, "zone", None),
    }


def dict_to_location_json(location_dict: dict) -> str:
    """Convert location dict to JSON string."""
    return json.dumps(location_dict) if location_dict else None


def time_window_to_dict(time_window) -> dict:
    """Convert TimeWindow dataclass to dict."""
    if not time_window:
        return None
    return {
        "start_time": time_window.start_time.isoformat(),
        "end_time": time_window.end_time.isoformat(),
    }

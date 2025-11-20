"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Enums matching domain models
class OrderStatusEnum(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    FAILED = "failed"


class VehicleStatusEnum(str, Enum):
    AVAILABLE = "available"
    IN_TRANSIT = "in_transit"
    LOADING = "loading"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"


class RouteStatusEnum(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DestinationCityEnum(str, Enum):
    NAKURU = "Nakuru"
    ELDORET = "Eldoret"
    KITALE = "Kitale"


class DecisionTypeEnum(str, Enum):
    DAILY_ROUTE_PLANNING = "daily_route_planning"
    ORDER_ARRIVAL = "order_arrival"
    REALTIME_ADJUSTMENT = "realtime_adjustment"
    BACKHAUL_CONSOLIDATION = "backhaul_consolidation"


class ActionTypeEnum(str, Enum):
    CREATE_ROUTE = "create_route"
    ACCEPT_ORDER = "accept_order"
    DEFER_ORDER = "defer_order"
    REJECT_ORDER = "reject_order"
    CONSOLIDATE_ROUTES = "consolidate_routes"
    NO_ACTION = "no_action"


# Location schemas
class LocationSchema(BaseModel):
    latitude: float
    longitude: float
    address: str
    place_id: Optional[str] = None
    zone: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TimeWindowSchema(BaseModel):
    start_time: datetime
    end_time: datetime

    model_config = ConfigDict(from_attributes=True)


# Order schemas
class OrderCreateRequest(BaseModel):
    customer_id: str
    customer_name: str
    pickup_location: LocationSchema
    destination_city: DestinationCityEnum
    destination_location: Optional[LocationSchema] = None
    weight_tonnes: float = Field(gt=0)
    volume_m3: float = Field(gt=0)
    time_window: TimeWindowSchema
    delivery_window: Optional[TimeWindowSchema] = None
    priority: int = Field(default=0, ge=0, le=2)
    special_handling: List[str] = Field(default_factory=list)
    customer_constraints: Dict[str, Any] = Field(default_factory=dict)
    price_kes: float = Field(default=0.0, ge=0)


class OrderResponse(BaseModel):
    order_id: str
    customer_id: str
    customer_name: str
    pickup_location: LocationSchema
    destination_city: DestinationCityEnum
    destination_location: Optional[LocationSchema] = None
    weight_tonnes: float
    volume_m3: float
    time_window: TimeWindowSchema
    delivery_window: Optional[TimeWindowSchema] = None
    priority: int
    special_handling: List[str]
    customer_constraints: Dict[str, Any]
    price_kes: float
    status: OrderStatusEnum
    assigned_route_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class OrderUpdateRequest(BaseModel):
    priority: Optional[int] = Field(None, ge=0, le=2)
    special_handling: Optional[List[str]] = None
    customer_constraints: Optional[Dict[str, Any]] = None
    status: Optional[OrderStatusEnum] = None


# Vehicle schemas
class VehicleCreateRequest(BaseModel):
    vehicle_type: str
    capacity_weight_tonnes: float
    capacity_volume_m3: float
    current_location: LocationSchema
    fuel_efficiency_km_per_liter: float = 8.5
    fuel_cost_per_km: Optional[float] = None
    driver_cost_per_hour: Optional[float] = None
    driver_id: Optional[str] = None
    maintenance_due: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class VehicleUpdateRequest(BaseModel):
    current_location: Optional[LocationSchema] = None
    status: Optional[VehicleStatusEnum] = None
    fuel_cost_per_km: Optional[float] = None
    driver_cost_per_hour: Optional[float] = None
    driver_id: Optional[str] = None
    maintenance_due: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class VehicleResponse(BaseModel):
    vehicle_id: str
    vehicle_type: str
    capacity_weight_tonnes: float
    capacity_volume_m3: float
    current_location: LocationSchema
    available_at: datetime
    status: VehicleStatusEnum
    assigned_route_id: Optional[str] = None
    fuel_efficiency_km_per_liter: float
    fuel_cost_per_km: Optional[float] = None
    driver_cost_per_hour: Optional[float] = None
    driver_id: Optional[str] = None
    maintenance_due: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# Customer schemas
class CustomerCreateRequest(BaseModel):
    customer_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class CustomerUpdateRequest(BaseModel):
    customer_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class CustomerResponse(BaseModel):
    customer_id: str
    customer_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Route schemas
class RouteStopSchema(BaseModel):
    stop_id: str
    order_ids: List[str]
    location: LocationSchema
    stop_type: str
    sequence_order: int
    estimated_arrival: datetime
    estimated_duration_minutes: int
    status: str
    actual_arrival: Optional[datetime] = None
    actual_duration_minutes: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class RouteResponse(BaseModel):
    route_id: str
    vehicle_id: str
    order_ids: List[str]
    stops: List[RouteStopSchema]
    destination_cities: List[DestinationCityEnum]
    total_distance_km: float
    estimated_duration_minutes: int
    estimated_cost_kes: float
    status: RouteStatusEnum
    estimated_fuel_cost: float
    estimated_time_cost: float
    estimated_delay_penalty: float
    actual_distance_km: Optional[float] = None
    actual_duration_minutes: Optional[int] = None
    actual_cost_kes: Optional[float] = None
    actual_fuel_cost: Optional[float] = None
    decision_id: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# Decision schemas
class DecisionRequest(BaseModel):
    decision_type: DecisionTypeEnum
    trigger_reason: Optional[str] = "Manual decision request from UI"
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PolicyDecisionSchema(BaseModel):
    policy_name: str
    recommended_action: ActionTypeEnum
    confidence_score: float
    expected_value: float
    routes: List[RouteResponse]
    reasoning: str
    computation_time_ms: float

    model_config = ConfigDict(from_attributes=True)


class DecisionResponse(BaseModel):
    decision_id: str
    decision_type: DecisionTypeEnum
    policy_name: str
    recommended_action: ActionTypeEnum
    confidence_score: float
    expected_value: float
    routes: List[RouteResponse]
    reasoning: str
    computation_time_ms: float
    timestamp: datetime
    committed: bool = False

    model_config = ConfigDict(from_attributes=True)


class PolicyRecommendation(BaseModel):
    """Individual policy recommendation for comparison."""
    policy_name: str
    recommended_action: ActionTypeEnum
    confidence_score: float
    expected_value: float
    routes: List[RouteResponse]
    reasoning: str
    policy_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class AgreementAnalysis(BaseModel):
    """Analysis of policy agreement/disagreement."""
    agreement_score: float  # 0.0 to 1.0
    consensus_action: str
    consensus_count: int
    total_policies: int
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    avg_confidence: float
    avg_expected_value: float


class PolicyComparisonResponse(BaseModel):
    """Complete 4-policy comparison response."""
    decision_type: DecisionTypeEnum
    timestamp: datetime

    # Individual policy recommendations
    pfa: PolicyRecommendation
    vfa: PolicyRecommendation
    cfa: PolicyRecommendation
    dla: PolicyRecommendation

    # Engine's recommended policy (from existing logic)
    recommended: PolicyRecommendation

    # Agreement analysis
    agreement_analysis: AgreementAnalysis

    # Metadata
    computation_time_ms: float
    trigger_reason: Optional[str] = ""


class DecisionCommitRequest(BaseModel):
    decision_id: str


class DecisionCommitResponse(BaseModel):
    success: bool
    action: str
    routes_created: List[str]
    orders_assigned: List[str]
    errors: List[str]
    message: str


# Operational Outcome schemas
class OperationalOutcomeRequest(BaseModel):
    route_id: str
    vehicle_id: str
    predicted_fuel_cost: float
    actual_fuel_cost: float
    predicted_duration_minutes: int
    actual_duration_minutes: int
    predicted_distance_km: float
    actual_distance_km: float
    on_time: bool
    delay_minutes: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    traffic_conditions: Dict[str, float] = Field(default_factory=dict)
    weather: str = ""
    day_of_week: str = ""
    customer_satisfaction_score: Optional[float] = Field(None, ge=0, le=1)
    notes: str = ""


class OperationalOutcomeResponse(BaseModel):
    outcome_id: str
    route_id: str
    learning_signals: Dict[str, Any]
    message: str


# System state schemas
class SystemStateResponse(BaseModel):
    pending_orders_count: int
    active_routes_count: int
    available_vehicles_count: int
    total_pending_weight: float
    total_pending_volume: float
    current_time: datetime
    eastleigh_window_active: bool
    traffic_conditions: Dict[str, float]
    weather: str


# Learning metrics schemas
class LearningMetricsResponse(BaseModel):
    cfa_metrics: Dict[str, Any]
    vfa_metrics: Dict[str, Any]
    pfa_metrics: Dict[str, Any]
    feedback_metrics: Dict[str, Any]
    last_updated: Optional[datetime] = None


# Error response schema
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Health check schema
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    components: Dict[str, str]

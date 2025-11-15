"""Decision and action schemas for Powell framework."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

from .domain import Route


class DecisionType(str, Enum):
    """Types of decisions the engine can make."""

    DAILY_ROUTE_PLANNING = "daily_route_planning"  # CFA/DLA Hybrid
    ORDER_ARRIVAL = "order_arrival"  # VFA or CFA
    REAL_TIME_ADJUSTMENT = "real_time_adjustment"  # PFA + CFA Hybrid
    BACKHAUL_OPPORTUNITY = "backhaul_opportunity"  # VFA


class ActionType(str, Enum):
    """Types of actions the engine can take."""

    CREATE_ROUTE = "create_route"
    ACCEPT_ORDER = "accept_order"
    DEFER_ORDER = "defer_order"
    REJECT_ORDER = "reject_order"
    ADJUST_ROUTE = "adjust_route"
    CONSOLIDATE_ORDERS = "consolidate_orders"


@dataclass
class PolicyDecision:
    """Output from a Powell policy class."""

    policy_name: str  # "PFA", "CFA", "VFA", "DLA"
    decision_type: DecisionType

    # Action recommendation
    recommended_action: ActionType
    routes: List[Route]  # Proposed routes to execute

    # Decision quality
    confidence_score: float  # 0.0-1.0
    expected_value: float  # Estimated business value (KES or utility)
    reasoning: str  # Human-readable explanation

    # Decision context
    considered_alternatives: int = 0  # How many alternatives were evaluated
    is_deterministic: bool = True  # Whether decision is deterministic or probabilistic

    # Metadata
    decision_id: str = ""
    policy_parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HybridDecision:
    """Output from hybrid policy combining multiple policy classes."""

    hybrid_name: str  # "CFA/VFA", "DLA/VFA", "PFA/CFA"
    primary_policy: PolicyDecision
    secondary_policy: PolicyDecision

    # Composite decision
    recommended_action: ActionType
    routes: List[Route]

    # Blending
    primary_weight: float  # 0.0-1.0
    secondary_weight: float  # 0.0-1.0

    # Decision quality
    confidence_score: float
    expected_value: float
    reasoning: str

    # Metadata
    decision_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DecisionContext:
    """Context information passed to policy classes for decision making."""

    # Trigger information
    decision_type: DecisionType
    trigger_reason: str  # "daily_planning", "new_order_arrived", "delay_detected"

    # Relevant orders and vehicles
    orders_to_consider: Dict[str, "Order"] = field(
        default_factory=dict
    )  # Order ID -> Order
    vehicles_available: Dict[str, "Vehicle"] = field(
        default_factory=dict
    )  # Vehicle ID -> Vehicle

    # Current system state (aggregated)
    total_pending_orders: int = 0
    total_available_capacity: tuple[float, float] = (0.0, 0.0)  # (weight, volume)
    fleet_utilization_percent: float = 0.0

    # Constraints
    hard_constraints: List[Dict[str, Any]] = field(default_factory=list)
    soft_constraints: List[Dict[str, Any]] = field(default_factory=list)

    # Environmental context
    current_time: datetime = field(default_factory=datetime.now)
    day_of_week: str = ""
    time_of_day: str = ""  # "morning", "midday", "evening"
    traffic_conditions: Dict[str, float] = field(default_factory=dict)

    # Learning state
    learned_model_confidence: Dict[str, float] = field(default_factory=dict)
    recent_accuracy_metrics: Dict[str, float] = field(default_factory=dict)

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

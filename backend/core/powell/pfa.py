"""Policy Function Approximation - Rule-based decision making."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any
import logging

from ..models.state import SystemState
from ..models.domain import Order, Vehicle, Route, RouteStatus, OrderStatus
from ..models.decision import PolicyDecision, DecisionContext, DecisionType, ActionType

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """Learned decision rule from pattern mining."""

    rule_id: str
    name: str  # Human-readable name

    # Rule components
    conditions: List[Callable[[SystemState, DecisionContext], bool]]
    action: ActionType
    target_orders: List[str] = field(default_factory=list)  # Order IDs rule applies to

    # Rule confidence
    confidence: float = 0.9  # 0.0-1.0
    support: float = 0.5  # Fraction of historical decisions matching this rule

    # Performance tracking
    successful_applications: int = 0
    failed_applications: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None

    def apply(self, state: SystemState, context: DecisionContext) -> bool:
        """Check if all conditions are met."""
        return all(condition(state, context) for condition in self.conditions)

    def get_success_rate(self) -> float:
        """Calculate historical success rate."""
        total = self.successful_applications + self.failed_applications
        if total == 0:
            return 0.0
        return self.successful_applications / total


class PolicyFunctionApproximation:
    """Rule-based policy using learned decision patterns.

    PFA learns from operational patterns (e.g., "Eastleigh orders should go out
    during 8:30-9:45", "Fresh food gets priority", "Majid unavailable Wed 2PM+")
    and applies these rules when making decisions.

    This is the most explainable policy class - rules are human-readable.
    """

    def __init__(self, learning_state_dict: Optional[Dict[str, Any]] = None):
        """Initialize PFA with optional learned rules."""
        self.rules: List[Rule] = []
        self.rule_index: Dict[str, Rule] = {}

        if learning_state_dict:
            self._load_rules_from_dict(learning_state_dict)

    def _load_rules_from_dict(self, rules_dict: List[Dict[str, Any]]):
        """Load rules from learning state dictionary."""
        # Deserialize rules from a serializable structure.
        # Expected format: list of dicts with keys:
        #  - rule_id, name, action (string matching ActionType member name), confidence, support
        #  - conditions: list of {"type": "destination_city"|"special_handling"|"priority", "value": ...}
        if not rules_dict:
            # No persisted rules - keep rule set empty by default to avoid hardcoding
            return

        for rd in rules_dict:
            try:
                conds = []
                for c in rd.get("conditions", []):
                    ctype = c.get("type")
                    val = c.get("value")

                    if ctype == "destination_city":

                        def make_dest_condition(target):
                            return lambda s, ctx, target=target: any(
                                getattr(o.destination_city, "value", None) == target
                                or getattr(o.destination_city, "name", None) == target
                                for o in ctx.orders_to_consider.values()
                            )

                        conds.append(make_dest_condition(val))

                    elif ctype == "special_handling":

                        def make_tag_condition(tag):
                            return lambda s, ctx, tag=tag: any(
                                tag in getattr(o, "special_handling", [])
                                for o in ctx.orders_to_consider.values()
                            )

                        conds.append(make_tag_condition(val))

                    elif ctype == "priority":

                        def make_priority_condition(min_prio):
                            m = int(min_prio)
                            return lambda s, ctx, m=m: any(
                                getattr(o, "priority", 0) >= m
                                for o in ctx.orders_to_consider.values()
                            )

                        conds.append(make_priority_condition(val))

                    else:
                        # Unknown condition type: skip
                        continue

                action_name = rd.get("action")
                if action_name and hasattr(ActionType, action_name):
                    action = getattr(ActionType, action_name)
                else:
                    action = ActionType.DEFER_ORDER

                new_rule = Rule(
                    rule_id=rd.get("rule_id", f"rule_{len(self.rules)}"),
                    name=rd.get("name", "imported_rule"),
                    conditions=conds,
                    action=action,
                    confidence=float(rd.get("confidence", 0.5)),
                    support=float(rd.get("support", 0.5)),
                )

                self.add_rule(new_rule)
            except Exception:
                logger.exception("Failed to load rule from dict - skipping")

    def _initialize_business_rules(self):
        """Initialize with known business rules (hard constraints)."""
        # IMPORTANT: removed hardcoded business-specific rules from codebase.
        # PFA should not hardcode business names/locations. Rules must be
        # learned or loaded from configuration/learning state. This method
        # is intentionally left minimal to avoid baked-in templates.
        # If you need hard constraints (e.g., regulatory windows), provide
        # them via configuration passed into the engine or persisted
        # `learning_state_dict` so they can be audited and managed.
        return

    def add_rule(self, rule: Rule):
        """Register a learned rule."""
        self.rules.append(rule)
        self.rule_index[rule.rule_id] = rule

    def evaluate(self, state: SystemState, context: DecisionContext) -> PolicyDecision:
        """Apply PFA to make a decision.

        Workflow:
        1. Filter rules applicable to current context
        2. Sort by confidence * support (rule quality)
        3. Apply highest-confidence applicable rule
        4. If no rule applies, return no-op decision
        """

        # Find all applicable rules
        applicable_rules = []
        for rule in self.rules:
            if rule.apply(state, context):
                applicable_rules.append(rule)

        # Sort by quality (confidence * support * historical success)
        applicable_rules.sort(
            key=lambda r: r.confidence * r.support * (r.get_success_rate() + 0.1),
            reverse=True,
        )

        if not applicable_rules:
            # No rules apply - return neutral decision
            return PolicyDecision(
                policy_name="PFA",
                decision_type=context.decision_type,
                recommended_action=ActionType.DEFER_ORDER,
                routes=[],
                confidence_score=0.0,
                expected_value=0.0,
                reasoning="No learned rules applicable to this context",
            )

        # Apply best rule
        best_rule = applicable_rules[0]
        best_rule.last_applied = datetime.now()

        # Generate routes based on rule recommendation
        routes = self._generate_routes_for_rule(state, context, best_rule)

        # Calculate expected value
        expected_value = sum(state.get_estimated_route_value(r) for r in routes)

        return PolicyDecision(
            policy_name="PFA",
            decision_type=context.decision_type,
            recommended_action=best_rule.action,
            routes=routes,
            confidence_score=best_rule.confidence
            * (best_rule.get_success_rate() + 0.1),
            expected_value=expected_value,
            reasoning=f"Applied learned rule: {best_rule.name}",
            considered_alternatives=len(applicable_rules),
            is_deterministic=True,
            policy_parameters={
                "rule_id": best_rule.rule_id,
                "rule_name": best_rule.name,
                "confidence": best_rule.confidence,
                "support": best_rule.support,
            },
        )

    def _generate_routes_for_rule(
        self, state: SystemState, context: DecisionContext, rule: Rule
    ) -> List[Route]:
        """Generate routes that satisfy the rule's action."""

        if rule.action == ActionType.CREATE_ROUTE:
            # Group orders and assign to available vehicles
            routes = self._group_orders_into_routes(
                state, context.orders_to_consider, context.vehicles_available
            )
            return routes

        elif rule.action == ActionType.ACCEPT_ORDER:
            # Create route for first available order
            if context.orders_to_consider:
                order = next(iter(context.orders_to_consider.values()))
                suitable_vehicles = state.get_vehicles_with_capacity_for(order)

                if suitable_vehicles:
                    vehicle = next(iter(suitable_vehicles.values()))
                    route = self._create_single_order_route(state, order, vehicle)
                    return [route]
            return []

        elif rule.action == ActionType.DEFER_ORDER:
            # No routes generated for defer
            return []

        return []

    def _group_orders_into_routes(
        self, state: SystemState, orders: Dict[str, Order], vehicles: Dict[str, Vehicle]
    ) -> List[Route]:
        """Simple greedy routing: group orders by destination, assign to vehicles."""
        routes = []
        assigned_orders = set()

        for city_enum in [o.destination_city for o in orders.values()]:
            city_orders = {
                oid: o
                for oid, o in orders.items()
                if o.destination_city == city_enum and oid not in assigned_orders
            }

            if not city_orders:
                continue

            # Find vehicle with capacity
            for vehicle in vehicles.values():
                total_weight = sum(o.weight_tonnes for o in city_orders.values())
                total_volume = sum(o.volume_m3 for o in city_orders.values())

                if vehicle.has_capacity_for(total_weight, total_volume):
                    route = self._create_route_from_orders(state, city_orders, vehicle)
                    routes.append(route)
                    assigned_orders.update(city_orders.keys())
                    break

        return routes

    def _create_single_order_route(
        self, state: SystemState, order: Order, vehicle: Vehicle
    ) -> Route:
        """Create simple route for single order."""
        route_id = f"route_{len(state.active_routes)}_{int(datetime.now().timestamp())}"

        route = Route(
            route_id=route_id,
            vehicle_id=vehicle.vehicle_id,
            order_ids=[order.order_id],
            stops=[],  # Simplified - no stop details
            destination_cities=[order.destination_city],
            total_distance_km=0.0,  # Placeholder
            estimated_duration_minutes=0,  # Placeholder
            estimated_cost_kes=0.0,  # Placeholder
            status=RouteStatus.PLANNED,
        )

        return route

    def _create_route_from_orders(
        self, state: SystemState, orders: Dict[str, Order], vehicle: Vehicle
    ) -> Route:
        """Create route from group of orders."""
        route_id = f"route_{len(state.active_routes)}_{int(datetime.now().timestamp())}"

        route = Route(
            route_id=route_id,
            vehicle_id=vehicle.vehicle_id,
            order_ids=list(orders.keys()),
            stops=[],  # Simplified
            destination_cities=list(set(o.destination_city for o in orders.values())),
            total_distance_km=0.0,  # Placeholder
            estimated_duration_minutes=0,  # Placeholder
            estimated_cost_kes=0.0,  # Placeholder
            status=RouteStatus.PLANNED,
        )

        return route

    def update_from_feedback(self, outcome_dict: Dict[str, Any]):
        """Update rule confidence/support based on operational feedback.

        Args:
            outcome_dict: Contains fields like success, rule_id, prediction_error, etc.
        """
        rule_id = outcome_dict.get("rule_id")
        success = outcome_dict.get("success", False)

        if rule_id in self.rule_index:
            rule = self.rule_index[rule_id]

            if success:
                rule.successful_applications += 1
            else:
                rule.failed_applications += 1

            # Adjust confidence gradually (exponential smoothing)
            alpha = 0.1  # Learning rate
            new_success_rate = rule.get_success_rate()
            rule.confidence = alpha * new_success_rate + (1 - alpha) * rule.confidence

    def mine_rules_from_state(
        self, state, min_support: int = 3, min_success_rate: float = 0.8
    ):
        """Mine simple candidate rules from recent operational outcomes stored in state.

        Currently mines two types of patterns:
        - Destination city consolidation rules (if many successful deliveries to same city)
        - Special-handling rules (e.g., 'fresh_food')

        When a pattern meets support and success thresholds, a new Rule is created
        and added to the PFA rule set.
        """

        # Aggregate statistics from recent outcomes
        city_counts = {}
        city_success = {}
        special_counts = {}
        special_success = {}

        for outcome in state.recent_outcomes:
            # find associated route (completed routes map)
            route = state.completed_routes.get(
                outcome.route_id
            ) or state.active_routes.get(outcome.route_id)
            if not route:
                continue

            success = bool(outcome.on_time)

            for oid in route.order_ids:
                order = state.pending_orders.get(oid) or None
                if not order:
                    # If order not found in pending, try customers or ignore
                    continue

                city = order.destination_city
                city_counts[city] = city_counts.get(city, 0) + 1
                city_success[city] = city_success.get(city, 0) + (1 if success else 0)

                for tag in order.special_handling:
                    special_counts[tag] = special_counts.get(tag, 0) + 1
                    special_success[tag] = special_success.get(tag, 0) + (
                        1 if success else 0
                    )

        # Create rules for cities
        for city, count in city_counts.items():
            success_rate = city_success.get(city, 0) / count if count > 0 else 0.0
            if count >= min_support and success_rate >= min_success_rate:
                # Build condition closure
                def make_city_condition(target_city):
                    return lambda s, c: any(
                        o.destination_city == target_city
                        for o in c.orders_to_consider.values()
                    )

                cond = make_city_condition(city)
                rule_id = f"rule_city_{str(city)}"
                if rule_id not in self.rule_index:
                    new_rule = Rule(
                        rule_id=rule_id,
                        name=f"Consolidate to {city.value}",
                        conditions=[cond],
                        action=ActionType.CREATE_ROUTE,
                        confidence=success_rate,
                        support=float(count) / max(1, len(state.recent_outcomes)),
                    )
                    self.add_rule(new_rule)

        # Create rules for special handling tags
        for tag, count in special_counts.items():
            success_rate = special_success.get(tag, 0) / count if count > 0 else 0.0
            if count >= min_support and success_rate >= min_success_rate:

                def make_tag_condition(t):
                    return lambda s, c: any(
                        t in o.special_handling for o in c.orders_to_consider.values()
                    )

                rule_id = f"rule_tag_{tag}"
                if rule_id not in self.rule_index:
                    new_rule = Rule(
                        rule_id=rule_id,
                        name=f"Handle tag {tag}",
                        conditions=[make_tag_condition(tag)],
                        action=ActionType.CREATE_ROUTE,
                        confidence=success_rate,
                        support=float(count) / max(1, len(state.recent_outcomes)),
                    )
                    self.add_rule(new_rule)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "action": (
                        r.action.name if hasattr(r.action, "name") else str(r.action)
                    ),
                    "confidence": r.confidence,
                    "support": r.support,
                    "success_rate": r.get_success_rate(),
                    "applications": r.successful_applications + r.failed_applications,
                }
                for r in self.rules
            ]
        }

    def export_rules_for_learning_state(self) -> List[Dict[str, Any]]:
        """Export rules into a serializable format suitable for LearningState.pfa_rules.

        This is a best-effort exporter: it preserves rule metadata and attempts
        to infer simple condition descriptors from human-readable rule names
        (e.g., 'Consolidate to <city>' → destination_city condition).
        Complex condition closures will be exported with empty `conditions`.
        """
        exported = []
        for r in self.rules:
            rule_entry = {
                "rule_id": r.rule_id,
                "name": r.name,
                "action": r.action.name if hasattr(r.action, "name") else str(r.action),
                "confidence": float(r.confidence),
                "support": float(r.support),
                "conditions": [],
            }

            # Infer simple condition types from rule name
            if r.name.startswith("Consolidate to "):
                city = r.name.replace("Consolidate to ", "")
                rule_entry["conditions"].append(
                    {"type": "destination_city", "value": city}
                )

            if r.name.startswith("Handle tag "):
                tag = r.name.replace("Handle tag ", "")
                rule_entry["conditions"].append(
                    {"type": "special_handling", "value": tag}
                )

            exported.append(rule_entry)

        return exported

"""Policy Function Approximation - Rule-based decision making."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any, Set
import logging
import uuid

from ..models.state import SystemState
from ..models.domain import Order, Vehicle, Route, RouteStatus, OrderStatus
from ..models.decision import PolicyDecision, DecisionContext, DecisionType, ActionType

# sde learning components
from ..learning.pattern_mining import PatternMiningCoordinator
from ..learning.exploration import ExplorationCoordinator, EpsilonGreedy

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

        # sde pattern mining coordinator
        self.pattern_coordinator = PatternMiningCoordinator(
            min_support=0.1,  # 10% frequency threshold
            min_confidence=0.5,  # 50% confidence threshold
            min_lift=1.2,  # 20% better than random
            max_rules=100,  # Keep top 100 rules
        )

        # Exploration for rule selection
        self.rule_exploration = ExplorationCoordinator(
            strategy=EpsilonGreedy(epsilon=0.1, epsilon_decay=0.995),
            track_statistics=True,
        )

        # Track recent outcomes for pattern mining
        self.recent_outcomes: List[Dict[str, Any]] = []

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
        """Apply PFA to make a decision with exploration.

        Enhanced workflow:
        1. Filter rules applicable to current context
        2. Compute rule quality scores (confidence * support * success_rate)
        3. Use exploration strategy to select rule (ε-greedy)
        4. Apply selected rule
        5. If no rule applies, return no-op decision
        """

        # Find all applicable rules
        applicable_rules = []
        for rule in self.rules:
            if rule.apply(state, context):
                applicable_rules.append(rule)

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

        # Compute rule quality values for exploration
        rule_values = {}
        for rule in applicable_rules:
            quality = rule.confidence * rule.support * (rule.get_success_rate() + 0.1)
            rule_values[rule.rule_id] = quality

        # Use exploration coordinator to select rule
        selected_rule_id = self.rule_exploration.select_action(
            actions=[r.rule_id for r in applicable_rules],
            action_values=rule_values,
        )

        # Find the selected rule
        selected_rule = next(
            r for r in applicable_rules if r.rule_id == selected_rule_id
        )
        selected_rule.last_applied = datetime.now()

        # Generate routes based on rule recommendation
        routes = self._generate_routes_for_rule(state, context, selected_rule)

        # Calculate expected value
        expected_value = sum(state.get_estimated_route_value(r) for r in routes)

        return PolicyDecision(
            policy_name="PFA",
            decision_type=context.decision_type,
            recommended_action=selected_rule.action,
            routes=routes,
            confidence_score=selected_rule.confidence
            * (selected_rule.get_success_rate() + 0.1),
            expected_value=expected_value,
            reasoning=f"Applied learned rule: {selected_rule.name} (exploration)",
            considered_alternatives=len(applicable_rules),
            is_deterministic=False,  # Now uses exploration
            policy_parameters={
                "rule_id": selected_rule.rule_id,
                "rule_name": selected_rule.name,
                "confidence": selected_rule.confidence,
                "support": selected_rule.support,
                "exploration_rate": self.rule_exploration.get_exploration_rate(),
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
        route_id = f"route_pfa_{uuid.uuid4().hex[:12]}"

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
        route_id = f"route_pfa_{uuid.uuid4().hex[:12]}"

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
        """Mine association rules using Apriori algorithm from recent outcomes.

        Enhanced with sde pattern mining:
        - Apriori algorithm for frequent pattern discovery
        - Association rule learning with confidence, support, lift
        - Multi-feature pattern combinations
        - Automatic rule quality scoring

        Extracts features from outcomes and discovers patterns like:
        - "IF (destination=Eastleigh AND priority=high) THEN create_express_route"
        - "IF (special_tag=fresh_food AND time=morning) THEN consolidate_orders"
        """

        if not hasattr(state, "recent_outcomes") or len(state.recent_outcomes) < 20:
            logger.debug(
                f"Not enough outcomes for pattern mining (need 20+, have {len(getattr(state, 'recent_outcomes', []))})"
            )
            return

        # Convert outcomes to transactions for pattern mining
        for outcome in state.recent_outcomes:
            # Find associated route
            route = state.completed_routes.get(
                outcome.route_id
            ) or state.active_routes.get(outcome.route_id)
            if not route:
                continue

            # Extract context features
            features: Set[str] = set()
            actions: Set[str] = set()

            # Route-level features
            if hasattr(outcome, "route_id") and outcome.route_id:
                features.add(f"route_exists")

            # Time-based features
            if hasattr(state.environment, "current_time"):
                hour = state.environment.current_time.hour
                if 6 <= hour < 12:
                    features.add("time_morning")
                elif 12 <= hour < 18:
                    features.add("time_afternoon")
                else:
                    features.add("time_evening")

                # Day of week
                day = state.environment.current_time.strftime("%A")
                features.add(f"day_{day}")

            # Extract order-level features
            for oid in route.order_ids:
                order = state.pending_orders.get(oid)
                if not order:
                    continue

                # Destination city
                if hasattr(order, "destination_city"):
                    city_name = getattr(
                        order.destination_city, "value", str(order.destination_city)
                    )
                    features.add(f"destination_{city_name}")

                # Priority
                if hasattr(order, "priority"):
                    if order.priority >= 2:
                        features.add("priority_high")
                    elif order.priority == 1:
                        features.add("priority_medium")
                    else:
                        features.add("priority_low")

                # Special handling
                if hasattr(order, "special_handling") and order.special_handling:
                    for tag in order.special_handling:
                        features.add(f"tag_{tag}")

                # Order value (binned)
                if hasattr(order, "price_kes"):
                    if order.price_kes > 5000:
                        features.add("value_high")
                    elif order.price_kes > 2000:
                        features.add("value_medium")
                    else:
                        features.add("value_low")

            # Extract action features (what was done)
            if hasattr(outcome, "on_time") and outcome.on_time:
                actions.add("delivered_on_time")
            else:
                actions.add("delivered_late")

            if hasattr(route, "vehicle_type"):
                actions.add(f"vehicle_{route.vehicle_type}")

            if len(route.order_ids) > 3:
                actions.add("consolidated_route")
            elif len(route.order_ids) == 1:
                actions.add("single_order_route")
            else:
                actions.add("small_batch_route")

            # Determine reward (success metric)
            success = bool(outcome.on_time) if hasattr(outcome, "on_time") else False
            reward = 1.0 if success else -0.5

            # Add transaction to pattern mining coordinator
            self.pattern_coordinator.add_transaction(
                transaction_id=outcome.route_id,
                features=features,
                actions=actions,
                context={"outcome": outcome, "route": route},
                reward=reward,
            )

            # Store in recent outcomes
            outcome_dict = {
                "route_id": outcome.route_id,
                "on_time": success,
                "features": list(features),
                "actions": list(actions),
                "reward": reward,
            }
            self.recent_outcomes.append(outcome_dict)

            # Limit history
            if len(self.recent_outcomes) > 1000:
                self.recent_outcomes.pop(0)

        # Mine patterns and generate rules
        num_rules = self.pattern_coordinator.mine_and_update_rules(
            force=len(state.recent_outcomes) >= 20
        )

        if num_rules > 0:
            logger.info(
                f"PFA: Mined {num_rules} association rules using Apriori algorithm"
            )

            # Convert mined rules to PFA Rule objects
            self._convert_mined_rules_to_pfa_rules()

    def _convert_mined_rules_to_pfa_rules(self):
        """Convert mined association rules to PFA Rule objects."""
        for assoc_rule in self.pattern_coordinator.active_rules[:20]:  # Top 20 rules
            if not assoc_rule.active:
                continue

            # Skip if already exists
            if assoc_rule.rule_id in self.rule_index:
                # Update existing rule performance
                existing = self.rule_index[assoc_rule.rule_id]
                existing.confidence = assoc_rule.confidence
                existing.support = assoc_rule.support
                continue

            # Create condition functions from antecedent
            conditions = []
            rule_name_parts = []

            for item in sorted(assoc_rule.antecedent):
                item_str = str(item)

                if item_str.startswith("destination_"):
                    city_name = item_str.replace("destination_", "")
                    rule_name_parts.append(f"dest={city_name}")

                    def make_dest_condition(city):
                        return lambda s, c, city=city: any(
                            getattr(o.destination_city, "value", None) == city
                            or getattr(o.destination_city, "name", None) == city
                            for o in c.orders_to_consider.values()
                        )

                    conditions.append(make_dest_condition(city_name))

                elif item_str.startswith("tag_"):
                    tag = item_str.replace("tag_", "")
                    rule_name_parts.append(f"tag={tag}")

                    def make_tag_condition(t):
                        return lambda s, c, t=t: any(
                            t in (o.special_handling if o.special_handling else [])
                            for o in c.orders_to_consider.values()
                        )

                    conditions.append(make_tag_condition(tag))

                elif item_str == "priority_high":
                    rule_name_parts.append("high_priority")

                    def priority_high_condition(s, c):
                        return any(
                            o.priority >= 2 for o in c.orders_to_consider.values()
                        )

                    conditions.append(priority_high_condition)

                elif item_str.startswith("time_"):
                    time_period = item_str.replace("time_", "")
                    rule_name_parts.append(f"time={time_period}")
                    # Time conditions are harder to check from context, skip for now

            # Determine action from consequent
            action_type = ActionType.CREATE_ROUTE  # Default
            for item in assoc_rule.consequent:
                item_str = str(item)
                if "consolidated" in item_str:
                    action_type = ActionType.CREATE_ROUTE
                    rule_name_parts.append("→ consolidate")
                elif "single_order" in item_str:
                    action_type = ActionType.CREATE_ROUTE
                    rule_name_parts.append("→ immediate")

            # Create human-readable name
            if not rule_name_parts:
                rule_name_parts = ["Generic rule"]
            rule_name = " ".join(rule_name_parts)

            # Create PFA Rule
            new_rule = Rule(
                rule_id=assoc_rule.rule_id,
                name=rule_name,
                conditions=(
                    conditions if conditions else [lambda s, c: True]
                ),  # Always-true fallback
                action=action_type,
                confidence=assoc_rule.confidence,
                support=assoc_rule.support,
                successful_applications=assoc_rule.successes,
                failed_applications=assoc_rule.failures,
            )

            self.add_rule(new_rule)
            logger.debug(
                f"Created PFA rule: {rule_name} "
                f"(confidence={assoc_rule.confidence:.2f}, lift={assoc_rule.lift:.2f})"
            )

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

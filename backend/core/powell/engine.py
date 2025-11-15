"""Powell Sequential Decision Engine - Main coordinator."""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

from ..models.state import SystemState
from ..models.decision import (
    PolicyDecision,
    HybridDecision,
    DecisionContext,
    DecisionType,
    ActionType,
)
from ..models.domain import OrderStatus

from .pfa import PolicyFunctionApproximation
from .cfa import CostFunctionApproximation, CostParameters
from .vfa import ValueFunctionApproximation
from .dla import DirectLookaheadApproximation
from .hybrids import CFAVFAHybrid, DLAVFAHybrid, PFACFAHybrid
from backend.utils.config import load_model_config

logger = logging.getLogger(__name__)


class PowellEngine:
    """Sequential Decision Engine implementing Powell Framework.

    The engine selects optimal policy for each decision context, executes it,
    and learns from outcomes.

    Decision workflow:
    1. Analyze current state and decision context
    2. Select appropriate policy (or hybrid) based on context
    3. Execute policy to get decision
    4. Commit decision and create routes/actions
    5. Receive feedback and learn

    Policy Selection Logic:
    - Daily Route Planning (morning, many pending orders) → CFA/VFA Hybrid
    - Order Arrival (new order received) → VFA or CFA based on backhaul opportunity
    - Real-Time Adjustment (delay/issue detected) → PFA/CFA Hybrid
    - Backhaul Opportunity → VFA (long-term value assessment)
    """

    def __init__(self):
        """Initialize Powell Engine with all policy classes."""

        # Load model configuration (YAML/JSON)
        config = load_model_config()

        # Keep raw config for downstream components
        self.config = config

        # Individual policies (allow config to override hyperparams)
        vfa_cfg = config.get("vfa", {})
        cfa_cfg = config.get("cfa", {})
        dla_cfg = config.get("dla", {})

        self.pfa = PolicyFunctionApproximation()

        # CFA: construct parameters then apply overrides from config
        cfa_params = CostParameters()
        if isinstance(cfa_cfg, dict):
            tm = cfa_cfg.get("traffic_multipliers")
            if isinstance(tm, dict):
                cfa_params.traffic_multipliers.update(tm)
            dm = cfa_cfg.get("distance_matrix")
            if isinstance(dm, dict):
                # Expect keys as string tuples; update where possible
                try:
                    for k, v in dm.items():
                        if isinstance(k, str) and "," in k:
                            a, b = [x.strip() for x in k.split(",", 1)]
                            cfa_params.distance_matrix[(a, b)] = float(v)
                        elif isinstance(k, (list, tuple)) and len(k) == 2:
                            cfa_params.distance_matrix[(k[0], k[1])] = float(v)
                except Exception:
                    # If malformed, skip safely
                    pass
            # Vehicle-specific fuel/driver costs
            fpk = cfa_cfg.get("fuel_cost_per_km_by_vehicle")
            if isinstance(fpk, dict):
                try:
                    cfa_params.fuel_cost_per_km_by_vehicle.update(
                        {k: float(v) for k, v in fpk.items()}
                    )
                except Exception:
                    pass

            dch = cfa_cfg.get("driver_cost_per_hour_by_vehicle")
            if isinstance(dch, dict):
                try:
                    cfa_params.driver_cost_per_hour_by_vehicle.update(
                        {k: float(v) for k, v in dch.items()}
                    )
                except Exception:
                    pass

        self.cfa = CostFunctionApproximation(cfa_params)

        # VFA
        self.vfa = ValueFunctionApproximation(
            state_feature_dim=vfa_cfg.get("state_feature_dim", 20),
            use_pytorch=vfa_cfg.get("use_pytorch", True),
            special_handling_tags=vfa_cfg.get("special_handling_tags", None),
        )

        # DLA
        self.dla = DirectLookaheadApproximation(vfa=self.vfa)

        # Hybrid policies
        self.cfa_vfa_hybrid = CFAVFAHybrid(self.cfa, self.vfa)
        self.dla_vfa_hybrid = DLAVFAHybrid(self.dla, self.vfa)
        self.pfa_cfa_hybrid = PFACFAHybrid(self.pfa, self.cfa)

        # Decision history
        self.decision_history: List[Dict[str, Any]] = []
        self.executed_routes: List[str] = []

    def make_decision(
        self,
        state: SystemState,
        decision_type: DecisionType,
        orders_to_consider: Dict[str, any] = None,
        trigger_reason: str = "",
    ) -> Union[PolicyDecision, HybridDecision]:
        """Make optimal decision for given context.

        Args:
            state: Current system state
            decision_type: Type of decision (daily planning, order arrival, etc.)
            orders_to_consider: Specific orders to evaluate (if None, use pending)
            trigger_reason: Human-readable reason for decision

        Returns:
            Either single PolicyDecision or HybridDecision with recommended actions
        """

        # Build decision context
        context = self._build_decision_context(
            state, decision_type, orders_to_consider, trigger_reason
        )

        # Select appropriate policy/hybrid based on context
        decision = self._select_and_execute_policy(state, context, decision_type)

        # Record decision
        self._record_decision(state, context, decision)

        return decision

    def _build_decision_context(
        self,
        state: SystemState,
        decision_type: DecisionType,
        orders_to_consider: Optional[Dict[str, any]],
        trigger_reason: str,
    ) -> DecisionContext:
        """Build decision context from system state."""

        # Use provided orders or all pending
        if orders_to_consider is None:
            orders = state.get_unassigned_orders()
        else:
            orders = orders_to_consider

        available_vehicles = state.get_available_vehicles()

        # Calculate context metrics
        total_pending = len(state.pending_orders)
        avail_capacity = state.get_available_capacity_by_vehicle()
        total_capacity = (
            sum(c[0] for c in avail_capacity.values()),
            sum(c[1] for c in avail_capacity.values()),
        )

        # Average utilization
        utilizations = [
            state.get_vehicle_utilization_percent(v) for v in available_vehicles.keys()
        ]
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0.0

        # Time analysis
        current_time = state.environment.current_time
        hour = current_time.hour
        if hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "midday"
        else:
            time_of_day = "evening"

        return DecisionContext(
            decision_type=decision_type,
            trigger_reason=trigger_reason,
            orders_to_consider=orders,
            vehicles_available=available_vehicles,
            total_pending_orders=total_pending,
            total_available_capacity=total_capacity,
            fleet_utilization_percent=avg_util,
            current_time=current_time,
            day_of_week=current_time.strftime("%A"),
            time_of_day=time_of_day,
            traffic_conditions=state.environment.traffic_conditions.copy(),
            learned_model_confidence=state.get_learning_confidence_vector(),
            recent_accuracy_metrics={
                "cfa_fuel": state.learning.cfa_accuracy_fuel,
                "cfa_time": state.learning.cfa_accuracy_time,
                "vfa": state.learning.vfa_accuracy,
                "pfa": state.learning.pfa_coverage,
            },
        )

    def _select_and_execute_policy(
        self, state: SystemState, context: DecisionContext, decision_type: DecisionType
    ) -> Union[PolicyDecision, HybridDecision]:
        """Select appropriate policy based on decision context and execute it."""

        if decision_type == DecisionType.DAILY_ROUTE_PLANNING:
            # Morning optimization: CFA for cost + VFA for strategic value
            logger.info("Daily route planning - using CFA/VFA hybrid")
            return self.cfa_vfa_hybrid.evaluate(state, context)

        elif decision_type == DecisionType.ORDER_ARRIVAL:
            # New order arrived: check for backhaul opportunity
            if self._has_backhaul_opportunity(state):
                # Use VFA to assess long-term value
                logger.info("Order arrival with backhaul opportunity - using VFA")
                return self.vfa.evaluate(state, context)
            else:
                # Standard order: use CFA for cost optimization
                logger.info("Standard order arrival - using CFA")
                return self.cfa.evaluate(state, context)

        elif decision_type == DecisionType.REAL_TIME_ADJUSTMENT:
            # Disruption detected: use learned rules + re-optimize
            # PFA provides constraints, CFA re-optimizes
            logger.info("Real-time adjustment - using PFA/CFA hybrid")
            return self.pfa_cfa_hybrid.evaluate(state, context)

        elif decision_type == DecisionType.BACKHAUL_OPPORTUNITY:
            # Strategic decision: VFA assesses long-term value
            logger.info("Backhaul opportunity assessment - using VFA")
            return self.vfa.evaluate(state, context)

        else:
            # Default: use CFA for immediate optimization
            logger.warning(
                f"Unknown decision type {decision_type} - falling back to CFA"
            )
            return self.cfa.evaluate(state, context)

    def _has_backhaul_opportunity(self, state: SystemState) -> bool:
        """Check if current state has backhaul opportunities."""
        opportunities = state.get_backhaul_opportunities()
        return len(opportunities) > 0

    def _record_decision(
        self,
        state: SystemState,
        context: DecisionContext,
        decision: Union[PolicyDecision, HybridDecision],
    ):
        """Record decision for audit trail and learning."""

        record = {
            "timestamp": datetime.now().isoformat(),
            "state_id": state.state_id,
            "decision_type": context.decision_type.value,
            "trigger_reason": context.trigger_reason,
            "policy_used": (
                decision.policy_name
                if isinstance(decision, PolicyDecision)
                else decision.hybrid_name
            ),
            "recommended_action": decision.recommended_action.value,
            "confidence_score": decision.confidence_score,
            "expected_value": decision.expected_value,
            "routes_proposed": [r.route_id for r in decision.routes],
            "reasoning": decision.reasoning,
        }

        self.decision_history.append(record)
        logger.info(
            f"Decision recorded: {decision.policy_name if isinstance(decision, PolicyDecision) else decision.hybrid_name} → {decision.recommended_action.value}"
        )

    def commit_decision(
        self, decision: Union[PolicyDecision, HybridDecision], state: SystemState
    ) -> Dict[str, any]:
        """Commit decision and execute recommended actions.

        Args:
            decision: Decision from make_decision()
            state: Current system state

        Returns:
            Execution result with route IDs and status
        """

        result = {
            "success": False,
            "action": decision.recommended_action.value,
            "routes_created": [],  # list of Route objects to be committed by StateManager
            "orders_assigned": [],
            "errors": [],
        }

        if decision.recommended_action == ActionType.CREATE_ROUTE:
            # Validate routes and return them for the caller (StateManager) to commit
            for route in decision.routes:
                try:
                    if self._validate_route(route, state):
                        self.executed_routes.append(route.route_id)
                        result["routes_created"].append(route)
                        result["orders_assigned"].extend(route.order_ids)
                    else:
                        result["errors"].append(
                            f"Route {route.route_id} validation failed"
                        )
                except Exception as e:
                    result["errors"].append(
                        f"Error preparing route {route.route_id}: {str(e)}"
                    )
                    logger.error(f"Route commit error: {e}")

            result["success"] = len(result["routes_created"]) > 0

        elif decision.recommended_action == ActionType.ACCEPT_ORDER:
            # Accept and assign order — return route for commit
            if decision.routes:
                route = decision.routes[0]
                self.executed_routes.append(route.route_id)
                result["routes_created"].append(route)
                result["orders_assigned"].extend(route.order_ids)
                result["success"] = True

        elif decision.recommended_action == ActionType.DEFER_ORDER:
            # No action needed
            result["success"] = True

        elif decision.recommended_action == ActionType.ADJUST_ROUTE:
            # Modify existing routes (not implemented in base)
            result["success"] = True

        return result

    def _validate_route(self, route, state: SystemState) -> bool:
        """Validate route is feasible."""

        # Check vehicle exists and has capacity
        vehicle = state.fleet.get(route.vehicle_id)
        if not vehicle:
            return False

        if not route.is_feasible(vehicle, state.pending_orders):
            return False

        # Check orders exist and are assignable
        for order_id in route.order_ids:
            if order_id not in state.pending_orders:
                return False

        return True

    def learn_from_feedback(self, outcome: Dict[str, any]):
        """Update learned models based on operational feedback.

        Args:
            outcome: Feedback from executed route
                - route_id, vehicle_id
                - predicted_fuel_cost, actual_fuel_cost
                - predicted_duration_minutes, actual_duration_minutes
                - success (bool), on_time (bool)
                - customer_satisfaction_score
        """

        try:
            # Update CFA parameters
            self.cfa.update_from_feedback(outcome)

            # Update PFA rules
            rule_id = outcome.get("applied_rule_id")
            if rule_id:
                self.pfa.update_from_feedback(
                    {"rule_id": rule_id, "success": outcome.get("success", False)}
                )

            # TD-learning update for VFA
            if "next_state" in outcome:
                # This would be called with actual state transitions in production
                reward = outcome.get("profit", 0.0)
                # self.vfa.td_learning_update(state, action, reward, next_state)

        except Exception as e:
            logger.error(f"Learning error: {e}")

    def get_decision_history(self, limit: int = 100) -> List[Dict[str, any]]:
        """Retrieve recent decision history."""
        return self.decision_history[-limit:]

    def get_policy_performance(self) -> Dict[str, any]:
        """Analyze performance of each policy."""

        policy_stats = {}

        for record in self.decision_history:
            policy = record.get("policy_used", "unknown")

            if policy not in policy_stats:
                policy_stats[policy] = {
                    "count": 0,
                    "total_expected_value": 0.0,
                    "avg_confidence": 0.0,
                }

            policy_stats[policy]["count"] += 1
            policy_stats[policy]["total_expected_value"] += record.get(
                "expected_value", 0.0
            )
            policy_stats[policy]["avg_confidence"] += record.get(
                "confidence_score", 0.0
            )

        # Compute averages
        for policy, stats in policy_stats.items():
            count = max(1, stats["count"])
            stats["avg_confidence"] = stats["avg_confidence"] / count
            stats["avg_expected_value"] = stats["total_expected_value"] / count

        return policy_stats

    def get_learned_state(self) -> Dict[str, any]:
        """Export current learned state for persistence."""
        return {
            "timestamp": datetime.now().isoformat(),
            "pfa": self.pfa.to_dict(),
            "cfa": self.cfa.to_dict(),
            "vfa": self.vfa.to_dict(),
            "dla": self.dla.to_dict(),
            "decision_count": len(self.decision_history),
            "routes_executed": len(self.executed_routes),
        }

    def restore_learned_state(self, state_dict: Dict[str, any]):
        """Restore learned state from persistence (PFA rules, model weights, etc.)."""
        logger.info(f"Restoring learned state from {state_dict.get('timestamp')}")
        try:
            # Restore PFA rules
            pfa_rules = None
            if isinstance(state_dict.get("pfa"), dict):
                pfa_rules = state_dict.get("pfa", {}).get("rules")
            if not pfa_rules and state_dict.get("pfa_rules"):
                pfa_rules = state_dict.get("pfa_rules")
            if not pfa_rules and isinstance(state_dict.get("pfa"), list):
                pfa_rules = state_dict.get("pfa")

            if pfa_rules:
                try:
                    self.pfa._load_rules_from_dict(pfa_rules)
                    logger.info(
                        f"Loaded {len(self.pfa.rules)} PFA rules from persisted state"
                    )
                except Exception as e:
                    logger.exception(f"Failed to load PFA rules: {e}")

            # Restore VFA weights and state
            if isinstance(state_dict.get("vfa"), dict):
                try:
                    self.vfa.restore_from_dict(state_dict["vfa"])
                except Exception as e:
                    logger.debug(f"Failed to restore VFA state: {e}")

        except Exception:
            logger.exception("Error while restoring learned state")

"""Event-driven orchestration of decisions, execution, and learning."""

from typing import Dict, List, Optional, Any, Coroutine, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio

from ..core.models.state import SystemState
from ..core.models.decision import DecisionType, ActionType, DecisionContext
from ..core.models.domain import OperationalOutcome
from ..core.powell.engine import PowellEngine
from .state_manager import StateManager

logger = logging.getLogger(__name__)


class EventPriority(str, Enum):
    """Event priority levels."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Soon
    NORMAL = "normal"  # When convenient
    LOW = "low"  # Batch processing


class Event:
    """Base event class."""

    def __init__(
        self,
        event_type: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        timestamp: Optional[datetime] = None,
    ):
        self.event_type = event_type
        self.data = data
        self.priority = priority
        self.timestamp = timestamp or datetime.now()
        self.processed = False

    def __lt__(self, other):
        """Priority queue ordering (higher priority first)."""
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        return (
            priority_order[self.priority.value] < priority_order[other.priority.value]
        )


class EventOrchestrator:
    """Orchestrates decision flow, execution, and learning.

    Responsibilities:
    - Event reception and prioritization
    - Policy selection and execution
    - Route commitment and execution
    - Outcome recording and learning
    - Async/sync event handling
    """

    def __init__(self, engine: PowellEngine, state_manager: StateManager):
        """Initialize orchestrator."""
        self.engine = engine
        self.state_manager = state_manager
        self.event_queue: List[Event] = []
        self.processed_events: List[Event] = []

        # Event handlers
        self.decision_handlers: List[Callable] = []
        self.execution_handlers: List[Callable] = []
        self.learning_handlers: List[Callable] = []

    def submit_event(self, event: Event) -> str:
        """Submit event to queue."""
        event_id = (
            f"evt_{len(self.processed_events)}_{int(event.timestamp.timestamp())}"
        )

        # Sort by priority
        self.event_queue.append(event)
        self.event_queue.sort()

        logger.info(
            f"Event submitted: {event_id} ({event.event_type}, priority={event.priority})"
        )
        return event_id

    def submit_order_arrived(self, order: Any) -> str:
        """Convenience: submit order arrival event."""
        event = Event(
            event_type="order_arrived",
            data={"order": order},
            priority=EventPriority.HIGH,
        )
        return self.submit_event(event)

    def submit_route_outcome(self, outcome: OperationalOutcome) -> str:
        """Convenience: submit route completion outcome."""
        event = Event(
            event_type="route_outcome",
            data={"outcome": outcome},
            priority=EventPriority.NORMAL,
        )
        return self.submit_event(event)

    def process_event(self, event: Event) -> Dict[str, Any]:
        """Process single event through decision → execution → learning."""

        logger.info(f"Processing event: {event.event_type}")

        # Get current state
        state = self.state_manager.get_current_state()

        # Step 1: Determine decision type
        decision_type = self._map_event_to_decision_type(event)

        # Step 2: Make decision
        decision = self.engine.make_decision(
            state,
            decision_type=decision_type,
            trigger_reason=event.event_type,
        )

        # If VFA present, capture per-route pre-decision features so we can form (s,a,s')
        pre_features_by_route: Dict[str, List[float]] = {}
        try:
            if (
                hasattr(self.engine, "vfa")
                and decision
                and getattr(decision, "routes", None)
            ):
                # For each candidate route, build a focused DecisionContext and extract features
                for route in decision.routes:
                    try:
                        # Build orders dict for this route
                        orders = {
                            oid: state.pending_orders.get(oid)
                            for oid in getattr(route, "order_ids", [])
                            if state.pending_orders.get(oid)
                        }
                        vehicle = None
                        if getattr(route, "vehicle_id", None):
                            vehicle_obj = state.fleet.get(route.vehicle_id)
                            if vehicle_obj:
                                vehicle = {vehicle_obj.vehicle_id: vehicle_obj}

                        ctx = DecisionContext(
                            decision_type=DecisionType.ORDER_ARRIVAL,
                            trigger_reason=event.event_type,
                            orders_to_consider=orders,
                            vehicles_available=vehicle or {},
                            total_pending_orders=len(state.pending_orders),
                            total_available_capacity=state.get_available_capacity_by_vehicle(),
                            fleet_utilization_percent=state.get_fleet_utilization(),
                            current_time=state.environment.current_time,
                            day_of_week=state.environment.current_time.strftime("%A"),
                            time_of_day="",
                            traffic_conditions=state.environment.traffic_conditions.copy(),
                            learned_model_confidence=state.get_learning_confidence_vector(),
                            recent_accuracy_metrics={},
                        )

                        pre_features = self.engine.vfa.extract_state_features(
                            state, ctx
                        )
                        pre_features_by_route[
                            getattr(route, "route_id", f"route_{id(route)}")
                        ] = pre_features
                    except Exception:
                        continue
        except Exception:
            pre_features_by_route = {}

        logger.info(
            f"Decision made: {decision.policy_name if hasattr(decision, 'policy_name') else decision.hybrid_name} "
            f"→ {decision.recommended_action}"
        )

        # Fire decision handlers
        for handler in self.decision_handlers:
            try:
                handler(state, decision)
            except Exception as e:
                logger.error(f"Error in decision handler: {e}")

        # Step 3: Commit decision if action is concrete
        execution_result = None
        if decision.recommended_action != ActionType.DEFER_ORDER:
            execution_result = self.engine.commit_decision(decision, state)

            logger.info(
                f"Decision committed: {execution_result['action']}, routes={execution_result['routes_created']}"
            )

            # Update state with created routes
            if execution_result["routes_created"]:
                for route_obj in execution_result["routes_created"]:
                    try:
                        # Apply route creation to state via StateManager for immutable update
                        if hasattr(self.state_manager, "apply_event"):
                            self.state_manager.apply_event(
                                "route_created", {"route": route_obj}
                            )
                    except Exception as e:
                        logger.error(f"Failed to apply route_created event: {e}")

                # After applying route-created events, capture post-decision features
                try:
                    post_state = self.state_manager.get_current_state()
                    post_state_features = None
                    if hasattr(self.engine, "vfa"):
                        try:
                            post_state_features = (
                                self.engine.vfa.extract_state_features_from_state(
                                    post_state
                                )
                            )
                        except Exception:
                            post_state_features = None

                    # Record pending experiences keyed by route id so learning can
                    # complete them later when outcomes arrive. Use per-route pre/post features
                    if getattr(self.engine, "vfa", None):
                        for route_obj in execution_result["routes_created"]:
                            try:
                                rid = getattr(route_obj, "route_id", None)
                                pre_feats = pre_features_by_route.get(rid)
                                # If we didn't compute per-route pre-features, fall back to global pre/post
                                if pre_feats is None:
                                    try:
                                        pre_feats = self.engine.vfa.extract_state_features_from_state(
                                            state
                                        )
                                    except Exception:
                                        pre_feats = None

                                post_feats = None
                                try:
                                    # If we captured a post_state_features global vector use it
                                    post_feats = post_state_features
                                except Exception:
                                    post_feats = None

                                self.engine.vfa.add_pending_experience(
                                    rid,
                                    pre_feats,
                                    decision.recommended_action,
                                    post_feats,
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Failed to store pending VFA experience for {getattr(route_obj, 'route_id', 'unknown')}: {e}"
                                )
                except Exception:
                    logger.debug(
                        "Failed to capture post-decision state/features for VFA"
                    )

            # Fire execution handlers
            for handler in self.execution_handlers:
                try:
                    handler(execution_result)
                except Exception as e:
                    logger.error(f"Error in execution handler: {e}")

        # Step 4: Learn from event (if outcome)
        if event.event_type == "route_outcome":
            outcome = event.data.get("outcome")
            if outcome:
                self.engine.learn_from_feedback(
                    {
                        "route_id": outcome.route_id,
                        "predicted_fuel_cost": outcome.predicted_fuel_cost,
                        "actual_fuel_cost": outcome.actual_fuel_cost,
                        "predicted_duration_minutes": outcome.predicted_duration_minutes,
                        "actual_duration_minutes": outcome.actual_duration_minutes,
                        "success": outcome.on_time,
                    }
                )

                logger.info(f"Learning update: route {outcome.route_id}")

                # Fire learning handlers
                for handler in self.learning_handlers:
                    try:
                        try:
                            ret = handler(outcome, state)
                        except TypeError:
                            # Fallback to single-arg handlers
                            ret = handler(outcome)

                        # If handler returned learning updates (e.g., PFA rules), persist them
                        if isinstance(ret, dict) and ret.get("pfa_rules"):
                            try:
                                self.state_manager.apply_event(
                                    "learning_updated",
                                    {"pfa_rules": ret.get("pfa_rules")},
                                )
                                logger.info(
                                    f"Persisted {len(ret.get('pfa_rules'))} PFA rules to learning state"
                                )
                            except Exception as e:
                                logger.error(f"Failed to persist learning updates: {e}")
                    except Exception as e:
                        logger.error(f"Error in learning handler: {e}")

        event.processed = True
        self.processed_events.append(event)

        return {
            "event_type": event.event_type,
            "decision_type": decision_type.value if decision_type else None,
            "decision": {
                "policy": (
                    decision.policy_name
                    if hasattr(decision, "policy_name")
                    else decision.hybrid_name
                ),
                "action": decision.recommended_action.value,
                "confidence": decision.confidence_score,
                "value": decision.expected_value,
            },
            "execution": execution_result,
        }

    def _map_event_to_decision_type(self, event: Event) -> DecisionType:
        """Map event type to decision type."""

        if event.event_type == "order_arrived":
            return DecisionType.ORDER_ARRIVAL

        elif event.event_type == "daily_planning":
            return DecisionType.DAILY_ROUTE_PLANNING

        elif event.event_type == "delay_detected":
            return DecisionType.REAL_TIME_ADJUSTMENT

        elif event.event_type == "backhaul_opportunity":
            return DecisionType.BACKHAUL_OPPORTUNITY

        else:
            return DecisionType.ORDER_ARRIVAL  # Default

    def process_all_events(self) -> List[Dict[str, Any]]:
        """Process all queued events in priority order."""
        results = []

        while self.event_queue:
            event = self.event_queue.pop(0)  # Priority queue pops highest priority
            result = self.process_event(event)
            results.append(result)

        return results

    async def process_events_async(self) -> List[Dict[str, Any]]:
        """Process events asynchronously (for integration with async API)."""
        results = []

        while self.event_queue:
            event = self.event_queue.pop(0)
            result = await self._process_event_async(event)
            results.append(result)

        return results

    async def _process_event_async(self, event: Event) -> Dict[str, Any]:
        """Process single event asynchronously."""
        # Wrapper for sync processing to use in async context
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process_event, event
        )

    def register_decision_handler(self, handler: Callable):
        """Register handler called after decision made."""
        self.decision_handlers.append(handler)

    def register_execution_handler(self, handler: Callable):
        """Register handler called after decision executed."""
        self.execution_handlers.append(handler)

    def register_learning_handler(self, handler: Callable):
        """Register handler called after learning update."""
        self.learning_handlers.append(handler)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        priority_counts = {}
        for event in self.event_queue:
            priority = event.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        return {
            "total_queued": len(self.event_queue),
            "by_priority": priority_counts,
            "processed_total": len(self.processed_events),
        }

    def get_processing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent processing history."""
        # Would return detailed stats about recent events
        return []

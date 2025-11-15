"""Immutable state management for consistency and auditability."""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import replace
import logging
import uuid

from ..core.models.state import (
    SystemState,
    EnvironmentState,
    LearningState,
)
from ..core.models.domain import (
    Order,
    Vehicle,
    Route,
    Customer,
    OperationalOutcome,
    OrderStatus,
)

logger = logging.getLogger(__name__)


class StateManager:
    """Manages immutable system state with event-driven transitions.

    Responsibilities:
    - Current state holder
    - State transition validation
    - State history/audit trail
    - Transaction-like semantics (atomic updates)
    """

    def __init__(self, initial_state: Optional[SystemState] = None):
        """Initialize state manager."""
        if initial_state is not None:
            self.current_state = initial_state
        else:
            # Create minimal default SystemState with current environment
            env = EnvironmentState(current_time=datetime.now(), traffic_conditions={})
            self.current_state = SystemState(environment=env)
        self.state_history: List[Dict[str, Any]] = []
        self.state_transitions: List[Dict[str, Any]] = []
        self._event_handlers: Dict[str, List[Callable]] = {}

    def set_current_state(self, new_state: SystemState):
        """Set current state and apply configuration hooks (vehicles, time-windows).

        Use this instead of assigning `current_state` directly so configuration
        defaults from `model_config.yaml` and business overrides are applied.
        """
        # Apply vehicle and business configuration defaults
        try:
            from ..utils.config import apply_vehicle_configurations, load_model_config

            # Apply per-vehicle and per-type defaults
            apply_vehicle_configurations(new_state, load_model_config())
            # Apply business time windows if present in model config
            cfg = load_model_config()
            business = cfg.get("business", {}) if isinstance(cfg, dict) else {}
            default_windows = business.get("default_time_windows")
            if default_windows:
                # Merge into environment.time_windows without overwriting existing
                tw = dict(new_state.environment.time_windows)
                for k, v in default_windows.items():
                    if k not in tw:
                        tw[k] = v
                new_state.environment.time_windows = tw
        except Exception:
            logger.debug("Failed to apply configuration defaults to state")

        self.current_state = new_state

    def get_current_state(self) -> SystemState:
        """Get immutable current state."""
        return self.current_state

    def apply_event(self, event_type: str, event_data: Dict[str, Any]) -> SystemState:
        """Apply an event that transitions to new state.

        Args:
            event_type: Type of event (order_received, route_started, outcome_recorded, etc.)
            event_data: Event payload

        Returns:
            New state after event
        """

        old_state = self.current_state
        new_state = None

        # Dispatch to handler
        if event_type == "order_received":
            new_state = self._handle_order_received(old_state, event_data)

        elif event_type == "route_created":
            new_state = self._handle_route_created(old_state, event_data)

        elif event_type == "route_started":
            new_state = self._handle_route_started(old_state, event_data)

        elif event_type == "route_completed":
            new_state = self._handle_route_completed(old_state, event_data)

        elif event_type == "outcome_recorded":
            new_state = self._handle_outcome_recorded(old_state, event_data)

        elif event_type == "environment_updated":
            new_state = self._handle_environment_updated(old_state, event_data)

        elif event_type == "learning_updated":
            new_state = self._handle_learning_updated(old_state, event_data)

        else:
            logger.warning(f"Unknown event type: {event_type}")
            return old_state

        if new_state:
            # Record transition
            self._record_transition(event_type, event_data, old_state, new_state)

            # Update current state
            self.current_state = new_state

            # Fire event handlers
            self._fire_event_handlers(event_type, new_state)

            logger.info(
                f"State transition: {event_type} → {len(self.state_transitions)} transitions recorded"
            )

        return self.current_state

    def _handle_order_received(
        self, state: SystemState, event_data: Dict[str, Any]
    ) -> SystemState:
        """Handle new order received event."""
        order = event_data["order"]

        new_orders = state.pending_orders.copy()
        new_orders[order.order_id] = order

        return state.clone_with_updates(pending_orders=new_orders)

    def _handle_route_created(
        self, state: SystemState, event_data: Dict[str, Any]
    ) -> SystemState:
        """Handle route created event."""
        route = event_data["route"]

        new_routes = state.active_routes.copy()
        new_routes[route.route_id] = route

        # Update order statuses
        new_orders = state.pending_orders.copy()
        for order_id in route.order_ids:
            if order_id in new_orders:
                order = new_orders[order_id]
                order.status = OrderStatus.ASSIGNED
                order.assigned_route_id = route.route_id
                new_orders[order_id] = order

        return state.clone_with_updates(
            active_routes=new_routes, pending_orders=new_orders
        )

    def _handle_route_started(
        self, state: SystemState, event_data: Dict[str, Any]
    ) -> SystemState:
        """Handle route execution started event."""
        from backend.core.models.domain import RouteStatus

        route_id = event_data["route_id"]

        new_routes = state.active_routes.copy()
        if route_id in new_routes:
            old_route = new_routes[route_id]
            new_route = replace(
                old_route,
                status=RouteStatus.IN_PROGRESS,
                started_at=datetime.now()
            )
            new_routes[route_id] = new_route

        return state.clone_with_updates(active_routes=new_routes)

    def _handle_route_completed(
        self, state: SystemState, event_data: Dict[str, Any]
    ) -> SystemState:
        """Handle route completion event."""
        from backend.core.models.domain import RouteStatus

        route_id = event_data["route_id"]

        new_routes = state.active_routes.copy()
        completed_routes = state.completed_routes.copy()

        if route_id in new_routes:
            route = new_routes.pop(route_id)
            route = replace(
                route,
                status=RouteStatus.COMPLETED,
                completed_at=datetime.now()
            )
            completed_routes[route_id] = route

        return state.clone_with_updates(
            active_routes=new_routes, completed_routes=completed_routes
        )

    def _handle_outcome_recorded(
        self, state: SystemState, event_data: Dict[str, Any]
    ) -> SystemState:
        """Handle operational outcome recorded event."""
        outcome = event_data["outcome"]

        new_outcomes = list(state.recent_outcomes)
        new_outcomes.append(outcome)

        # Keep only recent outcomes (last 100)
        new_outcomes = new_outcomes[-100:]

        return state.clone_with_updates(recent_outcomes=new_outcomes)

    def _handle_environment_updated(
        self, state: SystemState, event_data: Dict[str, Any]
    ) -> SystemState:
        """Handle environment state update event."""
        new_env = (
            state.environment.clone_with_updates(**event_data)
            if hasattr(state.environment, "clone_with_updates")
            else replace(state.environment, **event_data)
        )

        return state.clone_with_updates(environment=new_env)

    def _handle_learning_updated(
        self, state: SystemState, event_data: Dict[str, Any]
    ) -> SystemState:
        """Handle learning state update event."""
        new_learning = replace(state.learning, **event_data)
        return state.clone_with_updates(learning=new_learning)

    def _record_transition(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        old_state: SystemState,
        new_state: SystemState,
    ):
        """Record state transition for audit trail."""
        transition = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "old_state_id": old_state.state_id,
            "new_state_id": new_state.state_id,
            "changes": self._compute_changes(old_state, new_state),
        }

        self.state_transitions.append(transition)

    def _compute_changes(
        self, old_state: SystemState, new_state: SystemState
    ) -> Dict[str, str]:
        """Compute what changed between states."""
        changes = {}

        if len(old_state.pending_orders) != len(new_state.pending_orders):
            changes["orders"] = (
                f"{len(old_state.pending_orders)} → {len(new_state.pending_orders)}"
            )

        if len(old_state.active_routes) != len(new_state.active_routes):
            changes["active_routes"] = (
                f"{len(old_state.active_routes)} → {len(new_state.active_routes)}"
            )

        if len(old_state.completed_routes) != len(new_state.completed_routes):
            changes["completed_routes"] = (
                f"{len(old_state.completed_routes)} → {len(new_state.completed_routes)}"
            )

        return changes

    def on_event(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def _fire_event_handlers(self, event_type: str, new_state: SystemState):
        """Fire registered event handlers."""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(new_state)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent state transitions."""
        return self.state_transitions[-limit:]

    def rollback_to_state(self, state_id: str) -> bool:
        """Attempt to rollback to previous state (not recommended in production)."""
        # In production: use database snapshots instead
        logger.warning(f"Rollback attempted to state {state_id}")
        return False

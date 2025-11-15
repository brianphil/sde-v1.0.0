"""Decision endpoints for Powell Sequential Decision Engine."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid
import logging

from backend.api.schemas import (
    DecisionRequest,
    DecisionResponse,
    DecisionCommitRequest,
    DecisionCommitResponse,
    RouteResponse,
    SystemStateResponse,
    LearningMetricsResponse,
)
from backend.core.models.decision import DecisionType
from backend.core.models.state import SystemState

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for decisions (replace with database in production)
decision_store: dict = {}


def get_app_state():
    """Get application state from main app."""
    from backend.api.main import app_state
    return app_state


@router.post("/decisions/make", response_model=DecisionResponse)
async def make_decision(
    request: DecisionRequest,
    app_state=Depends(get_app_state),
):
    """Make a routing decision using the Powell engine.

    This endpoint evaluates the current system state and returns a recommended
    decision based on the specified decision type and trigger reason.

    Args:
        request: Decision request with type, trigger reason, and context

    Returns:
        Decision with recommended action, routes, confidence, and reasoning
    """
    try:
        logger.info(f"Making decision: {request.decision_type.value}")

        # Get current state
        current_state = app_state.state_manager.get_current_state()

        if not current_state:
            raise HTTPException(
                status_code=400,
                detail="No system state available. Initialize the system first.",
            )

        # Map request decision type to domain model
        decision_type = DecisionType[request.decision_type.value.upper()]

        # Make decision
        start_time = datetime.now()
        decision = app_state.engine.make_decision(
            state=current_state,
            decision_type=decision_type,
            trigger_reason=request.trigger_reason,
        )
        computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Generate decision ID
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"

        # Store decision for later commit
        decision_store[decision_id] = {
            "decision": decision,
            "state": current_state,
            "timestamp": datetime.now(),
            "committed": False,
        }

        # Convert routes to response schema
        route_responses = []
        for route in decision.routes:
            route_responses.append(
                RouteResponse(
                    route_id=route.route_id,
                    vehicle_id=route.vehicle_id,
                    order_ids=route.order_ids,
                    stops=[],  # Simplified for now
                    destination_cities=route.destination_cities,
                    total_distance_km=route.total_distance_km,
                    estimated_duration_minutes=route.estimated_duration_minutes,
                    estimated_cost_kes=route.estimated_cost_kes,
                    status=route.status,
                    estimated_fuel_cost=route.estimated_fuel_cost,
                    estimated_time_cost=route.estimated_time_cost,
                    estimated_delay_penalty=route.estimated_delay_penalty,
                    created_at=route.created_at,
                )
            )

        # Build response
        policy_name = getattr(
            decision, "policy_name", getattr(decision, "hybrid_name", "unknown")
        )

        response = DecisionResponse(
            decision_id=decision_id,
            decision_type=request.decision_type,
            policy_name=policy_name,
            recommended_action=decision.recommended_action,
            confidence_score=decision.confidence_score,
            expected_value=decision.expected_value,
            routes=route_responses,
            reasoning=getattr(decision, "reasoning", ""),
            computation_time_ms=computation_time_ms,
            timestamp=datetime.now(),
            committed=False,
        )

        logger.info(
            f"Decision made: {decision_id} - {policy_name} - {decision.recommended_action.value}"
        )

        return response

    except Exception as e:
        logger.error(f"Error making decision: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to make decision: {str(e)}")


@router.post("/decisions/{decision_id}/commit", response_model=DecisionCommitResponse)
async def commit_decision(
    decision_id: str,
    app_state=Depends(get_app_state),
):
    """Commit a previously made decision to execute it.

    This applies the decision to the system state, creating routes and
    assigning orders as specified by the decision.

    Args:
        decision_id: ID of the decision to commit

    Returns:
        Commit result with created routes and assigned orders
    """
    try:
        logger.info(f"Committing decision: {decision_id}")

        # Retrieve decision
        if decision_id not in decision_store:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

        stored = decision_store[decision_id]

        if stored["committed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Decision {decision_id} already committed",
            )

        # Commit decision
        result = app_state.engine.commit_decision(
            stored["decision"],
            stored["state"],
        )

        # Mark as committed
        decision_store[decision_id]["committed"] = True

        # Extract route IDs from Route objects if needed
        routes_created = result["routes_created"]
        route_ids = []

        if routes_created:
            # Import route_store from routes module
            from backend.api.routes.routes import route_store

            if hasattr(routes_created[0], 'route_id'):
                # Result contains Route objects, extract IDs and store routes
                for route in routes_created:
                    route_ids.append(route.route_id)
                    route_store[route.route_id] = route

                    # Apply route_created event to state manager
                    app_state.state_manager.apply_event(
                        "route_created",
                        {"route": route}
                    )
            else:
                # Already string IDs
                route_ids = routes_created

        response = DecisionCommitResponse(
            success=not bool(result.get("errors")),
            action=result["action"],
            routes_created=route_ids,
            orders_assigned=result["orders_assigned"],
            errors=result.get("errors", []),
            message=f"Decision {decision_id} committed successfully"
            if not result.get("errors")
            else f"Decision committed with errors",
        )

        logger.info(
            f"Decision {decision_id} committed: {len(route_ids)} routes created"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error committing decision: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to commit decision: {str(e)}"
        )


@router.get("/decisions/{decision_id}", response_model=DecisionResponse)
async def get_decision(decision_id: str):
    """Get details of a specific decision.

    Args:
        decision_id: ID of the decision to retrieve

    Returns:
        Decision details
    """
    try:
        if decision_id not in decision_store:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

        stored = decision_store[decision_id]
        decision = stored["decision"]

        # Convert to response (simplified)
        policy_name = getattr(
            decision, "policy_name", getattr(decision, "hybrid_name", "unknown")
        )

        # Convert routes
        route_responses = []
        for route in decision.routes:
            route_responses.append(
                RouteResponse(
                    route_id=route.route_id,
                    vehicle_id=route.vehicle_id,
                    order_ids=route.order_ids,
                    stops=[],
                    destination_cities=route.destination_cities,
                    total_distance_km=route.total_distance_km,
                    estimated_duration_minutes=route.estimated_duration_minutes,
                    estimated_cost_kes=route.estimated_cost_kes,
                    status=route.status,
                    estimated_fuel_cost=route.estimated_fuel_cost,
                    estimated_time_cost=route.estimated_time_cost,
                    estimated_delay_penalty=route.estimated_delay_penalty,
                    created_at=route.created_at,
                )
            )

        return DecisionResponse(
            decision_id=decision_id,
            decision_type=DecisionType.DAILY_ROUTE_PLANNING,  # Simplified
            policy_name=policy_name,
            recommended_action=decision.recommended_action,
            confidence_score=decision.confidence_score,
            expected_value=decision.expected_value,
            routes=route_responses,
            reasoning=getattr(decision, "reasoning", ""),
            computation_time_ms=0.0,
            timestamp=stored["timestamp"],
            committed=stored["committed"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving decision: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve decision: {str(e)}"
        )


@router.get("/decisions", response_model=List[str])
async def list_decisions(
    limit: Optional[int] = 100,
    committed_only: Optional[bool] = None,
):
    """List all decision IDs.

    Args:
        limit: Maximum number of decisions to return
        committed_only: Filter to only committed decisions

    Returns:
        List of decision IDs
    """
    try:
        decision_ids = list(decision_store.keys())

        if committed_only is not None:
            decision_ids = [
                did for did in decision_ids
                if decision_store[did]["committed"] == committed_only
            ]

        # Sort by timestamp (most recent first)
        decision_ids.sort(
            key=lambda did: decision_store[did]["timestamp"],
            reverse=True,
        )

        return decision_ids[:limit]

    except Exception as e:
        logger.error(f"Error listing decisions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list decisions: {str(e)}"
        )


@router.get("/state", response_model=SystemStateResponse)
async def get_system_state(app_state=Depends(get_app_state)):
    """Get current system state summary.

    Returns:
        System state with orders, routes, vehicles, and environment
    """
    try:
        current_state = app_state.state_manager.get_current_state()

        if not current_state:
            raise HTTPException(
                status_code=400,
                detail="No system state available",
            )

        return SystemStateResponse(
            pending_orders_count=len(current_state.pending_orders),
            active_routes_count=len(current_state.active_routes),
            available_vehicles_count=len(current_state.get_available_vehicles()),
            total_pending_weight=current_state.get_total_pending_weight(),
            total_pending_volume=current_state.get_total_pending_volume(),
            current_time=current_state.environment.current_time,
            eastleigh_window_active=current_state.is_eastleigh_window_active(),
            traffic_conditions=current_state.environment.traffic_conditions,
            weather=current_state.environment.weather,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system state: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system state: {str(e)}"
        )


@router.get("/metrics/learning", response_model=LearningMetricsResponse)
async def get_learning_metrics(app_state=Depends(get_app_state)):
    """Get learning metrics from all policies.

    Returns:
        Metrics from CFA, VFA, PFA, and feedback processor
    """
    try:
        metrics = app_state.learning_coordinator.get_metrics()

        return LearningMetricsResponse(
            cfa_metrics=metrics.get("cfa_stats", {}),
            vfa_metrics=metrics.get("vfa_stats", {}),
            pfa_metrics=metrics.get("pfa_stats", {}),
            feedback_metrics=metrics.get("feedback_stats", {}),
            last_updated=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error getting learning metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get learning metrics: {str(e)}"
        )

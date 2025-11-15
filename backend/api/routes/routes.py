"""Route management endpoints for Powell Sequential Decision Engine."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid
import logging

from backend.api.schemas import (
    RouteResponse,
    RouteStatusEnum,
    OperationalOutcomeRequest,
    OperationalOutcomeResponse,
    RouteStopSchema,
)
from backend.core.models.domain import OperationalOutcome, RouteStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for routes (replace with database in production)
route_store: dict = {}


def get_app_state():
    """Get application state from main app."""
    from backend.api.main import app_state
    return app_state


@router.get("/routes", response_model=List[RouteResponse])
async def list_routes(
    status: Optional[RouteStatusEnum] = None,
    vehicle_id: Optional[str] = None,
    limit: Optional[int] = 100,
    app_state=Depends(get_app_state),
):
    """List all routes with optional filtering.

    Args:
        status: Filter by route status
        vehicle_id: Filter by vehicle ID
        limit: Maximum number of routes to return

    Returns:
        List of routes matching filters
    """
    try:
        # Get routes from state manager if available
        current_state = app_state.state_manager.get_current_state()

        if current_state:
            # Use routes from state
            routes = list(current_state.active_routes.values())
        else:
            # Fallback to in-memory store
            routes = list(route_store.values())

        # Apply filters
        if status:
            routes = [r for r in routes if r.status.value == status.value]

        if vehicle_id:
            routes = [r for r in routes if r.vehicle_id == vehicle_id]

        # Sort by created_at (most recent first)
        routes.sort(key=lambda r: r.created_at, reverse=True)

        # Limit results
        routes = routes[:limit]

        # Convert to response
        responses = []
        for route in routes:
            # Convert stops to schema
            stops_schema = []
            for stop in route.stops:
                stops_schema.append(
                    RouteStopSchema(
                        stop_id=stop.stop_id,
                        order_ids=stop.order_ids,
                        location=stop.location,
                        stop_type=stop.stop_type,
                        sequence_order=stop.sequence_order,
                        estimated_arrival=stop.estimated_arrival,
                        estimated_duration_minutes=stop.estimated_duration_minutes,
                        status=stop.status,
                        actual_arrival=stop.actual_arrival,
                        actual_duration_minutes=stop.actual_duration_minutes,
                    )
                )

            responses.append(
                RouteResponse(
                    route_id=route.route_id,
                    vehicle_id=route.vehicle_id,
                    order_ids=route.order_ids,
                    stops=stops_schema,
                    destination_cities=route.destination_cities,
                    total_distance_km=route.total_distance_km,
                    estimated_duration_minutes=route.estimated_duration_minutes,
                    estimated_cost_kes=route.estimated_cost_kes,
                    status=route.status,
                    estimated_fuel_cost=route.estimated_fuel_cost,
                    estimated_time_cost=route.estimated_time_cost,
                    estimated_delay_penalty=route.estimated_delay_penalty,
                    actual_distance_km=route.actual_distance_km,
                    actual_duration_minutes=route.actual_duration_minutes,
                    actual_cost_kes=route.actual_cost_kes,
                    actual_fuel_cost=route.actual_fuel_cost,
                    decision_id=route.decision_id,
                    created_at=route.created_at,
                    started_at=route.started_at,
                    completed_at=route.completed_at,
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Error listing routes: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list routes: {str(e)}"
        )


@router.get("/routes/{route_id}", response_model=RouteResponse)
async def get_route(
    route_id: str,
    app_state=Depends(get_app_state),
):
    """Get details of a specific route.

    Args:
        route_id: ID of the route to retrieve

    Returns:
        Route details with stops, orders, and performance metrics
    """
    try:
        # Try to get from state manager first
        current_state = app_state.state_manager.get_current_state()
        route = None

        if current_state and route_id in current_state.active_routes:
            route = current_state.active_routes[route_id]
        elif route_id in route_store:
            route = route_store[route_id]
        else:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        # Convert stops to schema
        stops_schema = []
        for stop in route.stops:
            stops_schema.append(
                RouteStopSchema(
                    stop_id=stop.stop_id,
                    order_ids=stop.order_ids,
                    location=stop.location,
                    stop_type=stop.stop_type,
                    sequence_order=stop.sequence_order,
                    estimated_arrival=stop.estimated_arrival,
                    estimated_duration_minutes=stop.estimated_duration_minutes,
                    status=stop.status,
                    actual_arrival=stop.actual_arrival,
                    actual_duration_minutes=stop.actual_duration_minutes,
                )
            )

        return RouteResponse(
            route_id=route.route_id,
            vehicle_id=route.vehicle_id,
            order_ids=route.order_ids,
            stops=stops_schema,
            destination_cities=route.destination_cities,
            total_distance_km=route.total_distance_km,
            estimated_duration_minutes=route.estimated_duration_minutes,
            estimated_cost_kes=route.estimated_cost_kes,
            status=route.status,
            estimated_fuel_cost=route.estimated_fuel_cost,
            estimated_time_cost=route.estimated_time_cost,
            estimated_delay_penalty=route.estimated_delay_penalty,
            actual_distance_km=route.actual_distance_km,
            actual_duration_minutes=route.actual_duration_minutes,
            actual_cost_kes=route.actual_cost_kes,
            actual_fuel_cost=route.actual_fuel_cost,
            decision_id=route.decision_id,
            created_at=route.created_at,
            started_at=route.started_at,
            completed_at=route.completed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving route: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve route: {str(e)}"
        )


@router.post("/routes/{route_id}/start")
async def start_route(
    route_id: str,
    app_state=Depends(get_app_state),
):
    """Mark a route as started (in progress).

    Args:
        route_id: ID of the route to start

    Returns:
        Success message
    """
    try:
        current_state = app_state.state_manager.get_current_state()

        if not current_state or route_id not in current_state.active_routes:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        route = current_state.active_routes[route_id]

        if route.status != RouteStatus.PLANNED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot start route in status: {route.status.value}",
            )

        # Apply route started event
        new_state = app_state.state_manager.apply_event(
            "route_started",
            {"route_id": route_id},
        )

        logger.info(f"Route {route_id} started successfully")

        return {
            "success": True,
            "message": f"Route {route_id} started",
            "route_id": route_id,
            "started_at": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting route: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start route: {str(e)}"
        )


@router.post("/routes/{route_id}/complete")
async def complete_route(
    route_id: str,
    app_state=Depends(get_app_state),
):
    """Mark a route as completed.

    Args:
        route_id: ID of the route to complete

    Returns:
        Success message
    """
    try:
        current_state = app_state.state_manager.get_current_state()

        if not current_state or route_id not in current_state.active_routes:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        route = current_state.active_routes[route_id]

        if route.status != RouteStatus.IN_PROGRESS:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot complete route in status: {route.status.value}",
            )

        # Apply route completed event
        new_state = app_state.state_manager.apply_event(
            "route_completed",
            {"route_id": route_id},
        )

        logger.info(f"Route {route_id} completed successfully")

        return {
            "success": True,
            "message": f"Route {route_id} completed",
            "route_id": route_id,
            "completed_at": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing route: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to complete route: {str(e)}"
        )


@router.post("/routes/{route_id}/outcome", response_model=OperationalOutcomeResponse)
async def record_outcome(
    route_id: str,
    request: OperationalOutcomeRequest,
    app_state=Depends(get_app_state),
):
    """Record operational outcome for a completed route.

    This triggers learning by feeding the outcome data to the Powell engine
    for parameter updates and model improvements.

    Args:
        route_id: ID of the route
        request: Operational outcome data with predictions vs actuals

    Returns:
        Outcome ID and learning signals generated
    """
    try:
        logger.info(f"Recording outcome for route: {route_id}")

        # Verify route exists (check both active and completed routes)
        current_state = app_state.state_manager.get_current_state()

        route = None
        if current_state:
            if route_id in current_state.active_routes:
                route = current_state.active_routes[route_id]
            elif route_id in current_state.completed_routes:
                route = current_state.completed_routes[route_id]

        if not route:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        # Generate outcome ID
        outcome_id = f"OUTCOME_{uuid.uuid4().hex[:12].upper()}"

        # Create operational outcome
        outcome = OperationalOutcome(
            outcome_id=outcome_id,
            route_id=request.route_id,
            vehicle_id=request.vehicle_id,
            predicted_fuel_cost=request.predicted_fuel_cost,
            actual_fuel_cost=request.actual_fuel_cost,
            predicted_duration_minutes=request.predicted_duration_minutes,
            actual_duration_minutes=request.actual_duration_minutes,
            predicted_distance_km=request.predicted_distance_km,
            actual_distance_km=request.actual_distance_km,
            on_time=request.on_time,
            delay_minutes=request.delay_minutes,
            successful_deliveries=request.successful_deliveries,
            failed_deliveries=request.failed_deliveries,
            traffic_conditions=request.traffic_conditions,
            weather=request.weather,
            day_of_week=request.day_of_week,
            customer_satisfaction_score=request.customer_satisfaction_score,
            notes=request.notes,
            recorded_at=datetime.now(),
        )

        # Process outcome through learning coordinator
        learning_signals = app_state.learning_coordinator.process_outcome(
            outcome,
            current_state,
        )

        # Record in state manager
        new_state = app_state.state_manager.apply_event(
            "outcome_recorded",
            {"outcome": outcome},
        )

        logger.info(
            f"Outcome {outcome_id} recorded and processed for learning"
        )

        return OperationalOutcomeResponse(
            outcome_id=outcome_id,
            route_id=route_id,
            learning_signals=learning_signals,
            message=f"Outcome recorded successfully. Learning signals generated for CFA, VFA, and PFA.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording outcome: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to record outcome: {str(e)}"
        )


@router.delete("/routes/{route_id}")
async def cancel_route(
    route_id: str,
    app_state=Depends(get_app_state),
):
    """Cancel a planned route.

    Only routes in PLANNED status can be cancelled. Routes in progress
    cannot be cancelled.

    Args:
        route_id: ID of the route to cancel

    Returns:
        Success message
    """
    try:
        current_state = app_state.state_manager.get_current_state()

        if not current_state or route_id not in current_state.active_routes:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        route = current_state.active_routes[route_id]

        if route.status != RouteStatus.PLANNED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel route in status: {route.status.value}. Only PLANNED routes can be cancelled.",
            )

        # Update route status to cancelled
        route.status = RouteStatus.CANCELLED

        logger.info(f"Route {route_id} cancelled successfully")

        return {
            "success": True,
            "message": f"Route {route_id} cancelled successfully",
            "route_id": route_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling route: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel route: {str(e)}"
        )

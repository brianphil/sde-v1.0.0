"""Route management endpoints for Powell Sequential Decision Engine."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import (
    RouteResponse,
    RouteStatusEnum,
    OperationalOutcomeRequest,
    OperationalOutcomeResponse,
    RouteStopSchema,
)
from backend.core.models.domain import (
    OperationalOutcome,
    RouteStatus,
    Route,
    RouteStop,
    Location,
)
from backend.db.database import get_db
from backend.db.models import RouteModel, RouteStopModel

logger = logging.getLogger(__name__)

router = APIRouter()


def get_app_state():
    """Get application state from main app."""
    from backend.api.main import app_state
    return app_state


@router.get("/routes", response_model=List[RouteResponse])
async def list_routes(
    status: Optional[RouteStatusEnum] = None,
    vehicle_id: Optional[str] = None,
    limit: Optional[int] = 100,
    db: AsyncSession = Depends(get_db),
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
        # Build query
        query = select(RouteModel)

        # Apply filters
        if status:
            # Map schema enum to domain enum
            domain_status = RouteStatus[status.value.upper()]
            query = query.where(RouteModel.status == domain_status)

        if vehicle_id:
            query = query.where(RouteModel.vehicle_id == vehicle_id)

        # Sort by created_at (most recent first)
        query = query.order_by(RouteModel.created_at.desc())

        # Limit results
        query = query.limit(limit)

        # Execute query
        result = await db.execute(query)
        route_models = result.scalars().all()

        # Convert to domain models and responses
        routes = []
        for route_model in route_models:
            # Load associated stops
            stops_result = await db.execute(
                select(RouteStopModel)
                .where(RouteStopModel.route_id == route_model.route_id)
                .order_by(RouteStopModel.sequence_order)
            )
            stop_models = stops_result.scalars().all()

            # Convert stops to domain models
            stops = [
                RouteStop(
                    stop_id=stop.stop_id,
                    order_ids=stop.order_ids,
                    location=Location(**stop.location),
                    stop_type=stop.stop_type,
                    sequence_order=stop.sequence_order,
                    estimated_arrival=stop.estimated_arrival,
                    estimated_duration_minutes=stop.estimated_duration_minutes,
                    status=stop.status,
                    actual_arrival=stop.actual_arrival,
                    actual_duration_minutes=stop.actual_duration_minutes,
                )
                for stop in stop_models
            ]

            # Convert route to domain model
            route = Route(
                route_id=route_model.route_id,
                vehicle_id=route_model.vehicle_id,
                order_ids=route_model.order_ids,
                stops=stops,
                destination_cities=route_model.destination_cities,
                total_distance_km=route_model.total_distance_km,
                estimated_duration_minutes=route_model.estimated_duration_minutes,
                estimated_cost_kes=route_model.estimated_cost_kes,
                status=route_model.status,
                estimated_fuel_cost=route_model.estimated_fuel_cost,
                estimated_time_cost=route_model.estimated_time_cost,
                estimated_delay_penalty=route_model.estimated_delay_penalty,
                actual_distance_km=route_model.actual_distance_km,
                actual_duration_minutes=route_model.actual_duration_minutes,
                actual_cost_kes=route_model.actual_cost_kes,
                actual_fuel_cost=route_model.actual_fuel_cost,
                decision_id=route_model.decision_id,
                created_at=route_model.created_at,
                started_at=route_model.started_at,
                completed_at=route_model.completed_at,
            )
            routes.append(route)

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


@router.get("/routes/active", response_model=List[RouteResponse])
async def list_active_routes(
    db: AsyncSession = Depends(get_db),
):
    """Get all active routes (in_progress status)."""
    from backend.api.schemas import RouteStatusEnum
    return await list_routes(status=RouteStatusEnum.IN_PROGRESS, db=db)


@router.get("/routes/completed", response_model=List[RouteResponse])
async def list_completed_routes(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Get all completed routes."""
    from backend.api.schemas import RouteStatusEnum
    return await list_routes(status=RouteStatusEnum.COMPLETED, limit=limit, db=db)


@router.get("/routes/{route_id}", response_model=RouteResponse)
async def get_route(
    route_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific route.

    Args:
        route_id: ID of the route to retrieve

    Returns:
        Route details with stops, orders, and performance metrics
    """
    try:
        # Load route from database
        result = await db.execute(
            select(RouteModel).where(RouteModel.route_id == route_id)
        )
        route_model = result.scalar_one_or_none()

        if not route_model:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        # Load associated stops
        stops_result = await db.execute(
            select(RouteStopModel)
            .where(RouteStopModel.route_id == route_id)
            .order_by(RouteStopModel.sequence_order)
        )
        stop_models = stops_result.scalars().all()

        # Convert stops to domain models
        stops = [
            RouteStop(
                stop_id=stop.stop_id,
                order_ids=stop.order_ids,
                location=Location(**stop.location),
                stop_type=stop.stop_type,
                sequence_order=stop.sequence_order,
                estimated_arrival=stop.estimated_arrival,
                estimated_duration_minutes=stop.estimated_duration_minutes,
                status=stop.status,
                actual_arrival=stop.actual_arrival,
                actual_duration_minutes=stop.actual_duration_minutes,
            )
            for stop in stop_models
        ]

        # Convert route to domain model
        route = Route(
            route_id=route_model.route_id,
            vehicle_id=route_model.vehicle_id,
            order_ids=route_model.order_ids,
            stops=stops,
            destination_cities=route_model.destination_cities,
            total_distance_km=route_model.total_distance_km,
            estimated_duration_minutes=route_model.estimated_duration_minutes,
            estimated_cost_kes=route_model.estimated_cost_kes,
            status=route_model.status,
            estimated_fuel_cost=route_model.estimated_fuel_cost,
            estimated_time_cost=route_model.estimated_time_cost,
            estimated_delay_penalty=route_model.estimated_delay_penalty,
            actual_distance_km=route_model.actual_distance_km,
            actual_duration_minutes=route_model.actual_duration_minutes,
            actual_cost_kes=route_model.actual_cost_kes,
            actual_fuel_cost=route_model.actual_fuel_cost,
            decision_id=route_model.decision_id,
            created_at=route_model.created_at,
            started_at=route_model.started_at,
            completed_at=route_model.completed_at,
        )

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
    db: AsyncSession = Depends(get_db),
    app_state=Depends(get_app_state),
):
    """Mark a route as started (in progress).

    Args:
        route_id: ID of the route to start

    Returns:
        Success message
    """
    try:
        # Load route from database
        result = await db.execute(
            select(RouteModel).where(RouteModel.route_id == route_id)
        )
        route_model = result.scalar_one_or_none()

        if not route_model:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        if route_model.status != RouteStatus.PLANNED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot start route in status: {route_model.status.value}",
            )

        # Update route status in database
        route_model.status = RouteStatus.IN_PROGRESS
        route_model.started_at = datetime.now()

        await db.commit()
        await db.refresh(route_model)

        # Also update state manager if available
        current_state = app_state.state_manager.get_current_state()
        if current_state and route_id in current_state.active_routes:
            new_state = app_state.state_manager.apply_event(
                "route_started",
                {"route_id": route_id},
            )

        logger.info(f"Route {route_id} started successfully")

        return {
            "success": True,
            "message": f"Route {route_id} started",
            "route_id": route_id,
            "started_at": route_model.started_at.isoformat(),
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
    db: AsyncSession = Depends(get_db),
    app_state=Depends(get_app_state),
):
    """Mark a route as completed.

    Args:
        route_id: ID of the route to complete

    Returns:
        Success message
    """
    try:
        # Load route from database
        result = await db.execute(
            select(RouteModel).where(RouteModel.route_id == route_id)
        )
        route_model = result.scalar_one_or_none()

        if not route_model:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        if route_model.status != RouteStatus.IN_PROGRESS:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot complete route in status: {route_model.status.value}",
            )

        # Update route status in database
        route_model.status = RouteStatus.COMPLETED
        route_model.completed_at = datetime.now()

        await db.commit()
        await db.refresh(route_model)

        # Also update state manager if available
        current_state = app_state.state_manager.get_current_state()
        if current_state and route_id in current_state.active_routes:
            new_state = app_state.state_manager.apply_event(
                "route_completed",
                {"route_id": route_id},
            )

        logger.info(f"Route {route_id} completed successfully")

        return {
            "success": True,
            "message": f"Route {route_id} completed",
            "route_id": route_id,
            "completed_at": route_model.completed_at.isoformat(),
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
    db: AsyncSession = Depends(get_db),
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
        # Load route from database
        result = await db.execute(
            select(RouteModel).where(RouteModel.route_id == route_id)
        )
        route_model = result.scalar_one_or_none()

        if not route_model:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        if route_model.status != RouteStatus.PLANNED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel route in status: {route_model.status.value}. Only PLANNED routes can be cancelled.",
            )

        # Update route status to cancelled in database
        route_model.status = RouteStatus.CANCELLED

        await db.commit()
        await db.refresh(route_model)

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

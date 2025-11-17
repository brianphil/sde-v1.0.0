"""Decision endpoints for Powell Sequential Decision Engine."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession

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
from backend.db.database import get_db
from backend.db.models import RouteModel, RouteStopModel, DecisionModel

logger = logging.getLogger(__name__)

router = APIRouter()


def get_app_state():
    """Get application state from main app."""
    from backend.api.main import app_state
    return app_state


@router.post("/decisions/make", response_model=DecisionResponse)
async def make_decision(
    request: DecisionRequest,
    db: AsyncSession = Depends(get_db),
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

        # Extract route IDs and order IDs from decision
        route_ids = [route.route_id for route in decision.routes]
        order_ids = []
        for route in decision.routes:
            order_ids.extend(route.order_ids)
        order_ids = list(set(order_ids))  # Remove duplicates

        # Get policy name
        policy_name = getattr(
            decision, "policy_name", getattr(decision, "hybrid_name", "unknown")
        )

        # Save decision to database
        decision_model = DecisionModel(
            decision_id=decision_id,
            decision_type=decision_type.value,
            policy_used=policy_name,
            state_snapshot=None,  # Could save simplified state if needed
            routes_created=route_ids,
            orders_routed=order_ids,
            total_cost_estimate=decision.expected_value,
            decision_confidence=decision.confidence_score,
            computation_time_ms=int(computation_time_ms),
            committed=False,
            executed=False,
            created_at=datetime.now(),
        )
        db.add(decision_model)
        await db.commit()
        await db.refresh(decision_model)

        logger.info(f"Decision {decision_id} saved to database")

        # Also store decision object temporarily for commit (routes need to be created)
        # We'll use a module-level cache with TTL
        if not hasattr(make_decision, '_decision_cache'):
            make_decision._decision_cache = {}
        make_decision._decision_cache[decision_id] = {
            "decision": decision,
            "state": current_state,
            "timestamp": datetime.now(),
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
            timestamp=decision_model.created_at,
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
    db: AsyncSession = Depends(get_db),
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

        # Load decision from database
        result_db = await db.execute(
            select(DecisionModel).where(DecisionModel.decision_id == decision_id)
        )
        decision_model = result_db.scalar_one_or_none()

        if not decision_model:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

        if decision_model.committed:
            raise HTTPException(
                status_code=400,
                detail=f"Decision {decision_id} already committed",
            )

        # Retrieve decision object from cache
        if not hasattr(make_decision, '_decision_cache') or decision_id not in make_decision._decision_cache:
            raise HTTPException(
                status_code=400,
                detail=f"Decision {decision_id} expired or not found in cache. Please make the decision again.",
            )

        stored = make_decision._decision_cache[decision_id]

        # Commit decision
        result = app_state.engine.commit_decision(
            stored["decision"],
            stored["state"],
        )

        # Update decision in database
        decision_model.committed = True
        decision_model.committed_at = datetime.now()
        await db.commit()
        await db.refresh(decision_model)

        logger.info(f"Decision {decision_id} marked as committed in database")

        # Remove from cache
        del make_decision._decision_cache[decision_id]

        # Extract route IDs from Route objects if needed
        routes_created = result["routes_created"]
        route_ids = []

        if routes_created:
            if hasattr(routes_created[0], 'route_id'):
                # Result contains Route objects, save to database
                for route in routes_created:
                    route_ids.append(route.route_id)

                    # Save route to database
                    route_model = RouteModel(
                        route_id=route.route_id,
                        vehicle_id=route.vehicle_id,
                        order_ids=route.order_ids,
                        destination_cities=[city.value if hasattr(city, 'value') else city for city in route.destination_cities],
                        total_distance_km=route.total_distance_km,
                        estimated_duration_minutes=route.estimated_duration_minutes,
                        estimated_cost_kes=route.estimated_cost_kes,
                        status=route.status,
                        estimated_fuel_cost=route.estimated_fuel_cost,
                        estimated_time_cost=route.estimated_time_cost,
                        estimated_delay_penalty=route.estimated_delay_penalty,
                        decision_id=decision_id,  # Link route to decision
                        created_at=route.created_at,
                    )
                    db.add(route_model)

                    # Save route stops to database
                    for stop in route.stops:
                        stop_model = RouteStopModel(
                            stop_id=stop.stop_id,
                            route_id=route.route_id,
                            order_ids=stop.order_ids,
                            location=stop.location.model_dump() if hasattr(stop.location, 'model_dump') else stop.location.__dict__,
                            stop_type=stop.stop_type,
                            sequence_order=stop.sequence_order,
                            estimated_arrival=stop.estimated_arrival,
                            estimated_duration_minutes=stop.estimated_duration_minutes,
                            status=stop.status,
                        )
                        db.add(stop_model)

                    # Apply route_created event to state manager
                    app_state.state_manager.apply_event(
                        "route_created",
                        {"route": route}
                    )

                # Commit all routes to database
                await db.commit()
                logger.info(f"Saved {len(route_ids)} routes to database")
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


@router.get("/decisions/{decision_id}")
async def get_decision(
    decision_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific decision from database.

    Args:
        decision_id: ID of the decision to retrieve

    Returns:
        Decision details from database
    """
    try:
        # Load decision from database
        result = await db.execute(
            select(DecisionModel).where(DecisionModel.decision_id == decision_id)
        )
        decision_model = result.scalar_one_or_none()

        if not decision_model:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

        # Load associated routes if committed
        routes_data = []
        if decision_model.committed and decision_model.routes_created:
            routes_result = await db.execute(
                select(RouteModel).where(
                    RouteModel.route_id.in_(decision_model.routes_created)
                )
            )
            route_models = routes_result.scalars().all()

            for route_model in route_models:
                routes_data.append({
                    "route_id": route_model.route_id,
                    "vehicle_id": route_model.vehicle_id,
                    "order_ids": route_model.order_ids,
                    "destination_cities": route_model.destination_cities,
                    "total_distance_km": route_model.total_distance_km,
                    "estimated_cost_kes": route_model.estimated_cost_kes,
                    "status": route_model.status.value,
                })

        return {
            "decision_id": decision_model.decision_id,
            "decision_type": decision_model.decision_type,
            "policy_used": decision_model.policy_used,
            "routes_created": decision_model.routes_created,
            "orders_routed": decision_model.orders_routed,
            "total_cost_estimate": decision_model.total_cost_estimate,
            "decision_confidence": decision_model.decision_confidence,
            "computation_time_ms": decision_model.computation_time_ms,
            "committed": decision_model.committed,
            "committed_at": decision_model.committed_at.isoformat() if decision_model.committed_at else None,
            "created_at": decision_model.created_at.isoformat(),
            "routes": routes_data,
        }

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
    db: AsyncSession = Depends(get_db),
):
    """List all decision IDs from database.

    Args:
        limit: Maximum number of decisions to return
        committed_only: Filter to only committed decisions

    Returns:
        List of decision IDs
    """
    try:
        # Build query
        query = select(DecisionModel)

        # Apply filter
        if committed_only is not None:
            query = query.where(DecisionModel.committed == committed_only)

        # Sort by created_at (most recent first)
        query = query.order_by(DecisionModel.created_at.desc())

        # Limit results
        query = query.limit(limit)

        # Execute query
        result = await db.execute(query)
        decision_models = result.scalars().all()

        # Extract decision IDs
        decision_ids = [decision_model.decision_id for decision_model in decision_models]

        return decision_ids

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

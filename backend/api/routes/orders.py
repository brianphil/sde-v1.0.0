"""Order management endpoints for Powell Sequential Decision Engine."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import (
    OrderCreateRequest,
    OrderResponse,
    OrderUpdateRequest,
    OrderStatusEnum,
)
from backend.core.models.domain import Order, OrderStatus, TimeWindow, Location
from backend.services.event_orchestrator import Event, EventPriority
from backend.db.database import get_db
from backend.db.models import OrderModel

logger = logging.getLogger(__name__)

router = APIRouter()


def get_app_state():
    """Get application state from main app."""
    from backend.api.main import app_state
    return app_state


@router.post("/orders", response_model=OrderResponse, status_code=201)
async def create_order(
    request: OrderCreateRequest,
    db: AsyncSession = Depends(get_db),
    app_state=Depends(get_app_state),
):
    """Create a new delivery order.

    This creates an order and optionally triggers a decision event
    if the order requires immediate attention.

    Args:
        request: Order creation request with customer, locations, and requirements

    Returns:
        Created order with assigned ID
    """
    try:
        # Generate order ID
        order_id = f"ORD_{uuid.uuid4().hex[:8].upper()}"

        logger.info(f"Creating order: {order_id} for customer {request.customer_id}")

        # Convert schema to domain model
        order = Order(
            order_id=order_id,
            customer_id=request.customer_id,
            customer_name=request.customer_name,
            pickup_location=request.pickup_location,
            destination_city=request.destination_city,
            destination_location=request.destination_location,
            weight_tonnes=request.weight_tonnes,
            volume_m3=request.volume_m3,
            time_window=TimeWindow(
                start_time=request.time_window.start_time,
                end_time=request.time_window.end_time,
            ),
            delivery_window=TimeWindow(
                start_time=request.delivery_window.start_time,
                end_time=request.delivery_window.end_time,
            ) if request.delivery_window else None,
            priority=request.priority,
            special_handling=request.special_handling,
            customer_constraints=request.customer_constraints,
            price_kes=request.price_kes,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Save order to database
        order_model = OrderModel(
            order_id=order.order_id,
            customer_id=order.customer_id,
            pickup_location=order.pickup_location.model_dump(),
            destination_city=order.destination_city,
            destination_location=order.destination_location.model_dump() if order.destination_location else None,
            weight_tonnes=order.weight_tonnes,
            volume_m3=order.volume_m3,
            time_window_start=order.time_window.start_time,
            time_window_end=order.time_window.end_time,
            delivery_window_start=order.delivery_window.start_time if order.delivery_window else None,
            delivery_window_end=order.delivery_window.end_time if order.delivery_window else None,
            priority=order.priority,
            special_handling=order.special_handling,
            customer_constraints=order.customer_constraints,
            price_kes=order.price_kes,
            status=order.status,
        )
        db.add(order_model)
        await db.commit()
        await db.refresh(order_model)

        logger.info(f"Order {order_id} saved to database")

        # Submit event to orchestrator if immediate processing needed
        if request.priority >= 1:  # High or urgent priority
            event = Event(
                "order_arrived",
                {"order": order},
                priority=EventPriority.HIGH if request.priority == 2 else EventPriority.NORMAL,
            )
            app_state.orchestrator.submit_event(event)
            logger.info(f"Order {order_id} submitted for immediate processing")

        # Add order to system state
        current_state = app_state.state_manager.get_current_state()
        if current_state:
            new_state = app_state.state_manager.apply_event(
                "order_received",
                {"order": order},
            )
            logger.info(f"Order {order_id} added to system state")

        # Convert to response
        response = OrderResponse(
            order_id=order.order_id,
            customer_id=order.customer_id,
            customer_name=order.customer_name,
            pickup_location=order.pickup_location,
            destination_city=order.destination_city,
            destination_location=order.destination_location,
            weight_tonnes=order.weight_tonnes,
            volume_m3=order.volume_m3,
            time_window=order.time_window,
            delivery_window=order.delivery_window,
            priority=order.priority,
            special_handling=order.special_handling,
            customer_constraints=order.customer_constraints,
            price_kes=order.price_kes,
            status=order.status,
            assigned_route_id=order.assigned_route_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )

        logger.info(f"Order {order_id} created successfully")
        return response

    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create order: {str(e)}")


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific order.

    Args:
        order_id: ID of the order to retrieve

    Returns:
        Order details
    """
    try:
        # Load order from database
        result = await db.execute(
            select(OrderModel).where(OrderModel.order_id == order_id)
        )
        order_model = result.scalar_one_or_none()

        if not order_model:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Convert ORM model to domain model
        order = Order(
            order_id=order_model.order_id,
            customer_id=order_model.customer_id,
            customer_name=order_model.customer_id,  # Could load from customer table
            pickup_location=Location(**order_model.pickup_location),
            destination_city=order_model.destination_city,
            destination_location=Location(**order_model.destination_location) if order_model.destination_location else None,
            weight_tonnes=order_model.weight_tonnes,
            volume_m3=order_model.volume_m3,
            time_window=TimeWindow(
                start_time=order_model.time_window_start,
                end_time=order_model.time_window_end,
            ),
            delivery_window=TimeWindow(
                start_time=order_model.delivery_window_start,
                end_time=order_model.delivery_window_end,
            ) if order_model.delivery_window_start else None,
            priority=order_model.priority,
            special_handling=order_model.special_handling,
            customer_constraints=order_model.customer_constraints,
            price_kes=order_model.price_kes,
            status=order_model.status,
            assigned_route_id=order_model.assigned_route_id,
            created_at=order_model.created_at,
            updated_at=order_model.updated_at,
        )

        return OrderResponse(
            order_id=order.order_id,
            customer_id=order.customer_id,
            customer_name=order.customer_name,
            pickup_location=order.pickup_location,
            destination_city=order.destination_city,
            destination_location=order.destination_location,
            weight_tonnes=order.weight_tonnes,
            volume_m3=order.volume_m3,
            time_window=order.time_window,
            delivery_window=order.delivery_window,
            priority=order.priority,
            special_handling=order.special_handling,
            customer_constraints=order.customer_constraints,
            price_kes=order.price_kes,
            status=order.status,
            assigned_route_id=order.assigned_route_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving order: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve order: {str(e)}"
        )


@router.get("/orders", response_model=List[OrderResponse])
async def list_orders(
    status: Optional[OrderStatusEnum] = None,
    customer_id: Optional[str] = None,
    limit: Optional[int] = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all orders with optional filtering.

    Args:
        status: Filter by order status
        customer_id: Filter by customer ID
        limit: Maximum number of orders to return

    Returns:
        List of orders matching filters
    """
    try:
        # Build query
        query = select(OrderModel)

        # Apply filters
        if status:
            # Map schema enum to domain enum
            domain_status = OrderStatus[status.value.upper()]
            query = query.where(OrderModel.status == domain_status)

        if customer_id:
            query = query.where(OrderModel.customer_id == customer_id)

        # Sort by created_at (most recent first)
        query = query.order_by(OrderModel.created_at.desc())

        # Limit results
        query = query.limit(limit)

        # Execute query
        result = await db.execute(query)
        order_models = result.scalars().all()

        # Convert to response
        responses = []
        for order_model in order_models:
            # Convert to domain model first
            order = Order(
                order_id=order_model.order_id,
                customer_id=order_model.customer_id,
                customer_name=order_model.customer_id,  # Could load from customer table
                pickup_location=Location(**order_model.pickup_location),
                destination_city=order_model.destination_city,
                destination_location=Location(**order_model.destination_location) if order_model.destination_location else None,
                weight_tonnes=order_model.weight_tonnes,
                volume_m3=order_model.volume_m3,
                time_window=TimeWindow(
                    start_time=order_model.time_window_start,
                    end_time=order_model.time_window_end,
                ),
                delivery_window=TimeWindow(
                    start_time=order_model.delivery_window_start,
                    end_time=order_model.delivery_window_end,
                ) if order_model.delivery_window_start else None,
                priority=order_model.priority,
                special_handling=order_model.special_handling,
                customer_constraints=order_model.customer_constraints,
                price_kes=order_model.price_kes,
                status=order_model.status,
                assigned_route_id=order_model.assigned_route_id,
                created_at=order_model.created_at,
                updated_at=order_model.updated_at,
            )

            responses.append(
                OrderResponse(
                    order_id=order.order_id,
                    customer_id=order.customer_id,
                    customer_name=order.customer_name,
                    pickup_location=order.pickup_location,
                    destination_city=order.destination_city,
                    destination_location=order.destination_location,
                    weight_tonnes=order.weight_tonnes,
                    volume_m3=order.volume_m3,
                    time_window=order.time_window,
                    delivery_window=order.delivery_window,
                    priority=order.priority,
                    special_handling=order.special_handling,
                    customer_constraints=order.customer_constraints,
                    price_kes=order.price_kes,
                    status=order.status,
                    assigned_route_id=order.assigned_route_id,
                    created_at=order.created_at,
                    updated_at=order.updated_at,
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Error listing orders: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list orders: {str(e)}"
        )


@router.put("/orders/{order_id}", response_model=OrderResponse)
async def update_order(
    order_id: str,
    request: OrderUpdateRequest,
    db: AsyncSession = Depends(get_db),
    app_state=Depends(get_app_state),
):
    """Update an existing order.

    Only certain fields can be updated (priority, special_handling, status, etc.).
    Cannot update order after it has been assigned to a route.

    Args:
        order_id: ID of the order to update
        request: Order update request with fields to change

    Returns:
        Updated order
    """
    try:
        # Load order from database
        result = await db.execute(
            select(OrderModel).where(OrderModel.order_id == order_id)
        )
        order_model = result.scalar_one_or_none()

        if not order_model:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Check if order can be updated
        if order_model.status in [OrderStatus.IN_TRANSIT, OrderStatus.DELIVERED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot update order in status: {order_model.status.value}",
            )

        # Update fields
        if request.priority is not None:
            order_model.priority = request.priority

        if request.special_handling is not None:
            order_model.special_handling = request.special_handling

        if request.customer_constraints is not None:
            order_model.customer_constraints = request.customer_constraints

        if request.status is not None:
            # Map schema enum to domain enum
            order_model.status = OrderStatus[request.status.value.upper()]

        order_model.updated_at = datetime.now()

        # Save to database
        await db.commit()
        await db.refresh(order_model)

        logger.info(f"Order {order_id} updated successfully")

        # Convert to domain model for response
        order = Order(
            order_id=order_model.order_id,
            customer_id=order_model.customer_id,
            customer_name=order_model.customer_id,
            pickup_location=Location(**order_model.pickup_location),
            destination_city=order_model.destination_city,
            destination_location=Location(**order_model.destination_location) if order_model.destination_location else None,
            weight_tonnes=order_model.weight_tonnes,
            volume_m3=order_model.volume_m3,
            time_window=TimeWindow(
                start_time=order_model.time_window_start,
                end_time=order_model.time_window_end,
            ),
            delivery_window=TimeWindow(
                start_time=order_model.delivery_window_start,
                end_time=order_model.delivery_window_end,
            ) if order_model.delivery_window_start else None,
            priority=order_model.priority,
            special_handling=order_model.special_handling,
            customer_constraints=order_model.customer_constraints,
            price_kes=order_model.price_kes,
            status=order_model.status,
            assigned_route_id=order_model.assigned_route_id,
            created_at=order_model.created_at,
            updated_at=order_model.updated_at,
        )

        return OrderResponse(
            order_id=order.order_id,
            customer_id=order.customer_id,
            customer_name=order.customer_name,
            pickup_location=order.pickup_location,
            destination_city=order.destination_city,
            destination_location=order.destination_location,
            weight_tonnes=order.weight_tonnes,
            volume_m3=order.volume_m3,
            time_window=order.time_window,
            delivery_window=order.delivery_window,
            priority=order.priority,
            special_handling=order.special_handling,
            customer_constraints=order.customer_constraints,
            price_kes=order.price_kes,
            status=order.status,
            assigned_route_id=order.assigned_route_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating order: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update order: {str(e)}"
        )


@router.delete("/orders/{order_id}")
async def delete_order(
    order_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete (cancel) an order.

    Only pending orders can be deleted. Orders already assigned to routes
    must be unassigned first.

    Args:
        order_id: ID of the order to delete

    Returns:
        Success message
    """
    try:
        # Load order from database
        result = await db.execute(
            select(OrderModel).where(OrderModel.order_id == order_id)
        )
        order_model = result.scalar_one_or_none()

        if not order_model:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Check if order can be deleted
        if order_model.assigned_route_id:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete order assigned to route: {order_model.assigned_route_id}",
            )

        if order_model.status not in [OrderStatus.PENDING, OrderStatus.CANCELLED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete order in status: {order_model.status.value}",
            )

        # Mark as cancelled instead of deleting
        order_model.status = OrderStatus.CANCELLED
        order_model.updated_at = datetime.now()

        await db.commit()

        logger.info(f"Order {order_id} cancelled successfully")

        return {
            "success": True,
            "message": f"Order {order_id} cancelled successfully",
            "order_id": order_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting order: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete order: {str(e)}"
        )

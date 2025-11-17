"""Vehicle management endpoints for Powell Sequential Decision Engine."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import (
    VehicleCreateRequest,
    VehicleUpdateRequest,
    VehicleResponse,
    VehicleStatusEnum,
)
from backend.core.models.domain import Vehicle, VehicleStatus, Location
from backend.db.database import get_db
from backend.db.models import VehicleModel

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/vehicles", response_model=VehicleResponse, status_code=201)
async def create_vehicle(
    request: VehicleCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new vehicle.

    Args:
        request: Vehicle creation request with type, capacity, and location

    Returns:
        Created vehicle with assigned ID
    """
    try:
        # Generate vehicle ID
        vehicle_id = f"VEH_{uuid.uuid4().hex[:8].upper()}"

        logger.info(f"Creating vehicle: {vehicle_id} of type {request.vehicle_type}")

        # Save vehicle to database
        vehicle_model = VehicleModel(
            vehicle_id=vehicle_id,
            vehicle_type=request.vehicle_type,
            capacity_weight_tonnes=request.capacity_weight_tonnes,
            capacity_volume_m3=request.capacity_volume_m3,
            current_location=request.current_location.model_dump(),
            available_at=datetime.now(),
            status=VehicleStatus.AVAILABLE,
            fuel_efficiency_km_per_liter=request.fuel_efficiency_km_per_liter,
            fuel_cost_per_km=request.fuel_cost_per_km,
            driver_cost_per_hour=request.driver_cost_per_hour,
            driver_id=request.driver_id,
            maintenance_due=request.maintenance_due,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        db.add(vehicle_model)
        await db.commit()
        await db.refresh(vehicle_model)

        logger.info(f"Vehicle {vehicle_id} created successfully")

        return VehicleResponse(
            vehicle_id=vehicle_model.vehicle_id,
            vehicle_type=vehicle_model.vehicle_type,
            capacity_weight_tonnes=vehicle_model.capacity_weight_tonnes,
            capacity_volume_m3=vehicle_model.capacity_volume_m3,
            current_location=Location(**vehicle_model.current_location),
            available_at=vehicle_model.available_at,
            status=vehicle_model.status,
            assigned_route_id=vehicle_model.assigned_route_id,
            fuel_efficiency_km_per_liter=vehicle_model.fuel_efficiency_km_per_liter,
            fuel_cost_per_km=vehicle_model.fuel_cost_per_km,
            driver_cost_per_hour=vehicle_model.driver_cost_per_hour,
            driver_id=vehicle_model.driver_id,
            maintenance_due=vehicle_model.maintenance_due,
            created_at=vehicle_model.created_at,
            updated_at=vehicle_model.updated_at,
        )

    except Exception as e:
        logger.error(f"Error creating vehicle: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create vehicle: {str(e)}")


@router.get("/vehicles/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(
    vehicle_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific vehicle.

    Args:
        vehicle_id: ID of the vehicle to retrieve

    Returns:
        Vehicle details
    """
    try:
        # Load vehicle from database
        result = await db.execute(
            select(VehicleModel).where(VehicleModel.vehicle_id == vehicle_id)
        )
        vehicle_model = result.scalar_one_or_none()

        if not vehicle_model:
            raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")

        return VehicleResponse(
            vehicle_id=vehicle_model.vehicle_id,
            vehicle_type=vehicle_model.vehicle_type,
            capacity_weight_tonnes=vehicle_model.capacity_weight_tonnes,
            capacity_volume_m3=vehicle_model.capacity_volume_m3,
            current_location=Location(**vehicle_model.current_location),
            available_at=vehicle_model.available_at,
            status=vehicle_model.status,
            assigned_route_id=vehicle_model.assigned_route_id,
            fuel_efficiency_km_per_liter=vehicle_model.fuel_efficiency_km_per_liter,
            fuel_cost_per_km=vehicle_model.fuel_cost_per_km,
            driver_cost_per_hour=vehicle_model.driver_cost_per_hour,
            driver_id=vehicle_model.driver_id,
            maintenance_due=vehicle_model.maintenance_due,
            created_at=vehicle_model.created_at,
            updated_at=vehicle_model.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving vehicle: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve vehicle: {str(e)}"
        )


@router.get("/vehicles", response_model=List[VehicleResponse])
async def list_vehicles(
    status: Optional[VehicleStatusEnum] = None,
    vehicle_type: Optional[str] = None,
    available_only: Optional[bool] = False,
    limit: Optional[int] = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all vehicles with optional filtering.

    Args:
        status: Filter by vehicle status
        vehicle_type: Filter by vehicle type
        available_only: Filter to only available vehicles
        limit: Maximum number of vehicles to return

    Returns:
        List of vehicles matching filters
    """
    try:
        # Build query
        query = select(VehicleModel)

        # Apply filters
        if status:
            # Map schema enum to domain enum
            domain_status = VehicleStatus[status.value.upper()]
            query = query.where(VehicleModel.status == domain_status)

        if vehicle_type:
            query = query.where(VehicleModel.vehicle_type == vehicle_type)

        if available_only:
            query = query.where(VehicleModel.status == VehicleStatus.AVAILABLE)
            query = query.where(VehicleModel.available_at <= datetime.now())

        # Sort by available_at (soonest first)
        query = query.order_by(VehicleModel.available_at.asc())

        # Limit results
        query = query.limit(limit)

        # Execute query
        result = await db.execute(query)
        vehicle_models = result.scalars().all()

        # Convert to response
        responses = []
        for vehicle_model in vehicle_models:
            responses.append(
                VehicleResponse(
                    vehicle_id=vehicle_model.vehicle_id,
                    vehicle_type=vehicle_model.vehicle_type,
                    capacity_weight_tonnes=vehicle_model.capacity_weight_tonnes,
                    capacity_volume_m3=vehicle_model.capacity_volume_m3,
                    current_location=Location(**vehicle_model.current_location),
                    available_at=vehicle_model.available_at,
                    status=vehicle_model.status,
                    assigned_route_id=vehicle_model.assigned_route_id,
                    fuel_efficiency_km_per_liter=vehicle_model.fuel_efficiency_km_per_liter,
                    fuel_cost_per_km=vehicle_model.fuel_cost_per_km,
                    driver_cost_per_hour=vehicle_model.driver_cost_per_hour,
                    driver_id=vehicle_model.driver_id,
                    maintenance_due=vehicle_model.maintenance_due,
                    created_at=vehicle_model.created_at,
                    updated_at=vehicle_model.updated_at,
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Error listing vehicles: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list vehicles: {str(e)}"
        )


@router.put("/vehicles/{vehicle_id}", response_model=VehicleResponse)
async def update_vehicle(
    vehicle_id: str,
    request: VehicleUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update an existing vehicle.

    Args:
        vehicle_id: ID of the vehicle to update
        request: Vehicle update request with fields to change

    Returns:
        Updated vehicle
    """
    try:
        # Load vehicle from database
        result = await db.execute(
            select(VehicleModel).where(VehicleModel.vehicle_id == vehicle_id)
        )
        vehicle_model = result.scalar_one_or_none()

        if not vehicle_model:
            raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")

        # Update fields
        if request.current_location is not None:
            vehicle_model.current_location = request.current_location.model_dump()

        if request.status is not None:
            # Map schema enum to domain enum
            vehicle_model.status = VehicleStatus[request.status.value.upper()]

            # If setting to AVAILABLE, update available_at
            if vehicle_model.status == VehicleStatus.AVAILABLE:
                vehicle_model.available_at = datetime.now()

        if request.fuel_cost_per_km is not None:
            vehicle_model.fuel_cost_per_km = request.fuel_cost_per_km

        if request.driver_cost_per_hour is not None:
            vehicle_model.driver_cost_per_hour = request.driver_cost_per_hour

        if request.driver_id is not None:
            vehicle_model.driver_id = request.driver_id

        if request.maintenance_due is not None:
            vehicle_model.maintenance_due = request.maintenance_due

        vehicle_model.updated_at = datetime.now()

        # Save to database
        await db.commit()
        await db.refresh(vehicle_model)

        logger.info(f"Vehicle {vehicle_id} updated successfully")

        return VehicleResponse(
            vehicle_id=vehicle_model.vehicle_id,
            vehicle_type=vehicle_model.vehicle_type,
            capacity_weight_tonnes=vehicle_model.capacity_weight_tonnes,
            capacity_volume_m3=vehicle_model.capacity_volume_m3,
            current_location=Location(**vehicle_model.current_location),
            available_at=vehicle_model.available_at,
            status=vehicle_model.status,
            assigned_route_id=vehicle_model.assigned_route_id,
            fuel_efficiency_km_per_liter=vehicle_model.fuel_efficiency_km_per_liter,
            fuel_cost_per_km=vehicle_model.fuel_cost_per_km,
            driver_cost_per_hour=vehicle_model.driver_cost_per_hour,
            driver_id=vehicle_model.driver_id,
            maintenance_due=vehicle_model.maintenance_due,
            created_at=vehicle_model.created_at,
            updated_at=vehicle_model.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating vehicle: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update vehicle: {str(e)}"
        )


@router.delete("/vehicles/{vehicle_id}")
async def delete_vehicle(
    vehicle_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete (deactivate) a vehicle.

    Vehicles are marked as OUT_OF_SERVICE instead of being physically deleted.

    Args:
        vehicle_id: ID of the vehicle to delete

    Returns:
        Success message
    """
    try:
        # Load vehicle from database
        result = await db.execute(
            select(VehicleModel).where(VehicleModel.vehicle_id == vehicle_id)
        )
        vehicle_model = result.scalar_one_or_none()

        if not vehicle_model:
            raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")

        # Check if vehicle is assigned to a route
        if vehicle_model.assigned_route_id:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete vehicle assigned to route: {vehicle_model.assigned_route_id}",
            )

        # Mark as out of service instead of deleting
        vehicle_model.status = VehicleStatus.OUT_OF_SERVICE
        vehicle_model.updated_at = datetime.now()

        await db.commit()

        logger.info(f"Vehicle {vehicle_id} marked as out of service")

        return {
            "success": True,
            "message": f"Vehicle {vehicle_id} marked as out of service",
            "vehicle_id": vehicle_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vehicle: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete vehicle: {str(e)}"
        )

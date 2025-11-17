"""Customer management endpoints for Powell Sequential Decision Engine."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import (
    CustomerCreateRequest,
    CustomerUpdateRequest,
    CustomerResponse,
)
from backend.db.database import get_db
from backend.db.models import CustomerModel

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/customers", response_model=CustomerResponse, status_code=201)
async def create_customer(
    request: CustomerCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new customer.

    Args:
        request: Customer creation request with name, contact info, and preferences

    Returns:
        Created customer with assigned ID
    """
    try:
        # Generate customer ID
        customer_id = f"CUST_{uuid.uuid4().hex[:8].upper()}"

        logger.info(f"Creating customer: {customer_id} - {request.customer_name}")

        # Save customer to database
        customer_model = CustomerModel(
            customer_id=customer_id,
            customer_name=request.customer_name,
            email=request.email,
            phone=request.phone,
            address=request.address,
            constraints=request.constraints,
            preferences=request.preferences,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        db.add(customer_model)
        await db.commit()
        await db.refresh(customer_model)

        logger.info(f"Customer {customer_id} created successfully")

        return CustomerResponse(
            customer_id=customer_model.customer_id,
            customer_name=customer_model.customer_name,
            email=customer_model.email,
            phone=customer_model.phone,
            address=customer_model.address,
            constraints=customer_model.constraints,
            preferences=customer_model.preferences,
            created_at=customer_model.created_at,
            updated_at=customer_model.updated_at,
        )

    except Exception as e:
        logger.error(f"Error creating customer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create customer: {str(e)}")


@router.get("/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific customer.

    Args:
        customer_id: ID of the customer to retrieve

    Returns:
        Customer details
    """
    try:
        # Load customer from database
        result = await db.execute(
            select(CustomerModel).where(CustomerModel.customer_id == customer_id)
        )
        customer_model = result.scalar_one_or_none()

        if not customer_model:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        return CustomerResponse(
            customer_id=customer_model.customer_id,
            customer_name=customer_model.customer_name,
            email=customer_model.email,
            phone=customer_model.phone,
            address=customer_model.address,
            constraints=customer_model.constraints,
            preferences=customer_model.preferences,
            created_at=customer_model.created_at,
            updated_at=customer_model.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving customer: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve customer: {str(e)}"
        )


@router.get("/customers", response_model=List[CustomerResponse])
async def list_customers(
    name_search: Optional[str] = None,
    email_search: Optional[str] = None,
    limit: Optional[int] = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all customers with optional filtering.

    Args:
        name_search: Filter by customer name (partial match)
        email_search: Filter by email (partial match)
        limit: Maximum number of customers to return

    Returns:
        List of customers matching filters
    """
    try:
        # Build query
        query = select(CustomerModel)

        # Apply filters
        if name_search:
            query = query.where(CustomerModel.customer_name.ilike(f"%{name_search}%"))

        if email_search:
            query = query.where(CustomerModel.email.ilike(f"%{email_search}%"))

        # Sort by created_at (newest first)
        query = query.order_by(CustomerModel.created_at.desc())

        # Limit results
        query = query.limit(limit)

        # Execute query
        result = await db.execute(query)
        customer_models = result.scalars().all()

        # Convert to response
        responses = []
        for customer_model in customer_models:
            responses.append(
                CustomerResponse(
                    customer_id=customer_model.customer_id,
                    customer_name=customer_model.customer_name,
                    email=customer_model.email,
                    phone=customer_model.phone,
                    address=customer_model.address,
                    constraints=customer_model.constraints,
                    preferences=customer_model.preferences,
                    created_at=customer_model.created_at,
                    updated_at=customer_model.updated_at,
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Error listing customers: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list customers: {str(e)}"
        )


@router.put("/customers/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: str,
    request: CustomerUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update an existing customer.

    Args:
        customer_id: ID of the customer to update
        request: Customer update request with fields to change

    Returns:
        Updated customer
    """
    try:
        # Load customer from database
        result = await db.execute(
            select(CustomerModel).where(CustomerModel.customer_id == customer_id)
        )
        customer_model = result.scalar_one_or_none()

        if not customer_model:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        # Update fields
        if request.customer_name is not None:
            customer_model.customer_name = request.customer_name

        if request.email is not None:
            customer_model.email = request.email

        if request.phone is not None:
            customer_model.phone = request.phone

        if request.address is not None:
            customer_model.address = request.address

        if request.constraints is not None:
            customer_model.constraints = request.constraints

        if request.preferences is not None:
            customer_model.preferences = request.preferences

        customer_model.updated_at = datetime.now()

        # Save to database
        await db.commit()
        await db.refresh(customer_model)

        logger.info(f"Customer {customer_id} updated successfully")

        return CustomerResponse(
            customer_id=customer_model.customer_id,
            customer_name=customer_model.customer_name,
            email=customer_model.email,
            phone=customer_model.phone,
            address=customer_model.address,
            constraints=customer_model.constraints,
            preferences=customer_model.preferences,
            created_at=customer_model.created_at,
            updated_at=customer_model.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating customer: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update customer: {str(e)}"
        )


@router.delete("/customers/{customer_id}")
async def delete_customer(
    customer_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a customer.

    Note: Customers with existing orders cannot be deleted due to foreign key constraints.
    Consider marking them as inactive in preferences instead.

    Args:
        customer_id: ID of the customer to delete

    Returns:
        Success message
    """
    try:
        # Load customer from database
        result = await db.execute(
            select(CustomerModel).where(CustomerModel.customer_id == customer_id)
        )
        customer_model = result.scalar_one_or_none()

        if not customer_model:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        # Check if customer has orders (relationship will be loaded via selectin)
        if customer_model.orders and len(customer_model.orders) > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete customer with existing orders. Customer has {len(customer_model.orders)} order(s).",
            )

        # Delete customer
        await db.delete(customer_model)
        await db.commit()

        logger.info(f"Customer {customer_id} deleted successfully")

        return {
            "success": True,
            "message": f"Customer {customer_id} deleted successfully",
            "customer_id": customer_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting customer: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete customer: {str(e)}"
        )

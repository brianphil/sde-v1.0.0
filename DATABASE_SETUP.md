# Database Setup Guide

## Overview

The Powell Sequential Decision Engine uses **SQLite** for development with async SQLAlchemy 2.0 and Alembic for migrations. This guide covers database setup, migrations, and common operations.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database and create tables
python manage_db.py init

# 3. Seed with initial data (admin user + sample data)
python manage_db.py seed

# 4. Verify setup
python manage_db.py check
```

## Database Configuration

### Environment Variables

Create or update `.env` file:

```env
# Database
DATABASE_URL=sqlite+aiosqlite:///./senga.db
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=0

# JWT Authentication
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Database Location

- **Development**: `./senga.db` (SQLite file in project root)
- **Production**: Update `DATABASE_URL` to PostgreSQL connection string

## Database Schema

### Core Tables

1. **users** - User authentication and authorization
2. **customers** - Customer profiles and preferences
3. **orders** - Order details with status tracking
4. **vehicles** - Fleet management
5. **routes** - Planned and active routes
6. **route_stops** - Individual stops on routes
7. **decisions** - Decision audit trail
8. **operational_outcomes** - Learning feedback from completed routes

## Using the Database CLI

### Available Commands

```bash
# Initialize database (creates tables)
python manage_db.py init

# Seed with sample data
python manage_db.py seed

# Check database health
python manage_db.py check

# Reset database (WARNING: deletes all data)
python manage_db.py reset
```

### Seed Data

The `seed` command creates:

**Admin User:**
- Username: `admin`
- Password: `admin123`
- Email: `admin@senga.com`
- Role: `admin` (superuser)

**Test User:**
- Username: `testuser`
- Password: `test123`
- Email: `test@senga.com`
- Role: `user`

**Sample Customers:**
- `CUST_MAJID` - Majid Retailers (Nairobi, Nakuru, Eldoret)
- `CUST_TOPTIER` - Top Tier Logistics (Nairobi, Kisumu)
- `CUST_EXPRESS` - Express Deliveries (Nairobi, Mombasa)

**Sample Vehicles:**
- `VEH_001` - 5T truck (Nairobi Depot)
- `VEH_002` - 5T truck (Nairobi Depot)
- `VEH_003` - 10T truck (Nairobi Depot)

## Database Migrations with Alembic

### Create a New Migration

After modifying models in `backend/db/models.py`:

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description of changes"

# Review the generated migration in alembic/versions/
# Edit if necessary, then apply:
alembic upgrade head
```

### Migration Commands

```bash
# Show current migration version
alembic current

# Show migration history
alembic history

# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Downgrade to specific version
alembic downgrade <revision_id>

# Downgrade all
alembic downgrade base
```

### Creating Manual Migrations

```bash
# Create empty migration file
alembic revision -m "add custom index"

# Edit the generated file in alembic/versions/
# Implement upgrade() and downgrade() functions
```

## Accessing the Database in Code

### In API Endpoints

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.database import get_db
from backend.db.models import OrderModel

@router.post("/orders")
async def create_order(
    order_data: OrderCreate,
    db: AsyncSession = Depends(get_db)
):
    # Create new order
    new_order = OrderModel(
        order_id=generate_order_id(),
        customer_id=order_data.customer_id,
        pickup_location=order_data.pickup_location.model_dump(),
        # ... other fields
    )

    db.add(new_order)
    await db.commit()
    await db.refresh(new_order)

    return new_order
```

### Direct Database Access

```python
from backend.db.database import get_session
from sqlalchemy import select

async def get_active_orders():
    async with get_session() as session:
        result = await session.execute(
            select(OrderModel)
            .where(OrderModel.status == OrderStatus.PENDING)
            .order_by(OrderModel.time_window_start)
        )
        orders = result.scalars().all()
        return orders
```

### Query Examples

```python
from sqlalchemy import select, and_, or_, func

# Simple query
result = await session.execute(
    select(VehicleModel).where(VehicleModel.status == VehicleStatus.AVAILABLE)
)
vehicles = result.scalars().all()

# Join query
result = await session.execute(
    select(OrderModel)
    .join(RouteModel)
    .where(RouteModel.status == RouteStatus.IN_PROGRESS)
)
orders = result.scalars().all()

# Aggregate query
result = await session.execute(
    select(func.count(OrderModel.id))
    .where(OrderModel.status == OrderStatus.COMPLETED)
)
count = result.scalar()

# Complex filter
result = await session.execute(
    select(OrderModel).where(
        and_(
            OrderModel.status == OrderStatus.PENDING,
            OrderModel.priority >= 1,
            OrderModel.weight_tonnes <= 5.0
        )
    )
)
```

## Authentication

### User Registration

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "newuser@example.com",
    "password": "SecurePass123",
    "full_name": "New User"
  }'
```

### User Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Using Authenticated Endpoints

```bash
# Get current user info
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# Access protected endpoint
curl http://localhost:8000/api/v1/orders \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Token Refresh

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```

## Database Models

### Converting Domain Models to ORM Models

When saving domain model instances (e.g., `Order`, `Vehicle`) to the database:

```python
from backend.core.models.domain import Order, Location, OrderStatus
from backend.db.models import OrderModel

# Domain model instance
order = Order(
    order_id="ORD_001",
    customer_id="CUST_001",
    pickup_location=Location(
        latitude=-1.2921,
        longitude=36.8219,
        address="Nairobi Depot"
    ),
    destination_city=DestinationCity.NAKURU,
    # ... other fields
)

# Convert to ORM model for database
order_model = OrderModel(
    order_id=order.order_id,
    customer_id=order.customer_id,
    pickup_location=order.pickup_location.model_dump(),  # Serialize to JSON
    destination_city=order.destination_city,
    weight_tonnes=order.weight_tonnes,
    volume_m3=order.volume_m3,
    time_window_start=order.time_window_start,
    time_window_end=order.time_window_end,
    priority=order.priority,
    special_handling=order.special_handling,
    price_kes=order.price_kes,
    status=order.status,
)

db.add(order_model)
await db.commit()
```

### Converting ORM Models to Domain Models

When loading from database to use in business logic:

```python
# Query from database
result = await session.execute(
    select(OrderModel).where(OrderModel.order_id == "ORD_001")
)
order_model = result.scalar_one()

# Convert to domain model
order = Order(
    order_id=order_model.order_id,
    customer_id=order_model.customer_id,
    pickup_location=Location(**order_model.pickup_location),  # Deserialize from JSON
    destination_city=order_model.destination_city,
    weight_tonnes=order_model.weight_tonnes,
    volume_m3=order_model.volume_m3,
    time_window_start=order_model.time_window_start,
    time_window_end=order_model.time_window_end,
    priority=order_model.priority,
    special_handling=order_model.special_handling,
    price_kes=order_model.price_kes,
    status=order_model.status,
)
```

## JSON Storage in SQLite

SQLite stores complex data (locations, arrays) as JSON:

```python
# Storing Location objects
pickup_location = Column(JSON, nullable=False)
# Stored as: {"latitude": -1.2921, "longitude": 36.8219, "address": "Nairobi"}

# Storing arrays
order_ids = Column(JSON, nullable=False)
# Stored as: ["ORD_001", "ORD_002", "ORD_003"]

# Storing complex objects
state_snapshot = Column(JSON, nullable=True)
# Stored as: {"orders": [...], "vehicles": [...], "timestamp": "..."}
```

## Testing Database Operations

```bash
# Run tests with database
pytest tests/test_database.py

# Run tests with coverage
pytest tests/test_database.py --cov=backend.db

# Run all API tests (includes database)
pytest tests/test_api.py -v
```

## Common Issues & Solutions

### Issue: "No module named 'aiosqlite'"

```bash
pip install aiosqlite
```

### Issue: "Table already exists"

```bash
# Reset database
python manage_db.py reset

# Or manually delete
rm senga.db
python manage_db.py init
```

### Issue: Alembic can't find models

Ensure all models are imported in `alembic/env.py`:

```python
from backend.db.database import Base
from backend.db.models import *  # Imports all models
```

### Issue: Migration conflicts

```bash
# Check current state
alembic current
alembic history

# If needed, stamp to specific version
alembic stamp head

# Or reset migrations
rm -rf alembic/versions/*.py
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

## Production Considerations

### Switching to PostgreSQL

1. Update `.env`:
```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/senga_db
```

2. Update models (if using PostgreSQL-specific features):
```python
from sqlalchemy.dialects.postgresql import JSONB

# Change JSON to JSONB for better performance
pickup_location = Column(JSONB, nullable=False)
```

3. Install PostgreSQL driver:
```bash
pip install asyncpg
```

4. Run migrations:
```bash
alembic upgrade head
```

### Database Backups

```bash
# SQLite backup
cp senga.db senga_backup_$(date +%Y%m%d).db

# PostgreSQL backup
pg_dump -h localhost -U user -d senga_db > backup.sql
```

### Connection Pooling

Configure in `.env`:
```env
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
```

## API Documentation

Once the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation including authentication endpoints.

## Health Check

```bash
# Check if database is connected
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "components": {
    "engine": "healthy",
    "state_manager": "healthy",
    "orchestrator": "healthy",
    "learning": "healthy",
    "database": "healthy"
  }
}
```

## Next Steps

1. Run database initialization: `python manage_db.py init`
2. Seed with sample data: `python manage_db.py seed`
3. Start the API server: `python -m backend.api.main`
4. Test authentication: Login with `admin/admin123`
5. Explore API docs at http://localhost:8000/docs

## Support

For database-related issues:
1. Check this guide
2. Review `backend/db/database.py` and `backend/db/models.py`
3. Check Alembic documentation: https://alembic.sqlalchemy.org/
4. Check SQLAlchemy 2.0 docs: https://docs.sqlalchemy.org/

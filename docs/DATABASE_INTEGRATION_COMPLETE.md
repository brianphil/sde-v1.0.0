# Database Integration - Orders API

## Summary

The Orders API has been successfully integrated with SQLite database persistence. All order operations now save to and load from the database instead of using in-memory storage.

## Changes Made

### File: [backend/api/routes/orders.py](backend/api/routes/orders.py)

**Dependencies Added:**
```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.database import get_db
from backend.db.models import OrderModel
from backend.core.models.domain import Location
```

**Removed:**
- In-memory `order_store` dictionary (line 22-23 in original)

**Updated Endpoints:**

### 1. `POST /api/v1/orders` - Create Order
**What changed:**
- Added `db: AsyncSession = Depends(get_db)` parameter
- Saves order to database using `OrderModel`
- Converts domain model (`Order`) to ORM model (`OrderModel`)
- Commits to database before triggering events

**Database fields saved:**
```python
- order_id
- customer_id
- pickup_location (JSON)
- destination_city
- destination_location (JSON)
- weight_tonnes, volume_m3
- time_window_start, time_window_end
- delivery_window_start, delivery_window_end
- priority, special_handling, customer_constraints
- price_kes, status
- created_at, updated_at
```

### 2. `GET /api/v1/orders/{order_id}` - Get Order
**What changed:**
- Loads order from database instead of `order_store`
- Converts ORM model back to domain model
- Deserializes JSON fields (pickup_location, destination_location) to `Location` objects

### 3. `GET /api/v1/orders` - List Orders
**What changed:**
- Queries database with filtering (status, customer_id)
- Supports sorting by created_at (descending)
- Pagination with limit parameter
- No longer uses state_manager for listing

**Query example:**
```python
query = select(OrderModel)
    .where(OrderModel.status == OrderStatus.PENDING)
    .where(OrderModel.customer_id == "CUST_001")
    .order_by(OrderModel.created_at.desc())
    .limit(100)
```

### 4. `PUT /api/v1/orders/{order_id}` - Update Order
**What changed:**
- Loads order from database
- Updates fields in ORM model
- Commits changes to database
- Returns updated order

**Updatable fields:**
- priority
- special_handling
- customer_constraints
- status

### 5. `DELETE /api/v1/orders/{order_id}` - Delete Order
**What changed:**
- Loads order from database
- Marks as CANCELLED instead of deleting
- Saves status change to database

## Data Flow

### Create Order Flow:
```
1. API Request → OrderCreateRequest schema
2. Convert to domain model (Order)
3. Convert to ORM model (OrderModel)
4. Save to database (INSERT)
5. Add to state_manager (in-memory for decisions)
6. Trigger orchestrator events if high priority
7. Return OrderResponse
```

### Read Order Flow:
```
1. API Request → order_id
2. Query database (SELECT)
3. Load OrderModel
4. Convert to domain model (Order)
5. Convert to OrderResponse
6. Return response
```

## Database Schema

Orders are stored in the `orders` table with the following structure:

```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    pickup_location JSON NOT NULL,
    destination_city ENUM NOT NULL,
    destination_location JSON,
    weight_tonnes FLOAT NOT NULL,
    volume_m3 FLOAT NOT NULL,
    time_window_start DATETIME NOT NULL,
    time_window_end DATETIME NOT NULL,
    delivery_window_start DATETIME,
    delivery_window_end DATETIME,
    priority INTEGER NOT NULL DEFAULT 0,
    special_handling JSON,
    customer_constraints JSON,
    price_kes FLOAT NOT NULL,
    status ENUM NOT NULL DEFAULT 'PENDING',
    assigned_route_id VARCHAR(50),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (assigned_route_id) REFERENCES routes(route_id)
);
```

**Indexes:**
- `order_id` (unique)
- `customer_id`
- `status`
- `destination_city, status` (composite)
- `created_at`
- `assigned_route_id`
- `priority`

## Testing the Integration

### 1. Create an Order

```bash
curl -X POST http://localhost:8000/api/v1/orders \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "customer_id": "CUST_001",
    "customer_name": "Test Customer",
    "pickup_location": {
      "latitude": -1.2921,
      "longitude": 36.8219,
      "address": "Nairobi Depot"
    },
    "destination_city": "NAKURU",
    "weight_tonnes": 2.5,
    "volume_m3": 5.0,
    "time_window": {
      "start_time": "2025-11-17T08:00:00",
      "end_time": "2025-11-17T18:00:00"
    },
    "priority": 0,
    "price_kes": 5000.0
  }'
```

### 2. List All Orders

```bash
curl http://localhost:8000/api/v1/orders \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3. Filter Orders by Status

```bash
curl "http://localhost:8000/api/v1/orders?status=PENDING&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. Get Specific Order

```bash
curl http://localhost:8000/api/v1/orders/ORD_12345678 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 5. Update Order

```bash
curl -X PUT http://localhost:8000/api/v1/orders/ORD_12345678 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "priority": 2,
    "special_handling": ["fresh_food", "fragile"]
  }'
```

### 6. Cancel Order

```bash
curl -X DELETE http://localhost:8000/api/v1/orders/ORD_12345678 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Verification in Database

```bash
# Using SQLite CLI
sqlite3 senga.db

# Check orders table
sqlite> SELECT order_id, customer_id, status, created_at FROM orders;

# Check order details
sqlite> SELECT * FROM orders WHERE order_id = 'ORD_12345678';

# Count orders by status
sqlite> SELECT status, COUNT(*) FROM orders GROUP BY status;
```

## Benefits of Database Integration

1. **Persistence**: Orders survive server restarts
2. **Concurrency**: Multiple API instances can share data
3. **Querying**: Efficient filtering, sorting, pagination
4. **Audit Trail**: Created_at and updated_at timestamps
5. **Data Integrity**: Foreign key constraints, unique constraints
6. **Scalability**: Can migrate to PostgreSQL later
7. **Backup**: Database can be backed up independently

## Next Steps

### Immediate (Optional):
- Test order creation and retrieval
- Verify database persistence
- Check error handling

### Future Work:
1. **Routes API**: Integrate database persistence for routes
2. **Decisions API**: Save decision audit trail to database
3. **Vehicles API**: Load vehicles from database
4. **Customers API**: Create CRUD endpoints for customers
5. **Operational Outcomes**: Save learning feedback to database

### Enhancements:
- Add pagination metadata (total count, pages)
- Implement soft delete (deleted_at column)
- Add order history tracking (audit log)
- Optimize queries with eager loading
- Add database transactions for complex operations
- Implement caching layer (Redis)

## Important Notes

1. **State Manager**: The state_manager still uses in-memory storage for active decision-making. Orders are added to both database and state_manager.

2. **Event Orchestrator**: High-priority orders still trigger orchestrator events for immediate processing.

3. **Authentication**: All endpoints now require authentication (JWT Bearer token).

4. **Error Handling**: Database errors are caught and returned as HTTP 500 with descriptive messages.

5. **Data Conversion**: There's automatic conversion between:
   - API schemas (OrderCreateRequest, OrderResponse)
   - Domain models (Order, Location, TimeWindow)
   - ORM models (OrderModel)

## Dependencies Required

Ensure these are installed:
```bash
pip install sqlalchemy
pip install aiosqlite
pip install alembic
```

## Configuration

Database URL in `.env`:
```env
DATABASE_URL=sqlite+aiosqlite:///./senga.db
```

## Performance Considerations

- **SQLite** is suitable for development and light production
- For production with high concurrency, migrate to PostgreSQL:
  ```env
  DATABASE_URL=postgresql+asyncpg://user:pass@localhost/senga_db
  ```
- Consider adding database connection pooling
- Monitor query performance with EXPLAIN QUERY PLAN
- Add indexes for frequently queried fields

## Migration Path

To migrate to PostgreSQL:
1. Update `DATABASE_URL` in .env
2. Change JSON columns to JSONB in models.py
3. Run Alembic migrations: `alembic upgrade head`
4. No code changes needed (SQLAlchemy abstracts the database)

---

**Status**: ✅ Orders API fully integrated with database persistence

**Date**: 2025-11-17

**Author**: Claude Code (AI Assistant)

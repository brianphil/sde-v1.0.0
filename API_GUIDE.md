# Powell Sequential Decision Engine - API Guide

## Overview

The Powell SDE API provides RESTful endpoints for making routing decisions, managing orders and routes, and receiving real-time updates via WebSocket.

**Base URL:** `http://localhost:8000`
**API Version:** v1
**API Prefix:** `/api/v1`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# From the senga-sde directory
python -m backend.api.main
```

Or using uvicorn directly:

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

## Authentication

Currently, the API does not require authentication. In production, implement JWT or API key authentication.

## API Endpoints

### Health & Status

#### GET `/`
Root endpoint - basic health check

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0",
  "components": {
    "engine": "initialized",
    "state_manager": "initialized",
    "orchestrator": "initialized",
    "learning": "initialized"
  }
}
```

#### GET `/health`
Detailed health check with component status

---

### Decision Management

#### POST `/api/v1/decisions/make`
Make a routing decision using the Powell engine

**Request Body:**
```json
{
  "decision_type": "daily_route_planning",
  "trigger_reason": "Daily morning optimization",
  "context": {}
}
```

**Decision Types:**
- `daily_route_planning` - Daily batch route optimization
- `order_arrival` - Real-time decision when order arrives
- `realtime_adjustment` - Adjust existing routes
- `backhaul_consolidation` - Consolidate return loads

**Response:**
```json
{
  "decision_id": "dec_abc123def456",
  "decision_type": "daily_route_planning",
  "policy_name": "CFA/VFA",
  "recommended_action": "create_route",
  "confidence_score": 0.85,
  "expected_value": 12500.0,
  "routes": [
    {
      "route_id": "ROUTE_001",
      "vehicle_id": "VEH_001",
      "order_ids": ["ORD_001", "ORD_002"],
      "destination_cities": ["Nakuru", "Eldoret"],
      "total_distance_km": 250.5,
      "estimated_duration_minutes": 320,
      "estimated_cost_kes": 8500.0,
      "status": "planned"
    }
  ],
  "reasoning": "Hybrid CFA/VFA: Optimized for cost efficiency...",
  "computation_time_ms": 145.2,
  "timestamp": "2024-01-15T10:30:00",
  "committed": false
}
```

#### POST `/api/v1/decisions/{decision_id}/commit`
Commit a decision to execute it

**Response:**
```json
{
  "success": true,
  "action": "create_route",
  "routes_created": ["ROUTE_001", "ROUTE_002"],
  "orders_assigned": ["ORD_001", "ORD_002", "ORD_003"],
  "errors": [],
  "message": "Decision dec_abc123def456 committed successfully"
}
```

#### GET `/api/v1/decisions/{decision_id}`
Get decision details

#### GET `/api/v1/decisions`
List all decisions

**Query Parameters:**
- `limit` (int, default=100) - Maximum results
- `committed_only` (bool) - Filter by commit status

#### GET `/api/v1/state`
Get current system state summary

**Response:**
```json
{
  "pending_orders_count": 5,
  "active_routes_count": 3,
  "available_vehicles_count": 2,
  "total_pending_weight": 12.5,
  "total_pending_volume": 25.0,
  "current_time": "2024-01-15T10:30:00",
  "eastleigh_window_active": true,
  "traffic_conditions": {
    "CBD": 0.5,
    "Eastleigh": 0.3
  },
  "weather": "clear"
}
```

#### GET `/api/v1/metrics/learning`
Get learning metrics from all policies

**Response:**
```json
{
  "cfa_metrics": {
    "samples_observed": 150,
    "prediction_accuracy_fuel": 0.89
  },
  "vfa_metrics": {
    "trained_samples": 200,
    "total_loss": 0.045
  },
  "pfa_metrics": {
    "rules_count": 15,
    "average_confidence": 0.82
  },
  "feedback_metrics": {
    "on_time_mean": 0.92,
    "success_rate_mean": 0.88
  },
  "last_updated": "2024-01-15T10:30:00"
}
```

---

### Order Management

#### POST `/api/v1/orders`
Create a new delivery order

**Request Body:**
```json
{
  "customer_id": "CUST_001",
  "customer_name": "Majid Retailers",
  "pickup_location": {
    "latitude": -1.2921,
    "longitude": 36.8219,
    "address": "Eastleigh, Nairobi"
  },
  "destination_city": "Nakuru",
  "destination_location": {
    "latitude": -0.3031,
    "longitude": 35.2684,
    "address": "Nakuru CBD"
  },
  "weight_tonnes": 2.5,
  "volume_m3": 4.0,
  "time_window": {
    "start_time": "2024-01-15T08:30:00",
    "end_time": "2024-01-15T09:45:00"
  },
  "priority": 1,
  "special_handling": ["fresh_food"],
  "price_kes": 2500.0
}
```

**Response:** (Status 201)
```json
{
  "order_id": "ORD_A1B2C3D4",
  "customer_id": "CUST_001",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00",
  ...
}
```

#### GET `/api/v1/orders/{order_id}`
Get order details

#### GET `/api/v1/orders`
List all orders

**Query Parameters:**
- `status` (enum) - Filter by status: pending, assigned, in_transit, delivered, cancelled, failed
- `customer_id` (string) - Filter by customer
- `limit` (int, default=100) - Maximum results

#### PUT `/api/v1/orders/{order_id}`
Update an existing order

**Request Body:**
```json
{
  "priority": 2,
  "special_handling": ["fresh_food", "fragile"],
  "status": "assigned"
}
```

#### DELETE `/api/v1/orders/{order_id}`
Cancel an order

**Response:**
```json
{
  "success": true,
  "message": "Order ORD_A1B2C3D4 cancelled successfully",
  "order_id": "ORD_A1B2C3D4"
}
```

---

### Route Management

#### GET `/api/v1/routes`
List all routes

**Query Parameters:**
- `status` (enum) - planned, in_progress, completed, cancelled
- `vehicle_id` (string) - Filter by vehicle
- `limit` (int, default=100) - Maximum results

**Response:**
```json
[
  {
    "route_id": "ROUTE_001",
    "vehicle_id": "VEH_001",
    "order_ids": ["ORD_001", "ORD_002"],
    "stops": [],
    "destination_cities": ["Nakuru"],
    "total_distance_km": 150.0,
    "estimated_duration_minutes": 180,
    "estimated_cost_kes": 5000.0,
    "status": "planned",
    "created_at": "2024-01-15T10:30:00"
  }
]
```

#### GET `/api/v1/routes/{route_id}`
Get route details

#### POST `/api/v1/routes/{route_id}/start`
Mark route as started

**Response:**
```json
{
  "success": true,
  "message": "Route ROUTE_001 started",
  "route_id": "ROUTE_001",
  "started_at": "2024-01-15T11:00:00"
}
```

#### POST `/api/v1/routes/{route_id}/complete`
Mark route as completed

#### POST `/api/v1/routes/{route_id}/outcome`
Record operational outcome (triggers learning)

**Request Body:**
```json
{
  "route_id": "ROUTE_001",
  "vehicle_id": "VEH_001",
  "predicted_fuel_cost": 1500.0,
  "actual_fuel_cost": 1450.0,
  "predicted_duration_minutes": 180,
  "actual_duration_minutes": 175,
  "predicted_distance_km": 150.0,
  "actual_distance_km": 148.0,
  "on_time": true,
  "delay_minutes": 0,
  "successful_deliveries": 2,
  "failed_deliveries": 0,
  "traffic_conditions": {
    "CBD": 0.3,
    "Nakuru": 0.2
  },
  "weather": "clear",
  "customer_satisfaction_score": 0.95,
  "notes": "Delivery completed successfully"
}
```

**Response:**
```json
{
  "outcome_id": "OUTCOME_XYZ123ABC456",
  "route_id": "ROUTE_001",
  "learning_signals": {
    "cfa_signals": {
      "fuel_error": -50.0,
      "time_error": -5
    },
    "vfa_signals": {...},
    "pfa_signals": {...}
  },
  "message": "Outcome recorded successfully. Learning signals generated for CFA, VFA, and PFA."
}
```

#### DELETE `/api/v1/routes/{route_id}`
Cancel a planned route

---

### WebSocket - Real-Time Updates

#### WS `/api/v1/ws`
WebSocket connection for real-time events

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws');

ws.onopen = () => {
  console.log('Connected to Powell Engine WebSocket');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};
```

**Subscribe to Events:**
```javascript
ws.send(JSON.stringify({
  action: 'subscribe',
  events: ['decision_made', 'route_created', 'outcome_recorded']
}));
```

**Available Events:**
- `decision_made` - New decision created
- `decision_committed` - Decision executed
- `order_created` - New order received
- `route_created` - New route created
- `route_started` - Route started
- `route_completed` - Route completed
- `outcome_recorded` - Outcome recorded
- `learning_updated` - Models updated

**Event Message Format:**
```json
{
  "type": "route_created",
  "data": {
    "route_id": "ROUTE_001",
    "vehicle_id": "VEH_001",
    "order_count": 2
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

**Client Actions:**
- `subscribe` - Subscribe to events
- `unsubscribe` - Unsubscribe from events
- `get_subscriptions` - Get current subscriptions
- `ping` - Ping server (responds with pong)

---

## Example Workflows

### Workflow 1: Daily Route Planning

```bash
# 1. Check system state
curl -X GET http://localhost:8000/api/v1/state

# 2. Make decision
curl -X POST http://localhost:8000/api/v1/decisions/make \
  -H "Content-Type: application/json" \
  -d '{
    "decision_type": "daily_route_planning",
    "trigger_reason": "Daily morning optimization"
  }'

# 3. Commit decision
curl -X POST http://localhost:8000/api/v1/decisions/dec_abc123/commit

# 4. Start route
curl -X POST http://localhost:8000/api/v1/routes/ROUTE_001/start

# 5. Complete route
curl -X POST http://localhost:8000/api/v1/routes/ROUTE_001/complete

# 6. Record outcome
curl -X POST http://localhost:8000/api/v1/routes/ROUTE_001/outcome \
  -H "Content-Type: application/json" \
  -d '{
    "route_id": "ROUTE_001",
    "vehicle_id": "VEH_001",
    "predicted_fuel_cost": 1500.0,
    "actual_fuel_cost": 1450.0,
    "predicted_duration_minutes": 180,
    "actual_duration_minutes": 175,
    "predicted_distance_km": 150.0,
    "actual_distance_km": 148.0,
    "on_time": true,
    "successful_deliveries": 2,
    "failed_deliveries": 0
  }'
```

### Workflow 2: Real-Time Order Processing

```bash
# 1. Create urgent order
curl -X POST http://localhost:8000/api/v1/orders \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_VIP",
    "customer_name": "VIP Customer",
    "pickup_location": {
      "latitude": -1.2921,
      "longitude": 36.8219,
      "address": "Nairobi CBD"
    },
    "destination_city": "Eldoret",
    "weight_tonnes": 1.5,
    "volume_m3": 3.0,
    "time_window": {
      "start_time": "2024-01-15T14:00:00",
      "end_time": "2024-01-15T16:00:00"
    },
    "priority": 2,
    "price_kes": 3500.0
  }'

# 2. System automatically triggers decision (order_arrival event)

# 3. Check learning metrics
curl -X GET http://localhost:8000/api/v1/metrics/learning
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- **200 OK** - Successful GET/PUT/DELETE
- **201 Created** - Successful POST
- **400 Bad Request** - Invalid input
- **404 Not Found** - Resource not found
- **500 Internal Server Error** - Server error

**Error Response Format:**
```json
{
  "error": "Validation Error",
  "detail": "weight_tonnes must be greater than 0",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## Rate Limiting

Currently no rate limiting. In production, implement rate limiting per client/API key.

---

## Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Make decision
curl -X POST http://localhost:8000/api/v1/decisions/make \
  -H "Content-Type: application/json" \
  -d '{"decision_type": "daily_route_planning", "trigger_reason": "test"}'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Make decision
response = requests.post(
    'http://localhost:8000/api/v1/decisions/make',
    json={
        'decision_type': 'daily_route_planning',
        'trigger_reason': 'Morning optimization'
    }
)
decision = response.json()
print(f"Decision ID: {decision['decision_id']}")

# Commit decision
response = requests.post(
    f"http://localhost:8000/api/v1/decisions/{decision['decision_id']}/commit"
)
print(response.json())
```

### Using WebSocket (Python)

```python
import websockets
import asyncio
import json

async def test_websocket():
    uri = "ws://localhost:8000/api/v1/ws"
    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Connected: {welcome}")

        # Subscribe to events
        await websocket.send(json.dumps({
            "action": "subscribe",
            "events": ["decision_made", "route_created"]
        }))

        # Listen for events
        async for message in websocket:
            data = json.loads(message)
            print(f"Event: {data['type']}")
            print(f"Data: {data['data']}")

asyncio.run(test_websocket())
```

---

## Next Steps

1. **Add Authentication** - Implement JWT or API key auth
2. **Add Database** - Replace in-memory stores with PostgreSQL
3. **Add Caching** - Use Redis for frequently accessed data
4. **Add Monitoring** - Integrate Prometheus/Grafana
5. **Deploy** - Containerize with Docker and deploy to cloud

---

## Support

For issues or questions, refer to the main project documentation or create an issue in the repository.

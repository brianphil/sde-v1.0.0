# Powell Engine - API Implementation Guide

## Overview

This guide specifies the REST API endpoints for the Powell Sequential Decision Engine, enabling external clients to request decisions, track routes, submit feedback, and monitor performance.

## Architecture Pattern

The API follows these principles:

- **Separation of Concerns**: API layer delegates to services
- **Event-Driven**: Decisions trigger events through EventOrchestrator
- **State Management**: All state immutability enforced by StateManager
- **Learning Integration**: Feedback flows to FeedbackProcessor automatically
- **WebSocket Support**: Real-time updates via subscription model

## API Endpoints

### 1. Decision Endpoints

#### Request Decision

```
POST /api/decisions
Content-Type: application/json

{
  "decision_type": "DAILY_ROUTE_PLANNING" | "ORDER_ARRIVAL" | "REAL_TIME_ADJUSTMENT" | "BACKHAUL_OPPORTUNITY",
  "context": {
    "order_ids": ["ORD_001", "ORD_002"],  # optional, for specific orders
    "vehicle_ids": ["VEH_001"],           # optional, for specific vehicles
    "urgency": "normal" | "high" | "critical",
    "metadata": {                         # optional
      "reason": "High traffic detected",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }
}

Response 200:
{
  "decision_id": "DEC_20240115_001",
  "timestamp": "2024-01-15T10:30:15Z",
  "policy_used": "CFAVFAHybrid",
  "confidence_score": 0.87,
  "expected_value": 15000,
  "routes": [
    {
      "route_id": "ROUTE_001",
      "vehicle_id": "VEH_001",
      "orders": ["ORD_001", "ORD_002"],
      "destinations": ["NAKURU", "ELDORET"],
      "estimated_cost": 2500.0,
      "estimated_revenue": 5500.0,
      "estimated_profit": 3000.0,
      "eta_minutes": 180
    }
  ],
  "reasoning": "CFA optimization with 40% rule-based constraints...",
  "alternatives": [
    {
      "policy": "VFA",
      "confidence": 0.81,
      "expected_value": 14200,
      "routes": [...]
    }
  ]
}

Response 400:
{
  "error": "INVALID_DECISION_TYPE",
  "message": "Unknown decision type: INVALID",
  "timestamp": "2024-01-15T10:30:15Z"
}

Response 503:
{
  "error": "SERVICE_UNAVAILABLE",
  "message": "Engine is initializing, please retry",
  "timestamp": "2024-01-15T10:30:15Z"
}
```

#### Get Decision Details

```
GET /api/decisions/{decision_id}

Response 200:
{
  "decision_id": "DEC_20240115_001",
  "created_at": "2024-01-15T10:30:15Z",
  "policy_used": "CFAVFAHybrid",
  "decision_type": "DAILY_ROUTE_PLANNING",
  "status": "PENDING_EXECUTION" | "COMMITTED" | "EXECUTED" | "CANCELED",
  "routes": [...],
  "confidence_score": 0.87,
  "expected_value": 15000,
  "execution_status": null  # when status = PENDING_EXECUTION
}
```

#### List Decisions

```
GET /api/decisions?limit=50&offset=0&status=COMMITTED

Response 200:
{
  "decisions": [...],
  "total": 342,
  "limit": 50,
  "offset": 0
}
```

### 2. Decision Execution Endpoints

#### Commit (Execute) Decision

```
POST /api/decisions/{decision_id}/commit
Content-Type: application/json

{
  "confirm": true,
  "adjustments": {                    # optional
    "route_modifications": {
      "ROUTE_001": {
        "remove_orders": ["ORD_005"]
      }
    }
  }
}

Response 200:
{
  "decision_id": "DEC_20240115_001",
  "status": "COMMITTED",
  "routes_created": 2,
  "orders_assigned": 4,
  "vehicles_deployed": 2,
  "committed_at": "2024-01-15T10:31:00Z",
  "routes_execution": [
    {
      "route_id": "ROUTE_001",
      "vehicle_id": "VEH_001",
      "start_time": "2024-01-15T10:35:00Z",
      "first_stop": "Eastleigh Store"
    }
  ]
}

Response 409:
{
  "error": "DECISION_ALREADY_COMMITTED",
  "message": "Decision DEC_20240115_001 was already committed at 2024-01-15T10:30:45Z"
}
```

#### Cancel Decision

```
POST /api/decisions/{decision_id}/cancel
Content-Type: application/json

{
  "reason": "Traffic conditions changed"
}

Response 200:
{
  "decision_id": "DEC_20240115_001",
  "status": "CANCELED",
  "reason": "Traffic conditions changed",
  "canceled_at": "2024-01-15T10:32:00Z"
}
```

### 3. Order Endpoints

#### Create Order

```
POST /api/orders
Content-Type: application/json

{
  "customer_id": "MAJID",
  "customer_name": "Majid Retailers",
  "pickup_location": {
    "latitude": -1.2921,
    "longitude": 36.8219,
    "address": "Eastleigh Store",
    "zone": "Eastleigh"
  },
  "destination_city": "NAKURU",
  "destination_location": {
    "latitude": -0.3031,
    "longitude": 35.2684,
    "address": "Nakuru CBD"
  },
  "weight_tonnes": 2.5,
  "volume_m3": 4.0,
  "time_window": {
    "start": "2024-01-15T08:30:00Z",
    "end": "2024-01-15T09:45:00Z"
  },
  "priority": 0,
  "special_handling": ["fragile", "fresh_food"],
  "price_kes": 2500.0
}

Response 201:
{
  "order_id": "ORD_001",
  "status": "PENDING_ASSIGNMENT",
  "created_at": "2024-01-15T10:30:15Z"
}
```

#### Get Order Status

```
GET /api/orders/{order_id}

Response 200:
{
  "order_id": "ORD_001",
  "status": "ASSIGNED_TO_ROUTE" | "IN_TRANSIT" | "DELIVERED" | "FAILED",
  "route_id": "ROUTE_001",
  "assigned_vehicle": "VEH_001",
  "expected_pickup": "2024-01-15T10:35:00Z",
  "expected_delivery": "2024-01-15T12:55:00Z",
  "actual_pickup": null,
  "actual_delivery": null,
  "tracking": {
    "current_location": {"latitude": -1.25, "longitude": 36.85},
    "distance_to_delivery_km": 45,
    "eta_minutes": 65
  }
}
```

#### List Orders

```
GET /api/orders?customer_id=MAJID&status=PENDING_ASSIGNMENT&limit=100

Response 200:
{
  "orders": [...],
  "total": 45,
  "filtered_by": {
    "customer_id": "MAJID",
    "status": "PENDING_ASSIGNMENT"
  }
}
```

### 4. Route Endpoints

#### Get Route Details

```
GET /api/routes/{route_id}

Response 200:
{
  "route_id": "ROUTE_001",
  "vehicle_id": "VEH_001",
  "driver_id": "DRIVER_001",
  "status": "PLANNED" | "IN_TRANSIT" | "COMPLETED" | "FAILED",
  "orders": [
    {
      "order_id": "ORD_001",
      "sequence": 1,
      "status": "PENDING_PICKUP",
      "destination": "Nakuru CBD"
    }
  ],
  "stops": [
    {
      "sequence": 1,
      "location": "Eastleigh Store",
      "type": "PICKUP",
      "orders": ["ORD_001", "ORD_002"],
      "eta": "2024-01-15T10:35:00Z",
      "actual_arrival": null
    }
  ],
  "summary": {
    "total_orders": 4,
    "total_weight": 12.5,
    "total_volume": 20.0,
    "planned_distance_km": 250.0,
    "planned_duration_minutes": 320,
    "planned_cost_kes": 2500.0,
    "planned_revenue_kes": 8000.0
  },
  "tracking": {
    "current_location": {"latitude": -1.25, "longitude": 36.85},
    "completed_stops": 2,
    "completed_orders": 2,
    "failed_orders": 0
  }
}
```

#### List Routes

```
GET /api/routes?vehicle_id=VEH_001&status=IN_TRANSIT&date=2024-01-15

Response 200:
{
  "routes": [...],
  "total": 3,
  "filtered_by": {
    "vehicle_id": "VEH_001",
    "status": "IN_TRANSIT",
    "date": "2024-01-15"
  }
}
```

### 5. Feedback Endpoints

#### Submit Route Outcome

```
POST /api/outcomes
Content-Type: application/json

{
  "route_id": "ROUTE_001",
  "vehicle_id": "VEH_001",
  "predicted_fuel_cost": 2500.0,
  "actual_fuel_cost": 2400.0,
  "predicted_duration_minutes": 320,
  "actual_duration_minutes": 310,
  "predicted_distance_km": 250.0,
  "actual_distance_km": 248.0,
  "on_time": true,
  "successful_deliveries": 4,
  "failed_deliveries": 0,
  "customer_satisfaction_score": 0.95,
  "notes": "Route completed successfully, good traffic conditions"
}

Response 201:
{
  "outcome_id": "OUTCOME_001",
  "processed_at": "2024-01-15T14:30:00Z",
  "learning_signals": {
    "cfa_fuel_error_kes": -100.0,
    "cfa_time_error_minutes": -10,
    "vfa_reward_points": 950,
    "pfa_rule_validation": {"rule_1": true, "rule_2": false},
    "dla_forecast_accuracy": 0.96
  }
}
```

#### Get Feedback Statistics

```
GET /api/feedback/stats?start_date=2024-01-01&end_date=2024-01-15&vehicle_id=VEH_001

Response 200:
{
  "period": {
    "start": "2024-01-01",
    "end": "2024-01-15"
  },
  "statistics": {
    "total_routes": 45,
    "successful_routes": 43,
    "success_rate": 0.9556,
    "on_time_delivery_rate": 0.9111,
    "average_fuel_cost_error": -50.5,
    "average_time_error_minutes": -5.2,
    "average_customer_satisfaction": 0.92,
    "total_revenue_kes": 150000.0,
    "total_cost_kes": 45000.0,
    "total_profit_kes": 105000.0
  },
  "vehicle_stats": {
    "VEH_001": {
      "routes": 15,
      "success_rate": 0.9333,
      "on_time_rate": 0.9333,
      "efficiency": 0.92
    }
  }
}
```

### 6. Performance Analytics Endpoints

#### Policy Performance

```
GET /api/analytics/policies?since=7d

Response 200:
{
  "period": "last_7_days",
  "policies": {
    "PFA": {
      "usage_count": 12,
      "average_confidence": 0.78,
      "success_rate": 0.8333,
      "average_value": 12500,
      "preferred_for": ["BACKHAUL_OPPORTUNITY"]
    },
    "CFA": {
      "usage_count": 45,
      "average_confidence": 0.85,
      "success_rate": 0.9111,
      "average_value": 15000,
      "preferred_for": ["DAILY_ROUTE_PLANNING", "ORDER_ARRIVAL"]
    },
    "VFA": {
      "usage_count": 28,
      "average_confidence": 0.82,
      "success_rate": 0.8929,
      "average_value": 14200,
      "preferred_for": ["ORDER_ARRIVAL"]
    },
    "DLA": {
      "usage_count": 8,
      "average_confidence": 0.81,
      "success_rate": 0.875,
      "average_value": 13800,
      "preferred_for": ["DAILY_ROUTE_PLANNING"]
    }
  },
  "recommendations": [
    "CFA performing best for daily planning",
    "VFA learning rate is good, accuracy improving"
  ]
}
```

#### System Health

```
GET /api/health

Response 200:
{
  "status": "HEALTHY",
  "engine_version": "1.0.0",
  "uptime_seconds": 864000,
  "active_routes": 12,
  "pending_decisions": 3,
  "queue_size": 5,
  "model_status": {
    "cfa": "READY",
    "vfa": "READY",
    "pfa": "READY",
    "dla": "READY"
  },
  "recent_errors": []
}
```

### 7. WebSocket Events (Real-Time Updates)

#### Subscribe to Route Updates

```
WebSocket: ws://api.senga.local/ws/routes/{route_id}

Event: route_status_changed
{
  "type": "route_status_changed",
  "route_id": "ROUTE_001",
  "new_status": "IN_TRANSIT",
  "timestamp": "2024-01-15T10:35:00Z"
}

Event: location_update
{
  "type": "location_update",
  "route_id": "ROUTE_001",
  "location": {"latitude": -1.25, "longitude": 36.85},
  "eta_next_stop_minutes": 15,
  "timestamp": "2024-01-15T10:45:00Z"
}

Event: order_status_changed
{
  "type": "order_status_changed",
  "route_id": "ROUTE_001",
  "order_id": "ORD_001",
  "new_status": "DELIVERED",
  "timestamp": "2024-01-15T11:15:00Z"
}

Event: route_completed
{
  "type": "route_completed",
  "route_id": "ROUTE_001",
  "summary": {
    "orders_delivered": 4,
    "orders_failed": 0,
    "actual_duration_minutes": 310,
    "actual_cost_kes": 2400.0,
    "actual_revenue_kes": 8000.0
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

## Implementation Notes

### Error Handling

All endpoints return standard error responses:

```json
{
  "error": "ERROR_CODE",
  "message": "Human readable description",
  "details": {
    "field": "Additional context"
  },
  "timestamp": "2024-01-15T10:30:15Z",
  "request_id": "REQ_ABC123"
}
```

### Authentication

- Every request must include `Authorization: Bearer <token>`
- Token validation against user service (future integration)
- Rate limiting: 1000 requests per minute per token

### Pagination

- Default limit: 50
- Maximum limit: 500
- Use `offset` for pagination (not `page`)

### Timestamps

- All timestamps in ISO 8601 format with timezone
- Server timezone: UTC
- Client timestamps converted to UTC automatically

## Implementation Sequence

1. **Phase 1**: Decision and execution endpoints (POST /decisions, POST /commit)
2. **Phase 2**: Query endpoints (GET /decisions, GET /orders, GET /routes)
3. **Phase 3**: Feedback endpoints (POST /outcomes)
4. **Phase 4**: Analytics endpoints (GET /analytics)
5. **Phase 5**: WebSocket integration
6. **Phase 6**: Authentication and rate limiting

## Example Request Flow

```
1. Client: POST /api/orders (create new order)
   ↓
2. Engine: Emit 'order_arrived' event
   ↓
3. EventOrchestrator: Queue ORDER_ARRIVAL decision
   ↓
4. Client: POST /api/decisions (request decision)
   ↓
5. PowellEngine: Select policy, make decision
   ↓
6. Client receives decision with routes
   ↓
7. Client: POST /api/decisions/{id}/commit (execute)
   ↓
8. Routes created, vehicles dispatched
   ↓
9. Client: Subscribe to ws://api/routes/{route_id}
   ↓
10. Real-time location/status updates via WebSocket
    ↓
11. Client: POST /api/outcomes (submit feedback)
    ↓
12. FeedbackProcessor: Generate learning signals
    ↓
13. Policies update parameters
    ↓
14. Next decision benefits from learning
```

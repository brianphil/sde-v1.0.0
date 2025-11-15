# Backhaul & Order Consolidation Workflow

## Executive Summary

**Current Status**: Consolidation logic is **partially implemented** but lacks a complete end-to-end workflow.

**What's Working**:
- ✅ Backhaul opportunity detection (`SystemState.get_backhaul_opportunities()`)
- ✅ VFA policy evaluates backhaul decisions
- ✅ Event mapping for `backhaul_opportunity` → `DecisionType.BACKHAUL_OPPORTUNITY`

**What's Missing**:
- ❌ No trigger mechanism to check for backhaul opportunities automatically
- ❌ Consolidation optimization is minimal (only takes first order per route)
- ❌ No consolidated route creation logic
- ❌ No learning/feedback loop for consolidation decisions
- ❌ No API endpoint for backhaul/consolidation requests

---

## What is Order Consolidation?

**Backhaul Consolidation** = Adding orders to active routes that have spare capacity, especially on return trips.

### Business Value
- **Reduce empty miles**: Vehicles returning empty can pick up orders
- **Increase vehicle utilization**: Fill spare capacity on partially loaded routes
- **Lower costs**: Marginal cost of adding order to existing route << new route cost
- **Higher profit**: Same delivery fees with lower operational costs

### Example Scenario
```
Current State:
  - Route R1: Nairobi → Eldoret (delivering 3 tonnes, capacity 5T)
  - Available capacity on return: 2 tonnes
  - Pending order O123: Eldoret → Nairobi (1.5 tonnes)

Consolidation Decision:
  → Add order O123 to route R1 (backhaul)
  → Marginal fuel cost: ~200 KES (vs. new route: 5,000 KES)
  → Profit increase: ~4,800 KES
```

---

## Current Implementation Analysis

### 1. Backhaul Detection (✅ Working)

**Location**: `backend/core/models/state.py:315-342`

```python
def get_backhaul_opportunities(self) -> List[tuple[Route, List[Order]]]:
    """Identify active routes with backhaul opportunities."""
    opportunities = []

    for route in self.active_routes.values():
        # Calculate remaining capacity
        used_w, used_v = self.get_used_capacity_by_vehicle().get(
            route.vehicle_id, (0.0, 0.0)
        )
        vehicle = self.fleet.get(route.vehicle_id)
        avail_w, avail_v = vehicle.get_remaining_capacity(used_w, used_v)

        # Find orders that fit
        fitting_orders = []
        for order in self.get_unassigned_orders().values():
            if order.can_fit_in_vehicle(avail_w, avail_v):
                fitting_orders.append(order)

        if fitting_orders:
            opportunities.append((route, fitting_orders))

    return opportunities
```

**What it does**: Finds all active routes with spare capacity and lists orders that could fit.

**Limitation**: Doesn't consider:
- Geographic feasibility (is order on route path?)
- Time window compatibility
- Profitability (is consolidation worth it?)

### 2. Consolidation Optimizer (⚠️ Minimal)

**Location**: `backend/services/route_optimizer.py:107-124`

```python
def optimize_backhaul_consolidation(
    self, state: SystemState
) -> List[Tuple[Route, Order]]:
    """Identify backhaul opportunities and consolidation options."""

    opportunities = []
    backhaul_opps = state.get_backhaul_opportunities()

    for route, available_orders in backhaul_opps:
        for order in available_orders[:1]:  # Only first order per route
            opportunities.append((route, order))

    return opportunities
```

**Issues**:
- ❌ Only takes first order (no optimization)
- ❌ Doesn't evaluate profitability
- ❌ Doesn't check route compatibility
- ❌ Doesn't create updated routes

### 3. VFA Backhaul Evaluation (⚠️ Simplistic)

**Location**: `backend/core/powell/vfa.py:448-464`

```python
if context.decision_type == DecisionType.BACKHAUL_OPPORTUNITY:
    accept_value = current_value + sum(
        o.price_kes * 0.1 for o in context.orders_to_consider.values()
    )  # 10% margin estimate

    if accept_value > best_value:
        best_action = ActionType.ACCEPT_ORDER
        # Create dummy routes (in production, would optimize)
        for vehicle in context.vehicles_available.values():
            route = self._create_simple_route(state, orders_dict, vehicle)
            best_routes.append(route)
```

**Issues**:
- ❌ Uses fixed 10% margin (ignores actual costs)
- ❌ Creates "dummy routes" instead of updating existing routes
- ❌ Doesn't optimize consolidation sequencing
- ❌ No geographic validation

### 4. Event Triggering (⚠️ Manual Only)

**Location**: `backend/services/event_orchestrator.py:327-343`

```python
def _map_event_to_decision_type(self, event: Event) -> DecisionType:
    # ...
    elif event.event_type == "backhaul_opportunity":
        return DecisionType.BACKHAUL_OPPORTUNITY
    # ...
```

**Issue**: There's no automatic detection. Someone must manually submit a `backhaul_opportunity` event.

**Missing**: Periodic checks or event triggers when:
- Route completes delivery leg
- New order arrives near active route
- Vehicle utilization drops below threshold

---

## Proposed Clear Workflow

### End-to-End Consolidation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ TRIGGER: Backhaul Opportunity Detected                         │
│ - Route completes outbound leg (vehicle returning)             │
│ - New order arrives near active route path                     │
│ - Periodic check (every 15 minutes)                            │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Identify Candidates                                    │
│ - Get active routes with spare capacity                        │
│ - Get unassigned orders                                        │
│ - Filter by geographic proximity (±50km from route path)       │
│ - Filter by time window compatibility                          │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Evaluate Profitability (VFA)                           │
│ - For each (route, order) pair:                                │
│   • Calculate marginal cost (fuel + time)                      │
│   • Calculate value (order price - marginal cost)              │
│   • Estimate long-term value (VFA neural network)              │
│   • Compute expected profit                                    │
│ - Rank by profitability                                        │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Optimize Consolidation                                 │
│ - Select top N profitable consolidations                       │
│ - Check conflicts (same order on multiple routes)              │
│ - Update route sequences with new stops                        │
│ - Validate capacity and time windows                           │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Create Decision                                        │
│ - Policy: VFA (backhaul is strategic decision)                 │
│ - Action: CREATE_ROUTE (updated route with consolidated order) │
│ - Confidence: Based on VFA training samples                    │
│ - Expected Value: Sum of consolidated order profits            │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Commit Decision                                        │
│ - Update route with new stops                                  │
│ - Assign orders to route                                       │
│ - Notify driver of route change                                │
│ - Update state (via StateManager)                              │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Execute & Track                                        │
│ - Driver picks up consolidated order                           │
│ - Real-time tracking                                           │
│ - Record actual costs (fuel, time)                             │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Learn from Outcome                                     │
│ - Compare predicted vs. actual marginal cost                   │
│ - Update VFA network with TD-learning                          │
│ - Adjust consolidation profitability estimates                 │
│ - Track consolidation success rate                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Gaps & Fixes

### Gap 1: No Automatic Trigger ❌

**Current**: Must manually submit `backhaul_opportunity` event

**Fix**: Add automatic triggers in `StateManager` or `EventOrchestrator`

```python
# backend/services/event_orchestrator.py

def check_backhaul_opportunities(self):
    """Periodic check for backhaul opportunities (call every 15 min)."""
    state = self.state_manager.get_current_state()
    opportunities = state.get_backhaul_opportunities()

    if opportunities:
        event = Event(
            event_type="backhaul_opportunity",
            data={"opportunities": opportunities},
            priority=EventPriority.NORMAL
        )
        self.submit_event(event)
        logger.info(f"Detected {len(opportunities)} backhaul opportunities")

def on_route_leg_completed(self, route_id: str):
    """Trigger when outbound leg completes (vehicle returning)."""
    self.check_backhaul_opportunities()

def on_order_arrived(self, order):
    """Check if new order fits on active routes."""
    state = self.state_manager.get_current_state()

    # Check if order is near any active route
    for route in state.active_routes.values():
        if self._is_order_near_route(order, route):
            self.check_backhaul_opportunities()
            break
```

### Gap 2: Minimal Optimization Logic ❌

**Current**: `optimize_backhaul_consolidation()` only takes first order

**Fix**: Implement profitability-based optimization

```python
# backend/services/route_optimizer.py

def optimize_backhaul_consolidation(
    self, state: SystemState
) -> List[Tuple[Route, Order, float]]:
    """Optimize backhaul consolidation by profitability.

    Returns:
        List of (route, order, profit) sorted by profit descending
    """

    opportunities = []
    backhaul_opps = state.get_backhaul_opportunities()

    for route, available_orders in backhaul_opps:
        vehicle = state.fleet.get(route.vehicle_id)

        for order in available_orders:
            # Check geographic feasibility
            if not self._is_order_on_route_path(order, route, max_detour_km=50):
                continue

            # Check time window compatibility
            if not self._check_time_window_compatible(order, route):
                continue

            # Calculate marginal cost
            marginal_cost = self._calculate_marginal_cost(order, route, vehicle)

            # Calculate profit
            profit = order.price_kes - marginal_cost

            if profit > 0:  # Only profitable consolidations
                opportunities.append((route, order, profit))

    # Sort by profit (highest first)
    opportunities.sort(key=lambda x: x[2], reverse=True)

    logger.info(f"Found {len(opportunities)} profitable consolidations")
    return opportunities

def _calculate_marginal_cost(
    self, order: Order, route: Route, vehicle: Vehicle
) -> float:
    """Calculate additional cost of adding order to route."""

    # Additional distance (detour from route path)
    extra_distance_km = self._calculate_detour_distance(order, route)

    # Marginal fuel cost
    fuel_cost = extra_distance_km * self._get_fuel_cost_per_km(vehicle)

    # Additional time cost
    extra_time_minutes = self._estimate_extra_time(order, route)
    time_cost = (extra_time_minutes / 60.0) * 500  # 500 KES/hour

    # Minimal handling cost
    handling_cost = 200  # Fixed per order

    return fuel_cost + time_cost + handling_cost

def _is_order_on_route_path(
    self, order: Order, route: Route, max_detour_km: float = 50
) -> bool:
    """Check if order is geographically feasible for route."""

    # Get route waypoints
    waypoints = [stop.location for stop in route.stops]

    # Check if order pickup/delivery is within max_detour of route path
    for waypoint in waypoints:
        if self._distance_km(order.delivery_location, waypoint) < max_detour_km:
            return True

    return False

def _check_time_window_compatible(self, order: Order, route: Route) -> bool:
    """Check if order time window fits route schedule."""

    # Get route estimated completion time
    route_end_time = route.estimated_end_time

    # Check if order can be delivered before its deadline
    order_deadline = order.delivery_window_end

    # Add buffer time for detour
    buffer_minutes = 60

    return route_end_time + timedelta(minutes=buffer_minutes) <= order_deadline
```

### Gap 3: VFA Doesn't Update Routes ❌

**Current**: Creates "dummy routes" instead of updating existing routes

**Fix**: Implement route update logic in VFA

```python
# backend/core/powell/vfa.py

def evaluate(self, state: SystemState, context: DecisionContext) -> PolicyDecision:
    """Evaluate backhaul consolidation with route updates."""

    # ... (existing feature extraction)

    best_routes = []

    if context.decision_type == DecisionType.BACKHAUL_OPPORTUNITY:
        # Get consolidated routes from RouteOptimizer
        consolidations = self.route_optimizer.optimize_backhaul_consolidation(state)

        for route, order, profit in consolidations[:5]:  # Top 5
            # Create updated route with new stop
            updated_route = self._add_order_to_route(route, order, state)

            # Estimate value of this consolidation
            marginal_value = profit + self._estimate_long_term_value(
                order, updated_route, state
            )

            if marginal_value > 0:
                best_routes.append(updated_route)
                accept_value += marginal_value

    # ... (rest of decision logic)

def _add_order_to_route(
    self, route: Route, order: Order, state: SystemState
) -> Route:
    """Create updated route with consolidated order."""

    from copy import deepcopy
    from ..models.domain import Route, RouteStop

    # Copy route
    updated_route = deepcopy(route)

    # Add new stop for order pickup (if needed)
    if order.pickup_location != route.stops[0].location:
        pickup_stop = RouteStop(
            location=order.pickup_location,
            stop_type="pickup",
            order_id=order.order_id,
            planned_arrival=self._estimate_arrival_time(route, order.pickup_location)
        )
        # Insert at optimal position (TSP-style)
        optimal_position = self._find_optimal_stop_position(
            updated_route, pickup_stop
        )
        updated_route.stops.insert(optimal_position, pickup_stop)

    # Add delivery stop
    delivery_stop = RouteStop(
        location=order.delivery_location,
        stop_type="delivery",
        order_id=order.order_id,
        planned_arrival=self._estimate_arrival_time(
            updated_route, order.delivery_location
        )
    )
    delivery_position = self._find_optimal_stop_position(
        updated_route, delivery_stop
    )
    updated_route.stops.insert(delivery_position, delivery_stop)

    # Update route IDs
    updated_route.order_ids.append(order.order_id)
    updated_route.route_id = f"{route.route_id}_consolidated_{order.order_id}"

    return updated_route
```

### Gap 4: No Learning Feedback ❌

**Current**: No consolidation-specific feedback tracking

**Fix**: Track consolidation outcomes separately

```python
# backend/core/learning/feedback_processor.py

def process_consolidation_outcome(
    self, outcome: OperationalOutcome, was_consolidated: bool
) -> Dict[str, Any]:
    """Process outcome for routes with consolidated orders."""

    signals = self.process_outcome(outcome, state)  # Standard processing

    if was_consolidated:
        # Additional consolidation-specific metrics
        consolidation_profit = (
            outcome.actual_revenue - outcome.actual_fuel_cost
        )

        signals['consolidation_metrics'] = {
            'profit': consolidation_profit,
            'marginal_cost_error': (
                outcome.predicted_marginal_cost - outcome.actual_marginal_cost
            ),
            'success': outcome.on_time and outcome.successful_deliveries > 0,
            'detour_distance_km': outcome.actual_distance_km - outcome.predicted_distance_km
        }

        # Update VFA with consolidation-specific reward
        consolidation_reward = consolidation_profit / 100.0  # Normalize
        signals['vfa_signals']['consolidation_reward'] = consolidation_reward

    return signals
```

---

## Complete Usage Example

### Step-by-Step: Handling a Backhaul Opportunity

```python
from backend.core.powell.engine import PowellEngine
from backend.services.event_orchestrator import EventOrchestrator, Event, EventPriority
from backend.services.state_manager import StateManager
from backend.core.models.decision import DecisionType
from datetime import datetime

# Initialize system
engine = PowellEngine()
state_manager = StateManager()
orchestrator = EventOrchestrator(engine, state_manager)

# ===== SCENARIO =====
# Route R1: Nairobi → Eldoret (outbound leg complete, returning with 2T spare capacity)
# Order O123: Eldoret → Nairobi (1.5T)

# Step 1: Trigger backhaul check (automatic or manual)
event = Event(
    event_type="backhaul_opportunity",
    data={
        "trigger": "route_leg_completed",
        "route_id": "R1",
    },
    priority=EventPriority.NORMAL
)
orchestrator.submit_event(event)

# Step 2: Process event (orchestrator handles workflow)
result = orchestrator.process_event(event)

print(f"Decision Type: {result['decision_type']}")
# Output: backhaul_opportunity

print(f"Policy Used: {result['decision']['policy']}")
# Output: VFA

print(f"Action: {result['decision']['action']}")
# Output: create_route (updated route with consolidated order)

print(f"Confidence: {result['decision']['confidence']:.1%}")
# Output: 72.5%

print(f"Expected Value: {result['decision']['value']} KES")
# Output: 4800 KES (profit from consolidation)

# Step 3: Execute decision (routes already created by commit_decision)
if result['execution']:
    for route in result['execution']['routes_created']:
        print(f"Updated Route: {route.route_id}")
        print(f"  Orders: {route.order_ids}")
        print(f"  Stops: {len(route.stops)}")
        # Output:
        # Updated Route: R1_consolidated_O123
        #   Orders: ['O100', 'O101', 'O102', 'O123']
        #   Stops: 5

# Step 4: Track outcome (after route execution)
from backend.core.models.domain import OperationalOutcome

outcome = OperationalOutcome(
    outcome_id="OUT_R1_consolidated",
    route_id="R1_consolidated_O123",
    vehicle_id="VEH_001",
    predicted_fuel_cost=5200,
    actual_fuel_cost=5350,  # Slightly higher
    predicted_duration_minutes=300,
    actual_duration_minutes=310,
    predicted_distance_km=280,
    actual_distance_km=285,
    on_time=True,
    successful_deliveries=4,  # All 4 orders delivered
    failed_deliveries=0,
    customer_satisfaction_score=0.95
)

# Submit outcome for learning
orchestrator.submit_route_outcome(outcome)
results = orchestrator.process_all_events()

# VFA learns: consolidation was profitable (slightly higher cost but all delivered on time)
# Next consolidation decision will have improved cost estimates
```

---

## API Endpoint Design

### POST /consolidation/check

Check for backhaul/consolidation opportunities now.

```json
POST /api/v1/consolidation/check

Request:
{
  "route_id": "R1",  // Optional: check specific route
  "trigger": "route_leg_completed"
}

Response:
{
  "opportunities": [
    {
      "route_id": "R1",
      "vehicle_id": "VEH_001",
      "available_capacity": {
        "weight_tonnes": 2.0,
        "volume_m3": 3.5
      },
      "fitting_orders": [
        {
          "order_id": "O123",
          "weight": 1.5,
          "volume": 2.0,
          "estimated_profit": 4800,
          "confidence": 0.85
        }
      ]
    }
  ],
  "total_opportunities": 1,
  "total_potential_profit": 4800
}
```

### POST /consolidation/evaluate

Evaluate specific consolidation decision.

```json
POST /api/v1/consolidation/evaluate

Request:
{
  "route_id": "R1",
  "order_ids": ["O123", "O124"]
}

Response:
{
  "decision_id": "DEC_12345",
  "policy": "VFA",
  "action": "accept_consolidation",
  "confidence": 0.85,
  "expected_profit": 4800,
  "updated_route": {
    "route_id": "R1_consolidated",
    "order_ids": ["O100", "O101", "O102", "O123"],
    "stops": [...],
    "estimated_cost": 5200,
    "estimated_revenue": 10000,
    "estimated_profit": 4800
  },
  "reasoning": "Profitable consolidation with 2T spare capacity, minimal detour (5km)"
}
```

### POST /consolidation/commit

Commit consolidation decision (update route).

```json
POST /api/v1/consolidation/commit

Request:
{
  "decision_id": "DEC_12345"
}

Response:
{
  "route_id": "R1_consolidated",
  "orders_added": ["O123"],
  "driver_notified": true,
  "estimated_completion": "2024-01-15T18:30:00Z",
  "status": "active"
}
```

---

## Implementation Priority

### Phase 1: Core Consolidation Logic (High Priority)
1. ✅ Fix `optimize_backhaul_consolidation()` with profitability ranking
2. ✅ Implement `_calculate_marginal_cost()`
3. ✅ Add geographic feasibility check (`_is_order_on_route_path()`)
4. ✅ Implement route update logic (`_add_order_to_route()`)

### Phase 2: Automatic Triggers (Medium Priority)
5. ✅ Add `check_backhaul_opportunities()` periodic task
6. ✅ Trigger on route leg completion
7. ✅ Trigger on new order arrival near active routes

### Phase 3: Learning & Optimization (Medium Priority)
8. ✅ Track consolidation-specific outcomes
9. ✅ Update VFA with consolidation rewards
10. ✅ Improve consolidation profitability estimates over time

### Phase 4: API Integration (Low Priority)
11. ✅ Build `/consolidation/check` endpoint
12. ✅ Build `/consolidation/evaluate` endpoint
13. ✅ Build `/consolidation/commit` endpoint

---

## Summary

**Problem**: Consolidation workflow is incomplete and unclear.

**Root Causes**:
1. No automatic triggering mechanism
2. Minimal optimization (only takes first order)
3. No actual route update logic
4. No learning feedback for consolidations

**Solution**: Implement the 7-step workflow with:
- Automatic triggers (periodic + event-based)
- Profitability-based optimization
- Route update logic in VFA
- Consolidation-specific learning signals

**Next Steps**:
1. Implement fixes for Gaps 1-4
2. Add comprehensive tests for consolidation
3. Create API endpoints
4. Document in API spec

This will transform consolidation from a placeholder to a fully functional, profitable feature.

# Senga SDE Framework - AI Agent Instructions

## Executive Context

**Senga Sequential Decision Engine (SDE)** is an event-driven, learning-based route optimization system for mesh logistics (Nairobi → Nakuru/Eldoret/Kitale). The system implements Warren Powell's Sequential Decision Analytics framework using all four policy classes (PFA, CFA, VFA, DLA) and their hybrids to make continuous decisions and learn from operational feedback.

**Core Problem Solved:** Senga operates a warehouseless mesh network requiring intelligent consolidation of orders into optimal routes. The system must:

- Dynamically pool and route orders as they arrive
- Optimize multi-destination pickup and delivery sequences
- Learn from operational outcomes to improve predictions
- Balance immediate costs against long-term network efficiency

**Success Criteria:**

- ✅ Functional: Engine makes automated routing decisions for incoming orders
- ✅ Learning: System improves cost/time predictions from operational feedback
- ✅ Operational: Generates executable route plans for operations team
- ✅ Scalable: Handles order arrival rate and fleet size without degradation

## Architecture Overview

**Senga** is a Spatial Decision Engine that combines real-time route optimization with reinforcement learning. It processes delivery orders, optimizes routes via Powell algorithms, and learns from feedback to improve future decisions.

### Core Components

#### Backend Architecture (FastAPI)

- **API Layer** (`backend/api/`): RESTful routes for decisions, orders; WebSocket for real-time updates
- **Core Models** (`backend/core/models/`): Domain entities (Order, Vehicle, Route, Customer, State)
- **Powell Engine** (`backend/core/powell/`): Algorithm suite (PFA, VFA, CFA, DLA, hybrids) for optimization
- **Learning System** (`backend/core/learning/`): TD-learning feedback loop, parameter updates, pattern mining
- **Services** (`backend/services/`):
  - `event_orchestrator.py`: Coordinates decision workflow and state transitions
  - `learning_coordinator.py`: Manages model training and parameter updates
  - `route_optimizer.py`: Applies Powell algorithms to generate optimized routes
  - `state_manager.py`: Maintains consistent application state
  - `external/`: Integrations (Google Places, Traffic API)
- **Database** (`backend/db/`): SQLAlchemy models and migrations for persistence

#### Frontend (React + Vite)

- **Dashboard**: Displays decisions, performance metrics, route visualization
- **Real-time Updates**: WebSocket client (`hooks/useWebSocket.js`) for live decision queue
- **Components**: Decision Queue, Feedback Form, Route Map, Performance Metrics

### Data Flow

1. **Decision Request** → API receives order/context → StateManager prepares state
2. **Optimization** → EventOrchestrator triggers RouteOptimizer → Powell algorithms generate candidate routes
3. **Decision Output** → Decision returned to frontend via API/WebSocket
4. **Feedback Loop** → FeedbackForm submits performance data → LearningCoordinator processes → TD-Learning updates parameters
5. **Pattern Discovery** → PatternMining analyzes feedback patterns → Engine refines future decisions

## Key Development Patterns

### Powell's Four Policy Classes - Decision Context

All policies answer: **Given state S_t, what action A_t should we take?**

#### 1. PFA - Policy Function Approximation (Rules-Based)

Direct function mapping state → action without optimization.

- **When to use**: Learned operational rules, heuristics, constraints
- **Examples**: "Start Eastleigh pickups at 8:30 AM", "Avoid Majid deliveries Wed 2PM+"
- **Learnable via**: Supervised learning from outcomes, pattern mining

```python
# Implementation pattern
class PolicyFunction:
    def get_action(self, state: State) -> Action:
        # Direct mapping, no optimization
        if state.customer == "Majid" and state.day == "Wednesday" and state.time > "14:00":
            return Action(type="skip", reason="delivery_block")
        return Action(type="schedule", priority=self.compute_priority(state))

    def learn(self, feedback: OperationalOutcome):
        # Pattern mining: detect recurring successful/failed patterns
        self.rules = extract_patterns(self.history + [feedback])
```

#### 2. CFA - Cost Function Approximation (Optimization)

Parameterized optimization problem: `A^π(S_t) = argmin_a C(S_t, a | θ)`

- **When to use**: Single-shot route optimization, cost minimization
- **Objective**: Minimize `fuel_cost + time_cost + delay_penalty`
- **Learnable via**: Compare predicted vs actual costs, adjust parameters

```python
# Implementation pattern
class CostFunctionPolicy:
    def get_action(self, state: State) -> Action:
        def objective(route_sequence):
            return (self.theta['fuel_weight'] * fuel_cost(route_sequence) +
                   self.theta['time_weight'] * time_cost(route_sequence) +
                   self.theta['delay_penalty'] * expected_delay(route_sequence))
        result = minimize(objective, constraints=self.constraints)
        return Action(route=result.x)

    def learn(self, predicted_cost, actual_cost):
        # Adjust parameters based on prediction error
        error = actual_cost - predicted_cost
        self.theta = gradient_update(self.theta, error)
```

#### 3. VFA - Value Function Approximation (Long-term)

Approximate downstream value: `A^π(S_t) = argmax_a {C(S_t, a) + γ * V̄(S_{t+1})}`

- **When to use**: Strategic decisions with long-term impact (backhaul acceptance, fleet expansion)
- **Learnable via**: Temporal Difference learning (compare predicted value to observed outcome)

```python
# Implementation pattern (uses neural network)
class VFAPolicy:
    def get_action(self, state: State, feasible_actions) -> Action:
        best_action = None
        best_value = float('-inf')
        for action in feasible_actions:
            immediate_reward = compute_immediate_reward(state, action)
            next_state = simulate_transition(state, action)
            future_value = self.V(next_state).item()  # NN inference
            total_value = immediate_reward + self.gamma * future_value
            if total_value > best_value:
                best_value, best_action = total_value, action
        return best_action

    def learn(self, state, reward, next_state):
        # TD update: V(s) ← V(s) + α[r + γV(s') - V(s)]
        current = self.V(state)
        target = reward + self.gamma * self.V(next_state).detach()
        loss = MSELoss(current, target)
        optimizer.step(loss)
```

#### 4. DLA - Direct Lookahead Approximation (Multi-period)

Explicitly plan into future periods:

- **Deterministic**: Optimize over next H periods assuming known future
- **Stochastic**: Optimize over scenarios (two-stage stochastic program)
- **When to use**: Multi-day route planning, demand forecasting
- **Learnable via**: Adjust forecast parameters, buffer sizes, scenario generation

```python
# Implementation pattern
class DeterministicLookahead:
    def get_action(self, state: State, forecast: DemandForecast, horizon=7) -> Action:
        # Build multi-period optimization model (days 0 to horizon-1)
        model = build_optimization_model(state, forecast, horizon)
        solution = solve_milp(model)
        return solution.day_0_actions  # Take first day, re-optimize tomorrow
```

#### Hybrid Policies (The Real Power)

Combine multiple classes for enhanced decisions:

```python
# CFA/VFA Hybrid: Optimize immediate costs while maximizing future value
def objective_with_value(route):
    immediate_cost = cfa.compute_cost(route)
    next_state = simulate(state, route)
    future_value = vfa.V(next_state)
    return immediate_cost - gamma * future_value  # Minimize cost, maximize value

# DLA/VFA Hybrid: Lookahead H periods with learned terminal values
def lookahead_objective(decisions):
    total_cost = sum((gamma ** t) * compute_cost(decisions[t]) for t in range(H))
    total_cost += (gamma ** H) * vfa.V(state_at_horizon)  # Terminal value from VFA
    return total_cost
```

### Model Organization

- **Domain Models** (`backend/core/models/`): Pure data classes representing Orders, Vehicles, Routes, Customers
- **State Representation** (`backend/core/models/state.py`): Captures full decision context (pending orders, fleet status, environment, learning state)
- **Database Models** (`backend/db/models.py`): ORM layer (SQLAlchemy) for persistence

## Critical Decision Workflows

### 1. Daily Route Planning (Trigger: Orders Pending / Morning Optimization)

**Policy**: CFA/DLA Hybrid | **Objective**: Minimize costs while balancing long-term network efficiency

```python
# Workflow
async def daily_route_planning(pending_orders, available_fleet) -> List[Route]:
    # Group orders by destination (Nakuru, Eldoret, Kitale)
    by_destination = group_orders_by_destination(pending_orders)

    routes = []
    for destination, order_group in by_destination.items():
        if should_use_lookahead(order_group):  # Multi-day impact?
            # Use DLA: Optimize over 7-day horizon
            route = dla_policy.plan_route(order_group, fleet, horizon=7)
        else:
            # Use CFA: Single-day optimization (minimize fuel + time + penalties)
            route = cfa_policy.optimize_route(order_group, fleet)
        routes.append(route)

    # Check consolidation: Can multiple routes be combined?
    return consolidate_routes(routes, available_fleet)
```

**Critical Constraints**:

- Eastleigh CBD pickups: 8:30-9:45 AM window only (learned traffic pattern)
- Multi-destination routes: One truck serves Nakuru→Eldoret→Kitale
- Vehicle capacity not exceeded
- Customer time windows respected

### 2. Order Arrival (Trigger: New Order → API)

**Policy**: VFA (if backhaul) or CFA (if standard) | **Objective**: Accept/assign/defer decision

```python
# Workflow
async def handle_new_order(order: Order, state: SystemState) -> Decision:
    if order.is_backhaul():
        # VFA: Long-term value assessment
        # "Accept this cargo, tie up truck, miss future high-value cargo?"
        decision = vfa_policy.evaluate_backhaul(order, state)
        if decision.accept:
            route = assign_to_return_trip(order, state.fleet)
        else:
            return Decision(action="defer", reason="opportunity_cost_too_high")
    else:
        # Find compatible existing routes
        candidates = find_compatible_routes(order, state.active_routes)
        if candidates:
            # CFA: Optimize insertion (minimize cost increase)
            best_route = cfa_policy.optimize_insertion(order, candidates)
        else:
            # DLA: Evaluate if new route economical
            best_route = dla_policy.evaluate_new_route(order, state)

    return Decision(action="accept", route_id=best_route.route_id)
```

### 3. Real-Time Route Adjustment (Trigger: Delay Detected / Breakdown)

**Policy**: PFA + CFA Hybrid | **Objective**: Minimize disruption while recovering from delay

```python
# Workflow
async def handle_delay(event: DelayEvent, state: SystemState) -> Decision:
    affected_route = state.get_route(event.route_id)

    # Step 1: Check learned rules (PFA)
    rule_decision = pfa_policy.get_contingency_action(event, affected_route)

    if rule_decision.confidence > 0.8:
        # High-confidence rule exists (learned from past similar delays)
        return rule_decision
    else:
        # No clear rule: Re-optimize remaining route (CFA)
        remaining_stops = affected_route.get_remaining_stops()
        new_sequence = cfa_policy.re_optimize(
            stops=remaining_stops,
            current_location=event.current_location,
            current_time=event.timestamp,
            constraints=affected_route.constraints
        )
        return Decision(action="resequence", new_sequence=new_sequence)
```

### 4. Route Completion & Feedback Loop (Trigger: Driver Reports Completion)

**Policy**: Learning Phase | **Objective**: Update models with operational outcomes

```python
# Workflow
async def handle_route_completion(route: Route, feedback: OperationalFeedback):
    # Extract actuals vs predictions
    predictions = {
        'fuel_cost': route.estimated_fuel_cost,
        'duration': route.estimated_duration,
        'distance': route.estimated_distance
    }
    actuals = {
        'fuel_cost': feedback.actual_fuel_cost,
        'duration': feedback.actual_duration,
        'distance': feedback.actual_distance
    }

    # Learning Step 1: VFA - Temporal Difference update
    vfa_learner.update(
        state=route.initial_state,
        reward=compute_profit(feedback),  # Actual profit
        next_state=route.final_state
    )

    # Learning Step 2: CFA - Parameter adjustment
    for metric in ['fuel', 'duration']:
        error = actuals[f'{metric}_cost'] - predictions[f'{metric}_cost']
        cfa_learner.adjust_parameters(metric, error, route.context)

    # Learning Step 3: PFA - Pattern mining
    pattern_miner.add_outcome(feedback)
    new_patterns = pattern_miner.extract_patterns()
    for pattern in new_patterns:
        if pattern.confidence > 0.85:
            pfa_policy.add_rule(pattern)

    # Persist updated models
    save_updated_models([vfa_learner, cfa_learner, pfa_policy])
```

## Service Layer Patterns

Services in `backend/services/` follow a coordinator pattern:

```python
# Example: EventOrchestrator orchestrates the decision flow
class EventOrchestrator:
    async def handle_event(self, event: Event):
        # 1. Update state with event
        self.state_manager.apply_event(event)
        current_state = self.state_manager.get_state()

        # 2. Determine if decision needed
        if self.requires_decision(event, current_state):
            # 3. Trigger decision chain (which policy class to use?)
            decision = await self.decision_engine.make_decision(
                state=current_state,
                trigger=event
            )

            # 4. Apply decision and notify
            await self.apply_decision(decision)
            await self.broadcast_update(decision)

        # 5. Learning opportunity?
        if event.type == EventType.FEEDBACK_RECEIVED:
            await self.decision_engine.learn_from_feedback(event.data)

class RouteOptimizer:
    def optimize(self, state: SystemState, policy_hint: str = None) -> Route:
        # Policy selection logic (or override with hint)
        if policy_hint == 'CFA':
            return self.cfa_policy.optimize(state)
        elif policy_hint == 'DLA':
            return self.dla_policy.optimize(state)
        else:
            # Auto-select based on state context
            if self.is_multi_day_impact(state):
                return self.dla_policy.optimize(state)
            else:
                return self.cfa_policy.optimize(state)
```

### Powell Algorithm Integration

- **Engine** (`backend/core/powell/engine.py`): Main coordinator that selects policy class based on decision context
- **Algorithm Classes** (pfa.py, vfa.py, cfa.py, dla.py): Individual optimization strategies
- **Hybrids** (hybrids.py): Combines multiple algorithms (CFA/VFA, DLA/VFA, PFA/CFA) for complex decisions
- **Decision Context**: Engine examines state to choose optimal policy:
  - Immediate routing need → CFA (cost minimization)
  - Long-term network value? → VFA (value maximization)
  - Multi-day impact? → DLA (lookahead planning)
  - Learned patterns available? → PFA hybrid (rules-based)

### State Representation (Critical for All Decisions)

The system state `S_t` contains all information needed for optimization:

```python
@dataclass
class SystemState:
    """Complete decision context"""
    pending_orders: List[Order]  # Awaiting assignment
    fleet: List[Vehicle]  # Current vehicles + status + location
    active_routes: List[Route]  # In-progress routes
    environment: Environment  # Traffic, weather, time-based context
    learning: LearningState  # Model versions, confidence scores
    timestamp: datetime
```

**Key State Features for Algorithms**:

- **For CFA**: Order locations, vehicle capacity, traffic conditions, cost parameters θ
- **For VFA**: Fleet availability, backhaul opportunities, future demand forecast
- **For DLA**: Demand forecast for horizon days, vehicle constraints, consolidation opportunities
- **For PFA**: Historical patterns, time-of-day constraints, customer special constraints

### Learning Integration

- **Feedback Processing** (`backend/core/learning/feedback_processor.py`): Ingests operational outcomes (actual times, costs, delays, customer satisfaction)
- **TD-Learning** (`backend/core/learning/td_learning.py`): Updates VFA neural network weights using temporal difference: `V(s) ← V(s) + α[r + γV(s') - V(s)]`
- **Parameter Updates** (`backend/core/learning/parameter_update.py`): Adjusts CFA parameters (fuel model, time buffers, delay penalties) based on prediction errors
- **Pattern Mining** (`backend/core/learning/pattern_mining.py`): Discovers high-confidence decision patterns for PFA (e.g., "Eastleigh: 92% on-time if start by 8:30 AM", "Majid: 87% conflict if Wed 2PM+")
- **Learning Trigger**: Route completion events trigger all three learning processes simultaneously, model versions incremented

### WebSocket Real-time Updates

- `backend/api/routes/websocket.py`: Manages persistent client connections
- Frontend hooks `useWebSocket.js`: Subscribes to `routes`, `decisions`, `system_state` channels
- **Events Broadcasting**:
  - Route updates (current location, next stop, ETA)
  - New decisions (especially manual override prompts if human-in-loop enabled)
  - Model updates (VFA/CFA/PFA version bumps, performance improvements)
  - Performance metrics (on-time %, cost accuracy, fleet utilization)

## Operational Constraints & Business Rules

**These drive all algorithm decisions:**

### Pickup Windows (Network-Wide)

- **Eastleigh CBD**: 8:30-9:45 AM only (learned from traffic analysis, non-negotiable)
- Drives pickup sequencing: Eastleigh always first in route to hit window

### Customer-Specific Constraints

- **Majid Fresh Foods**: No Wed 2PM+ deliveries (HQ delivery conflict, learned from pattern mining)
- **Fresh Food Customers**: Priority sequencing (pickup first to minimize time sensitivity)
- **General**: Time windows respected (not violated in CFA/DLA optimization)
- Stored in `customers` table, checked in PFA rules during all decisions

### Multi-Destination Routes (Mesh Model)

- One truck may serve: Nairobi (pickup) → Nakuru → Eldoret → Kitale (deliveries)
- Sequence optimization critical: CFA/DLA must respect multi-stop consolidation
- Return journey: Currently empty, but system must support loaded backhauls (VFA decisions)

### Fleet Capacity Constraints

- **5T vehicles**: 5 tonnes max weight, 30 m³ volume
- **10T vehicles**: 10 tonnes max weight, 60 m³ volume
- CFA/DLA constraints: sum(order_weights) ≤ vehicle_capacity, sum(volumes) ≤ vehicle_volume

## Development Workflows

### Running the Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn backend.api.main:app --reload

# The system runs event loop in EventOrchestrator continuously
```

### Running the Frontend

```bash
# Install dependencies
cd frontend
npm install

# Start Vite dev server
npm run dev
# Frontend connects to ws://localhost:8000/ws
```

### Database Setup

- Migrations in `backend/db/migrations/`
- ORM models in `backend/db/models.py` (SQLAlchemy)
- Create new migration after schema changes: `alembic revision --autogenerate`
- Run migrations to sync schema before testing

### Testing Workflows

**Unit Tests (Algorithms)**:

```bash
# Test CFA optimizer
pytest backend/core/powell/test_cfa.py -v

# Test VFA TD-learning
pytest backend/core/learning/test_td_learning.py -v

# Test PFA pattern mining
pytest backend/core/learning/test_pattern_mining.py -v
```

**Integration Tests (Decision Flow)**:

```bash
# Test order arrival → decision creation → route generation
pytest backend/api/test_decisions.py::test_order_arrival_to_route -v

# Test feedback loop → model learning
pytest backend/core/learning/test_feedback_loop.py -v
```

### Adding New Features

**New Powell Algorithm:**

1. Create class in `backend/core/powell/` with base interface methods
2. Implement `optimize(state: SystemState) -> Route`
3. Register in `engine.py` decision logic or selector function
4. Add learning/parameter persistence if needed
5. Test with sample state, verify route validity

**New Decision Type:**

1. Add schema in `backend/core/models/decision.py` (Pydantic model)
2. Add API endpoint in `backend/api/routes/decisions.py` or `orders.py`
3. Update EventOrchestrator to recognize trigger event
4. Add frontend component in `frontend/src/components/`
5. Add WebSocket broadcast if real-time update needed

**New Feedback Channel:**

1. Add handler in `backend/core/learning/feedback_processor.py`
2. Create POST endpoint in `backend/api/routes/` (e.g., `/feedback/delays`)
3. Update LearningCoordinator to process new feedback type
4. Trigger learning pipeline (VFA/CFA/PFA updates)
5. Add UI form in frontend's FeedbackForm

### API Examples

**Create Order (triggers decision)**:

```bash
POST http://localhost:8000/api/v1/orders
{
    "customer_id": "CUST-123",
    "pickup_address": "Eastleigh 1st Avenue, Nairobi",
    "destination_city": "Nakuru",
    "weight_tonnes": 2.5,
    "time_window_start": "2025-01-17T08:00:00",
    "time_window_end": "2025-01-17T18:00:00"
}

Response: 201 Created
{
    "order_id": "ORD-001",
    "decision": {
        "assigned_route": "RTE-001",
        "vehicle_id": "TRK-5T-001",
        "estimated_pickup": "08:30",
        "policy_used": "CFA",
        "confidence": 0.89
    }
}
```

**Submit Feedback (triggers learning)**:

```bash
POST http://localhost:8000/api/v1/feedback
{
    "route_id": "RTE-001",
    "actual_fuel_cost": 3500,
    "actual_duration_minutes": 450,
    "delays": [{"location": "Naivasha", "duration_minutes": 20}],
    "successful_deliveries": 3,
    "failed_deliveries": 0
}

Response: 200 OK
{
    "feedback_id": "FBK-001",
    "learning_triggered": true,
    "models_updated": ["CFA", "VFA"]
}
```

## Project-Specific Conventions

### Configuration & Logging

- `backend/utils/config.py`: Centralized configuration (environment-based, e.g., SENGA_ENV=dev/prod)
- `backend/utils/logging.py`: Structured JSON logging with request_id tracing
- `backend/utils/metrics.py`: Performance metrics collection (route optimization time, decision latency, etc.)

### External API Integration

- `backend/services/external/google_places.py`: Location geocoding service (retry logic, caching)
- `backend/services/external/traffic_api.py`: Traffic data integration (error handling with fallback)
- Implement exponential backoff (3s → 6s → 12s) for all external API calls
- Cache results where appropriate (geocoding valid 7 days, traffic 5 min)

### Frontend API Client

- `frontend/src/api/client.js`: Axios instance with base URL (configurable for dev/prod)
- Hook pattern in `frontend/src/hooks/`: Custom hooks for decisions, routes, WebSocket subscriptions
- Use React Query or SWR for server state management (queries/mutations/caching)

## Critical Integration Points

1. **Order → Decision Flow**:
   - Order API endpoint → StateManager.build_state() → Engine.select_policy() → RouteOptimizer.optimize() → Decision logged
2. **Feedback → Learning Loop**:
   - Feedback endpoint → FeedbackProcessor → VFA.learn() + CFA.update_params() + PFA.mine_patterns() → Models persisted → next decision uses updated parameters
3. **Real-time Sync**:
   - WebSocket broadcasts from backend (route updates, decisions) → frontend useWebSocket hook → Dashboard re-renders
4. **External Data**:
   - RouteOptimizer queries Google Places (geocoding) + Traffic API (congestion) during CFA/DLA optimization
5. **Database Persistence**:
   - All orders, routes, decisions, feedback stored via SQLAlchemy → enables historical analysis, learning, compliance audit trail

---

**Last Updated**: PRD v1.0 integrated into instructions. Refer to `PRD.md` for full specifications on business model, success criteria, and technical architecture.

## Common Pitfalls to Avoid

### Algorithm Selection & State Validation

- ❌ **Wrong**: Always using CFA regardless of decision context
- ✅ **Right**: Check if multi-day impact exists → use DLA; if backhaul opportunity → use VFA; if pattern available → use PFA
- ❌ **Wrong**: Running VFA without complete state features (missing demand forecast)
- ✅ **Right**: Validate state has required features before selecting algorithm

### State Immutability

- ❌ **Wrong**: Mutating StateManager state directly → causes decision inconsistencies
- ✅ **Right**: StateManager state is immutable; create new instances for updates via `state.clone().apply_update(...)`

### Operational Constraints Enforcement

- ❌ **Wrong**: Ignoring Eastleigh CBD 8:30-9:45 AM window in CFA optimization
- ✅ **Right**: Add hard constraint in optimization: `if "Eastleigh" in pickup_zone: start_time in [08:30, 09:45]`
- ❌ **Wrong**: Assigning Majid deliveries on Wednesday after 2 PM
- ✅ **Right**: Check PFA rules before assignment: `if customer=="Majid" and day=="Wed" and time > 14:00: skip`

### Feedback Timing & Attribution

- ❌ **Wrong**: Recording feedback with wrong timestamp → learning from misaligned data
- ✅ **Right**: Ensure feedback timestamp ≤ actual completion time, link to original decision_id for learning attribution
- ❌ **Wrong**: Learning from feedback before route actually completes (premature learning)
- ✅ **Right**: Wait for `ROUTE_COMPLETED` event with full actual outcomes before triggering learning

### WebSocket Reliability

- ❌ **Wrong**: Assuming WebSocket always connected → frontend stale UI
- ✅ **Right**: Implement automatic reconnection in `useWebSocket.js` with exponential backoff (3s, 6s, 12s max)
- ❌ **Wrong**: Broadcasting all state changes → network overload
- ✅ **Right**: Throttle updates (max 1/100ms per channel); only broadcast deltas (changed fields)

### External API Rate Limits & Failures

- ❌ **Wrong**: Calling Google Places API once per order → hits rate limits, slow decisions
- ✅ **Right**: Cache geocoding results; batch requests; implement exponential backoff (3 retries max)
- ❌ **Wrong**: Route optimization fails silently if Traffic API down
- ✅ **Right**: Graceful fallback → use last-known traffic factors; log errors; retry asynchronously

### Model Learning & Drift

- ❌ **Wrong**: Continuously updating VFA on every route → overfits to recent noise
- ✅ **Right**: Batch learning: collect 10-20 routes, then update; validate improvements before persisting
- ❌ **Wrong**: Overwriting CFA parameters without bounds → time buffers become 0 or 999
- ✅ **Right**: Gradient update with bounds: `new_param = clip(old_param + α*error, min, max)`
- ❌ **Wrong**: PFA rule confidence never decreases → stale rules never removed
- ✅ **Right**: Prune rules with <70% confidence quarterly; require minimum support (5+ occurrences)

### Multi-Destination Route Complexity

- ❌ **Wrong**: Optimizing Nakuru + Eldoret + Kitale stops independently
- ✅ **Right**: Optimize as single multi-stop sequence in CFA; consider consolidation benefits in objective
- ❌ **Wrong**: Not accounting for travel time between cities
- ✅ **Right**: Include inter-city segments in time/cost calculations (Nakuru→Eldoret buffer, etc.)

### Backhaul VFA Decisions

- ❌ **Wrong**: Always accepting backhaul cargo (ties up capacity)
- ✅ **Right**: Use VFA to evaluate: immediate revenue vs opportunity cost of missing future high-value cargo
- ❌ **Wrong**: No fallback when VFA network offline
- ✅ **Right**: Have simple heuristic fallback (e.g., "accept if margin > 40%")

### Common Coding Errors

- ❌ **Wrong**: `for order in pending_orders: optimize_individually()` → no consolidation
- ✅ **Right**: `group_orders_by_destination() → optimize_groups_jointly()`
- ❌ **Wrong**: `decision.route.vehicle_id` without null check → KeyError
- ✅ **Right**: `decision.route.vehicle_id if decision.route else None` or use Pydantic validation
- ❌ **Wrong**: Hardcoded cost weights (fuel=1000, time=2000) → non-transferable CFA
- ✅ **Right**: Store weights in `cfa_params.json`; load via config; update via learning

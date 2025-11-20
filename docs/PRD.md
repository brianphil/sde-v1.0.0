# Senga Sequential Decision Engine (SDE)

## Product Requirements Document

---

## 1. Executive Summary

### 1.1 Product Vision

Build an intelligent Sequential Decision Engine that enables Senga's mesh logistics network through continuous, learning-based optimization of route planning, vehicle assignment, and order consolidation.

### 1.2 Core Problem

Senga operates a warehouseless mesh network (Nairobi → Nakuru/Eldoret/Kitale) requiring intelligent consolidation of orders into optimal routes. Manual planning cannot:

- Dynamically pool and route orders as they arrive
- Optimize multi-destination pickup and delivery sequences
- Learn from operational outcomes to improve predictions
- Balance immediate costs against long-term network efficiency

### 1.3 Solution

A continuous, event-driven decision engine implementing Warren Powell's Sequential Decision Analytics framework using all four policy classes (PFA, CFA, VFA, DLA) and their hybrids to make and learn from sequential decisions.

### 1.4 Success Criteria

- **Functional:** Engine makes automated routing decisions for incoming orders
- **Learning:** System improves cost/time predictions from operational feedback
- **Operational:** Generates executable route plans for operations team
- **Scalable:** Handles order arrival rate and fleet size without performance degradation

---

## 2. Product Context

### 2.1 Senga's Business Model

**Direct mesh consolidation without warehouses:**

1. Orders arrive continuously from Nairobi customers
2. Destinations: Nakuru, Eldoret, Kitale (multi-destination routes)
3. Trucks: 5T and 10T capacity
4. Current state: Empty return hauls (optimize outbound only)
5. Future capability: Intelligent backhaul utilization (system must enable, not require)

**Operational Flow:**

```
Order arrives →
  Pool by destination →
    Design pickup route (sequence customers in Nairobi zones) →
      Assign to vehicle →
        Execute pickup →
          Execute delivery (may stop at multiple towns)
```

**Key Operational Constraints:**

- Eastleigh CBD pickups: 8:30-9:45 AM only (traffic window)
- Customer-specific constraints (e.g., Majid Fresh Foods: no Wed 2PM+ deliveries)
- Multi-destination routes: One truck serves Nakuru→Eldoret→Kitale
- Return hauls: Currently empty, but system must support loaded returns when available

### 2.2 Why Powell's Framework

The mesh model creates a **sequential decision problem**:

- **Decision → Information → Decision → Information** cycle
- Today's route assignment affects tomorrow's fleet availability
- Current pickup sequence impacts delivery timing
- Operational outcomes reveal true costs/times (learning signal)

Traditional optimization (single-shot planning) fails because:

1. Decisions are made continuously (orders arrive all day)
2. Information evolves (traffic, customer availability, vehicle status)
3. Uncertainty exists (actual times ≠ predicted times)
4. Learning is essential (patterns emerge from operational data)

---

## 3. Powell's Framework - Technical Foundation

### 3.1 The Four Policy Classes

**All policies answer:** Given state S_t, what action A_t should we take?

#### 3.1.1 PFA - Policy Function Approximation

**Definition:** Direct function mapping state → action (no optimization)

**Mathematical Form:**

```
A^π(S_t | θ) = f(S_t, θ)
```

**For Senga:**

- Learned operational rules: "Start Eastleigh pickups at 8:30 AM sharp"
- Customer pattern policies: "Avoid Majid deliveries Wednesday 2PM+"
- Simple heuristics: "Prioritize fresh food customers first"

**Implementation:**

```python
class PolicyFunction:
    def __init__(self, parameters: dict):
        self.theta = parameters  # Tunable parameters

    def get_action(self, state: State) -> Action:
        # Direct mapping, no optimization
        if state.customer == "Majid" and state.day == "Wednesday" and state.time > "14:00":
            return Action(type="skip", reason="delivery_block")
        return Action(type="schedule", priority=self.compute_priority(state))

    def update_parameters(self, feedback: OperationalOutcome):
        # Learn from outcomes
        self.theta = self.learn(self.theta, feedback)
```

**Learnable via:** Supervised learning from operational outcomes, reinforcement learning from success/failure

#### 3.1.2 CFA - Cost Function Approximation

**Definition:** Parameterized optimization problem (typically deterministic)

**Mathematical Form:**

```
A^π(S_t | θ) = argmin_a C(S_t, a | θ)
subject to: constraints(S_t, a, θ)
```

**For Senga:**
Route optimization with learned parameters:

```
minimize: fuel_cost(route) + time_cost(route) + delay_penalty(route)
subject to:
  - pickup_windows[customer] (learned from patterns)
  - traffic_buffers[zone, time] (learned from history)
  - vehicle_capacity[truck_type]
  - sequence_constraints (pickup before delivery)
```

**Implementation:**

```python
from scipy.optimize import minimize

class CostFunctionPolicy:
    def __init__(self, cost_parameters: dict):
        self.theta = cost_parameters  # Learned cost weights, buffers, penalties

    def get_action(self, state: State) -> Action:
        # Formulate optimization problem
        def objective(x):
            route_sequence = decode_solution(x)
            return (
                self.theta['fuel_weight'] * fuel_cost(route_sequence) +
                self.theta['time_weight'] * time_cost(route_sequence) +
                self.theta['delay_penalty'] * expected_delay(route_sequence, self.theta['traffic_buffer'])
            )

        constraints = self.build_constraints(state)
        result = minimize(objective, x0=initial_solution, constraints=constraints)
        return Action(route=decode_solution(result.x))

    def update_parameters(self, predicted_cost: float, actual_cost: float):
        # Learn: actual costs vs predicted costs
        error = actual_cost - predicted_cost
        self.theta = gradient_update(self.theta, error)
```

**Learnable via:** Compare predicted costs to actual costs, adjust parameters (cost weights, time buffers, delay penalties)

#### 3.1.3 VFA - Value Function Approximation

**Definition:** Approximate downstream value using Bellman's equation

**Mathematical Form:**

```
A^π(S_t) = argmax_a {C(S_t, a) + γ * V̄(S^a_{t+1} | θ)}
```

Where V̄ is approximate value function (often neural network)

**For Senga:**
Strategic decisions requiring long-term value estimation:

- Accept backhaul cargo? (immediate revenue vs future opportunity cost)
- Add new route? (expansion cost vs long-term network value)
- Upgrade to 10T truck? (capital cost vs utilization benefit)

**Implementation:**

```python
import torch
import torch.nn as nn

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Value estimate
        )

    def forward(self, state_tensor):
        return self.network(state_tensor)

class VFAPolicy:
    def __init__(self, value_function: ValueFunction, gamma: float = 0.95):
        self.V = value_function
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.V.parameters())

    def get_action(self, state: State, feasible_actions: List[Action]) -> Action:
        best_action = None
        best_value = float('-inf')

        for action in feasible_actions:
            immediate_reward = self.compute_immediate_reward(state, action)
            next_state = self.simulate_transition(state, action)
            future_value = self.V(next_state.to_tensor()).item()

            total_value = immediate_reward + self.gamma * future_value
            if total_value > best_value:
                best_value = total_value
                best_action = action

        return best_action

    def learn(self, state: State, action: Action, reward: float, next_state: State):
        # Temporal Difference learning
        current_value = self.V(state.to_tensor())
        target_value = reward + self.gamma * self.V(next_state.to_tensor()).detach()

        loss = nn.MSELoss()(current_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

**Learnable via:** Temporal difference learning - compare predicted value to (reward + discounted next value)

#### 3.1.4 DLA - Direct Lookahead Approximation

**Definition:** Explicitly plan into future to make current decision

**Two Subclasses:**

**4a. Deterministic DLA (Rolling Horizon):**

```
Optimize over next H periods assuming deterministic future,
take first period's decision, re-optimize tomorrow
```

**4b. Stochastic DLA (Scenario-based):**

```
Sample/enumerate future scenarios,
optimize over scenarios,
take robust first-stage decision
```

**For Senga:**
Multi-day route planning with lookahead:

```python
class DeterministicLookahead:
    def __init__(self, horizon_days: int = 7):
        self.H = horizon_days

    def get_action(self, state: State, forecast: DemandForecast) -> Action:
        # Build deterministic model for next H days
        model = self.build_optimization_model(state, forecast, self.H)

        # Solve: minimize total cost over H days
        # subject to: fleet constraints, customer constraints, continuity
        solution = solve_milp(model)

        # Extract day 0 decision, discard rest (will re-optimize tomorrow)
        return solution.day_0_actions

    def build_optimization_model(self, state, forecast, horizon):
        # Multi-period optimization problem
        # Decision variables: route[d, r], assign[d, v, r] for d in [0..H-1]
        # Objective: sum over d of costs[d]
        # Constraints: fleet continuity, capacity, time windows
        pass

class StochasticLookahead:
    def __init__(self, horizon_days: int = 7, num_scenarios: int = 100):
        self.H = horizon_days
        self.N = num_scenarios

    def get_action(self, state: State, demand_model: DemandModel) -> Action:
        # Generate scenarios for uncertain demand
        scenarios = [demand_model.sample() for _ in range(self.N)]

        # Solve two-stage stochastic program:
        # Stage 1 (now): Choose routes/assignments
        # Stage 2 (scenarios): Recourse decisions
        # Objective: minimize stage_1_cost + E[stage_2_cost over scenarios]

        solution = solve_two_stage_sp(state, scenarios)
        return solution.first_stage_decision
```

**Learnable via:**

- Deterministic: Learn forecast parameters, buffer sizes, cost estimates
- Stochastic: Learn demand distributions, scenario generation parameters

### 3.2 Hybrid Policies (The Power Move)

**Real optimization power comes from combining classes:**

#### 3.2.1 CFA/VFA Hybrid

```python
class CFAVFAHybrid:
    def __init__(self, cost_params: dict, value_function: ValueFunction):
        self.cfa = CostFunctionPolicy(cost_params)
        self.vfa = VFAPolicy(value_function)

    def get_action(self, state: State) -> Action:
        # CFA: Optimize immediate costs
        # VFA: Evaluate downstream value

        def objective_with_value(x):
            route = decode(x)
            immediate_cost = self.cfa.compute_cost(route)
            next_state = simulate(state, route)
            future_value = self.vfa.V(next_state.to_tensor()).item()
            return immediate_cost - self.gamma * future_value  # Minimize cost, maximize value

        result = minimize(objective_with_value, x0, constraints=self.cfa.constraints)
        return decode(result.x)
```

#### 3.2.2 DLA/VFA Hybrid (Lookahead with Learned Terminal Values)

```python
class DLAVFAHybrid:
    def __init__(self, horizon: int, value_function: ValueFunction):
        self.H = horizon
        self.V = value_function

    def get_action(self, state: State) -> Action:
        # Lookahead H periods, use VFA to estimate value at horizon

        def lookahead_objective(decisions):
            total_cost = 0
            current_state = state

            for t in range(self.H):
                cost_t = compute_cost(current_state, decisions[t])
                total_cost += (self.gamma ** t) * cost_t
                current_state = transition(current_state, decisions[t])

            # Terminal value from VFA (approximates cost-to-go beyond horizon)
            terminal_value = self.V(current_state.to_tensor()).item()
            total_cost += (self.gamma ** self.H) * terminal_value

            return total_cost

        solution = optimize_over_horizon(lookahead_objective)
        return solution[0]  # First period decision
```

#### 3.2.3 PFA/CFA Hybrid (Rules with Optimization)

```python
class PFACFAHybrid:
    def __init__(self, policy_rules: PolicyFunction, optimizer: CostFunctionPolicy):
        self.pfa = policy_rules
        self.cfa = optimizer

    def get_action(self, state: State) -> Action:
        # PFA generates constraints/priorities
        constraints = self.pfa.get_constraints(state)
        priorities = self.pfa.get_priorities(state)

        # CFA optimizes subject to PFA-generated constraints
        def objective(x):
            route = decode(x)
            cost = self.cfa.compute_cost(route)
            priority_penalty = sum(priorities[c] * position[c] for c in customers)
            return cost + priority_penalty

        result = minimize(objective, x0, constraints=constraints)
        return decode(result.x)
```

### 3.3 State Space Definition

**The system state S_t contains all information needed to make decisions:**

```python
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class Order:
    order_id: str
    customer_id: str
    pickup_location: tuple[float, float]  # (lat, lon) from Google Places
    destination: str  # "Nakuru", "Eldoret", "Kitale"
    weight: float  # tonnes
    volume: float  # cubic meters
    time_window_start: datetime
    time_window_end: datetime
    priority: int
    special_constraints: Dict[str, any]  # e.g., {"fresh_food": True}

@dataclass
class Vehicle:
    vehicle_id: str
    type: str  # "5T", "10T"
    capacity_weight: float
    capacity_volume: float
    current_location: tuple[float, float]
    available_at: datetime
    assigned_route: Optional[str]
    status: str  # "available", "in_transit", "loading", "maintenance"

@dataclass
class Route:
    route_id: str
    vehicle_id: str
    orders: List[str]  # order_ids
    pickup_sequence: List[str]  # customer_ids in pickup order
    delivery_sequence: List[str]  # destination stops in order
    estimated_cost: float
    estimated_duration: float
    status: str  # "planned", "in_progress", "completed"
    created_at: datetime

@dataclass
class Environment:
    current_time: datetime
    traffic_conditions: Dict[str, float]  # zone -> congestion_factor
    weather: Dict[str, any]
    historical_patterns: Dict[str, any]  # Learned patterns

@dataclass
class LearningState:
    model_parameters: Dict[str, any]  # CFA/VFA/PFA/DLA parameters
    confidence_scores: Dict[str, float]  # Prediction confidence
    last_updated: datetime

@dataclass
class SystemState:
    """Complete state S_t at time t"""
    pending_orders: List[Order]
    fleet: List[Vehicle]
    active_routes: List[Route]
    environment: Environment
    learning: LearningState
    timestamp: datetime
```

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                          │
│  - Real-time dashboard (route visualization)                │
│  - Decision review/approval interface                       │
│  - Performance monitoring                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Python)                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              API Layer (FastAPI)                    │   │
│  │  /orders/create, /routes/get, /decisions/batch     │   │
│  │  WebSocket: /ws/updates                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Event-Driven Orchestrator                 │   │
│  │  - Listens for: Order arrival, vehicle updates,    │   │
│  │    time ticks, feedback                            │   │
│  │  - Triggers: Decision chain                        │   │
│  │  - Manages: Decision queue, state updates          │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Powell Framework Engine (Core)              │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │   PFA    │  │   CFA    │  │   VFA    │         │   │
│  │  │ (Rules)  │  │ (Optim)  │  │ (Value)  │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │       ↓             ↓             ↓                │   │
│  │  ┌─────────────────────────────────────────┐      │   │
│  │  │          DLA Coordinator                │      │   │
│  │  │  (Decides which policy class to use)    │      │   │
│  │  └─────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Services Layer                         │   │
│  │  - State Manager (current system state)            │   │
│  │  - Learning Coordinator (update models)            │   │
│  │  - Route Executor (GPS tracking, feedback)         │   │
│  │  - External APIs (Google Places, traffic)          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    SQLite Database                          │
│  - orders, vehicles, routes, customers                      │
│  - decisions (log of all decisions + outcomes)              │
│  - models (VFA neural nets, CFA parameters, PFA rules)      │
│  - operational_history (for learning)                       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Event-Driven Decision Flow

**Core Principle:** System is **reactive** to events, not scheduled batch processing.

```python
class EventType(Enum):
    ORDER_ARRIVED = "order_arrived"
    VEHICLE_AVAILABLE = "vehicle_available"
    ROUTE_COMPLETED = "route_completed"
    TIME_TICK = "time_tick"  # Periodic re-optimization
    TRAFFIC_UPDATE = "traffic_update"
    FEEDBACK_RECEIVED = "feedback_received"

class Event:
    type: EventType
    data: Dict
    timestamp: datetime

class DecisionOrchestrator:
    """
    Main event loop - listens for events and triggers decision chain
    """
    def __init__(self):
        self.state_manager = StateManager()
        self.decision_engine = PowellEngine()
        self.event_queue = asyncio.Queue()

    async def run(self):
        while True:
            event = await self.event_queue.get()
            await self.handle_event(event)

    async def handle_event(self, event: Event):
        # Update system state
        self.state_manager.apply_event(event)
        current_state = self.state_manager.get_state()

        # Determine if decision needed
        if self.requires_decision(event, current_state):
            # Trigger decision chain
            decision = await self.decision_engine.make_decision(
                state=current_state,
                trigger=event
            )

            # Apply decision
            await self.apply_decision(decision)

            # Notify UI
            await self.broadcast_update(decision)

        # Learning opportunity?
        if event.type == EventType.FEEDBACK_RECEIVED:
            await self.decision_engine.learn_from_feedback(event.data)
```

**Example Event Flows:**

**1. New Order Arrives:**

```
Order Created (API) →
  Event: ORDER_ARRIVED →
    Update State: Add to pending_orders →
      Decision Needed? YES →
        Policy Selection:
          - If order destination matches existing route → CFA (re-optimize sequence)
          - If new route needed → DLA (lookahead for multi-day impact)
          - If backhaul opportunity → VFA (long-term value assessment)
        Generate Decision →
          Apply: Assign to route, update vehicle, notify coordinator →
            Log Decision →
              WebSocket: Update UI
```

**2. Route Completed (Feedback Loop):**

```
Driver Reports Completion →
  Event: ROUTE_COMPLETED →
    Update State: Vehicle available, route done →
      Extract Actuals:
        - Actual pickup times vs predicted
        - Actual fuel cost vs predicted
        - Actual delays encountered
      Learning:
        - VFA: Update value estimates (was backhaul decision good?)
        - CFA: Update cost parameters (were time buffers accurate?)
        - PFA: Reinforce/adjust rules (did Eastleigh window work?)
      Store Outcome →
        Decision: Vehicle reassignment if pending orders exist
```

### 4.3 Module Structure

```
senga-sde/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── orders.py          # Order CRUD endpoints
│   │   │   ├── routes.py          # Route management
│   │   │   ├── decisions.py       # Decision review/approval
│   │   │   └── websocket.py       # Real-time updates
│   │   └── main.py                # FastAPI app
│   │
│   ├── core/
│   │   ├── models/
│   │   │   ├── state.py           # State space definitions
│   │   │   ├── decision.py        # Decision schemas
│   │   │   └── domain.py          # Order, Vehicle, Route, Customer
│   │   │
│   │   ├── powell/
│   │   │   ├── pfa.py             # Policy Function Approximation
│   │   │   ├── cfa.py             # Cost Function Approximation
│   │   │   ├── vfa.py             # Value Function Approximation
│   │   │   ├── dla.py             # Direct Lookahead Approximation
│   │   │   ├── hybrids.py         # CFA/VFA, DLA/VFA, PFA/CFA
│   │   │   └── engine.py          # Main Powell Engine (coordinates all)
│   │   │
│   │   └── learning/
│   │       ├── td_learning.py     # Temporal Difference for VFA
│   │       ├── parameter_update.py # CFA parameter learning
│   │       ├── pattern_mining.py  # Extract patterns for PFA
│   │       └── feedback_processor.py
│   │
│   ├── services/
│   │   ├── state_manager.py       # In-memory state + persistence
│   │   ├── event_orchestrator.py  # Event loop, decision triggers
│   │   ├── route_optimizer.py     # Optimization utilities (scipy, OR-tools)
│   │   ├── learning_coordinator.py # Manage model training
│   │   └── external/
│   │       ├── google_places.py   # Geocoding
│   │       └── traffic_api.py     # Traffic data (if available)
│   │
│   ├── db/
│   │   ├── database.py            # SQLAlchemy setup
│   │   ├── models.py              # ORM models
│   │   └── migrations/            # Schema versions
│   │
│   └── utils/
│       ├── config.py              # Configuration management
│       ├── logging.py             # Structured logging
│       └── metrics.py             # Performance tracking
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx      # Main dashboard
│   │   │   ├── RouteMap.jsx       # Visualize routes
│   │   │   ├── DecisionQueue.jsx  # Pending decisions
│   │   │   ├── PerformanceMetrics.jsx
│   │   │   └── FeedbackForm.jsx   # Operational feedback input
│   │   │
│   │   ├── hooks/
│   │   │   ├── useWebSocket.js    # Real-time connection
│   │   │   ├── useDecisions.js    # Decision state
│   │   │   └── useRoutes.js       # Route state
│   │   │
│   │   ├── api/
│   │   │   └── client.js          # API client
│   │   │
│   │   └── App.jsx
│   │
│   └── package.json
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Python project config
└── README.md
```

---

## 5. Detailed Functional Requirements

### 5.1 Core Decision Workflows

#### 5.1.1 Daily Route Planning

**Trigger:** Morning optimization (e.g., 6:00 AM) or continuous as orders accumulate

**Decision Type:** CFA/DLA Hybrid

**Process:**

```python
async def daily_route_planning(state: SystemState) -> List[Route]:
    """
    Generate optimized route plan for pending orders
    Uses CFA for base optimization + DLA for multi-day lookahead
    """

    # Get all pending orders
    orders = state.pending_orders
    available_fleet = [v for v in state.fleet if v.status == "available"]

    # Group orders by destination
    by_destination = group_orders_by_destination(orders)

    # For each destination cluster
    routes = []
    for destination, order_group in by_destination.items():

        # Check if multi-day lookahead needed
        if should_use_lookahead(order_group, state):
            # DLA: Optimize over next 7 days
            route = dla_policy.plan_route(
                orders=order_group,
                fleet=available_fleet,
                horizon=7,
                state=state
            )
        else:
            # CFA: Single-day optimization
            route = cfa_policy.optimize_route(
                orders=order_group,
                fleet=available_fleet,
                state=state
            )

        routes.append(route)

    # Consolidation check: Can routes be combined?
    routes = consolidate_routes(routes, available_fleet)

    return routes
```

**CFA Optimization Problem:**

```python
def optimize_single_route(orders: List[Order], vehicles: List[Vehicle], params: CFAParams):
    """
    Formulate and solve constrained optimization problem
    """
    from scipy.optimize import linprog, minimize
    import numpy as np

    # Decision variables:
    # x[i,j] = 1 if customer i visited before customer j
    # v[k] = 1 if vehicle k assigned to route
    # t[i] = time of arrival at customer i

    n_customers = len(orders)
    n_vehicles = len(vehicles)

    # Objective: minimize total cost
    def objective(x):
        route_sequence = decode_route(x[:n_customers**2])
        vehicle_idx = decode_vehicle(x[n_customers**2:])

        fuel = compute_fuel_cost(route_sequence, vehicles[vehicle_idx])
        time = compute_time_cost(route_sequence, params.traffic_model)
        penalty = compute_delay_penalty(route_sequence, orders, params.delay_weight)

        return fuel + time + penalty

    # Constraints
    constraints = []

    # 1. Each customer visited exactly once
    # 2. Route is connected (no sub-tours)
    # 3. Vehicle capacity not exceeded
    # 4. Time windows satisfied
    # 5. Precedence constraints (e.g., Eastleigh before other zones)

    constraints.extend(build_tsp_constraints(n_customers))
    constraints.extend(build_capacity_constraints(orders, vehicles))
    constraints.extend(build_time_window_constraints(orders, params))
    constraints.extend(build_precedence_constraints(orders, params.pfa_rules))

    # Solve
    result = minimize(
        objective,
        x0=initial_solution(orders, vehicles),
        constraints=constraints,
        method='SLSQP'
    )

    return decode_full_solution(result.x, orders, vehicles)
```

**DLA Lookahead (7-day horizon):**

```python
def lookahead_planning(current_orders: List[Order],
                       state: SystemState,
                       horizon: int = 7) -> Route:
    """
    Multi-day optimization with demand forecast
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    # Forecast demand for next 7 days (learned from historical patterns)
    forecast = state.learning.demand_forecast.predict(horizon)

    # Build multi-day optimization model
    # Days: 0 (today), 1, 2, ..., 6
    # Decision: route[d], assign[d, v] for each day d
    # Objective: sum over d of cost[d] + terminal_value[state_7]

    # Use VFA for terminal value at horizon
    def terminal_value(state_at_day_7):
        return vfa_policy.V(state_at_day_7.to_tensor()).item()

    # Solve multi-stage optimization
    # Return only day 0 decision (will re-optimize tomorrow with new info)
    solution = solve_multistage_vrp(
        days=range(horizon),
        orders_by_day=[current_orders] + forecast,
        fleet=state.fleet,
        terminal_value_fn=terminal_value
    )

    return solution.day_0_route
```

#### 5.1.2 Order Acceptance Decision

**Trigger:** New order arrives

**Decision Type:** VFA (if backhaul) or CFA (if standard)

**Process:**

```python
async def handle_new_order(order: Order, state: SystemState) -> Decision:
    """
    Decide: Accept order? If yes, assign to which route?
    """

    # Is this a backhaul opportunity (return trip)?
    if order.is_backhaul():
        # Use VFA: Long-term network value vs immediate revenue
        decision = vfa_policy.evaluate_backhaul(order, state)

        if decision.accept:
            # Assign to return trip
            route = assign_to_return_trip(order, state.fleet)
        else:
            # Decline (opportunity cost too high)
            route = None

    else:
        # Standard order: Can we fit it into existing route?
        candidate_routes = find_compatible_routes(order, state.active_routes)

        if candidate_routes:
            # CFA: Optimize insertion (minimize cost increase)
            best_route = cfa_policy.optimize_insertion(
                order=order,
                candidates=candidate_routes,
                state=state
            )
        else:
            # Need new route: Use DLA to check if economical
            best_route = dla_policy.evaluate_new_route(order, state)

    return Decision(
        order_id=order.order_id,
        action="accept" if best_route else "defer",
        route_id=best_route.route_id if best_route else None,
        reasoning=decision.explanation
    )
```

**VFA Backhaul Evaluation:**

```python
class BackhaulVFA:
    def __init__(self, value_network: ValueFunction):
        self.V = value_network

    def evaluate_backhaul(self, order: Order, state: SystemState) -> Decision:
        """
        Accept backhaul? Compare immediate revenue to opportunity cost
        """

        # Immediate reward: Revenue from backhaul
        immediate_revenue = order.price

        # Opportunity cost: What if we leave truck empty for future high-value cargo?
        # Compute via VFA

        # State if we accept backhaul
        state_accept = state.clone()
        state_accept.assign_backhaul(order)

        # State if we decline (truck returns empty but available sooner)
        state_decline = state.clone()
        state_decline.return_empty(order.vehicle_id)

        # Compare values
        value_accept = immediate_revenue + self.gamma * self.V(state_accept.to_tensor()).item()
        value_decline = self.V(state_decline.to_tensor()).item()

        return Decision(
            accept=(value_accept > value_decline),
            value_accept=value_accept,
            value_decline=value_decline,
            explanation=f"Accept value: {value_accept:.2f}, Decline value: {value_decline:.2f}"
        )
```

#### 5.1.3 Real-Time Route Adjustment

**Trigger:** Delay detected, vehicle breakdown, customer unavailable

**Decision Type:** PFA + CFA hybrid

**Process:**

```python
async def handle_delay(event: DelayEvent, state: SystemState) -> Decision:
    """
    Real-time adjustment: What to do when route is delayed?
    """

    affected_route = state.get_route(event.route_id)

    # PFA: Check learned rules for this scenario
    rule_decision = pfa_policy.get_contingency_action(event, affected_route)

    if rule_decision.confidence > 0.8:
        # High confidence rule exists (learned from past)
        return rule_decision

    else:
        # No clear rule: Use CFA to re-optimize remaining route
        remaining_stops = affected_route.get_remaining_stops()

        # Re-optimize sequence + timing
        new_sequence = cfa_policy.re_optimize(
            stops=remaining_stops,
            current_location=event.current_location,
            current_time=event.timestamp,
            constraints=affected_route.constraints
        )

        return Decision(
            action="resequence",
            new_sequence=new_sequence,
            reasoning="Re-optimized due to delay"
        )
```

### 5.2 Learning Workflows

#### 5.2.1 VFA Learning (Temporal Difference)

**Trigger:** Route completed (feedback received)

**Process:**

```python
async def learn_from_route_completion(route: Route, actual_outcome: Outcome):
    """
    Update value function based on actual outcomes
    """

    # Extract experience tuple
    state_before = route.initial_state
    action = route.decisions_made
    reward = compute_reward(actual_outcome)  # Profit - costs - penalties
    state_after = route.final_state

    # TD update: V(s) ← V(s) + α[r + γV(s') - V(s)]
    vfa_learner.update(
        state=state_before,
        action=action,
        reward=reward,
        next_state=state_after
    )

    # Persist updated model
    await save_model(vfa_learner.V, "value_function_v{version}.pt")
```

**Implementation:**

```python
class VFALearner:
    def __init__(self, value_function: ValueFunction, learning_rate: float = 0.01, gamma: float = 0.95):
        self.V = value_function
        self.alpha = learning_rate
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.V.parameters(), lr=learning_rate)

    def update(self, state: SystemState, action: Action, reward: float, next_state: SystemState):
        """
        Temporal Difference learning update
        """
        # Convert to tensors
        s_tensor = state.to_tensor()
        s_next_tensor = next_state.to_tensor()

        # Current value estimate
        v_current = self.V(s_tensor)

        # TD target: r + γ * V(s')
        with torch.no_grad():
            v_next = self.V(s_next_tensor)
        td_target = reward + self.gamma * v_next

        # TD error
        td_error = td_target - v_current

        # Update via gradient descent
        loss = td_error ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log
        logger.info(f"VFA Update: reward={reward:.2f}, v_current={v_current.item():.2f}, "
                   f"v_next={v_next.item():.2f}, td_error={td_error.item():.2f}")
```

#### 5.2.2 CFA Learning (Parameter Adjustment)

**Trigger:** Route completed (compare predicted vs actual costs)

**Process:**

```python
async def learn_cost_parameters(route: Route, actual_outcome: Outcome):
    """
    Adjust CFA cost model parameters based on prediction errors
    """

    # Extract predictions vs actuals
    predicted_fuel = route.estimated_fuel_cost
    actual_fuel = actual_outcome.fuel_cost

    predicted_time = route.estimated_duration
    actual_time = actual_outcome.actual_duration

    # Compute errors
    fuel_error = actual_fuel - predicted_fuel
    time_error = actual_time - predicted_time

    # Update parameters
    cfa_learner.update_fuel_model(fuel_error, route.context)
    cfa_learner.update_time_model(time_error, route.context)
    cfa_learner.update_traffic_buffers(time_error, route.traffic_conditions)

    # Persist
    await save_parameters(cfa_learner.params, "cfa_params_v{version}.json")
```

**Implementation:**

```python
class CFAParameterLearner:
    def __init__(self, initial_params: dict):
        self.params = initial_params
        # params = {
        #   'fuel_per_km': 0.15,  # liters/km
        #   'fuel_price': 150,    # KES/liter
        #   'time_buffer_eastleigh': 15,  # minutes
        #   'delay_penalty_weight': 5000,  # KES/hour
        #   ...
        # }

    def update_fuel_model(self, error: float, context: dict):
        """
        Adjust fuel cost estimation
        error = actual - predicted (positive means underestimated)
        """
        # Simple gradient update
        route_distance = context['distance_km']
        fuel_per_km_error = error / route_distance / self.params['fuel_price']

        self.params['fuel_per_km'] += 0.01 * fuel_per_km_error  # Small learning rate

        # Bound check
        self.params['fuel_per_km'] = max(0.1, min(0.3, self.params['fuel_per_km']))

    def update_traffic_buffers(self, time_error: float, traffic_conditions: dict):
        """
        Adjust time buffers for different zones based on actual delays
        """
        for zone, congestion in traffic_conditions.items():
            if time_error > 10:  # Underestimated by >10 minutes
                # Increase buffer
                buffer_key = f'time_buffer_{zone}'
                self.params[buffer_key] = self.params.get(buffer_key, 10) + 2
            elif time_error < -10:  # Overestimated
                # Decrease buffer (but keep minimum)
                buffer_key = f'time_buffer_{zone}'
                self.params[buffer_key] = max(5, self.params.get(buffer_key, 10) - 1)
```

#### 5.2.3 PFA Learning (Pattern Mining)

**Trigger:** Periodic (e.g., weekly) or after N route completions

**Process:**

```python
async def mine_operational_patterns():
    """
    Extract patterns from operational history to create/update PFA rules
    """

    # Get recent operational data
    recent_outcomes = await db.query(
        "SELECT * FROM operational_history WHERE timestamp > ?",
        (datetime.now() - timedelta(days=30),)
    )

    # Mine patterns
    patterns = pattern_miner.extract_patterns(recent_outcomes)

    # Examples of patterns discovered:
    # - "Eastleigh CBD: 92% on-time if start by 8:30 AM"
    # - "Majid Fresh Foods: 87% conflict if Wednesday 2PM+"
    # - "Nakuru route: Average 15 min delay at Naivasha on Fridays"

    # Convert to PFA rules
    for pattern in patterns:
        if pattern.confidence > 0.85:
            pfa_policy.add_rule(pattern)

    # Prune low-confidence rules
    pfa_policy.prune_rules(confidence_threshold=0.7)

    # Persist
    await save_rules(pfa_policy.rules, "pfa_rules_v{version}.json")
```

**Implementation:**

```python
class PatternMiner:
    def extract_patterns(self, outcomes: List[OperationalOutcome]) -> List[Pattern]:
        """
        Mine frequent patterns from operational data
        """
        patterns = []

        # Customer time window patterns
        customer_patterns = self.mine_customer_patterns(outcomes)
        patterns.extend(customer_patterns)

        # Traffic patterns
        traffic_patterns = self.mine_traffic_patterns(outcomes)
        patterns.extend(traffic_patterns)

        # Seasonal patterns
        seasonal_patterns = self.mine_seasonal_patterns(outcomes)
        patterns.extend(seasonal_patterns)

        return patterns

    def mine_customer_patterns(self, outcomes):
        """
        Example: Detect customer availability windows
        """
        from collections import defaultdict

        # Group by customer
        by_customer = defaultdict(list)
        for outcome in outcomes:
            by_customer[outcome.customer_id].append(outcome)

        patterns = []
        for customer_id, customer_outcomes in by_customer.items():
            # Check for day-of-week patterns
            by_day = defaultdict(list)
            for o in customer_outcomes:
                day = o.timestamp.strftime('%A')
                by_day[day].append(o)

            # Detect blocked times
            for day, day_outcomes in by_day.items():
                failed_deliveries = [o for o in day_outcomes if o.delivery_failed]
                if len(failed_deliveries) > 3:
                    # Pattern detected
                    blocked_hours = [o.timestamp.hour for o in failed_deliveries]
                    avg_blocked_hour = sum(blocked_hours) / len(blocked_hours)

                    patterns.append(Pattern(
                        type="customer_availability",
                        customer_id=customer_id,
                        day=day,
                        blocked_after_hour=int(avg_blocked_hour),
                        confidence=len(failed_deliveries) / len(day_outcomes),
                        support=len(failed_deliveries)
                    ))

        return patterns
```

### 5.3 Data Requirements

#### 5.3.1 Input Data (Required for Operation)

**Orders:**

```python
{
    "order_id": "ORD-20250116-001",
    "customer_id": "CUST-123",
    "customer_name": "Majid Fresh Foods",
    "pickup_address": "Eastleigh 1st Avenue, Nairobi",  # Google Places geocoding
    "pickup_location": {"lat": -1.2833, "lon": 36.8333},
    "destination_city": "Nakuru",
    "destination_address": "...",
    "destination_location": {"lat": -0.3031, "lon": 36.0800},
    "weight_tonnes": 2.5,
    "volume_m3": 8.0,
    "time_window_start": "2025-01-17T08:00:00",
    "time_window_end": "2025-01-17T18:00:00",
    "priority": "high",
    "special_handling": ["fresh_food", "fragile"],
    "price_kes": 15000,
    "created_at": "2025-01-16T14:23:00"
}
```

**Fleet:**

```python
{
    "vehicle_id": "TRK-5T-001",
    "type": "5T",
    "capacity_weight_tonnes": 5.0,
    "capacity_volume_m3": 30.0,
    "fuel_efficiency_km_per_liter": 8.5,
    "current_location": {"lat": -1.2921, "lon": 36.8219},  # Nairobi depot
    "status": "available",
    "available_at": "2025-01-17T06:00:00",
    "maintenance_due": "2025-02-01",
    "driver_id": "DRV-042"
}
```

**Customers (Profile):**

```python
{
    "customer_id": "CUST-123",
    "name": "Majid Fresh Foods",
    "locations": [
        {"address": "...", "lat": ..., "lon": ..., "type": "pickup"},
        {"address": "...", "lat": ..., "lon": ..., "type": "delivery"}
    ],
    "constraints": {
        "delivery_blocked": [
            {"day": "Wednesday", "time_start": "14:00", "time_end": "15:30", "reason": "HQ deliveries"}
        ],
        "preferred_windows": [
            {"days": ["Tuesday", "Thursday"], "time_start": "13:30", "time_end": "14:00"}
        ]
    },
    "priority_level": "high",
    "fresh_food_customer": true
}
```

#### 5.3.2 Historical Data (Required for Learning)

**Operational History:**

```sql
CREATE TABLE operational_history (
    id INTEGER PRIMARY KEY,
    route_id TEXT,
    vehicle_id TEXT,
    date DATE,

    -- Predicted values (from decision engine)
    predicted_fuel_cost REAL,
    predicted_duration_minutes INTEGER,
    predicted_distance_km REAL,

    -- Actual values (from execution)
    actual_fuel_cost REAL,
    actual_duration_minutes INTEGER,
    actual_distance_km REAL,

    -- Performance
    on_time BOOLEAN,
    delay_minutes INTEGER,

    -- Context
    traffic_conditions JSON,  -- {"eastleigh": 0.8, "thika_road": 1.2}
    weather TEXT,
    day_of_week TEXT,

    -- Outcomes
    successful_deliveries INTEGER,
    failed_deliveries INTEGER,
    customer_satisfaction_score REAL,

    created_at TIMESTAMP
);
```

**Decision Log:**

```sql
CREATE TABLE decision_log (
    id INTEGER PRIMARY KEY,
    decision_id TEXT UNIQUE,
    timestamp TIMESTAMP,

    -- Context
    trigger_event TEXT,  -- "order_arrived", "route_completed", etc.
    system_state JSON,   -- Snapshot of state when decision made

    -- Decision
    policy_class TEXT,  -- "CFA", "VFA", "DLA", "PFA", "CFA/VFA"
    action_type TEXT,   -- "assign_route", "accept_backhaul", etc.
    action_details JSON,

    -- Reasoning
    confidence REAL,
    parameters_used JSON,  -- Which model parameters were used

    -- Outcome (updated later)
    executed BOOLEAN,
    overridden BOOLEAN,
    override_reason TEXT,
    actual_outcome JSON,

    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

#### 5.3.3 Model Storage

**Value Functions (VFA):**

```python
# PyTorch model saved as .pt file
{
    "model_state_dict": ...,  # Neural network weights
    "optimizer_state_dict": ...,
    "version": "v1.2.3",
    "training_history": {
        "episodes": 10000,
        "avg_td_error": 0.05,
        "last_updated": "2025-01-16T12:00:00"
    },
    "architecture": {
        "input_dim": 128,
        "hidden_layers": [128, 64],
        "output_dim": 1
    }
}
```

**Cost Parameters (CFA):**

```json
{
  "version": "v2.1.0",
  "fuel_model": {
    "fuel_per_km_liters": 0.15,
    "fuel_price_kes_per_liter": 150.0,
    "confidence": 0.92
  },
  "time_model": {
    "base_speed_kmh": 45,
    "traffic_buffers": {
      "eastleigh_morning": 15,
      "thika_road_evening": 25,
      "default": 10
    }
  },
  "delay_penalties": {
    "late_delivery_kes_per_hour": 5000,
    "customer_dissatisfaction_penalty": 10000
  },
  "last_updated": "2025-01-16T08:00:00",
  "learning_rate": 0.01
}
```

**Policy Rules (PFA):**

```json
{
  "version": "v1.5.2",
  "rules": [
    {
      "rule_id": "R001",
      "type": "timing_constraint",
      "condition": {
        "zone": "Eastleigh CBD",
        "time_of_day": "morning"
      },
      "action": {
        "start_by": "08:30",
        "latest_start": "08:35"
      },
      "confidence": 0.92,
      "support": 156,
      "learned_from": "traffic_pattern_analysis",
      "created_at": "2025-01-10"
    },
    {
      "rule_id": "R002",
      "type": "customer_constraint",
      "condition": {
        "customer_id": "CUST-123",
        "day_of_week": "Wednesday",
        "time_after": "14:00"
      },
      "action": {
        "avoid_delivery": true,
        "reason": "HQ_delivery_block"
      },
      "confidence": 0.87,
      "support": 34,
      "learned_from": "customer_pattern_mining",
      "created_at": "2025-01-12"
    }
  ],
  "last_updated": "2025-01-15T18:00:00"
}
```

### 5.4 External Integrations

#### 5.4.1 Google Places API (Address Geocoding)

**Purpose:** Convert addresses to coordinates for routing

```python
import googlemaps

class GooglePlacesService:
    def __init__(self, api_key: str):
        self.client = googlemaps.Client(key=api_key)

    async def geocode_address(self, address: str) -> dict:
        """
        Convert address to lat/lon coordinates
        """
        try:
            result = self.client.geocode(address)
            if result:
                location = result[0]['geometry']['location']
                return {
                    'lat': location['lat'],
                    'lon': location['lng'],
                    'formatted_address': result[0]['formatted_address'],
                    'place_id': result[0]['place_id']
                }
            else:
                raise ValueError(f"No results for address: {address}")
        except Exception as e:
            logger.error(f"Geocoding failed: {e}")
            raise

    async def reverse_geocode(self, lat: float, lon: float) -> dict:
        """
        Convert coordinates to address
        """
        result = self.client.reverse_geocode((lat, lon))
        if result:
            return {
                'address': result[0]['formatted_address'],
                'place_id': result[0]['place_id']
            }
        return None
```

#### 5.4.2 Traffic Data (Optional - Google Maps Traffic API or similar)

**Purpose:** Real-time traffic conditions for better time estimates

```python
class TrafficService:
    def __init__(self, api_key: str):
        self.client = googlemaps.Client(key=api_key)

    async def get_traffic_conditions(self, origin: tuple, destination: tuple) -> dict:
        """
        Get current traffic conditions between two points
        """
        result = self.client.directions(
            origin=origin,
            destination=destination,
            mode="driving",
            departure_time="now"
        )

        if result:
            leg = result[0]['legs'][0]
            return {
                'distance_km': leg['distance']['value'] / 1000,
                'duration_minutes': leg['duration']['value'] / 60,
                'duration_in_traffic_minutes': leg.get('duration_in_traffic', {}).get('value', 0) / 60,
                'traffic_factor': leg.get('duration_in_traffic', {}).get('value', 1) / leg['duration']['value']
            }
        return None
```

---

## 6. Non-Functional Requirements

### 6.1 Performance

**Response Times:**

- API endpoints: < 200ms (p95)
- Route optimization (CFA): < 30 seconds for 50 orders
- Lookahead planning (DLA): < 60 seconds for 7-day horizon
- VFA inference: < 100ms per decision
- WebSocket latency: < 50ms for state updates

**Throughput:**

- Handle 100 order arrivals per hour
- Support 20 concurrent vehicles
- Process 10 decision evaluations per second

**Resource Constraints:**

- Memory: < 2GB for SDE engine
- CPU: Optimize for single-core initially (multi-core for DLA lookahead)
- Database: SQLite handles up to 1M decision log entries

### 6.2 Scalability

**MVP Target:**

- 3-5 routes simultaneously active
- 50-100 orders per day
- 10-20 vehicles in fleet

**Growth Path:**

```
SQLite → PostgreSQL (when > 100K decisions/month)
Single server → Load balanced (when > 10 req/sec)
Sync optimization → Distributed workers (when DLA > 2 min)
```

### 6.3 Reliability

**Error Handling:**

- All optimization failures → fallback to simple heuristic (CFA → PFA)
- API failures → retry with exponential backoff
- Model inference errors → use last known good parameters
- Database errors → log and alert, continue with in-memory state

**Data Integrity:**

- Every decision logged before execution
- State snapshots every 15 minutes
- Daily backup of SQLite database

**Availability:**

- Target: 99% uptime during operational hours (6 AM - 10 PM EAT)
- Graceful degradation: If optimization fails, use rule-based fallback

### 6.4 Security

**Authentication:**

- API key authentication for external access
- JWT tokens for frontend ↔ backend

**Data Privacy:**

- Customer data encrypted at rest (SQLite encryption extension)
- PII (customer names, addresses) access-controlled

**Audit Trail:**

- All decisions logged with timestamp, user, parameters
- Model updates tracked with version history

### 6.5 Monitoring & Observability

**Metrics to Track:**

```python
# Performance metrics
route_optimization_duration_seconds
decision_latency_milliseconds
api_request_duration_seconds

# Business metrics
orders_processed_per_hour
routes_created_per_day
fleet_utilization_percentage
on_time_delivery_percentage

# Learning metrics
vfa_td_error
cfa_prediction_error (fuel, time)
pfa_rule_confidence
model_version

# System metrics
cpu_usage_percentage
memory_usage_mb
database_size_mb
active_websocket_connections
```

**Logging:**

- Structured JSON logs
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Contextual logging (include decision_id, route_id in all logs)

---

## 7. API Specifications

### 7.1 REST Endpoints

**Base URL:** `http://localhost:8000/api/v1`

#### Orders

```python
# Create order
POST /orders
{
    "customer_id": "CUST-123",
    "pickup_address": "Eastleigh 1st Avenue, Nairobi",
    "destination_city": "Nakuru",
    "weight_tonnes": 2.5,
    "volume_m3": 8.0,
    "time_window_start": "2025-01-17T08:00:00",
    "time_window_end": "2025-01-17T18:00:00",
    "priority": "high",
    "special_handling": ["fresh_food"]
}

Response: 201 Created
{
    "order_id": "ORD-20250116-001",
    "status": "pending",
    "decision": {
        "assigned_route": "RTE-001",
        "vehicle_id": "TRK-5T-001",
        "estimated_pickup": "2025-01-17T08:30:00",
        "confidence": 0.89
    }
}

# Get order
GET /orders/{order_id}

# List orders
GET /orders?status=pending&destination=Nakuru&limit=50
```

#### Routes

```python
# Get route
GET /routes/{route_id}

Response: 200 OK
{
    "route_id": "RTE-001",
    "vehicle_id": "TRK-5T-001",
    "status": "in_progress",
    "orders": ["ORD-001", "ORD-002", "ORD-003"],
    "pickup_sequence": [
        {"customer_id": "CUST-123", "address": "...", "eta": "08:30"},
        {"customer_id": "CUST-456", "address": "...", "eta": "09:15"}
    ],
    "delivery_sequence": [
        {"city": "Nakuru", "orders": ["ORD-001", "ORD-003"], "eta": "12:30"},
        {"city": "Eldoret", "orders": ["ORD-002"], "eta": "15:00"}
    ],
    "estimated_cost": 25000,
    "estimated_duration_minutes": 420,
    "created_at": "2025-01-17T06:00:00"
}

# List active routes
GET /routes?status=active

# Update route (manual override)
PATCH /routes/{route_id}
{
    "pickup_sequence": [...],  # Manual reordering
    "notes": "Customer requested earlier pickup"
}
```

#### Decisions

```python
# Trigger decision evaluation (manual)
POST /decisions/evaluate
{
    "decision_type": "route_planning",
    "context": {
        "orders": ["ORD-001", "ORD-002"],
        "force_policy": "DLA"  # Optional: force specific policy class
    }
}

Response: 200 OK
{
    "decision_id": "DEC-20250117-042",
    "policy_used": "DLA/VFA",
    "recommendation": {
        "action": "create_route",
        "route_plan": {...},
        "expected_cost": 25000,
        "expected_value": 35000
    },
    "confidence": 0.91,
    "alternatives": [...]  # Other considered options
}

# Get decision history
GET /decisions?start_date=2025-01-01&policy=VFA&limit=100

# Approve/reject decision (human-in-loop)
POST /decisions/{decision_id}/approve
POST /decisions/{decision_id}/reject
{
    "reason": "Customer called, changed time window"
}
```

#### Feedback

```python
# Submit operational feedback
POST /feedback
{
    "route_id": "RTE-001",
    "completed_at": "2025-01-17T14:30:00",
    "actual_fuel_cost": 3500,
    "actual_duration_minutes": 450,  # 30 min delay
    "delays": [
        {"location": "Naivasha", "duration_minutes": 20, "reason": "traffic"},
        {"customer_id": "CUST-456", "duration_minutes": 10, "reason": "customer_unavailable"}
    ],
    "successful_deliveries": 3,
    "failed_deliveries": 0,
    "notes": "Heavy traffic on Nakuru road"
}

Response: 200 OK
{
    "feedback_id": "FBK-001",
    "learning_triggered": true,
    "models_updated": ["CFA", "VFA", "PFA"]
}
```

#### System State

```python
# Get current system state snapshot
GET /system/state

Response: 200 OK
{
    "timestamp": "2025-01-17T10:00:00",
    "pending_orders": 12,
    "active_routes": 5,
    "available_vehicles": 3,
    "fleet_utilization": 0.72,
    "decision_queue_size": 2,
    "learning": {
        "vfa_version": "v1.2.3",
        "cfa_version": "v2.1.0",
        "pfa_version": "v1.5.2",
        "last_model_update": "2025-01-17T08:00:00"
    }
}

# Get performance metrics
GET /system/metrics?period=7d

Response: 200 OK
{
    "period": {"start": "2025-01-10", "end": "2025-01-17"},
    "orders_processed": 450,
    "routes_created": 65,
    "on_time_percentage": 87.3,
    "average_cost_accuracy": 0.92,  # Predicted vs actual
    "average_time_accuracy": 0.88,
    "fleet_utilization": 0.68,
    "decisions_by_policy": {
        "CFA": 120,
        "VFA": 15,
        "DLA": 8,
        "PFA": 45,
        "CFA/VFA": 22
    }
}
```

### 7.2 WebSocket Endpoints

**Connection:** `ws://localhost:8000/ws`

**Message Types:**

```python
# Client → Server: Subscribe to updates
{
    "type": "subscribe",
    "channels": ["routes", "decisions", "system_state"]
}

# Server → Client: Route update
{
    "type": "route_update",
    "data": {
        "route_id": "RTE-001",
        "status": "in_progress",
        "current_location": {"lat": -1.1, "lon": 36.9},
        "next_stop": "CUST-456",
        "eta_next_stop": "09:15"
    }
}

# Server → Client: Decision notification
{
    "type": "decision_created",
    "data": {
        "decision_id": "DEC-042",
        "type": "route_adjustment",
        "requires_approval": true,
        "expires_at": "2025-01-17T09:30:00"
    }
}

# Server → Client: Learning update
{
    "type": "model_updated",
    "data": {
        "model": "VFA",
        "version": "v1.2.4",
        "performance_improvement": 0.03,
        "updated_at": "2025-01-17T10:00:00"
    }
}
```

---

## 8. Database Schema

### 8.1 Core Tables

```sql
-- Orders
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    pickup_address TEXT,
    pickup_lat REAL,
    pickup_lon REAL,
    destination_city TEXT,
    destination_address TEXT,
    destination_lat REAL,
    destination_lon REAL,
    weight_tonnes REAL,
    volume_m3 REAL,
    time_window_start TIMESTAMP,
    time_window_end TIMESTAMP,
    priority TEXT,
    special_handling JSON,  -- ["fresh_food", "fragile"]
    price_kes REAL,
    status TEXT,  -- "pending", "assigned", "in_transit", "delivered", "cancelled"
    assigned_route_id TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (assigned_route_id) REFERENCES routes(route_id)
);

-- Vehicles
CREATE TABLE vehicles (
    vehicle_id TEXT PRIMARY KEY,
    type TEXT,  -- "5T", "10T"
    capacity_weight_tonnes REAL,
    capacity_volume_m3 REAL,
    fuel_efficiency_km_per_liter REAL,
    current_lat REAL,
    current_lon REAL,
    status TEXT,  -- "available", "in_transit", "loading", "maintenance"
    available_at TIMESTAMP,
    driver_id TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Routes
CREATE TABLE routes (
    route_id TEXT PRIMARY KEY,
    vehicle_id TEXT NOT NULL,
    status TEXT,  -- "planned", "in_progress", "completed", "cancelled"
    orders JSON,  -- ["ORD-001", "ORD-002", ...]
    pickup_sequence JSON,  -- [{customer_id, address, lat, lon, eta}, ...]
    delivery_sequence JSON,  -- [{city, orders[], eta}, ...]
    estimated_cost REAL,
    estimated_duration_minutes INTEGER,
    estimated_distance_km REAL,
    actual_cost REAL,
    actual_duration_minutes INTEGER,
    actual_distance_km REAL,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
);

-- Customers
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT,
    locations JSON,  -- [{address, lat, lon, type: "pickup"|"delivery"}, ...]
    constraints JSON,  -- {delivery_blocked: [...], preferred_windows: [...]}
    priority_level TEXT,
    fresh_food_customer BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 8.2 Decision & Learning Tables

```sql
-- Decision log
CREATE TABLE decision_log (
    decision_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP,
    trigger_event TEXT,
    system_state JSON,
    policy_class TEXT,  -- "PFA", "CFA", "VFA", "DLA", "CFA/VFA", etc.
    action_type TEXT,
    action_details JSON,
    confidence REAL,
    parameters_used JSON,
    executed BOOLEAN DEFAULT FALSE,
    overridden BOOLEAN DEFAULT FALSE,
    override_reason TEXT,
    actual_outcome JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Operational history
CREATE TABLE operational_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id TEXT NOT NULL,
    vehicle_id TEXT NOT NULL,
    date DATE,
    predicted_fuel_cost REAL,
    predicted_duration_minutes INTEGER,
    predicted_distance_km REAL,
    actual_fuel_cost REAL,
    actual_duration_minutes INTEGER,
    actual_distance_km REAL,
    on_time BOOLEAN,
    delay_minutes INTEGER,
    traffic_conditions JSON,
    weather TEXT,
    day_of_week TEXT,
    successful_deliveries INTEGER,
    failed_deliveries INTEGER,
    created_at TIMESTAMP,
    FOREIGN KEY (route_id) REFERENCES routes(route_id),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
);

-- Model versions
CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT,  -- "VFA", "CFA", "PFA"
    version TEXT,
    parameters JSON,  -- For CFA/PFA
    model_path TEXT,  -- For VFA (PyTorch .pt file path)
    performance_metrics JSON,
    created_at TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);

-- Feedback
CREATE TABLE feedback (
    feedback_id TEXT PRIMARY KEY,
    route_id TEXT NOT NULL,
    completed_at TIMESTAMP,
    actual_fuel_cost REAL,
    actual_duration_minutes INTEGER,
    delays JSON,  -- [{location, duration_minutes, reason}, ...]
    successful_deliveries INTEGER,
    failed_deliveries INTEGER,
    notes TEXT,
    learning_triggered BOOLEAN,
    models_updated JSON,  -- ["CFA", "VFA"]
    created_at TIMESTAMP,
    FOREIGN KEY (route_id) REFERENCES routes(route_id)
);
```

---

## 9. Implementation Phases

### Phase 1: Foundation (Weeks 1-3)

**Goal:** Basic system that can accept orders, run CFA optimization, generate routes

**Deliverables:**

1. FastAPI backend scaffold

   - API routes (orders, routes, vehicles)
   - SQLite database + SQLAlchemy models
   - Basic CRUD operations

2. State management

   - SystemState data model
   - In-memory state cache
   - State persistence

3. CFA Implementation (Priority #1)

   - Route optimization using scipy
   - Constraint handling (time windows, capacity)
   - Simple cost model (distance-based)

4. React dashboard (basic)
   - Order creation form
   - Route visualization (map)
   - Active routes list

**Success Criteria:**

- Create order via API → CFA optimizes → Route created → Visible in dashboard
- Can handle 10 orders, generate 3 routes

---

### Phase 2: Learning & PFA (Weeks 4-6)

**Goal:** System learns from outcomes, implements pattern-based rules

**Deliverables:**

1. PFA Implementation

   - Rule engine (time windows, customer constraints)
   - Pattern mining from operational history
   - Rule confidence scoring

2. CFA Learning

   - Parameter adjustment based on feedback
   - Cost model refinement (fuel, time, delays)
   - Traffic buffer learning

3. Feedback system

   - Route completion API
   - Actual vs predicted comparison
   - Learning triggers

4. Enhanced dashboard
   - Performance metrics
   - Prediction accuracy charts
   - Rule visualization

**Success Criteria:**

- Complete 20 routes → System learns patterns → CFA predictions improve
- PFA rules automatically generated from patterns
- Cost prediction error < 15%

---

### Phase 3: VFA & Strategic Decisions (Weeks 7-9)

**Goal:** Add value-based decisions for strategic choices

**Deliverables:**

1. VFA Implementation

   - Neural network value function (PyTorch)
   - Temporal difference learning
   - State encoding

2. Backhaul decision support

   - VFA evaluates long-term value
   - Accept/reject recommendations
   - Value tracking over time

3. Strategic planning features
   - Fleet expansion analysis
   - New route evaluation
   - Capacity planning

**Success Criteria:**

- System makes backhaul accept/reject decisions
- Value function converges (TD error < 0.1)
- Strategic recommendations generated

---

### Phase 4: DLA & Multi-Day Planning (Weeks 10-12)

**Goal:** Lookahead optimization for multi-day planning

**Deliverables:**

1. DLA Implementation

   - Deterministic lookahead (7-day horizon)
   - Rolling horizon optimization
   - Demand forecasting

2. DLA/VFA Hybrid

   - Terminal value from VFA
   - Integrated multi-day optimization

3. Advanced dashboard
   - Multi-day plan visualization
   - Scenario comparison
   - What-if analysis

**Success Criteria:**

- Generate 7-day lookahead plans
- DLA decisions improve long-term outcomes vs greedy CFA
- System handles 50+ orders/day

---

### Phase 5: Production Readiness (Weeks 13-16)

**Goal:** Hardening, monitoring, deployment

**Deliverables:**

1. Performance optimization

   - Caching strategies
   - Query optimization
   - Async processing for heavy tasks

2. Monitoring & observability

   - Prometheus metrics
   - Grafana dashboards
   - Alert system

3. Testing

   - Unit tests (pytest)
   - Integration tests
   - Scenario-based testing

4. Documentation

   - API documentation (OpenAPI/Swagger)
   - Deployment guide
   - User manual

5. Deployment infrastructure
   - Docker containerization
   - CI/CD pipeline
   - Backup & recovery

**Success Criteria:**

- 99% uptime over 2 weeks
- All APIs documented
- Automated tests pass
- Production deployment successful

---

## 10. Tech Stack Summary

### Backend

```
Python 3.11+
FastAPI - Web framework
Uvicorn - ASGI server
Pydantic - Data validation
SQLAlchemy - ORM
SQLite - Database (MVP)

# Powell Framework
scipy - Optimization (CFA, constraints)
numpy - Numerical computation
networkx - Graph algorithms (routing)
ortools - Vehicle routing, constraint programming

# Learning
torch (PyTorch) - Neural networks (VFA, PFA)
scikit-learn - ML utilities
pandas - Data manipulation

# Real-time
websockets - WebSocket support
asyncio - Async event handling

# External APIs
googlemaps - Google Places geocoding
```

### Frontend

```
React 18+
Axios - HTTP client
WebSocket client
Leaflet/Mapbox - Map visualization
Recharts - Performance charts
TailwindCSS - Styling
```

### Development

```
pytest - Testing
black - Code formatting
mypy - Type checking
ruff - Linting
```

---

## 11. Success Metrics (Post-Deployment)

### Functional Metrics

- **Decision Automation Rate:** % of orders automatically assigned without human intervention
- **Override Rate:** % of automated decisions overridden by ops team (target: <10%)
- **System Uptime:** % availability during operational hours (target: >99%)

### Learning Metrics

- **CFA Cost Prediction Error:** |actual - predicted| / actual (target: <10%)
- **CFA Time Prediction Error:** |actual - predicted| / actual (target: <15%)
- **VFA TD Error:** Convergence measure (target: <0.1 after 1000 episodes)
- **PFA Rule Confidence:** Avg confidence of applied rules (target: >0.85)

### Business Metrics

- **Route Efficiency:** Actual cost vs optimal cost (target: >85%)
- **Fleet Utilization:** % of fleet capacity used (track trend, no target yet)
- **On-Time Delivery:** % deliveries within time window (track)
- **Backhaul Acceptance:** % backhaul opportunities accepted (track VFA impact)

### Operational Metrics

- **Decision Latency:** Time from order arrival to route assignment (target: <30s)
- **Optimization Time:** CFA route optimization duration (target: <30s)
- **Learning Cycle Time:** Feedback to model update (target: <5 minutes)

---

## 12. Risks & Mitigations

### Technical Risks

**Risk:** Optimization doesn't converge or takes too long

- **Mitigation:** Fallback to heuristics (greedy assignment), time limits on optimization, warm-start solutions

**Risk:** Learning destabilizes (models get worse)

- **Mitigation:** Version control all models, A/B testing, rollback capability, human approval for major changes

**Risk:** Real-world operational data differs from assumptions

- **Mitigation:** Extensive logging, feedback loops, gradual deployment, override mechanisms

**Risk:** SQLite performance bottleneck

- **Mitigation:** Designed for migration path to PostgreSQL, monitor query performance, caching

### Operational Risks

**Risk:** Users don't trust AI decisions

- **Mitigation:** Explainable decisions (show reasoning), human-in-loop approval, gradual automation, transparency

**Risk:** Edge cases not handled

- **Mitigation:** Comprehensive error handling, fallback to manual, extensive testing, learn from production

**Risk:** Integration with existing systems fails

- **Mitigation:** Well-defined APIs, thorough testing, gradual rollout, manual backup processes

---

## 13. Out of Scope (MVP)

**Explicitly NOT included in MVP:**

- Multi-depot operations (only Nairobi origin)
- Real-time GPS tracking integration
- Mobile driver app
- Customer portal
- Automated invoicing/billing
- Multi-language support
- Advanced weather integration
- Dynamic pricing
- Third-party carrier integration
- Warehouse management (Senga has no warehouses)

**Future considerations (post-MVP):**

- Stochastic DLA (scenario-based)
- Multi-agent coordination (multiple concurrent optimizations)
- Online learning (immediate model updates)
- Advanced neural architectures (attention, transformers)
- Distributed computing for large-scale optimization

---

## 14. Appendix

### 14.1 Powell's Framework - Additional Resources

**Core References:**

- Powell, W.B. (2022). Sequential Decision Analytics and Modeling. _Foundations and Trends in Technology, Information and Operations Management_
- Powell's website: https://castle.princeton.edu/sda/

**The Four Classes Explained:**

- **PFA:** Direct state→action mapping, no optimization
- **CFA:** Parameterized optimization (deterministic approximation of uncertain problem)
- **VFA:** Bellman-based value approximation (ADP, RL)
- **DLA:** Explicit lookahead (deterministic or stochastic)

**Why All Four?**

- No single class is universally best
- Different problem characteristics favor different approaches
- Hybrids often outperform pure classes
- Real systems use multiple classes for different decisions

### 14.2 Senga-Specific Terminology

**Mesh Network:** Point-to-point logistics without central warehouses (aviation-inspired)

**Consolidation:** Grouping orders by destination before pickup

**Multi-destination Route:** Single truck serves multiple towns (Nakuru→Eldoret→Kitale)

**Backhaul:** Return trip cargo (currently empty, future opportunity)

**Eastleigh Window:** 8:30-9:45 AM pickup constraint due to traffic

**Fresh Food Priority:** Time-sensitive customers (e.g., Majid Fresh Foods)

### 14.3 Sample Calculations

**CFA Cost Function Example:**

```python
def route_cost(sequence, vehicle, params):
    """
    Total cost = fuel + time + delay penalties
    """
    # Distance cost
    total_km = sum(distance(sequence[i], sequence[i+1])
                   for i in range(len(sequence)-1))
    fuel_liters = total_km / vehicle.fuel_efficiency
    fuel_cost = fuel_liters * params['fuel_price']

    # Time cost
    base_time = total_km / params['avg_speed_kmh']
    traffic_buffer = sum(params[f'buffer_{zone}']
                         for zone in get_zones(sequence))
    total_time_hours = (base_time + traffic_buffer) / 60
    time_cost = total_time_hours * params['driver_cost_per_hour']

    # Delay penalties
    delays = [max(0, arrival_time[i] - order[i].time_window_end)
              for i in range(len(sequence))]
    delay_cost = sum(delays) * params['delay_penalty_per_hour']

    return fuel_cost + time_cost + delay_cost
```

**VFA State Encoding Example:**

```python
def encode_state(state: SystemState) -> torch.Tensor:
    """
    Convert system state to neural network input
    """
    features = []

    # Time features
    features.extend([
        state.timestamp.hour / 24.0,
        state.timestamp.weekday() / 7.0,
        state.timestamp.day / 31.0
    ])

    # Fleet features
    available_5t = sum(1 for v in state.fleet if v.type == "5T" and v.status == "available")
    available_10t = sum(1 for v in state.fleet if v.type == "10T" and v.status == "available")
    features.extend([available_5t / 10.0, available_10t / 5.0])

    # Order features (by destination)
    for city in ["Nakuru", "Eldoret", "Kitale"]:
        city_orders = [o for o in state.pending_orders if o.destination == city]
        features.extend([
            len(city_orders) / 20.0,  # Normalized count
            sum(o.weight_tonnes for o in city_orders) / 50.0  # Normalized weight
        ])

    # Route features
    features.append(len(state.active_routes) / 10.0)

    return torch.tensor(features, dtype=torch.float32)
```

---

## 15. Acceptance Criteria

**The system is ready for MVP deployment when:**

1. ✅ **Core Functionality:**

   - Orders can be created via API
   - CFA optimization generates routes
   - Routes assign vehicles and sequence customers
   - Dashboard displays routes on map

2. ✅ **Learning Capability:**

   - System logs operational feedback
   - CFA parameters update based on actuals
   - VFA learns from route completions
   - PFA mines patterns from history

3. ✅ **Performance:**

   - Route optimization completes in <30s for 50 orders
   - API response times <200ms
   - WebSocket updates <50ms latency
   - System handles 100 orders/day

4. ✅ **Reliability:**

   - 99% uptime over 1 week continuous operation
   - No data loss on restart
   - Graceful error handling (fallback to heuristics)

5. ✅ **Documentation:**

   - API endpoints documented (OpenAPI)
   - README with setup instructions
   - Code comments explain Powell framework usage

6. ✅ **Testing:**
   - Unit tests cover core functions (>80% coverage)
   - Integration tests validate decision flow
   - Manual testing scenarios passed (20+ routes)

---

## 16. Deployment Instructions

### Local Development Setup

```bash
# Clone repository
git clone <repo-url>
cd senga-sde

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Initialize database
python -m db.init_db

# Run backend
uvicorn api.main:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
npm start  # Runs on port 3000
```

### Environment Variables

```bash
# .env file
DATABASE_URL=sqlite:///./senga.db
GOOGLE_PLACES_API_KEY=<your-key>
LOG_LEVEL=INFO
WEBSOCKET_HEARTBEAT_SECONDS=30

# Model paths
VFA_MODEL_PATH=./models/vfa_v1.pt
CFA_PARAMS_PATH=./models/cfa_params_v1.json
PFA_RULES_PATH=./models/pfa_rules_v1.json
```

### Production Deployment (Docker)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build & run
docker build -t senga-sde .
docker run -p 8000:8000 -v $(pwd)/data:/app/data senga-sde
```

---

**END OF PRODUCT REQUIREMENTS DOCUMENT**

---

This PRD provides comprehensive, unambiguous specifications for building the Senga Sequential Decision Engine. A developer or AI assistant (like Copilot) should be able to:

1. Understand Powell's framework and why each policy class is needed
2. Implement all four classes (PFA, CFA, VFA, DLA) with actual algorithms
3. Build the event-driven decision system
4. Create the learning loops for continuous improvement
5. Develop the full-stack application (FastAPI + React)
6. Deploy and monitor the system

**Key distinguishing features of this PRD:**

- ✅ Mathematical rigor (actual Powell framework, not approximations)
- ✅ Learnable systems (all four classes learn from operational data)
- ✅ Hybrid policies (CFA/VFA, DLA/VFA, etc.)
- ✅ Event-driven architecture (not batch)
- ✅ Senga-specific context (mesh network, backhauls, African logistics)
- ✅ Complete tech stack (Python/FastAPI/PyTorch/React)
- ✅ Detailed API specs, DB schemas, code examples
- ✅ Clear phases and success metrics

Ready for implementation.

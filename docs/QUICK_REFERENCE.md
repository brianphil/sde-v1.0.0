# Powell Engine - Developer Quick Reference

## 🚀 TL;DR

A production-ready routing optimization engine with 4 decision policies, event-driven architecture, and continuous learning.

```python
# Create engine
engine = PowellEngine()

# Make decision
decision = engine.make_decision(
    state=current_state,
    decision_type=DecisionType.DAILY_ROUTE_PLANNING,
    trigger_reason="Morning optimization"
)

# Execute decision
result = engine.commit_decision(decision, current_state)

# Learn from outcome
engine.learn_from_feedback(operational_outcome)
```

---

## 📦 Core Classes

### 1. Policies

```python
from backend.core.powell.pfa import PolicyFunctionApproximation
from backend.core.powell.cfa import CostFunctionApproximation
from backend.core.powell.vfa import ValueFunctionApproximation
from backend.core.powell.dla import DirectLookaheadApproximation

# PFA: Rule-based decisions
pfa = PolicyFunctionApproximation()
decision = pfa.evaluate(state)

# CFA: Cost minimization
cfa = CostFunctionApproximation()
decision = cfa.evaluate(state)

# VFA: Neural network value estimation
vfa = ValueFunctionApproximation()
decision = vfa.evaluate(state)

# DLA: Multi-period planning
dla = DirectLookaheadApproximation()
decision = dla.evaluate(state)
```

### 2. Engine

```python
from backend.core.powell.engine import PowellEngine

engine = PowellEngine()

# Make a decision
decision = engine.make_decision(state, DecisionType.DAILY_ROUTE_PLANNING)

# Execute it
result = engine.commit_decision(decision, state)

# Learn from feedback
engine.learn_from_feedback(outcome)

# Get policy stats
stats = engine.get_policy_performance()
```

### 3. State Management

```python
from backend.core.models.state import SystemState, EnvironmentState, LearningState

# Create state
state = SystemState(
    pending_orders=orders_dict,
    fleet=vehicles_dict,
    customers=customers_dict,
    environment=EnvironmentState(current_time=datetime.now()),
    learning=LearningState()
)

# Query state
available_vehicles = state.get_available_vehicles()
backhaul_orders = state.get_backhaul_opportunities()
eastleigh_active = state.is_eastleigh_window_active()
```

### 4. Orchestration

```python
from backend.services.event_orchestrator import EventOrchestrator, Event, EventPriority

orchestrator = EventOrchestrator(engine, state_manager)

# Submit event
event = Event("order_arrived", {"order": order}, EventPriority.HIGH)
event_id = orchestrator.submit_event(event)

# Process all pending events
results = orchestrator.process_all_events()
```

### 5. Learning

```python
from backend.core.learning.feedback_processor import FeedbackProcessor
from backend.core.learning.td_learning import NeuralNetworkTDLearner

# Process feedback
processor = FeedbackProcessor()
signals = processor.process_outcome(operational_outcome)

# TD-learning for VFA
learner = NeuralNetworkTDLearner()
learner.td_learning_step(current_value, reward, next_value)
```

---

## 🎯 Decision Types

```python
from backend.core.models.decision import DecisionType

# Daily route optimization
decision = engine.make_decision(state, DecisionType.DAILY_ROUTE_PLANNING)

# New order arrival
decision = engine.make_decision(state, DecisionType.ORDER_ARRIVAL)

# Real-time adjustment
decision = engine.make_decision(state, DecisionType.REAL_TIME_ADJUSTMENT)

# Backhaul consolidation
decision = engine.make_decision(state, DecisionType.BACKHAUL_OPPORTUNITY)
```

---

## 🛠️ Common Tasks

### Create an Order

```python
from backend.core.models.domain import Order, Location, TimeWindow, DestinationCity

order = Order(
    order_id="ORD_123",
    customer_id="CUST_001",
    customer_name="ABC Corp",
    pickup_location=Location(
        latitude=-1.2921,
        longitude=36.8219,
        address="Nairobi CBD",
        zone="CBD"
    ),
    destination_city=DestinationCity.NAKURU,
    destination_location=Location(
        latitude=-0.3031,
        longitude=35.2684,
        address="Nakuru CBD"
    ),
    weight_tonnes=2.5,
    volume_m3=4.0,
    time_window=TimeWindow(
        start_time=datetime(2024, 1, 15, 8, 30),
        end_time=datetime(2024, 1, 15, 9, 45)
    ),
    price_kes=2500.0
)
```

### Create a Vehicle

```python
from backend.core.models.domain import Vehicle, Location, VehicleStatus

vehicle = Vehicle(
    vehicle_id="VEH_001",
    vehicle_type="5T",
    capacity_weight_tonnes=5.0,
    capacity_volume_m3=8.0,
    current_location=Location(latitude=-1.2921, longitude=36.8219),
    available_at=datetime.now(),
    status=VehicleStatus.AVAILABLE,
    driver_id="DRIVER_001"
)
```

### Build System State

```python
from backend.core.models.state import SystemState, EnvironmentState

state = SystemState(
    pending_orders={"ORD_001": order1, "ORD_002": order2},
    fleet={"VEH_001": vehicle1, "VEH_002": vehicle2},
    customers={"CUST_001": customer1},
    environment=EnvironmentState(
        current_time=datetime.now(),
        traffic_conditions={"CBD": 0.5, "Nakuru": 0.2},
        weather="clear"
    )
)
```

### Query State

```python
# Get available vehicles
available = state.get_available_vehicles()

# Get unassigned orders
pending = state.get_unassigned_orders()

# Get backhaul opportunities
backhauls = state.get_backhaul_opportunities()

# Get route profitability
profit = state.get_route_profitability("ROUTE_001")

# Check Eastleigh window
is_active = state.is_eastleigh_window_active()

# Get vehicles near location
nearby = state.get_vehicles_near_location(lat, lon, radius_km=50)
```

### Make a Decision and Execute

```python
# 1. Make decision
decision = engine.make_decision(
    state,
    decision_type=DecisionType.DAILY_ROUTE_PLANNING,
    trigger_reason="Morning optimization"
)

# 2. Inspect decision
print(f"Policy: {decision.policy_name}")
print(f"Confidence: {decision.confidence_score:.0%}")
print(f"Routes: {len(decision.routes)}")

# 3. Commit (execute) decision
result = engine.commit_decision(decision, state)

print(f"Routes created: {result['routes_created']}")
print(f"Orders assigned: {result['orders_assigned']}")
```

### Submit Feedback

```python
from backend.core.models.domain import OperationalOutcome

outcome = OperationalOutcome(
    outcome_id="OUTCOME_001",
    route_id="ROUTE_001",
    vehicle_id="VEH_001",
    predicted_fuel_cost=2500.0,
    actual_fuel_cost=2400.0,
    predicted_duration_minutes=320,
    actual_duration_minutes=310,
    predicted_distance_km=250.0,
    actual_distance_km=248.0,
    on_time=True,
    successful_deliveries=4,
    failed_deliveries=0,
    customer_satisfaction_score=0.95
)

# Feed into engine
engine.learn_from_feedback(outcome)
```

---

## 📊 Policy Performance

```python
# Get performance stats
stats = engine.get_policy_performance()

for policy_name, metrics in stats.items():
    print(f"{policy_name}:")
    print(f"  Uses: {metrics['usage_count']}")
    print(f"  Confidence: {metrics['avg_confidence']:.0%}")
    print(f"  Success: {metrics['success_rate']:.0%}")
    print(f"  Value: {metrics['avg_value']:.0f} KES")
```

---

## 🔄 Event Flow

```python
# 1. Event created
event = Event("order_arrived", {"order": order}, EventPriority.HIGH)

# 2. Event queued
event_id = orchestrator.submit_event(event)

# 3. Event processed
orchestrator.process_all_events()
# → Determines decision type
# → Calls engine.make_decision()
# → Commits decision
# → Calls learning if available
# → Calls registered handlers

# 4. Result available via WebSocket (Phase 6)
```

---

## 🧠 Learning Pipeline

```python
# 1. Collect outcome
outcome = OperationalOutcome(...)

# 2. Process outcome
processor = FeedbackProcessor()
signals = processor.process_outcome(outcome)
# Returns:
# {
#   'cfa_signals': {'fuel_error': -100, 'time_error': -5, ...},
#   'vfa_signals': {'reward': 0.95, ...},
#   'pfa_signals': {'rule_1': True, 'rule_2': False},
#   'dla_signals': {'forecast_accuracy': 0.96}
# }

# 3. Update parameters
engine.cfa.update_from_feedback(signals['cfa_signals'])
engine.vfa.td_learning_update(state, signals['vfa_signals']['reward'], next_state)
engine.pfa.update_from_feedback(signals['pfa_signals'])
engine.dla.update_forecast_accuracy(signals['dla_signals']['forecast_accuracy'])

# 4. Next decision uses updated parameters
```

---

## ⚙️ Configuration

```python
# Hyperparameters are in each policy class
# Example: CFA parameters

class CostParameters:
    fuel_per_km: float = 500.0              # KES per km
    driver_hourly_rate: float = 1000.0      # KES per hour
    delay_penalty_per_minute: float = 5.0   # KES per minute
    prediction_accuracy_fuel: float = 0.95   # 95% accurate

# Update parameters
cfa.params.fuel_per_km = 520.0
```

---

## 🔍 Debugging

### View State

```python
# Print full state
from pprint import pprint
pprint(state.__dict__)

# Check specific aspects
print(f"Pending: {len(state.pending_orders)} orders")
print(f"Active: {len(state.active_routes)} routes")
print(f"Fleet: {len(state.fleet)} vehicles")
```

### View Decision Details

```python
decision = engine.make_decision(state, DecisionType.DAILY_ROUTE_PLANNING)

print(f"Policy: {decision.policy_name}")
print(f"Confidence: {decision.confidence_score}")
print(f"Expected Value: {decision.expected_value}")
print(f"Reasoning: {decision.reasoning}")
print(f"Routes:")
for route in decision.routes:
    print(f"  - {route.route_id}: {len(route.order_ids)} orders")
```

### View Learning State

```python
state.learning.cfa_parameters
state.learning.vfa_weights  # Neural network weights
state.learning.pfa_rules     # Rule confidence scores
state.learning.dla_forecasts  # Forecast accuracy
```

---

## 📚 Imports Reference

```python
# Domain models
from backend.core.models.domain import (
    Order, Vehicle, Route, Customer, Location,
    TimeWindow, OperationalOutcome,
    OrderStatus, VehicleStatus, RouteStatus,
    DestinationCity
)

# State
from backend.core.models.state import SystemState, EnvironmentState, LearningState

# Decisions
from backend.core.models.decision import DecisionType, PolicyDecision, DecisionContext

# Policies
from backend.core.powell.pfa import PolicyFunctionApproximation
from backend.core.powell.cfa import CostFunctionApproximation
from backend.core.powell.vfa import ValueFunctionApproximation
from backend.core.powell.dla import DirectLookaheadApproximation
from backend.core.powell.hybrids import CFAVFAHybrid, DLAVFAHybrid, PFACFAHybrid
from backend.core.powell.engine import PowellEngine

# Services
from backend.services.state_manager import StateManager
from backend.services.event_orchestrator import EventOrchestrator, Event, EventPriority
from backend.services.route_optimizer import RouteOptimizer
from backend.core.learning.feedback_processor import FeedbackProcessor
from backend.core.learning.td_learning import NeuralNetworkTDLearner
```

---

## 🐛 Common Issues

### Issue: "PyTorch not installed"

**Solution**: VFA gracefully falls back to linear regression. Install torch for full capability:

```bash
pip install torch
```

### Issue: "State is immutable"

**Solution**: Don't try to modify state. Create new state instead:

```python
# ❌ Wrong
state.pending_orders["NEW_ORDER"] = order

# ✅ Right
new_state = state_manager.apply_event("order_received", {"order": order})
```

### Issue: "Decision not committing"

**Solution**: Check decision status is PENDING_EXECUTION:

```python
if decision.status == "PENDING_EXECUTION":
    result = engine.commit_decision(decision, state)
```

---

## 📈 Performance Tips

1. **Batch events**: Process multiple events in one call

   ```python
   orchestrator.process_all_events()  # Process all queued
   ```

2. **Reuse state**: Don't recreate state multiple times

   ```python
   state = state_manager.get_current_state()
   # Use state multiple times
   ```

3. **Cache VFA features**: Extract features once

   ```python
   features = vfa.extract_state_features(state)
   # Use for multiple predictions
   ```

4. **Batch TD-learning**: Update VFA with multiple outcomes
   ```python
   learner.batch_td_learning([outcome1, outcome2, ...])
   ```

---

## 🔗 See Also

- **Full API Spec**: `API_SPECIFICATION.md`
- **Integration Guide**: `INTEGRATION_GUIDE.md`
- **Engine Deep Dive**: `ENGINE_IMPLEMENTATION.md`
- **Demo Suite**: `demo.py`

---

_Quick Reference - Powell Sequential Decision Engine v1.0_

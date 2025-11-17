# Consolidation Engine Integration with Powell SDE

## Overview

The Consolidation Engine **prepares filtered order groups** but does NOT make routing decisions. Instead, it works WITH the Powell Sequential Decision Engine (CFA/VFA/PFA/DLA) to enable intelligent consolidation.

## Key Principle

```
❌ WRONG: Consolidation Engine makes routing decisions
✅ RIGHT: Consolidation Engine filters orders → Powell SDE decides optimal routes
```

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ORDER ARRIVAL                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│             CONSOLIDATION ENGINE                            │
│  • Classify order (bulk/consolidated/urgent)                │
│  • Bulk/Urgent → flag for immediate Powell SDE routing     │
│  • Consolidated → add to pool                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
┌─────────────┐         ┌──────────────────────┐
│ BULK/URGENT │         │ CONSOLIDATION POOL   │
│   ORDERS    │         │  • Wait for triggers │
└──────┬──────┘         │  • Geographic filter │
       │                │  • Service filter    │
       │                │  • Time filter       │
       │                └──────────┬───────────┘
       │                           │
       │                           ▼
       │                ┌────────────────────────┐
       │                │ CONSOLIDATION          │
       │                │ OPPORTUNITIES          │
       │                └──────────┬─────────────┘
       │                           │
       └───────────┬───────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   POWELL SDE                                │
│  • CFA: Optimize cost for each opportunity                 │
│  • VFA: Evaluate long-term value                           │
│  • PFA: Apply learned routing patterns                     │
│  • DLA: Consider future decisions                          │
│  → SELECT OPTIMAL ROUTES                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. **Powell Engine Initialization**

```python
# backend/core/powell/engine.py

from backend.core.consolidation import ConsolidationEngine

class PowellEngine:
    def __init__(self):
        # ... existing initialization ...

        # Add consolidation engine
        self.consolidation_engine = ConsolidationEngine()

        logger.info("Powell Engine initialized with Consolidation Engine")
```

### 2. **Order Arrival Handling**

```python
# When new order arrives
def handle_order_arrival(self, order: Order, state: SystemState):
    # Step 1: Consolidation engine classifies order
    result = self.consolidation_engine.process_new_order(order, state)

    # Step 2: Handle based on classification
    if result.bulk_order_ids or result.urgent_order_ids:
        # Immediate routing via Powell SDE
        orders_to_route = (result.bulk_order_ids + result.urgent_order_ids)

        # Build context for Powell SDE
        context = self._build_decision_context(
            state,
            DecisionType.ORDER_ARRIVAL,
            orders_to_consider={oid: state.pending_orders[oid] for oid in orders_to_route},
            trigger_reason=f"Bulk/Urgent order arrival: {', '.join(orders_to_route)}"
        )

        # Powell SDE makes routing decision
        decision = self._select_and_execute_policy(state, context, DecisionType.ORDER_ARRIVAL)

        return decision

    elif result.pooled_order_ids:
        # Added to consolidation pool
        logger.info(f"Order added to consolidation pool. Pool status: {result.pool_status}")

        # Check if consolidation should trigger
        if result.should_trigger_consolidation:
            return self.run_consolidation_decision(state)

    return None
```

### 3. **Consolidation Decision Workflow**

```python
def run_consolidation_decision(self, state: SystemState):
    """Run consolidation with Powell SDE decision-making."""

    # Step 1: Prepare consolidation opportunities
    opportunities = self.consolidation_engine.prepare_consolidation_opportunities(state)

    if not opportunities:
        logger.info("No consolidation opportunities available")
        return None

    logger.info(f"Prepared {len(opportunities)} consolidation opportunities for evaluation")

    # Step 2: For each opportunity, build decision context
    all_routes = []

    for opp in opportunities:
        # Get orders for this opportunity
        orders_to_consider = {
            oid: state.pending_orders[oid]
            for oid in opp.order_ids
            if oid in state.pending_orders
        }

        # Build context
        context = self._build_decision_context(
            state,
            DecisionType.DAILY_ROUTE_PLANNING,
            orders_to_consider=orders_to_consider,
            trigger_reason=f"Consolidation opportunity: {opp.cluster_id} "
                          f"({len(opp.order_ids)} orders, score={opp.compatibility_score:.2f})"
        )

        # Powell SDE evaluates this opportunity
        decision = self._select_and_execute_policy(
            state,
            context,
            DecisionType.DAILY_ROUTE_PLANNING
        )

        # Extract routes from decision
        if hasattr(decision, 'routes') and decision.routes:
            all_routes.extend(decision.routes)

            # Remove routed orders from pool
            routed_order_ids = []
            for route in decision.routes:
                routed_order_ids.extend(route.order_ids)

            self.consolidation_engine.remove_routed_orders(routed_order_ids)

    return all_routes
```

### 4. **Scheduled Consolidation**

```python
def daily_route_planning(self, state: SystemState):
    """Daily route planning with consolidation."""

    # Step 1: Check consolidation pool
    if self.consolidation_engine.pool.should_trigger_consolidation():
        logger.info("Consolidation triggered during daily planning")
        consolidation_routes = self.run_consolidation_decision(state)
    else:
        consolidation_routes = []

    # Step 2: Handle remaining pending orders
    remaining_orders = state.get_unassigned_orders()

    if remaining_orders:
        context = self._build_decision_context(
            state,
            DecisionType.DAILY_ROUTE_PLANNING,
            orders_to_consider=remaining_orders,
            trigger_reason="Daily route planning - remaining orders"
        )

        decision = self._select_and_execute_policy(
            state,
            context,
            DecisionType.DAILY_ROUTE_PLANNING
        )

        return consolidation_routes + (decision.routes if hasattr(decision, 'routes') else [])

    return consolidation_routes
```

---

## Decision Flow Examples

### Example 1: Bulk Order

```python
# Order arrives: 4.5T to Nakuru (90% utilization on 5T truck)

result = consolidation_engine.process_new_order(order, state)
# Result:
# - bulk_order_ids: ['ORD_001']
# - pooled_order_ids: []
# - should_trigger_consolidation: False

# → Powell SDE routes immediately
decision = powell_engine.make_decision(
    state,
    DecisionType.ORDER_ARRIVAL,
    orders_to_consider={'ORD_001': order},
    trigger_reason="Bulk order arrival"
)

# Powell SDE (CFA/VFA) evaluates:
# - CFA: Optimal vehicle selection, cost minimization
# - VFA: Long-term value assessment
# → Creates route with best vehicle
```

### Example 2: Consolidated Orders

```python
# 3 orders arrive to Nakuru: 1.2T, 1.5T, 1.8T
# Each below 60% utilization → classified as CONSOLIDATED

# Order 1
result = consolidation_engine.process_new_order(order1, state)
# pooled_order_ids: ['ORD_001']
# should_trigger_consolidation: False (pool size = 1)

# Order 2
result = consolidation_engine.process_new_order(order2, state)
# pooled_order_ids: ['ORD_002']
# should_trigger_consolidation: False (pool size = 2)

# Order 3
result = consolidation_engine.process_new_order(order3, state)
# pooled_order_ids: ['ORD_003']
# should_trigger_consolidation: True (cluster size = 3, trigger threshold met!)

# → Prepare consolidation opportunities
opportunities = consolidation_engine.prepare_consolidation_opportunities(state)
# Returns:
# [
#     ConsolidationOpportunity(
#         cluster_id='nairobi_nakuru_kisumu_Nakuru',
#         order_ids=['ORD_001', 'ORD_002', 'ORD_003'],
#         estimated_total_weight=4.5,
#         estimated_total_volume=8.0,
#         compatibility_score=0.95
#     )
# ]

# → Powell SDE evaluates opportunity
decision = powell_engine.make_decision(
    state,
    DecisionType.DAILY_ROUTE_PLANNING,
    orders_to_consider={'ORD_001': order1, 'ORD_002': order2, 'ORD_003': order3},
    trigger_reason="Consolidation opportunity: 3 orders to Nakuru"
)

# Powell SDE (CFA/VFA) evaluates:
# - CFA: Consolidate on 5T truck (90% utilization) OR separate routes?
# - VFA: Future value of consolidation
# - Cost: 1 trip (2,800 KES) vs 3 trips (4,500 KES)
# → Decides to consolidate (saves 1,700 KES)
```

### Example 3: Mixed Classifications

```python
# 5 orders arrive:
# - Order 1: 4.5T (BULK)
# - Order 2: 1.2T (CONSOLIDATED)
# - Order 3: 1.5T (CONSOLIDATED)
# - Order 4: 2.8T, Priority=2 (URGENT)
# - Order 5: 1.8T (CONSOLIDATED)

# Order 1 (BULK)
result = consolidation_engine.process_new_order(order1, state)
# → Powell SDE routes immediately

# Order 2-3, 5 (CONSOLIDATED)
# → Added to pool

# Order 4 (URGENT)
result = consolidation_engine.process_new_order(order4, state)
# → Powell SDE routes immediately (high priority)

# Later, when consolidation triggers:
opportunities = consolidation_engine.prepare_consolidation_opportunities(state)
# → Powell SDE evaluates consolidated orders (2, 3, 5)
```

---

## What Happens When Powell SDE Decides to Wait?

### Key Insight: Orders Stay in Pool by Default

**Important:** When consolidation opportunities are prepared, the orders are **still in the pool**. They are only removed when Powell SDE actually creates routes for them.

### Decision Flow

```python
def run_consolidation_decision(self, state: SystemState):
    """Run consolidation with Powell SDE decision-making."""

    # Step 1: Prepare opportunities from pool
    opportunities = self.consolidation_engine.prepare_consolidation_opportunities(state)

    all_routes = []

    for opp in opportunities:
        # Build context
        context = self._build_decision_context(
            state,
            DecisionType.DAILY_ROUTE_PLANNING,
            orders_to_consider={oid: state.pending_orders[oid] for oid in opp.order_ids},
            trigger_reason=f"Consolidation opportunity: {opp.cluster_id}"
        )

        # Powell SDE evaluates this opportunity
        decision = self._select_and_execute_policy(state, context, DecisionType.DAILY_ROUTE_PLANNING)

        # ✅ Powell SDE ACCEPTS: Creates routes
        if hasattr(decision, 'routes') and decision.routes:
            all_routes.extend(decision.routes)

            # Extract routed order IDs
            routed_order_ids = []
            for route in decision.routes:
                routed_order_ids.extend(route.order_ids)

            # NOW remove from pool
            self.consolidation_engine.remove_routed_orders(routed_order_ids)
            logger.info(f"Powell SDE accepted consolidation: routed {len(routed_order_ids)} orders")

        # ❌ Powell SDE DEFERS: No routes created
        else:
            # Orders remain in pool - no action needed!
            logger.info(
                f"Powell SDE deferred consolidation for {opp.cluster_id} "
                f"({len(opp.order_ids)} orders remain in pool)"
            )
            # Pool will continue monitoring and may present again later

    return all_routes
```

### Why Powell SDE Might Defer

Powell SDE (via CFA/VFA/PFA/DLA) may decide to wait for several reasons:

#### 1. **Cost-Function Approximation (CFA) Analysis**
```python
# CFA evaluates immediate cost
current_cost = estimate_consolidation_cost(opp.order_ids)
# Result: 3,500 KES for 2 orders

# CFA determines: Not cost-effective yet
# Reasoning: Waiting for more orders could improve utilization
```

#### 2. **Value-Function Approximation (VFA) Analysis**
```python
# VFA evaluates long-term value
immediate_value = cfa_cost
future_value = vfa.estimate_value(state_after_routing)

if future_value > immediate_value:
    # Better to wait - expecting more orders to same destination
    return None  # Defer consolidation
```

#### 3. **Policy-Function Approximation (PFA) Pattern**
```python
# PFA learned from past: "Tuesday 10am usually has 5+ Nakuru orders"
# Current: Tuesday 10am, only 2 Nakuru orders in pool
# PFA recommendation: Wait for more orders (pattern-based)
```

#### 4. **Deep Lookahead Approximation (DLA) Projection**
```python
# DLA projects: "3 more Nakuru orders expected in next 30 minutes"
# Decision: Wait for better consolidation opportunity
```

### Example Scenario: Powell SDE Defers Then Accepts

```python
# Time: 09:30 - Pool has 2 orders to Nakuru (1.2T, 1.5T)
# Consolidation trigger: cluster size = 2

opportunities = consolidation_engine.prepare_consolidation_opportunities(state)
# ConsolidationOpportunity(
#     cluster_id='nairobi_nakuru_kisumu_Nakuru',
#     order_ids=['ORD_001', 'ORD_002'],
#     estimated_total_weight=2.7,
#     compatibility_score=0.85
# )

decision = powell_engine.make_decision(state, context, DecisionType.DAILY_ROUTE_PLANNING)

# Powell SDE evaluates:
# - CFA: Cost = 2,800 KES for 2.7T load (54% utilization on 5T truck)
# - VFA: Future value higher if we wait
# - DLA: Predicts 2 more orders to Nakuru in next 60 minutes
# → Decision: DEFER (no routes created)

# Result: Orders stay in pool
logger.info("Powell SDE deferred: ORD_001, ORD_002 remain in pool")

# ---

# Time: 10:15 - 2 more orders arrive to Nakuru (1.8T, 1.3T)
result = consolidation_engine.process_new_order(order3, state)
result = consolidation_engine.process_new_order(order4, state)

# Pool now has 4 orders, trigger again
opportunities = consolidation_engine.prepare_consolidation_opportunities(state)
# ConsolidationOpportunity(
#     cluster_id='nairobi_nakuru_kisumu_Nakuru',
#     order_ids=['ORD_001', 'ORD_002', 'ORD_003', 'ORD_004'],
#     estimated_total_weight=5.8,
#     compatibility_score=0.92
# )

decision = powell_engine.make_decision(state, context, DecisionType.DAILY_ROUTE_PLANNING)

# Powell SDE re-evaluates:
# - CFA: Cost = 3,200 KES for 5.8T load (96% utilization on 6T truck!)
# - VFA: Immediate routing now optimal
# - Savings: 1 trip (3,200 KES) vs 4 trips (5,600 KES) = 2,400 KES saved
# → Decision: ACCEPT (creates consolidated route)

# Result: Route created, orders removed from pool
self.consolidation_engine.remove_routed_orders(['ORD_001', 'ORD_002', 'ORD_003', 'ORD_004'])
logger.info("Powell SDE accepted: Created consolidated route with 4 orders")
```

### Pool Behavior Summary

| Powell SDE Decision | Action | Orders in Pool | Future Opportunities |
|-------------------|--------|----------------|---------------------|
| **ACCEPT** (creates routes) | `remove_routed_orders()` called | Removed from pool | Won't be presented again |
| **DEFER** (no routes) | No action | Remain in pool | Will be presented again when triggers fire |

### Avoiding Repeated Evaluation

**Concern:** Will Powell SDE repeatedly evaluate the same opportunity?

**Answer:** Yes, but this is intentional and beneficial:

1. **State Changes:** Between evaluations, system state changes (new orders, vehicle availability, time)
2. **VFA Learning:** VFA value estimates improve over time
3. **Pool Triggers:** Different triggers (time, size) provide natural intervals
4. **DLA Projections:** Future state projections become more accurate as time progresses

**Optional Enhancement:** Track deferred opportunities to add backoff:

```python
# Optional: Add to ConsolidationPool
class ConsolidationPool:
    def __init__(self):
        self.deferred_opportunities: Dict[str, datetime] = {}
        self.defer_backoff_minutes = 30

    def should_reevaluate_opportunity(self, cluster_id: str) -> bool:
        """Check if enough time has passed since last deferral."""
        if cluster_id not in self.deferred_opportunities:
            return True

        last_defer = self.deferred_opportunities[cluster_id]
        minutes_elapsed = (datetime.now() - last_defer).total_seconds() / 60

        return minutes_elapsed >= self.defer_backoff_minutes

    def mark_deferred(self, cluster_id: str):
        """Mark opportunity as deferred."""
        self.deferred_opportunities[cluster_id] = datetime.now()

    def mark_accepted(self, cluster_id: str):
        """Clear deferral tracking when opportunity is accepted."""
        if cluster_id in self.deferred_opportunities:
            del self.deferred_opportunities[cluster_id]
```

### Time-Based Safety Net

Even if Powell SDE repeatedly defers, pool triggers provide safety:

```python
# Pool configuration
config = PoolConfiguration(
    max_pool_wait_time_minutes=120,  # 2 hours maximum
    scheduled_consolidation_times=["09:00", "14:00", "17:00"]
)

# After 2 hours, orders MUST be evaluated
# At scheduled times (09:00, 14:00, 17:00), forced evaluation occurs

# This ensures orders don't wait indefinitely
```

### Summary: Deferred Orders

✅ **Orders remain in pool** when Powell SDE defers (no action needed)
✅ **Pool continues monitoring** via triggers (size, time, cluster, schedule)
✅ **Re-evaluation occurs** when triggers fire again
✅ **State evolves** between evaluations (new orders, vehicles, time)
✅ **Safety nets** prevent indefinite waiting (max wait time, scheduled runs)
✅ **Optional backoff** can prevent excessive re-evaluation of same opportunity

**Key Takeaway:** The pool is persistent storage. Powell SDE decides "when to route," consolidation engine manages "what to present."

---

## Benefits of This Integration

### 1. **Separation of Concerns**
- Consolidation Engine: Filtering & compatibility
- Powell SDE: Optimal routing decisions

### 2. **Leverages Powell's Intelligence**
- CFA: Cost optimization
- VFA: Long-term value
- PFA: Learned patterns
- DLA: Future state consideration

### 3. **Flexible Decision-Making**
- Powell SDE can choose NOT to consolidate if cost-ineffective
- Considers current state, vehicle availability, learning

### 4. **Learning Integration**
- Powell SDE learns from consolidation outcomes
- Improves consolidation decisions over time

---

## Summary

**Consolidation Engine Role:**
- ✅ Classify orders (bulk/consolidated/urgent)
- ✅ Manage consolidation pool
- ✅ Filter compatible order groups
- ✅ Prepare opportunities for Powell SDE
- ❌ Does NOT make routing decisions

**Powell SDE Role:**
- ✅ Evaluate consolidation opportunities
- ✅ Decide optimal vehicle assignment
- ✅ Optimize route sequences
- ✅ Balance cost vs. value
- ✅ Learn from outcomes

**Result:** Intelligent consolidation that leverages Powell's full decision-making capabilities!

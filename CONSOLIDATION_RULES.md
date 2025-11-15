# Intelligent Consolidation Rules

## Overview

The Powell Sequential Decision Engine now includes **intelligent consolidation logic** that makes operationally sound routing decisions based on real-world business constraints.

## Problem Solved

**Before:** Engine was dispatching vehicles inefficiently:
- ❌ 10T truck with 200kg load (2% utilization)
- ❌ Fresh food mixed with hazardous cargo
- ❌ Urgent orders batched with low-priority orders
- ❌ Single order per vehicle despite consolidation opportunities

**After:** Engine makes intelligent decisions:
- ✅ Minimum 40% weight utilization enforced
- ✅ Cargo compatibility validated
- ✅ Priority levels respected
- ✅ Optimal vehicle selection for load size
- ✅ Multi-order consolidation when beneficial

---

## Consolidation Constraints

### 1. **Vehicle Utilization**

**Minimum Utilization (Default):**
- Weight: 40% minimum
- Volume: 30% minimum

**Maximum Utilization (Safety):**
- Weight: 95% maximum
- Volume: 95% maximum

**Example:**
```
❌ REJECTED: 10T truck (10,000kg capacity) with 1 order (200kg) = 2% utilization
✅ APPROVED: 5T truck (5,000kg capacity) with 3 orders (2,500kg) = 50% utilization
```

### 2. **Cargo Compatibility**

**Fresh Food Rules:**
- `allow_fresh_food_mixing: false` (default)
- Fresh food requires dedicated vehicle
- Cannot mix with other cargo types

**Hazardous Cargo Rules:**
- `allow_hazardous_mixing: false` (default)
- Hazardous materials require dedicated vehicle
- Strict safety compliance

**Fragile Cargo Rules:**
- `allow_fragile_mixing: true` (default)
- Can mix with compatible cargo
- Avoid heavy loads (>2T) when fragile present

**Example:**
```
❌ REJECTED: Fresh food + general cargo on same truck
✅ APPROVED: Fresh food only (dedicated vehicle)
✅ APPROVED: Fragile + light general cargo
```

### 3. **Priority Handling**

**Priority Levels:**
- 0 = Normal
- 1 = High
- 2 = Urgent

**Mixing Rules:**
- `max_priority_difference: 1` (default)
- Can mix adjacent priority levels (0+1 or 1+2)
- Cannot mix urgent (2) with normal (0)

**Example:**
```
❌ REJECTED: Urgent (priority=2) + Normal (priority=0) = difference of 2
✅ APPROVED: High (priority=1) + Normal (priority=0) = difference of 1
✅ APPROVED: Urgent (priority=2) + High (priority=1) = difference of 1
```

### 4. **Destination Constraints**

**Multi-City Routes:**
- `max_destinations_per_route: 3` (default)
- Prefer single-destination routes
- Multi-city allowed if cost-effective

**Detour Limits:**
- `max_detour_ratio: 1.3` (default)
- Consolidated route can be max 30% longer than direct

**Example:**
```
❌ REJECTED: Route visiting 4 cities (too complex)
✅ APPROVED: Route to Nakuru with 3 stops (same city)
✅ APPROVED: Route to Nakuru → Eldoret (2 cities, within detour limit)
```

### 5. **Cost Efficiency**

**Consolidation Justification:**
- `min_cost_savings_for_consolidation: 500 KES` (default)
- `min_orders_for_consolidation: 2` (default)
- Consolidation must provide meaningful savings

**Example:**
```
❌ REJECTED: Consolidate 2 orders, save 200 KES (below 500 KES threshold)
✅ APPROVED: Consolidate 3 orders, save 1,200 KES (above threshold)
```

---

## How It Works

### Step 1: Identify Consolidation Opportunities

Engine groups orders by:
1. **Destination city** (primary criteria)
2. **Special handling** (fresh food, hazardous)
3. **Priority level** (if strict separation enabled)

```python
consolidation_groups = get_consolidation_opportunities(
    orders,
    constraints
)

# Example output:
# {
#   "Nakuru": ["ORD_001", "ORD_002", "ORD_005"],  # 3 Nakuru orders
#   "Eldoret": ["ORD_003"],                       # 1 Eldoret order
#   "Nakuru_fresh": ["ORD_004"],                  # Fresh food (separate)
# }
```

### Step 2: Find Optimal Vehicle for Each Group

For each group, find **smallest vehicle** that meets utilization requirements:

```python
vehicle = validator.get_optimal_vehicle_for_load(
    total_weight=7.5,  # tonnes
    total_volume=12.5, # m³
    available_vehicles=[VEH_001, VEH_002, VEH_003]
)

# Checks:
# - VEH_001 (5T): Cannot fit 7.5T ❌
# - VEH_002 (5T): Cannot fit 7.5T ❌
# - VEH_003 (10T): Can fit, utilization = 75% ✅
```

### Step 3: Validate Route Against Constraints

```python
is_valid, violations, reasoning = validator.validate_route(
    route,
    vehicle,
    orders
)

# Validation checks:
# ✅ Utilization: 75% weight, 83% volume (within 40-95%)
# ✅ Cargo compatibility: All general cargo (compatible)
# ✅ Priority mixing: All priority=0 (no conflict)
# ✅ Destinations: 1 city (within limit)
# → Route APPROVED
```

### Step 4: Create Consolidated Routes

- Approved routes are created
- Rejected routes logged with reasoning
- Remaining orders processed individually

---

## Configuration

### Default Configuration

```python
from backend.core.powell.consolidation_rules import ConsolidationConstraints

constraints = ConsolidationConstraints(
    # Utilization
    min_weight_utilization=0.40,  # 40%
    min_volume_utilization=0.30,  # 30%
    max_weight_utilization=0.95,  # 95%
    max_volume_utilization=0.95,  # 95%

    # Cargo compatibility
    allow_fresh_food_mixing=False,
    allow_fragile_mixing=True,
    allow_hazardous_mixing=False,

    # Priority
    allow_priority_mixing=True,
    max_priority_difference=1,

    # Destinations
    prefer_same_destination=True,
    max_destinations_per_route=3,
    max_detour_ratio=1.3,

    # Cost
    min_cost_savings_for_consolidation=500.0,  # KES
    min_orders_for_consolidation=2,
)
```

### Custom Configuration

```python
# More aggressive consolidation
aggressive_constraints = ConsolidationConstraints(
    min_weight_utilization=0.30,  # Lower threshold
    allow_fresh_food_mixing=True,  # Allow mixing
    max_priority_difference=2,     # More flexible
    min_cost_savings_for_consolidation=200.0,  # Lower savings needed
)

# More conservative (safety-first)
conservative_constraints = ConsolidationConstraints(
    min_weight_utilization=0.60,  # Higher threshold
    allow_fresh_food_mixing=False,
    allow_fragile_mixing=False,
    max_priority_difference=0,    # No mixing
    max_destinations_per_route=1, # Single destination only
)
```

---

## Integration with CFA Policy

The CFA (Cost Function Approximation) policy now uses consolidation intelligence:

```python
from backend.core.powell.cfa import CostFunctionApproximation
from backend.core.powell.consolidation_rules import ConsolidationConstraints

# Create CFA with custom constraints
constraints = ConsolidationConstraints(
    min_weight_utilization=0.50,
)

cfa = CostFunctionApproximation(
    consolidation_constraints=constraints
)

# CFA will now:
# 1. Identify consolidation opportunities
# 2. Select optimal vehicle sizes
# 3. Validate all constraints
# 4. Only create operationally sound routes
```

---

## Validation Examples

### Example 1: Low Utilization Rejected

```
Route: ROUTE_001
Vehicle: VEH_003 (10T truck, 15m³ capacity)
Orders: 1 order (500kg, 2m³)

Validation:
❌ REJECTED - Low Utilization
   Weight: 500kg / 10,000kg = 5% (below 40% minimum)
   Volume: 2m³ / 15m³ = 13% (below 30% minimum)

Recommendation: Use smaller 5T truck or consolidate more orders
```

### Example 2: Cargo Incompatibility

```
Route: ROUTE_002
Vehicle: VEH_001 (5T truck)
Orders: 2 orders
  - ORD_001: Fresh food (2.5T)
  - ORD_002: General cargo (2.0T)

Validation:
❌ REJECTED - Incompatible Cargo
   Fresh food requires dedicated vehicle (cannot mix with other cargo)

Recommendation: Create separate routes for fresh food
```

### Example 3: Approved Consolidation

```
Route: ROUTE_003
Vehicle: VEH_002 (5T truck, 8m³ capacity)
Orders: 3 orders to Nakuru
  - ORD_001: 2.5T, 4m³ (Priority: 0)
  - ORD_002: 1.5T, 3m³ (Priority: 0)
  - ORD_003: 1.0T, 2m³ (Priority: 1)

Validation:
✅ APPROVED
   Weight: 5.0T / 5.0T = 100% ✗ Exceeds 95% → Adjust

Retry with 10T truck:
✅ APPROVED
   Weight: 5.0T / 10.0T = 50% ✅
   Volume: 9m³ / 15m³ = 60% ✅
   Cargo: All general (compatible) ✅
   Priority: Difference = 1 (acceptable) ✅
   Destination: 1 city (optimal) ✅

Route created successfully!
```

---

## Monitoring and Logging

The engine logs consolidation decisions:

```
INFO: Found 3 consolidation groups
INFO: ✅ Created consolidated route route_cfa_0_xxx with 3 orders on 5T truck
DEBUG: No suitable vehicle found for group Nakuru_fresh (2.5T, 4.0m³)
INFO: Processing 2 remaining orders individually
WARNING: ❌ Cannot create route for order ORD_004: Weight utilization 25% below minimum 40%
```

---

## Benefits

### 1. **Cost Efficiency**
- Eliminate under-utilized vehicles
- Optimal vehicle size selection
- Reduced fuel and driver costs

### 2. **Operational Safety**
- Cargo compatibility enforced
- Safe weight/volume limits
- Priority handling respected

### 3. **Business Logic**
- Rules reflect real-world constraints
- Configurable for different operations
- Transparent validation reasoning

### 4. **Learning**
- Engine learns which consolidations work
- Builds confidence over time
- Improves decision quality

---

## Testing

Run the workflow test to see consolidation in action:

```bash
python test_complete_workflow.py
```

**Expected behavior:**
- Multiple Nakuru orders consolidated onto appropriate vehicle
- Fresh food handled separately
- Utilization thresholds enforced
- Detailed logging of consolidation decisions

---

## Future Enhancements

1. **Time Window Constraints**
   - Validate delivery deadlines
   - Account for traffic conditions
   - Sequence stops optimally

2. **Dynamic Threshold Adjustment**
   - Learn optimal utilization thresholds
   - Adjust based on demand patterns
   - Account for seasonal variations

3. **Advanced Routing**
   - Multi-city route optimization
   - TSP/VRP integration
   - Real-time traffic data

4. **Customer Preferences**
   - Preferred delivery times
   - Special handling requirements
   - Vehicle type preferences

---

## Summary

The Powell engine now makes **intelligent, operationally sound** consolidation decisions that:

- ✅ Maximize vehicle utilization (40-95%)
- ✅ Respect cargo compatibility
- ✅ Handle priorities appropriately
- ✅ Select optimal vehicle sizes
- ✅ Provide transparent reasoning
- ✅ Learn from experience

This ensures **cost-effective, safe, and efficient** routing operations!

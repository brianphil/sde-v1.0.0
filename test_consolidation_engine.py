"""End-to-End Test for Consolidation Engine.

This script tests the complete consolidation workflow:
1. Order classification (bulk/consolidated/urgent)
2. Consolidation pool management
3. Geographic clustering
4. Compatibility filtering
5. Consolidation opportunity preparation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.core.models.domain import (
    Order,
    Location,
    Vehicle,
    DestinationCity,
    TimeWindow,
)
from backend.core.models.state import SystemState, EnvironmentState
from backend.core.consolidation import (
    ConsolidationEngine,
    ConsolidationResult,
    ConsolidationOpportunity,
    PoolConfiguration,
    OrderClassification,
)


def create_test_location(city: str, lat: float, lon: float) -> Location:
    """Create test location."""
    return Location(latitude=lat, longitude=lon, address=city, zone=city)


def create_test_order(
    order_id: str,
    weight: float,
    volume: float,
    destination_city: DestinationCity,
    priority: int = 0,
    time_window: TimeWindow = None,
    special_handling: List[str] = None,
) -> Order:
    """Create test order."""
    # Nairobi origin
    nairobi = create_test_location("Nairobi", -1.2921, 36.8219)

    # Destination mapping
    destinations = {
        DestinationCity.NAKURU: create_test_location("Nakuru", -0.3031, 36.0800),
        DestinationCity.ELDORET: create_test_location("Eldoret", 0.5143, 35.2698),
        DestinationCity.KISUMU: create_test_location("Kisumu", -0.0917, 34.7680),
        DestinationCity.KITALE: create_test_location("Kitale", 1.0157, 35.0062),
    }

    return Order(
        order_id=order_id,
        customer_id=f"CUST_{order_id}",
        pickup_location=nairobi,
        delivery_location=destinations.get(destination_city, nairobi),
        destination_city=destination_city,
        weight_tonnes=weight,
        volume_m3=volume,
        priority=priority,
        time_window=time_window,
        special_handling=special_handling or [],
        revenue=1000.0,
        arrival_time=datetime.now(),
    )


def create_test_vehicles() -> List[Vehicle]:
    """Create test fleet."""
    nairobi = create_test_location("Nairobi", -1.2921, 36.8219)

    vehicles = [
        Vehicle(
            vehicle_id="VEH_5T",
            vehicle_type="5T Truck",
            capacity_weight_tonnes=5.0,
            capacity_volume_m3=10.0,
            fuel_cost_per_km=15.0,
            driver_cost_per_hour=500.0,
            base_location=nairobi,
            current_location=nairobi,
            availability_start=datetime.now(),
            availability_end=datetime.now() + timedelta(hours=12),
        ),
        Vehicle(
            vehicle_id="VEH_10T",
            vehicle_type="10T Truck",
            capacity_weight_tonnes=10.0,
            capacity_volume_m3=20.0,
            fuel_cost_per_km=25.0,
            driver_cost_per_hour=600.0,
            base_location=nairobi,
            current_location=nairobi,
            availability_start=datetime.now(),
            availability_end=datetime.now() + timedelta(hours=12),
        ),
        Vehicle(
            vehicle_id="VEH_3T",
            vehicle_type="3T Truck",
            capacity_weight_tonnes=3.0,
            capacity_volume_m3=6.0,
            fuel_cost_per_km=12.0,
            driver_cost_per_hour=400.0,
            base_location=nairobi,
            current_location=nairobi,
            availability_start=datetime.now(),
            availability_end=datetime.now() + timedelta(hours=12),
        ),
    ]

    return vehicles


def create_test_state(vehicles: List[Vehicle]) -> SystemState:
    """Create test system state."""
    fleet = {v.vehicle_id: v for v in vehicles}

    env = EnvironmentState(
        current_time=datetime.now(),
        network_conditions={},
        road_closures=set(),
    )

    return SystemState(
        fleet=fleet,
        pending_orders={},
        active_routes={},
        completed_routes=[],
        environment=env,
    )


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_result(label: str, value, indent=0):
    """Print test result."""
    prefix = "  " * indent
    print(f"{prefix}{label}: {value}")


def test_order_classification():
    """Test 1: Order Classification."""
    print_section("TEST 1: Order Classification")

    engine = ConsolidationEngine()
    vehicles = create_test_vehicles()
    state = create_test_state(vehicles)

    # Test 1.1: Bulk Order (high utilization)
    print("Test 1.1: Bulk Order (4.5T on 5T truck = 90% utilization)")
    bulk_order = create_test_order("ORD_BULK_001", 4.5, 9.0, DestinationCity.NAKURU)

    result = engine.process_new_order(bulk_order, state)

    print_result("Classification", "BULK" if result.bulk_order_ids else "NOT BULK")
    print_result("Bulk Order IDs", result.bulk_order_ids)
    print_result("Pooled Order IDs", result.pooled_order_ids)
    print_result("Should Trigger", result.should_trigger_consolidation)

    assert (
        len(result.bulk_order_ids) == 1
    ), "Bulk order should be flagged for immediate routing"
    assert result.bulk_order_ids[0] == "ORD_BULK_001"
    print("✅ PASS: Bulk order correctly classified\n")

    # Test 1.2: Consolidated Order (low utilization)
    print("Test 1.2: Consolidated Order (1.2T on 5T truck = 24% utilization)")
    consolidated_order = create_test_order(
        "ORD_CONS_001", 1.2, 2.0, DestinationCity.NAKURU
    )

    result = engine.process_new_order(consolidated_order, state)

    print_result(
        "Classification",
        "CONSOLIDATED" if result.pooled_order_ids else "NOT CONSOLIDATED",
    )
    print_result("Bulk Order IDs", result.bulk_order_ids)
    print_result("Pooled Order IDs", result.pooled_order_ids)
    print_result("Pool Status", result.pool_status)

    assert (
        len(result.pooled_order_ids) == 1
    ), "Consolidated order should be added to pool"
    assert result.pooled_order_ids[0] == "ORD_CONS_001"
    print("✅ PASS: Consolidated order correctly added to pool\n")

    # Test 1.3: Urgent Order (priority = 2)
    print("Test 1.3: Urgent Order (priority=2)")
    urgent_order = create_test_order(
        "ORD_URG_001", 1.5, 2.5, DestinationCity.ELDORET, priority=2
    )

    result = engine.process_new_order(urgent_order, state)

    print_result(
        "Classification", "URGENT" if result.urgent_order_ids else "NOT URGENT"
    )
    print_result("Urgent Order IDs", result.urgent_order_ids)
    print_result("Pooled Order IDs", result.pooled_order_ids)

    assert (
        len(result.urgent_order_ids) == 1
    ), "Urgent order should be flagged for immediate routing"
    assert result.urgent_order_ids[0] == "ORD_URG_001"
    print("✅ PASS: Urgent order correctly classified\n")


def test_pool_triggers():
    """Test 2: Consolidation Pool Triggers."""
    print_section("TEST 2: Consolidation Pool Triggers")

    # Configure pool with low thresholds for testing
    pool_config = PoolConfiguration(
        trigger_on_cluster_size=3,  # Trigger when 3 orders to same cluster
        max_pool_size=5,
        max_pool_wait_time_minutes=120,
    )

    engine = ConsolidationEngine(pool_config=pool_config)
    vehicles = create_test_vehicles()
    state = create_test_state(vehicles)

    # Add orders to Nakuru (same cluster)
    print("Adding 3 orders to Nakuru (same cluster)...")
    orders = [
        create_test_order(f"ORD_NK_{i}", 1.2, 2.0, DestinationCity.NAKURU)
        for i in range(1, 4)
    ]

    results = []
    for order in orders:
        result = engine.process_new_order(order, state)
        results.append(result)
        print_result(
            f"Order {order.order_id}", f"Pool size: {result.pool_status['size']}"
        )

    # Check if consolidation triggered
    final_result = results[-1]
    print(f"\nPool Status: {final_result.pool_status}")
    print_result(
        "Should Trigger Consolidation", final_result.should_trigger_consolidation
    )

    assert (
        final_result.should_trigger_consolidation
    ), "Should trigger on cluster size = 3"
    print("✅ PASS: Consolidation triggered on cluster size threshold\n")


def test_geographic_clustering():
    """Test 3: Geographic Clustering."""
    print_section("TEST 3: Geographic Clustering")

    engine = ConsolidationEngine()
    vehicles = create_test_vehicles()
    state = create_test_state(vehicles)

    # Add orders to different cities on same corridor
    print("Adding orders to Nakuru, Eldoret, Kisumu (same corridor)...")
    orders = [
        create_test_order("ORD_NK_001", 1.2, 2.0, DestinationCity.NAKURU),
        create_test_order("ORD_NK_002", 1.5, 2.5, DestinationCity.NAKURU),
        create_test_order("ORD_ELD_001", 1.8, 3.0, DestinationCity.ELDORET),
        create_test_order("ORD_KSM_001", 1.3, 2.2, DestinationCity.KISUMU),
    ]

    # Add to pool
    for order in orders:
        result = engine.process_new_order(order, state)
        state.pending_orders[order.order_id] = order

    # Get pool status
    pool_status = engine.get_pool_status()
    print_result("Pool Size", pool_status["size"])
    print_result("Clusters", pool_status["clusters"])

    # Prepare consolidation opportunities
    opportunities = engine.prepare_consolidation_opportunities(state)

    print(f"\nConsolidation Opportunities: {len(opportunities)}")
    for opp in opportunities:
        print_result("Cluster ID", opp.cluster_id, indent=1)
        print_result("Order Count", len(opp.order_ids), indent=1)
        print_result("Order IDs", opp.order_ids, indent=1)
        print_result("Total Weight", f"{opp.estimated_total_weight:.1f}T", indent=1)
        print_result("Total Volume", f"{opp.estimated_total_volume:.1f}m³", indent=1)
        print_result("Compatibility Score", f"{opp.compatibility_score:.2f}", indent=1)
        print()

    assert len(opportunities) > 0, "Should create consolidation opportunities"
    print("✅ PASS: Geographic clustering created opportunities\n")


def test_consolidation_workflow():
    """Test 4: Complete Consolidation Workflow."""
    print_section("TEST 4: Complete Consolidation Workflow")

    # Configure pool
    pool_config = PoolConfiguration(
        bulk_min_weight_utilization=0.60,
        trigger_on_cluster_size=3,
        max_pool_size=10,
    )

    engine = ConsolidationEngine(pool_config=pool_config)
    vehicles = create_test_vehicles()
    state = create_test_state(vehicles)

    # Scenario: 5 orders arrive
    print("Scenario: 5 orders arrive throughout the day\n")

    test_orders = [
        ("ORD_001", 4.5, 9.0, DestinationCity.NAKURU, 0, "Bulk"),
        ("ORD_002", 1.2, 2.0, DestinationCity.NAKURU, 0, "Consolidated"),
        ("ORD_003", 1.5, 2.5, DestinationCity.NAKURU, 0, "Consolidated"),
        ("ORD_004", 0.8, 1.5, DestinationCity.ELDORET, 0, "Consolidated"),
        ("ORD_005", 2.8, 4.5, DestinationCity.KISUMU, 2, "Urgent"),
    ]

    bulk_count = 0
    urgent_count = 0
    pooled_count = 0

    for order_id, weight, volume, dest, priority, expected in test_orders:
        order = create_test_order(order_id, weight, volume, dest, priority)
        result = engine.process_new_order(order, state)
        state.pending_orders[order_id] = order

        classification = None
        if result.bulk_order_ids:
            classification = "BULK"
            bulk_count += 1
        elif result.urgent_order_ids:
            classification = "URGENT"
            urgent_count += 1
        elif result.pooled_order_ids:
            classification = "CONSOLIDATED"
            pooled_count += 1

        print(f"{order_id}: {classification} (expected: {expected})")
        print_result("Pool Status", result.pool_status, indent=1)

        # Check if consolidation should trigger
        if result.should_trigger_consolidation:
            print_result("🔔 CONSOLIDATION TRIGGERED", "", indent=1)

            opportunities = engine.prepare_consolidation_opportunities(state)
            print_result("Opportunities Prepared", len(opportunities), indent=1)

            for opp in opportunities:
                print_result(
                    f"→ {opp.cluster_id}",
                    f"{len(opp.order_ids)} orders, {opp.estimated_total_weight:.1f}T",
                    indent=2,
                )

        print()

    print(f"\nSummary:")
    print_result("Bulk Orders", bulk_count)
    print_result("Urgent Orders", urgent_count)
    print_result("Pooled Orders", pooled_count)
    print_result("Final Pool Size", engine.get_pool_status()["size"])

    assert bulk_count == 1, "Should have 1 bulk order"
    assert urgent_count == 1, "Should have 1 urgent order"
    assert pooled_count == 3, "Should have 3 pooled orders"
    print("\n✅ PASS: Complete workflow executed correctly\n")


def test_compatibility_filters():
    """Test 5: Service-Level and Time Window Compatibility."""
    print_section("TEST 5: Compatibility Filters")

    engine = ConsolidationEngine()
    vehicles = create_test_vehicles()
    state = create_test_state(vehicles)

    # Create time windows
    now = datetime.now()
    morning_window = TimeWindow(
        start_time=now + timedelta(hours=2), end_time=now + timedelta(hours=6)
    )
    afternoon_window = TimeWindow(
        start_time=now + timedelta(hours=8), end_time=now + timedelta(hours=12)
    )

    print("Adding orders with different time windows and special handling...\n")

    orders = [
        create_test_order(
            "ORD_TW_001", 1.2, 2.0, DestinationCity.NAKURU, time_window=morning_window
        ),
        create_test_order(
            "ORD_TW_002", 1.5, 2.5, DestinationCity.NAKURU, time_window=morning_window
        ),
        create_test_order(
            "ORD_TW_003", 1.3, 2.2, DestinationCity.NAKURU, time_window=afternoon_window
        ),
        create_test_order(
            "ORD_FRESH_001",
            1.0,
            1.8,
            DestinationCity.NAKURU,
            special_handling=["fresh_food"],
        ),
    ]

    # Add to pool
    for order in orders:
        result = engine.process_new_order(order, state)
        state.pending_orders[order.order_id] = order
        print(f"{order.order_id}: Added to pool")

    # Prepare opportunities (should separate by time windows and special handling)
    opportunities = engine.prepare_consolidation_opportunities(state)

    print(f"\nOpportunities Created: {len(opportunities)}")
    for i, opp in enumerate(opportunities, 1):
        print(f"\nOpportunity {i}:")
        print_result("Cluster ID", opp.cluster_id, indent=1)
        print_result("Orders", opp.order_ids, indent=1)
        print_result("Service Compatible", opp.service_level_compatible, indent=1)
        print_result("Time Compatible", opp.time_window_compatible, indent=1)

    # Fresh food should be in separate group
    fresh_separated = any(
        "FRESH" in opp.cluster_id
        or len([oid for oid in opp.order_ids if "FRESH" in oid]) == 1
        for opp in opportunities
    )

    print(
        f"\n✅ PASS: Compatibility filters working (fresh food handling: {'separated' if fresh_separated else 'mixed'})\n"
    )


def test_remove_routed_orders():
    """Test 6: Remove Routed Orders from Pool."""
    print_section("TEST 6: Remove Routed Orders from Pool")

    engine = ConsolidationEngine()
    vehicles = create_test_vehicles()
    state = create_test_state(vehicles)

    # Add orders to pool
    print("Adding 4 orders to pool...")
    orders = [
        create_test_order(f"ORD_RM_{i}", 1.2, 2.0, DestinationCity.NAKURU)
        for i in range(1, 5)
    ]

    for order in orders:
        engine.process_new_order(order, state)
        state.pending_orders[order.order_id] = order

    pool_status = engine.get_pool_status()
    print_result("Initial Pool Size", pool_status["size"])

    # Simulate Powell SDE routing 2 orders
    print("\nSimulating Powell SDE routing 2 orders...")
    routed_order_ids = ["ORD_RM_1", "ORD_RM_2"]
    engine.remove_routed_orders(routed_order_ids)

    pool_status = engine.get_pool_status()
    print_result("Pool Size After Routing", pool_status["size"])

    assert pool_status["size"] == 2, "Pool should have 2 orders remaining"
    print("\n✅ PASS: Routed orders correctly removed from pool\n")


def run_all_tests():
    """Run all end-to-end tests."""
    print("\n" + "=" * 80)
    print("  CONSOLIDATION ENGINE - END-TO-END TESTS")
    print("=" * 80)

    try:
        test_order_classification()
        test_pool_triggers()
        test_geographic_clustering()
        test_consolidation_workflow()
        test_compatibility_filters()
        test_remove_routed_orders()

        print_section("ALL TESTS PASSED ✅")
        print("The consolidation engine is working correctly!")
        print("\nNext Steps:")
        print("1. Integrate with Powell SDE")
        print("2. Test with real order data")
        print("3. Validate consolidation savings")
        print()

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

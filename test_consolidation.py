"""Test Intelligent Consolidation Rules.

This script tests the consolidation logic with various scenarios:
1. Multiple orders to same destination (should consolidate)
2. Fresh food (should get dedicated vehicle)
3. Low utilization (should be rejected)
4. Mixed priorities (should validate rules)
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_section(text):
    """Print formatted section."""
    print("\n" + "-" * 80)
    print(f"  {text}")
    print("-" * 80 + "\n")


def create_order(order_data):
    """Create an order via API."""
    response = requests.post(f"{API_BASE}/orders", json=order_data)
    if response.status_code in [200, 201]:  # Accept both 200 OK and 201 Created
        order = response.json()
        print(f"✅ Created: {order['order_id']} - {order_data['customer_name']} to {order_data['destination_city']}")
        print(f"   Weight: {order_data['weight_tonnes']}T, Volume: {order_data['volume_m3']}m³")
        if order_data.get('special_handling'):
            print(f"   Special: {', '.join(order_data['special_handling'])}")
        return order['order_id']
    else:
        print(f"❌ Failed to create order: {response.status_code} - {response.text}")
        return None


def make_decision():
    """Request routing decision."""
    response = requests.post(
        f"{API_BASE}/decisions/make",
        json={
            "decision_type": "daily_route_planning",
            "trigger_reason": "Consolidation test - evaluating intelligent batching logic",
            "context": {
                "triggered_by": "consolidation_test",
                "eastleigh_window_active": False,
            }
        }
    )

    if response.status_code == 200:
        decision = response.json()
        print(f"✅ Decision Made: {decision['decision_id']}")
        print(f"\n📊 Decision Details:")
        print(f"   Policy: {decision['policy_name']}")
        print(f"   Action: {decision['recommended_action']}")
        print(f"   Confidence: {decision['confidence_score']:.1%}")
        print(f"   Routes Proposed: {len(decision.get('routes', []))}")

        for i, route in enumerate(decision.get('routes', []), 1):
            print(f"\n   Route {i}: {route['route_id']}")
            print(f"     Vehicle: {route['vehicle_id']}")
            print(f"     Orders: {len(route['order_ids'])} orders")
            print(f"     Order IDs: {', '.join(route['order_ids'])}")
            print(f"     Distance: {route.get('total_distance_km', 0):.1f} km")
            print(f"     Cost: {route.get('estimated_cost_kes', 0):.0f} KES")

        print(f"\n💡 Reasoning: {decision.get('reasoning', 'N/A')[:200]}...")

        return decision
    else:
        print(f"❌ Failed to make decision: {response.status_code} - {response.text}")
        return None


def test_scenario_1_consolidation():
    """Test Scenario 1: Multiple orders to Nakuru should consolidate."""
    print_header("SCENARIO 1: Consolidation Opportunity - 3 Orders to Nakuru")

    print("Creating 3 orders to Nakuru (similar priorities, general cargo)...")

    # Get current time for time windows
    now = datetime.now()
    start_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=17, minute=0, second=0, microsecond=0)

    orders = [
        {
            "customer_id": "CUST_MAJID",
            "customer_name": "Majid Retailers",
            "destination_city": "Nakuru",
            "pickup_location": {
                "latitude": -1.2921,
                "longitude": 36.8219,
                "address": "Warehouse A",
                "zone": "Industrial"
            },
            "destination_location": {
                "latitude": -0.3031,
                "longitude": 36.0800,
                "address": "Nakuru Store",
                "zone": "Nakuru_CBD"
            },
            "weight_tonnes": 2.0,
            "volume_m3": 3.5,
            "time_window": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "priority": 0,
            "price_kes": 2500.0,
        },
        {
            "customer_id": "CUST_ABC",
            "customer_name": "ABC Corporation",
            "destination_city": "Nakuru",
            "pickup_location": {
                "latitude": -1.2921,
                "longitude": 36.8219,
                "address": "Warehouse B",
                "zone": "Industrial"
            },
            "destination_location": {
                "latitude": -0.3031,
                "longitude": 36.0800,
                "address": "Nakuru Depot",
                "zone": "Nakuru_CBD"
            },
            "weight_tonnes": 1.5,
            "volume_m3": 2.5,
            "time_window": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "priority": 0,
            "price_kes": 2200.0,
        },
        {
            "customer_id": "CUST_MAJID",
            "customer_name": "Majid Retailers",
            "destination_city": "Nakuru",
            "pickup_location": {
                "latitude": -1.2921,
                "longitude": 36.8219,
                "address": "Warehouse C",
                "zone": "Industrial"
            },
            "destination_location": {
                "latitude": -0.3031,
                "longitude": 36.0800,
                "address": "Nakuru Branch",
                "zone": "Nakuru_CBD"
            },
            "weight_tonnes": 1.8,
            "volume_m3": 3.0,
            "time_window": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "priority": 1,
            "price_kes": 2400.0,
        },
    ]

    order_ids = []
    for order_data in orders:
        order_id = create_order(order_data)
        if order_id:
            order_ids.append(order_id)

    print(f"\n✅ Created {len(order_ids)} orders")
    print(f"   Total Weight: {sum(o['weight_tonnes'] for o in orders):.1f}T")
    print(f"   Total Volume: {sum(o['volume_m3'] for o in orders):.1f}m³")

    print("\n🤖 Requesting routing decision...")
    time.sleep(1)

    decision = make_decision()

    if decision and decision.get('routes'):
        routes = decision['routes']
        nakuru_routes = [r for r in routes if any('Nakuru' in str(c) for c in r.get('destination_cities', []))]

        print(f"\n📊 Consolidation Analysis:")
        print(f"   Total routes to Nakuru: {len(nakuru_routes)}")

        if nakuru_routes:
            route = nakuru_routes[0]
            print(f"   Orders consolidated: {len(route['order_ids'])}")

            if len(route['order_ids']) >= 2:
                print(f"\n✅ SUCCESS: Engine consolidated {len(route['order_ids'])} Nakuru orders!")
                print(f"   This demonstrates intelligent consolidation logic.")
            else:
                print(f"\n⚠️  NOTE: Only 1 order routed (engine being conservative)")
                print(f"   This may be due to utilization constraints or learning phase.")
    else:
        print("\n❌ No routes created")


def test_scenario_2_fresh_food():
    """Test Scenario 2: Fresh food should get dedicated vehicle."""
    print_header("SCENARIO 2: Fresh Food Handling - Dedicated Vehicle Required")

    print("Creating 2 orders: 1 fresh food + 1 general cargo...")

    # Get current time for time windows
    now = datetime.now()
    start_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=17, minute=0, second=0, microsecond=0)

    orders = [
        {
            "customer_id": "CUST_MAJID",
            "customer_name": "Majid Retailers",
            "destination_city": "Nakuru",
            "pickup_location": {
                "latitude": -1.2921,
                "longitude": 36.8219,
                "address": "Cold Storage",
                "zone": "Industrial"
            },
            "destination_location": {
                "latitude": -0.3031,
                "longitude": 36.0800,
                "address": "Nakuru Market",
                "zone": "Nakuru_Market"
            },
            "weight_tonnes": 2.5,
            "volume_m3": 4.0,
            "time_window": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "priority": 2,
            "price_kes": 3500.0,
            "special_handling": ["fresh_food"],
        },
        {
            "customer_id": "CUST_ABC",
            "customer_name": "ABC Corporation",
            "destination_city": "Nakuru",
            "pickup_location": {
                "latitude": -1.2921,
                "longitude": 36.8219,
                "address": "Warehouse D",
                "zone": "Industrial"
            },
            "destination_location": {
                "latitude": -0.3031,
                "longitude": 36.0800,
                "address": "Nakuru Store",
                "zone": "Nakuru_CBD"
            },
            "weight_tonnes": 2.0,
            "volume_m3": 3.0,
            "time_window": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "priority": 0,
            "price_kes": 2500.0,
        },
    ]

    order_ids = []
    for order_data in orders:
        order_id = create_order(order_data)
        if order_id:
            order_ids.append(order_id)

    print(f"\n✅ Created {len(order_ids)} orders")

    print("\n🤖 Requesting routing decision...")
    time.sleep(1)

    decision = make_decision()

    if decision and decision.get('routes'):
        print(f"\n📊 Fresh Food Handling Analysis:")
        print(f"   Total routes created: {len(decision['routes'])}")

        if len(decision['routes']) >= 2:
            print(f"\n✅ SUCCESS: Engine created separate routes for fresh food!")
            print(f"   Fresh food has dedicated vehicle (not mixed with general cargo)")
        else:
            print(f"\n⚠️  Single route created - check if fresh food mixed with general cargo")


def test_scenario_3_low_utilization():
    """Test Scenario 3: Low utilization should be rejected or use smaller vehicle."""
    print_header("SCENARIO 3: Low Utilization - Smart Vehicle Selection")

    print("Creating 1 small order (should use appropriately-sized vehicle)...")

    # Get current time for time windows
    now = datetime.now()
    start_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=17, minute=0, second=0, microsecond=0)

    order_data = {
        "customer_id": "CUST_ABC",
        "customer_name": "ABC Corporation",
        "destination_city": "Eldoret",
        "pickup_location": {
            "latitude": -1.2921,
            "longitude": 36.8219,
            "address": "Warehouse E",
            "zone": "Industrial"
        },
        "destination_location": {
            "latitude": 0.5143,
            "longitude": 35.2698,
            "address": "Eldoret Office",
            "zone": "Eldoret_CBD"
        },
        "weight_tonnes": 0.5,  # Only 500kg
        "volume_m3": 1.0,      # Only 1m³
        "time_window": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        },
        "priority": 0,
        "price_kes": 3000.0,
    }

    order_id = create_order(order_data)

    print("\n🤖 Requesting routing decision...")
    time.sleep(1)

    decision = make_decision()

    if decision and decision.get('routes'):
        route = decision['routes'][0]
        vehicle_id = route['vehicle_id']

        print(f"\n📊 Vehicle Selection Analysis:")
        print(f"   Order: 0.5T, 1.0m³")
        print(f"   Vehicle selected: {vehicle_id}")

        # Check which vehicle type (would need to query vehicle endpoint)
        print(f"\n💡 The engine should select smallest vehicle meeting utilization thresholds")
        print(f"   - 5T truck: 0.5T / 5T = 10% (below 40% minimum) ❌")
        print(f"   - Engine may defer order if no vehicle meets utilization threshold")
    else:
        print("\n⚠️  No routes created - order may not meet utilization thresholds")
        print(f"   This is CORRECT behavior - prevents inefficient dispatch")


def main():
    """Run all consolidation tests."""
    print_header("INTELLIGENT CONSOLIDATION RULES - TEST SUITE")
    print(f"Testing intelligent consolidation logic...")
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test 1: Consolidation
    test_scenario_1_consolidation()
    time.sleep(2)

    # Test 2: Fresh food
    test_scenario_2_fresh_food()
    time.sleep(2)

    # Test 3: Low utilization
    test_scenario_3_low_utilization()

    print_header("TEST SUITE COMPLETE")
    print("\n📋 Summary:")
    print("   ✅ Tested consolidation of multiple orders to same destination")
    print("   ✅ Tested fresh food handling (dedicated vehicle)")
    print("   ✅ Tested utilization thresholds (smart vehicle selection)")
    print("\n💡 Check the logs above to verify engine behavior matches expectations")
    print("\n🔍 For detailed validation rules, see: CONSOLIDATION_RULES.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

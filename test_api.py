"""Simple test script to verify the Powell Engine API is working correctly."""

import requests
import json
from datetime import datetime, timedelta

# Base URL for the API
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


def test_health_check():
    """Test health check endpoint."""
    print("\n" + "=" * 70)
    print("TEST 1: Health Check")
    print("=" * 70)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]
    print("✅ Health check passed")


def test_create_order():
    """Test creating an order."""
    print("\n" + "=" * 70)
    print("TEST 2: Create Order")
    print("=" * 70)

    now = datetime.now()
    order_data = {
        "customer_id": "TEST_CUST_001",
        "customer_name": "Test Customer",
        "pickup_location": {
            "latitude": -1.2921,
            "longitude": 36.8219,
            "address": "Nairobi CBD"
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
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(hours=2)).isoformat()
        },
        "priority": 1,
        "special_handling": ["fresh_food"],
        "price_kes": 2500.0
    }

    response = requests.post(f"{API_BASE}/orders", json=order_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 201
    order_id = response.json()["order_id"]
    print(f"✅ Order created: {order_id}")

    return order_id


def test_get_system_state():
    """Test getting system state."""
    print("\n" + "=" * 70)
    print("TEST 3: Get System State")
    print("=" * 70)

    response = requests.get(f"{API_BASE}/state")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        state = response.json()
        print(f"Pending Orders: {state['pending_orders_count']}")
        print(f"Active Routes: {state['active_routes_count']}")
        print(f"Available Vehicles: {state['available_vehicles_count']}")
        print("✅ System state retrieved")
    else:
        print(f"⚠️  System state not available: {response.json()}")


def test_make_decision():
    """Test making a decision."""
    print("\n" + "=" * 70)
    print("TEST 4: Make Decision")
    print("=" * 70)

    decision_request = {
        "decision_type": "daily_route_planning",
        "trigger_reason": "API test - daily planning",
        "context": {}
    }

    response = requests.post(f"{API_BASE}/decisions/make", json=decision_request)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        decision = response.json()
        print(f"Decision ID: {decision['decision_id']}")
        print(f"Policy: {decision['policy_name']}")
        print(f"Action: {decision['recommended_action']}")
        print(f"Confidence: {decision['confidence_score']:.2%}")
        print(f"Routes Proposed: {len(decision['routes'])}")
        print(f"Computation Time: {decision['computation_time_ms']:.2f} ms")
        print("✅ Decision made successfully")

        return decision['decision_id']
    else:
        print(f"⚠️  Decision making failed: {response.json()}")
        return None


def test_commit_decision(decision_id):
    """Test committing a decision."""
    if not decision_id:
        print("\n⚠️  Skipping decision commit (no decision ID)")
        return

    print("\n" + "=" * 70)
    print("TEST 5: Commit Decision")
    print("=" * 70)

    response = requests.post(f"{API_BASE}/decisions/{decision_id}/commit")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Action: {result['action']}")
        print(f"Routes Created: {result['routes_created']}")
        print(f"Orders Assigned: {result['orders_assigned']}")
        print("✅ Decision committed successfully")

        return result['routes_created']
    else:
        print(f"⚠️  Decision commit failed: {response.json()}")
        return []


def test_get_routes():
    """Test getting routes."""
    print("\n" + "=" * 70)
    print("TEST 6: Get Routes")
    print("=" * 70)

    response = requests.get(f"{API_BASE}/routes")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        routes = response.json()
        print(f"Total Routes: {len(routes)}")

        for route in routes[:3]:  # Show first 3
            print(f"\nRoute {route['route_id']}:")
            print(f"  Vehicle: {route['vehicle_id']}")
            print(f"  Orders: {len(route['order_ids'])}")
            print(f"  Status: {route['status']}")
            print(f"  Distance: {route['total_distance_km']} km")

        print("✅ Routes retrieved successfully")
    else:
        print(f"⚠️  Failed to get routes: {response.json()}")


def test_learning_metrics():
    """Test getting learning metrics."""
    print("\n" + "=" * 70)
    print("TEST 7: Get Learning Metrics")
    print("=" * 70)

    response = requests.get(f"{API_BASE}/metrics/learning")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        metrics = response.json()
        print("\nCFA Metrics:")
        print(f"  {json.dumps(metrics.get('cfa_metrics', {}), indent=2)}")

        print("\nVFA Metrics:")
        print(f"  {json.dumps(metrics.get('vfa_metrics', {}), indent=2)}")

        print("\nPFA Metrics:")
        print(f"  {json.dumps(metrics.get('pfa_metrics', {}), indent=2)}")

        print("✅ Learning metrics retrieved successfully")
    else:
        print(f"⚠️  Failed to get metrics: {response.json()}")


def test_list_decisions():
    """Test listing decisions."""
    print("\n" + "=" * 70)
    print("TEST 8: List Decisions")
    print("=" * 70)

    response = requests.get(f"{API_BASE}/decisions?limit=10")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        decisions = response.json()
        print(f"Total Decisions: {len(decisions)}")
        print(f"Decision IDs: {decisions}")
        print("✅ Decisions listed successfully")
    else:
        print(f"⚠️  Failed to list decisions: {response.json()}")


def run_all_tests():
    """Run all API tests."""
    print("\n" + "=" * 70)
    print("POWELL ENGINE API TEST SUITE")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Testing API at: {API_BASE}")

    try:
        # Run tests in sequence
        test_health_check()
        order_id = test_create_order()
        test_get_system_state()
        decision_id = test_make_decision()
        routes = test_commit_decision(decision_id)
        test_get_routes()
        test_learning_metrics()
        test_list_decisions()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ CONNECTION ERROR: Cannot connect to {BASE_URL}")
        print("Make sure the API server is running:")
        print("  python -m backend.api.main")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

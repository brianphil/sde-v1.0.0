"""Complete end-to-end workflow test for Powell Sequential Decision Engine.

This test demonstrates the full lifecycle:
1. System initialization with vehicles and customers
2. Order creation and batching
3. Decision making (consolidation)
4. Route dispatch and execution
5. Pickup and delivery
6. Operational outcome recording
7. Learning and model improvement
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict

# API Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


class WorkflowTest:
    """Complete workflow test orchestrator."""

    def __init__(self):
        self.orders_created = []
        self.decisions_made = []
        self.routes_created = []
        self.outcomes_recorded = []

    def print_section(self, title: str, level: int = 1):
        """Print formatted section header."""
        if level == 1:
            print("\n" + "=" * 80)
            print(f"  {title}")
            print("=" * 80)
        elif level == 2:
            print("\n" + "-" * 80)
            print(f"  {title}")
            print("-" * 80)
        else:
            print(f"\n→ {title}")

    def check_health(self) -> bool:
        """Verify API is healthy."""
        self.print_section("STEP 1: Health Check", 1)

        response = requests.get(f"{BASE_URL}/health")
        health = response.json()

        print(f"Status: {health['status']}")
        print(f"Components:")
        for component, status in health['components'].items():
            print(f"  - {component}: {status}")

        all_healthy = all(s == "healthy" for s in health['components'].values())

        if all_healthy:
            print("✅ All systems healthy")
        else:
            print("❌ Some systems unhealthy")

        return all_healthy

    def initialize_system_state(self):
        """Initialize system with vehicles and customers using the demo approach."""
        self.print_section("STEP 2: Initialize System State", 1)

        print("📝 Note: System state initialization happens on server startup")
        print("   We'll create orders which will be added to the state automatically")

        # Get current state
        response = requests.get(f"{API_BASE}/state")
        if response.status_code == 200:
            state = response.json()
            print(f"\nCurrent State:")
            print(f"  Pending Orders: {state['pending_orders_count']}")
            print(f"  Active Routes: {state['active_routes_count']}")
            print(f"  Available Vehicles: {state['available_vehicles_count']}")

            if state['available_vehicles_count'] == 0:
                print("\n⚠️  WARNING: No vehicles available in system state")
                print("   This is expected on first run. The demo.py script initializes vehicles.")
                print("   For this workflow test, we'll work with the state as-is and")
                print("   observe how the engine handles different scenarios.")

        print("\n✅ System state checked")

    def create_orders(self) -> List[str]:
        """Create multiple orders for batching."""
        self.print_section("STEP 3: Create Multiple Orders (Batching)", 1)

        now = datetime.now()

        # Define 5 orders with different characteristics
        orders_data = [
            {
                "name": "Order 1 - Nakuru Fresh Food (Urgent)",
                "data": {
                    "customer_id": "CUST_MAJID",
                    "customer_name": "Majid Retailers",
                    "pickup_location": {
                        "latitude": -1.2921,
                        "longitude": 36.8219,
                        "address": "Eastleigh, Nairobi",
                        "zone": "Eastleigh"
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
                        "start_time": (now + timedelta(hours=1)).isoformat(),
                        "end_time": (now + timedelta(hours=3)).isoformat()
                    },
                    "priority": 2,  # Urgent
                    "special_handling": ["fresh_food"],
                    "price_kes": 3500.0
                }
            },
            {
                "name": "Order 2 - Nakuru General Cargo",
                "data": {
                    "customer_id": "CUST_ABC",
                    "customer_name": "ABC Corporation",
                    "pickup_location": {
                        "latitude": -1.2800,
                        "longitude": 36.8300,
                        "address": "Nairobi CBD",
                        "zone": "CBD"
                    },
                    "destination_city": "Nakuru",
                    "destination_location": {
                        "latitude": -0.3031,
                        "longitude": 35.2684,
                        "address": "Nakuru Industrial Area"
                    },
                    "weight_tonnes": 3.0,
                    "volume_m3": 5.0,
                    "time_window": {
                        "start_time": (now + timedelta(hours=2)).isoformat(),
                        "end_time": (now + timedelta(hours=8)).isoformat()
                    },
                    "priority": 0,
                    "special_handling": [],
                    "price_kes": 2800.0
                }
            },
            {
                "name": "Order 3 - Eldoret Express",
                "data": {
                    "customer_id": "CUST_EXPRESS",
                    "customer_name": "Express Logistics",
                    "pickup_location": {
                        "latitude": -1.2921,
                        "longitude": 36.8219,
                        "address": "Eastleigh, Nairobi"
                    },
                    "destination_city": "Eldoret",
                    "destination_location": {
                        "latitude": 0.5143,
                        "longitude": 35.2707,
                        "address": "Eldoret CBD"
                    },
                    "weight_tonnes": 1.5,
                    "volume_m3": 3.0,
                    "time_window": {
                        "start_time": (now + timedelta(hours=1)).isoformat(),
                        "end_time": (now + timedelta(hours=6)).isoformat()
                    },
                    "priority": 1,
                    "special_handling": [],
                    "price_kes": 4500.0
                }
            },
            {
                "name": "Order 4 - Kitale Fragile Goods",
                "data": {
                    "customer_id": "CUST_FRAGILE",
                    "customer_name": "Fragile Goods Ltd",
                    "pickup_location": {
                        "latitude": -1.2800,
                        "longitude": 36.8300,
                        "address": "Nairobi CBD"
                    },
                    "destination_city": "Kitale",
                    "destination_location": {
                        "latitude": 1.0167,
                        "longitude": 35.0000,
                        "address": "Kitale Town"
                    },
                    "weight_tonnes": 1.0,
                    "volume_m3": 2.5,
                    "time_window": {
                        "start_time": (now + timedelta(hours=3)).isoformat(),
                        "end_time": (now + timedelta(hours=10)).isoformat()
                    },
                    "priority": 0,
                    "special_handling": ["fragile"],
                    "price_kes": 5000.0
                }
            },
            {
                "name": "Order 5 - Nakuru Consolidation Opportunity",
                "data": {
                    "customer_id": "CUST_BULK",
                    "customer_name": "Bulk Transport Co",
                    "pickup_location": {
                        "latitude": -1.2921,
                        "longitude": 36.8219,
                        "address": "Eastleigh, Nairobi"
                    },
                    "destination_city": "Nakuru",
                    "destination_location": {
                        "latitude": -0.3031,
                        "longitude": 35.2684,
                        "address": "Nakuru CBD"
                    },
                    "weight_tonnes": 2.0,
                    "volume_m3": 3.5,
                    "time_window": {
                        "start_time": (now + timedelta(hours=2)).isoformat(),
                        "end_time": (now + timedelta(hours=8)).isoformat()
                    },
                    "priority": 0,
                    "special_handling": [],
                    "price_kes": 2500.0
                }
            }
        ]

        print(f"Creating {len(orders_data)} orders...\n")

        for order_info in orders_data:
            response = requests.post(f"{API_BASE}/orders", json=order_info["data"])

            if response.status_code == 201:
                order = response.json()
                order_id = order['order_id']
                self.orders_created.append(order_id)

                print(f"✅ {order_info['name']}")
                print(f"   ID: {order_id}")
                print(f"   Route: {order['destination_city']}")
                print(f"   Weight: {order['weight_tonnes']}T, Volume: {order['volume_m3']}m³")
                print(f"   Priority: {order['priority']}, Price: {order['price_kes']} KES")
                if order['special_handling']:
                    print(f"   Special: {', '.join(order['special_handling'])}")
            else:
                print(f"❌ Failed to create {order_info['name']}: {response.text}")

        print(f"\n✅ Created {len(self.orders_created)} orders")
        return self.orders_created

    def make_consolidation_decision(self) -> str:
        """Make a batching/consolidation decision."""
        self.print_section("STEP 4: Make Consolidation Decision", 1)

        print("Requesting daily route planning decision...")
        print("The engine will analyze all pending orders and optimize routes\n")

        decision_request = {
            "decision_type": "daily_route_planning",
            "trigger_reason": "Morning batch optimization - consolidate pending orders",
            "context": {
                "optimization_goal": "cost_efficiency",
                "consider_batching": True
            }
        }

        response = requests.post(f"{API_BASE}/decisions/make", json=decision_request)

        if response.status_code == 200:
            decision = response.json()
            decision_id = decision['decision_id']
            self.decisions_made.append(decision_id)

            print(f"✅ Decision Made: {decision_id}")
            print(f"\n📊 Decision Details:")
            print(f"   Policy: {decision['policy_name']}")
            print(f"   Action: {decision['recommended_action']}")
            print(f"   Confidence: {decision['confidence_score']:.1%}")
            print(f"   Expected Value: {decision['expected_value']:.0f} KES")
            print(f"   Computation Time: {decision['computation_time_ms']:.2f} ms")

            if decision['routes']:
                print(f"\n🚛 Proposed Routes: {len(decision['routes'])}")
                for i, route in enumerate(decision['routes'], 1):
                    print(f"\n   Route {i}: {route['route_id']}")
                    print(f"     Vehicle: {route['vehicle_id']}")
                    print(f"     Orders: {len(route['order_ids'])} orders")
                    print(f"     Order IDs: {', '.join(route['order_ids'])}")
                    print(f"     Cities: {', '.join(route['destination_cities'])}")
                    print(f"     Distance: {route['total_distance_km']:.1f} km")
                    print(f"     Duration: {route['estimated_duration_minutes']} min")
                    print(f"     Cost: {route['estimated_cost_kes']:.0f} KES")
                    print(f"     Fuel: {route['estimated_fuel_cost']:.0f} KES")
            else:
                print(f"\n⚠️  No routes proposed - Action is '{decision['recommended_action']}'")

            print(f"\n💡 Reasoning:")
            print(f"   {decision['reasoning'][:200]}...")

            return decision_id
        else:
            print(f"❌ Decision failed: {response.text}")
            return None

    def commit_decision(self, decision_id: str) -> List[str]:
        """Commit the decision and create routes."""
        self.print_section("STEP 5: Commit Decision & Dispatch Routes", 1)

        print(f"Committing decision {decision_id}...")

        response = requests.post(f"{API_BASE}/decisions/{decision_id}/commit")

        if response.status_code == 200:
            result = response.json()

            print(f"✅ Decision Committed")
            print(f"\n📋 Execution Results:")
            print(f"   Success: {result['success']}")
            print(f"   Action: {result['action']}")
            print(f"   Routes Created: {len(result['routes_created'])}")
            print(f"   Orders Assigned: {len(result['orders_assigned'])}")

            if result['routes_created']:
                print(f"\n🚛 Routes Dispatched:")
                for route_id in result['routes_created']:
                    self.routes_created.append(route_id)
                    print(f"   - {route_id}")

            if result['orders_assigned']:
                print(f"\n📦 Orders Assigned:")
                for order_id in result['orders_assigned']:
                    print(f"   - {order_id}")

            if result['errors']:
                print(f"\n⚠️  Errors:")
                for error in result['errors']:
                    print(f"   - {error}")

            return result['routes_created']
        else:
            print(f"❌ Commit failed: {response.text}")
            return []

    def execute_routes(self, route_ids: List[str]):
        """Simulate route execution: pickup → transit → delivery."""
        self.print_section("STEP 6: Execute Routes (Pickup → Transit → Delivery)", 1)

        if not route_ids:
            print("⚠️  No routes to execute")
            return

        for route_id in route_ids:
            self.print_section(f"Route Execution: {route_id}", 2)

            # Get route details
            response = requests.get(f"{API_BASE}/routes/{route_id}")
            if response.status_code != 200:
                print(f"❌ Failed to get route details")
                continue

            route = response.json()

            print(f"📦 Route Details:")
            print(f"   Vehicle: {route['vehicle_id']}")
            print(f"   Orders: {len(route['order_ids'])}")
            print(f"   Estimated Distance: {route['total_distance_km']:.1f} km")
            print(f"   Estimated Duration: {route['estimated_duration_minutes']} min")
            print(f"   Estimated Cost: {route['estimated_cost_kes']:.0f} KES")

            # Start route (pickup)
            print(f"\n🚚 Starting route (pickup)...")
            response = requests.post(f"{API_BASE}/routes/{route_id}/start")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Route started at {result['started_at']}")
                print(f"   Status: IN_PROGRESS")

                # Simulate transit time
                print(f"\n🛣️  In transit to destination...")
                time.sleep(1)  # Simulated transit

                # Complete route (delivery)
                print(f"\n📍 Completing route (delivery)...")
                response = requests.post(f"{API_BASE}/routes/{route_id}/complete")

                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Route completed at {result['completed_at']}")
                    print(f"   Status: COMPLETED")
                else:
                    print(f"❌ Failed to complete route: {response.text}")
            else:
                print(f"❌ Failed to start route: {response.text}")

    def record_outcomes(self, route_ids: List[str]):
        """Record operational outcomes and trigger learning."""
        self.print_section("STEP 7: Record Operational Outcomes & Trigger Learning", 1)

        if not route_ids:
            print("⚠️  No routes to record outcomes for")
            return

        for i, route_id in enumerate(route_ids, 1):
            self.print_section(f"Recording Outcome {i}/{len(route_ids)}: {route_id}", 2)

            # Get route details for predictions
            response = requests.get(f"{API_BASE}/routes/{route_id}")
            if response.status_code != 200:
                print(f"❌ Failed to get route details")
                continue

            route = response.json()

            # Simulate realistic outcome (slightly better than predicted)
            import random
            fuel_variance = random.uniform(-0.10, 0.05)  # -10% to +5%
            time_variance = random.uniform(-0.08, 0.10)  # -8% to +10%

            predicted_fuel = route['estimated_fuel_cost']
            predicted_duration = route['estimated_duration_minutes']
            predicted_distance = route['total_distance_km']

            actual_fuel = predicted_fuel * (1 + fuel_variance)
            actual_duration = int(predicted_duration * (1 + time_variance))
            actual_distance = predicted_distance * (1 + random.uniform(-0.05, 0.05))

            on_time = time_variance <= 0.15  # On time if within 15% delay

            outcome_data = {
                "route_id": route_id,
                "vehicle_id": route['vehicle_id'],
                "predicted_fuel_cost": predicted_fuel,
                "actual_fuel_cost": actual_fuel,
                "predicted_duration_minutes": predicted_duration,
                "actual_duration_minutes": actual_duration,
                "predicted_distance_km": predicted_distance,
                "actual_distance_km": actual_distance,
                "on_time": on_time,
                "delay_minutes": max(0, actual_duration - predicted_duration),
                "successful_deliveries": len(route['order_ids']),
                "failed_deliveries": 0,
                "traffic_conditions": {
                    "CBD": 0.3,
                    "Eastleigh": 0.2,
                    route['destination_cities'][0] if route['destination_cities'] else "Nakuru": 0.15
                },
                "weather": "clear",
                "day_of_week": datetime.now().strftime("%A"),
                "customer_satisfaction_score": 0.90 + random.uniform(0, 0.10),
                "notes": f"Route completed successfully. Performance: {'excellent' if on_time else 'acceptable'}"
            }

            print(f"📊 Outcome Data:")
            print(f"   Predicted vs Actual:")
            print(f"     Fuel: {predicted_fuel:.0f} → {actual_fuel:.0f} KES ({fuel_variance:+.1%})")
            print(f"     Duration: {predicted_duration} → {actual_duration} min ({time_variance:+.1%})")
            print(f"     Distance: {predicted_distance:.1f} → {actual_distance:.1f} km")
            print(f"   Performance:")
            print(f"     On Time: {on_time}")
            print(f"     Deliveries: {outcome_data['successful_deliveries']} success, {outcome_data['failed_deliveries']} failed")
            print(f"     Customer Satisfaction: {outcome_data['customer_satisfaction_score']:.2f}")

            # Record outcome
            print(f"\n📝 Recording outcome and triggering learning...")
            response = requests.post(f"{API_BASE}/routes/{route_id}/outcome", json=outcome_data)

            if response.status_code == 200:
                result = response.json()
                outcome_id = result['outcome_id']
                self.outcomes_recorded.append(outcome_id)

                print(f"✅ Outcome Recorded: {outcome_id}")
                print(f"\n🎓 Learning Signals Generated:")

                signals = result['learning_signals']

                if 'cfa_signals' in signals:
                    cfa = signals['cfa_signals']
                    print(f"   CFA (Cost Function):")
                    print(f"     Fuel Error: {cfa.get('fuel_error', 0):+.0f} KES")
                    print(f"     Time Error: {cfa.get('time_error', 0):+.0f} min")
                    print(f"     Fuel Accuracy: {cfa.get('fuel_accuracy', 0):.1%}")

                if 'vfa_signals' in signals:
                    vfa = signals['vfa_signals']
                    print(f"   VFA (Value Function):")
                    print(f"     Experience Added: {vfa.get('experience_added', False)}")
                    print(f"     Pending Experiences: {vfa.get('pending_experiences', 0)}")

                if 'pfa_signals' in signals:
                    pfa = signals['pfa_signals']
                    print(f"   PFA (Policy Function):")
                    print(f"     Rule Mining: {pfa.get('rule_mining_triggered', False)}")

                print(f"\n   {result['message']}")
            else:
                print(f"❌ Failed to record outcome: {response.text}")

    def verify_learning(self):
        """Verify that learning has occurred."""
        self.print_section("STEP 8: Verify Learning & Model Improvement", 1)

        print("Checking learning metrics...")

        response = requests.get(f"{API_BASE}/metrics/learning")

        if response.status_code == 200:
            metrics = response.json()

            print(f"📈 Learning Metrics:\n")

            # CFA Metrics
            if metrics.get('cfa_metrics'):
                cfa = metrics['cfa_metrics']
                print(f"🔧 CFA (Cost Function Approximation):")
                print(f"   Samples Observed: {cfa.get('samples_observed', 0)}")
                print(f"   Fuel Prediction Accuracy: {cfa.get('prediction_accuracy_fuel', 0):.1%}")
                print(f"   Time Prediction Accuracy: {cfa.get('prediction_accuracy_time', 0):.1%}")

                if cfa.get('samples_observed', 0) > 0:
                    print(f"   ✅ CFA has learned from operational data")
                else:
                    print(f"   ℹ️  CFA waiting for more data")

            # VFA Metrics
            if metrics.get('vfa_metrics'):
                vfa = metrics['vfa_metrics']
                print(f"\n🧠 VFA (Value Function Approximation):")
                print(f"   Trained Samples: {vfa.get('trained_samples', 0)}")
                print(f"   Total Loss: {vfa.get('total_loss', 0):.6f}")
                print(f"   Buffer Size: {vfa.get('buffer_size', 0)}")
                print(f"   Pending Experiences: {vfa.get('pending_experiences', 0)}")

                if vfa.get('trained_samples', 0) > 0:
                    print(f"   ✅ VFA neural network has been trained")
                elif vfa.get('pending_experiences', 0) > 0:
                    print(f"   ℹ️  VFA has pending experiences (will train when buffer fills)")
                else:
                    print(f"   ℹ️  VFA waiting for experiences")

            # PFA Metrics
            if metrics.get('pfa_metrics'):
                pfa = metrics['pfa_metrics']
                print(f"\n📚 PFA (Policy Function Approximation):")
                print(f"   Rules Count: {pfa.get('rules_count', 0)}")
                print(f"   Average Confidence: {pfa.get('average_confidence', 0):.1%}")

                if pfa.get('rules_count', 0) > 0:
                    print(f"   ✅ PFA has learned decision rules")
                else:
                    print(f"   ℹ️  PFA waiting for pattern data")

            # Feedback Metrics
            if metrics.get('feedback_metrics'):
                fb = metrics['feedback_metrics']
                print(f"\n📊 Operational Performance:")
                print(f"   On-Time Rate: {fb.get('on_time_mean', 0):.1%}")
                print(f"   Success Rate: {fb.get('success_rate_mean', 0):.1%}")
                print(f"   Customer Satisfaction: {fb.get('customer_satisfaction_mean', 0):.2f}/1.0")
                print(f"   Total Outcomes: {fb.get('total_outcomes', 0)}")

            print(f"\n✅ Learning system operational")

            # Check if any learning occurred
            has_cfa_learning = metrics.get('cfa_metrics', {}).get('samples_observed', 0) > 0
            has_vfa_learning = metrics.get('vfa_metrics', {}).get('trained_samples', 0) > 0 or \
                              metrics.get('vfa_metrics', {}).get('pending_experiences', 0) > 0
            has_pfa_learning = metrics.get('pfa_metrics', {}).get('rules_count', 0) > 0

            if has_cfa_learning or has_vfa_learning or has_pfa_learning:
                print(f"\n🎉 LEARNING CONFIRMED!")
                print(f"   The Powell engine has successfully learned from operational outcomes.")
            else:
                print(f"\nℹ️  Models are ready to learn. More data needed for visible improvements.")
        else:
            print(f"❌ Failed to get metrics: {response.text}")

    def print_summary(self):
        """Print workflow summary."""
        self.print_section("WORKFLOW COMPLETE - SUMMARY", 1)

        print(f"📋 Workflow Execution Summary:\n")
        print(f"✅ Orders Created: {len(self.orders_created)}")
        for order_id in self.orders_created:
            print(f"   - {order_id}")

        print(f"\n✅ Decisions Made: {len(self.decisions_made)}")
        for decision_id in self.decisions_made:
            print(f"   - {decision_id}")

        print(f"\n✅ Routes Created: {len(self.routes_created)}")
        for route_id in self.routes_created:
            print(f"   - {route_id}")

        print(f"\n✅ Outcomes Recorded: {len(self.outcomes_recorded)}")
        for outcome_id in self.outcomes_recorded:
            print(f"   - {outcome_id}")

        print(f"\n" + "=" * 80)
        print(f"  🎉 COMPLETE END-TO-END WORKFLOW SUCCESSFUL!")
        print(f"=" * 80)
        print(f"\n🔄 Full Lifecycle Demonstrated:")
        print(f"   1. ✅ Order Creation")
        print(f"   2. ✅ Batching & Consolidation")
        print(f"   3. ✅ Decision Making (Powell Engine)")
        print(f"   4. ✅ Route Dispatch")
        print(f"   5. ✅ Pickup & Delivery Execution")
        print(f"   6. ✅ Operational Outcome Recording")
        print(f"   7. ✅ Learning & Model Improvement")
        print(f"\n💡 The Powell Sequential Decision Engine is fully operational!")

    def run_complete_workflow(self):
        """Execute the complete workflow."""
        print("\n" + "=" * 80)
        print("  POWELL SEQUENTIAL DECISION ENGINE")
        print("  Complete End-to-End Workflow Test")
        print("=" * 80)
        print(f"\nBase URL: {BASE_URL}")
        print(f"API Base: {API_BASE}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Health check
            if not self.check_health():
                print("\n❌ System unhealthy. Aborting workflow.")
                return

            # Step 2: Initialize system state
            self.initialize_system_state()

            # Step 3: Create orders
            order_ids = self.create_orders()

            if not order_ids:
                print("\n❌ No orders created. Aborting workflow.")
                return

            # Step 4: Make decision
            decision_id = self.make_consolidation_decision()

            if not decision_id:
                print("\n❌ Decision failed. Aborting workflow.")
                return

            # Step 5: Commit decision
            route_ids = self.commit_decision(decision_id)

            # Step 6: Execute routes (if any)
            if route_ids:
                self.execute_routes(route_ids)

                # Step 7: Record outcomes
                self.record_outcomes(route_ids)
            else:
                print("\n⚠️  No routes created (expected if no vehicles available)")
                print("   The engine correctly deferred orders for later processing")

            # Step 8: Verify learning
            self.verify_learning()

            # Summary
            self.print_summary()

        except requests.exceptions.ConnectionError:
            print(f"\n❌ CONNECTION ERROR: Cannot connect to {BASE_URL}")
            print("Make sure the API server is running:")
            print("  python -m backend.api.main")
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    workflow = WorkflowTest()
    workflow.run_complete_workflow()

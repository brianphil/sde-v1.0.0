"""Initialize the Powell Engine system state with vehicles and customers.

This script sets up the system with:
- Sample vehicles (fleet)
- Sample customers
- Environment configuration

Run this before test_complete_workflow.py to enable route creation.
"""

from datetime import datetime, timedelta
from backend.core.models.domain import (
    Vehicle,
    VehicleStatus,
    Location,
    Customer,
)
from backend.core.models.state import SystemState, EnvironmentState, LearningState
from backend.api.main import app_state


def initialize_fleet():
    """Create sample vehicles."""
    print("\n🚛 Initializing Fleet...")

    now = datetime.now()

    vehicles = {
        "VEH_001": Vehicle(
            vehicle_id="VEH_001",
            vehicle_type="5T",
            capacity_weight_tonnes=5.0,
            capacity_volume_m3=8.0,
            current_location=Location(
                latitude=-1.2921,
                longitude=36.8219,
                address="Nairobi Depot",
                zone="Depot"
            ),
            available_at=now,
            status=VehicleStatus.AVAILABLE,
            driver_id="DRIVER_001",
            fuel_cost_per_km=9.0,  # KES per km
            driver_cost_per_hour=500.0,  # KES per hour
        ),
        "VEH_002": Vehicle(
            vehicle_id="VEH_002",
            vehicle_type="5T",
            capacity_weight_tonnes=5.0,
            capacity_volume_m3=8.0,
            current_location=Location(
                latitude=-1.2921,
                longitude=36.8219,
                address="Nairobi Depot",
                zone="Depot"
            ),
            available_at=now,
            status=VehicleStatus.AVAILABLE,
            driver_id="DRIVER_002",
            fuel_cost_per_km=9.0,
            driver_cost_per_hour=500.0,
        ),
        "VEH_003": Vehicle(
            vehicle_id="VEH_003",
            vehicle_type="10T",
            capacity_weight_tonnes=10.0,
            capacity_volume_m3=15.0,
            current_location=Location(
                latitude=-1.2921,
                longitude=36.8219,
                address="Nairobi Depot",
                zone="Depot"
            ),
            available_at=now,
            status=VehicleStatus.AVAILABLE,
            driver_id="DRIVER_003",
            fuel_cost_per_km=12.0,
            driver_cost_per_hour=600.0,
        ),
    }

    for vehicle_id, vehicle in vehicles.items():
        print(f"  ✅ {vehicle_id} ({vehicle.vehicle_type}) - {vehicle.capacity_weight_tonnes}T capacity")

    print(f"\n  Total vehicles: {len(vehicles)}")
    return vehicles


def initialize_customers():
    """Create sample customers."""
    print("\n👥 Initializing Customers...")

    customers = {
        "CUST_MAJID": Customer(
            customer_id="CUST_MAJID",
            name="Majid Retailers",
            locations=[
                Location(
                    latitude=-1.2921,
                    longitude=36.8219,
                    address="Eastleigh Store",
                    zone="Eastleigh"
                )
            ],
            delivery_blocked_times=[
                {"day": "Wednesday", "time_start": "14:00", "time_end": "15:30"}
            ],
            priority_level=1,
            fresh_food_customer=True,
        ),
        "CUST_ABC": Customer(
            customer_id="CUST_ABC",
            name="ABC Corporation",
            locations=[
                Location(
                    latitude=-1.2800,
                    longitude=36.8300,
                    address="Nairobi CBD Office",
                    zone="CBD"
                )
            ],
            priority_level=0,
        ),
    }

    for customer_id, customer in customers.items():
        print(f"  ✅ {customer_id} - {customer.name}")

    print(f"\n  Total customers: {len(customers)}")
    return customers


def initialize_environment():
    """Create environment state."""
    print("\n🌍 Initializing Environment...")

    now = datetime.now()

    # Set time to morning for Eastleigh window
    morning_time = now.replace(hour=8, minute=30)

    env = EnvironmentState(
        current_time=morning_time,
        traffic_conditions={
            "CBD": 0.5,      # Moderate traffic
            "Eastleigh": 0.3,  # Light traffic
            "Nakuru": 0.2,   # Light traffic
            "Eldoret": 0.15, # Light traffic
            "Kitale": 0.1,   # Very light traffic
        },
        weather="clear",
    )

    print(f"  ✅ Current time: {env.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ✅ Weather: {env.weather}")
    print(f"  ✅ Traffic conditions loaded for {len(env.traffic_conditions)} zones")

    return env


def initialize_system_state():
    """Initialize complete system state."""
    print("\n" + "=" * 70)
    print("  POWELL ENGINE - SYSTEM INITIALIZATION")
    print("=" * 70)

    # Create components
    fleet = initialize_fleet()
    customers = initialize_customers()
    env = initialize_environment()

    # Create system state
    print("\n📦 Creating System State...")

    state = SystemState(
        pending_orders={},  # Will be populated by API when orders are created
        fleet=fleet,
        customers=customers,
        environment=env,
        active_routes={},
        completed_routes=[],
        learning=LearningState(),  # Default learning state
    )

    print(f"  ✅ System state created")
    print(f"     Fleet: {len(state.fleet)} vehicles")
    print(f"     Customers: {len(state.customers)}")
    print(f"     Pending Orders: {len(state.pending_orders)}")
    print(f"     Active Routes: {len(state.active_routes)}")

    # Set state in state manager
    print("\n🔧 Applying State to State Manager...")

    if app_state.state_manager:
        app_state.state_manager.set_current_state(state)
        print(f"  ✅ State applied to state manager")

        # Verify
        current = app_state.state_manager.get_current_state()
        print(f"\n✅ Verification:")
        print(f"     Available vehicles: {len(current.get_available_vehicles())}")
        print(f"     Eastleigh window active: {current.is_eastleigh_window_active()}")
        print(f"     Total pending weight: {current.get_total_pending_weight()} tonnes")
    else:
        print(f"  ⚠️  State manager not initialized (app may not be running)")
        print(f"     State created but not applied")

    print("\n" + "=" * 70)
    print("  ✅ SYSTEM INITIALIZATION COMPLETE")
    print("=" * 70)
    print("\n📝 Next Steps:")
    print("   1. System is now ready for operations")
    print("   2. Run: python test_complete_workflow.py")
    print("   3. The engine will create routes with these vehicles")
    print("\n")

    return state


if __name__ == "__main__":
    print("\n⚠️  NOTE: This script should be run AFTER starting the API server")
    print("   Or it can be imported and used programmatically\n")

    try:
        # Check if app is running
        if app_state.state_manager is None:
            print("❌ App state not initialized. Make sure API server is running:")
            print("   python -m backend.api.main\n")
            print("   Then run this script in a separate terminal.")
        else:
            state = initialize_system_state()
            print(f"✅ System ready for workflow tests!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nIf API is not running, you can still use this in your code:")
        print("   from initialize_system import initialize_system_state")
        import traceback
        traceback.print_exc()

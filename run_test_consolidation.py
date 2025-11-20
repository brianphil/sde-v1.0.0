from datetime import datetime, timedelta
from backend.core.models.domain import (
    Order,
    Vehicle,
    Location,
    TimeWindow,
    DestinationCity,
    VehicleStatus,
)
from backend.core.models.state import SystemState, EnvironmentState
from backend.core.powell.engine import PowellEngine

# Create orders: 2T (Nakuru), 3T (Nakuru), 5T (Eldoret)
now = datetime.now()

order_a = Order(
    order_id="O_A",
    customer_id="C1",
    customer_name="Cust A",
    pickup_location=Location(latitude=0.0, longitude=0.0, address="Depot"),
    destination_city=DestinationCity.NAKURU,
    destination_location=None,
    weight_tonnes=2.0,
    volume_m3=1.5,
    time_window=TimeWindow(start_time=now, end_time=now + timedelta(hours=8)),
    price_kes=1000.0,
)

order_b = Order(
    order_id="O_B",
    customer_id="C2",
    customer_name="Cust B",
    pickup_location=Location(latitude=0.0, longitude=0.0, address="Depot"),
    destination_city=DestinationCity.NAKURU,
    destination_location=None,
    weight_tonnes=3.0,
    volume_m3=2.0,
    time_window=TimeWindow(start_time=now, end_time=now + timedelta(hours=8)),
    price_kes=1200.0,
)

order_c = Order(
    order_id="O_C",
    customer_id="C3",
    customer_name="Cust C",
    pickup_location=Location(latitude=0.0, longitude=0.0, address="Depot"),
    destination_city=DestinationCity.ELDORET,
    destination_location=None,
    weight_tonnes=5.0,
    volume_m3=4.0,
    time_window=TimeWindow(start_time=now, end_time=now + timedelta(hours=8)),
    price_kes=2000.0,
)

# Vehicles 5T and 10T available earlier than env time
vehicle_5t = Vehicle(
    vehicle_id="V5",
    vehicle_type="5T",
    capacity_weight_tonnes=5.0,
    capacity_volume_m3=8.0,
    current_location=Location(latitude=0.0, longitude=0.0, address="Depot"),
    available_at=now - timedelta(hours=1),
    status=VehicleStatus.AVAILABLE,
)

vehicle_10t = Vehicle(
    vehicle_id="V10",
    vehicle_type="10T",
    capacity_weight_tonnes=10.0,
    capacity_volume_m3=15.0,
    current_location=Location(latitude=0.0, longitude=0.0, address="Depot"),
    available_at=now - timedelta(hours=1),
    status=VehicleStatus.AVAILABLE,
)

orders = {o.order_id: o for o in [order_a, order_b, order_c]}
vehicles = {v.vehicle_id: v for v in [vehicle_5t, vehicle_10t]}

env = EnvironmentState(current_time=now + timedelta(minutes=10))
state = SystemState(pending_orders=orders, fleet=vehicles, environment=env)

engine = PowellEngine()

print("=== Test: consolidation scenario ===")
print(f"Pending orders: {len(state.pending_orders)}")
print(f"Available vehicles: {len(state.get_available_vehicles())}")

routes = engine.daily_route_planning_with_consolidation(state)

print(f"Routes produced: {len(routes)}")
for r in routes:
    print(
        f" - {r.route_id}: vehicle={r.vehicle_id}, orders={r.order_ids}, est_cost={r.estimated_cost_kes}, est_dist={r.total_distance_km}"
    )

# Also show consolidation pool status
pool_status = engine.consolidation_engine.get_pool_status()
print("Consolidation pool status:", pool_status)

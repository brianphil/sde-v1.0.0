"""Insert test routes into database."""
import asyncio
from datetime import datetime, timedelta
from backend.db.database import init_database, get_session
from backend.db.models import RouteModel, RouteStopModel
from backend.core.models.domain import RouteStatus


async def create_test_routes():
    """Create test routes in database."""
    await init_database()

    async with get_session() as session:
        # Route 1: Nakuru
        route1 = RouteModel(
            route_id='ROUTE_TEST_001',
            vehicle_id='VEH_001',
            order_ids=['ORD_6A839BE1'],
            destination_cities=['Nakuru'],
            total_distance_km=150.5,
            estimated_duration_minutes=180,
            estimated_cost_kes=3500.0,
            status=RouteStatus.PLANNED,
            estimated_fuel_cost=1500.0,
            estimated_time_cost=1800.0,
            estimated_delay_penalty=0.0,
            created_at=datetime.now(),
        )
        session.add(route1)

        stop1 = RouteStopModel(
            stop_id='STOP_001',
            route_id='ROUTE_TEST_001',
            order_ids=['ORD_6A839BE1'],
            location={'latitude': -0.2827, 'longitude': 36.0687, 'address': 'Nakuru City'},
            stop_type='delivery',
            sequence_order=1,
            estimated_arrival=datetime.now() + timedelta(hours=3),
            estimated_duration_minutes=30,
            status='planned',
        )
        session.add(stop1)

        # Route 2: Eldoret
        route2 = RouteModel(
            route_id='ROUTE_TEST_002',
            vehicle_id='VEH_002',
            order_ids=['ORD_66CBDE42'],
            destination_cities=['Eldoret'],
            total_distance_km=320.0,
            estimated_duration_minutes=300,
            estimated_cost_kes=7500.0,
            status=RouteStatus.PLANNED,
            estimated_fuel_cost=3200.0,
            estimated_time_cost=3000.0,
            estimated_delay_penalty=0.0,
            created_at=datetime.now(),
        )
        session.add(route2)

        stop2 = RouteStopModel(
            stop_id='STOP_002',
            route_id='ROUTE_TEST_002',
            order_ids=['ORD_66CBDE42'],
            location={'latitude': 0.5143, 'longitude': 35.2698, 'address': 'Eldoret Town'},
            stop_type='delivery',
            sequence_order=1,
            estimated_arrival=datetime.now() + timedelta(hours=5),
            estimated_duration_minutes=45,
            status='planned',
        )
        session.add(stop2)

        await session.commit()
        print('✓ Created 2 test routes with stops in database')
        print(f'  - ROUTE_TEST_001: Nakuru (150.5 km, 180 min, 3500 KES)')
        print(f'  - ROUTE_TEST_002: Eldoret (320.0 km, 300 min, 7500 KES)')


if __name__ == '__main__':
    asyncio.run(create_test_routes())

"""Database Management Script.

This script provides utilities for database operations:
- Initialize database
- Create tables
- Run migrations
- Seed initial data
- Reset database (development only)

Usage:
    python manage_db.py init          # Initialize database and create tables
    python manage_db.py seed          # Seed initial data (admin user, test customers)
    python manage_db.py reset         # Reset database (WARNING: Destroys all data)
    python manage_db.py check         # Check database health
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select

from backend.db.database import (
    init_database,
    close_database,
    create_tables,
    drop_tables,
    reset_database,
    check_database_health,
)
from backend.db.models import UserModel, CustomerModel, VehicleModel
from backend.db.database import get_session
from passlib.context import CryptContext
from datetime import datetime

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def init_db():
    """Initialize database and create all tables."""
    print("Initializing database...")
    await init_database()
    print("Creating tables...")
    await create_tables()
    print("[OK] Database initialized successfully!")
    await close_database()


async def seed_db():
    """Seed database with initial data."""
    print("Seeding database with initial data...")
    await init_database()

    async with get_session() as session:
        # Create admin user
        admin_exists = await session.execute(
            select(UserModel).where(UserModel.username == "admin")
        )
        if not admin_exists.scalar_one_or_none():
            admin_user = UserModel(
                username="admin",
                email="admin@senga.com",
                hashed_password=pwd_context.hash("admin123"),
                full_name="System Administrator",
                role="admin",
                is_active=True,
                is_superuser=True,
            )
            session.add(admin_user)
            print("  [OK] Created admin user (username: admin, password: admin123)")

        # Create test user
        user_exists = await session.execute(
            select(UserModel).where(UserModel.username == "testuser")
        )
        if not user_exists.scalar_one_or_none():
            test_user = UserModel(
                username="testuser",
                email="testuser@senga.com",
                hashed_password=pwd_context.hash("test123"),
                full_name="Test User",
                role="user",
                is_active=True,
                is_superuser=False,
            )
            session.add(test_user)
            print("  [OK] Created test user (username: testuser, password: test123)")

        # Create test customers
        customer1_exists = await session.execute(
            select(CustomerModel).where(CustomerModel.customer_id == "CUST_001")
        )
        if not customer1_exists.scalar_one_or_none():
            customer1 = CustomerModel(
                customer_id="CUST_001",
                customer_name="ABC Company",
                email="abc@example.com",
                phone="+254712345678",
                address="123 Industrial Area, Nairobi",
                constraints={"max_wait_hours": 24, "preferred_time": "morning"},
            )
            session.add(customer1)
            print("  [OK] Created test customer: ABC Company")

        customer2_exists = await session.execute(
            select(CustomerModel).where(CustomerModel.customer_id == "CUST_002")
        )
        if not customer2_exists.scalar_one_or_none():
            customer2 = CustomerModel(
                customer_id="CUST_002",
                customer_name="XYZ Logistics",
                email="xyz@example.com",
                phone="+254723456789",
                address="456 Mombasa Road, Nairobi",
                constraints={"max_wait_hours": 48},
            )
            session.add(customer2)
            print("  [OK] Created test customer: XYZ Logistics")

        # Create test vehicles
        vehicle1_exists = await session.execute(
            select(VehicleModel).where(VehicleModel.vehicle_id == "VEH_5T_001")
        )
        if not vehicle1_exists.scalar_one_or_none():
            vehicle1 = VehicleModel(
                vehicle_id="VEH_5T_001",
                vehicle_type="5T Truck",
                capacity_weight_tonnes=5.0,
                capacity_volume_m3=10.0,
                current_location={
                    "latitude": -1.2921,
                    "longitude": 36.8219,
                    "address": "Nairobi Depot",
                    "zone": "Nairobi",
                },
                available_at=datetime.now(),
                fuel_cost_per_km=15.0,
                driver_cost_per_hour=500.0,
            )
            session.add(vehicle1)
            print("  [OK] Created test vehicle: 5T Truck")

        vehicle2_exists = await session.execute(
            select(VehicleModel).where(VehicleModel.vehicle_id == "VEH_10T_001")
        )
        if not vehicle2_exists.scalar_one_or_none():
            vehicle2 = VehicleModel(
                vehicle_id="VEH_10T_001",
                vehicle_type="10T Truck",
                capacity_weight_tonnes=10.0,
                capacity_volume_m3=20.0,
                current_location={
                    "latitude": -1.2921,
                    "longitude": 36.8219,
                    "address": "Nairobi Depot",
                    "zone": "Nairobi",
                },
                available_at=datetime.now(),
                fuel_cost_per_km=25.0,
                driver_cost_per_hour=600.0,
            )
            session.add(vehicle2)
            print("  [OK] Created test vehicle: 10T Truck")

        await session.commit()

    print("[OK] Database seeded successfully!")
    await close_database()


async def reset_db():
    """Reset database (WARNING: Destroys all data)."""
    print("[WARNING] This will destroy ALL data in the database!")
    confirm = input("Type 'yes' to confirm: ")

    if confirm.lower() != "yes":
        print("Reset cancelled.")
        return

    print("Resetting database...")
    await init_database()
    await reset_database()
    print("[OK] Database reset successfully!")
    await close_database()


async def check_db():
    """Check database health."""
    print("Checking database health...")
    await init_database()

    health = await check_database_health()

    if health["status"] == "healthy":
        print("[OK] Database is healthy")
        if "pool_size" in health:
            print(f"  Pool size: {health['pool_size']}")
            print(f"  Checked out: {health['checked_out']}")
    else:
        print("[ERROR] Database is unhealthy")
        print(f"  Error: {health.get('error', 'Unknown error')}")

    await close_database()


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    try:
        if command == "init":
            await init_db()
        elif command == "seed":
            await seed_db()
        elif command == "reset":
            await reset_db()
        elif command == "check":
            await check_db()
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

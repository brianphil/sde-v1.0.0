"""Test script for Customers API integration."""
import asyncio
import httpx
from datetime import datetime
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


async def test_customers_api():
    """Test all customer CRUD operations."""
    base_url = "http://localhost:8000/api/v1"

    # Get auth token
    async with httpx.AsyncClient() as client:
        # Login
        login_response = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.text}")
            return

        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        print("✅ Logged in successfully\n")

        # Test 1: Create customer
        print("Test 1: Create Customer")
        customer_data = {
            "customer_name": "Test Logistics Ltd",
            "email": "contact@testlogistics.com",
            "phone": "+254-700-123456",
            "address": "123 Industrial Area, Nairobi",
            "constraints": {
                "delivery_blocked_times": [
                    {"day": "Monday", "time_start": "12:00", "time_end": "13:00"}
                ],
                "max_delivery_window_hours": 8
            },
            "preferences": {
                "preferred_driver": "DRIVER_001",
                "preferred_vehicle_type": "10T",
                "notification_email": True
            }
        }

        create_response = await client.post(
            f"{base_url}/customers",
            json=customer_data,
            headers=headers
        )

        if create_response.status_code == 201:
            customer = create_response.json()
            customer_id = customer["customer_id"]
            print(f"✅ Customer created: {customer_id}")
            print(f"   Name: {customer['customer_name']}")
            print(f"   Email: {customer['email']}")
            print(f"   Phone: {customer['phone']}")
            print()
        else:
            print(f"❌ Create failed: {create_response.status_code} - {create_response.text}")
            return

        # Test 2: Get customer
        print("Test 2: Get Customer")
        get_response = await client.get(
            f"{base_url}/customers/{customer_id}",
            headers=headers
        )

        if get_response.status_code == 200:
            customer = get_response.json()
            print(f"✅ Customer retrieved: {customer['customer_id']}")
            print(f"   Constraints: {customer.get('constraints')}")
            print(f"   Preferences: {customer.get('preferences')}")
            print()
        else:
            print(f"❌ Get failed: {get_response.status_code} - {get_response.text}")

        # Test 3: List customers
        print("Test 3: List Customers")
        list_response = await client.get(
            f"{base_url}/customers",
            headers=headers
        )

        if list_response.status_code == 200:
            customers = list_response.json()
            print(f"✅ Found {len(customers)} customer(s)")
            for c in customers:
                print(f"   - {c['customer_id']}: {c['customer_name']}")
            print()
        else:
            print(f"❌ List failed: {list_response.status_code} - {list_response.text}")

        # Test 4: Search by name
        print("Test 4: Search by Name")
        search_response = await client.get(
            f"{base_url}/customers?name_search=Test",
            headers=headers
        )

        if search_response.status_code == 200:
            customers = search_response.json()
            print(f"✅ Found {len(customers)} customer(s) matching 'Test'")
            print()
        else:
            print(f"❌ Search failed: {search_response.status_code} - {search_response.text}")

        # Test 5: Update customer
        print("Test 5: Update Customer")
        update_data = {
            "email": "newemail@testlogistics.com",
            "phone": "+254-700-999888",
            "preferences": {
                "preferred_driver": "DRIVER_002",
                "notification_sms": True
            }
        }

        update_response = await client.put(
            f"{base_url}/customers/{customer_id}",
            json=update_data,
            headers=headers
        )

        if update_response.status_code == 200:
            customer = update_response.json()
            print(f"✅ Customer updated: {customer_id}")
            print(f"   New Email: {customer['email']}")
            print(f"   New Phone: {customer['phone']}")
            print(f"   Updated Preferences: {customer.get('preferences')}")
            print()
        else:
            print(f"❌ Update failed: {update_response.status_code} - {update_response.text}")

        # Test 6: Delete customer (should succeed - no orders)
        print("Test 6: Delete Customer (no orders)")
        delete_response = await client.delete(
            f"{base_url}/customers/{customer_id}",
            headers=headers
        )

        if delete_response.status_code == 200:
            result = delete_response.json()
            print(f"✅ Customer deleted: {result['message']}")
            print()
        else:
            print(f"❌ Delete failed: {delete_response.status_code} - {delete_response.text}")

        # Test 7: Verify deletion
        print("Test 7: Verify Deletion")
        get_deleted_response = await client.get(
            f"{base_url}/customers/{customer_id}",
            headers=headers
        )

        if get_deleted_response.status_code == 404:
            print(f"✅ Customer successfully deleted (404 returned)")
            print()
        else:
            print(f"❌ Expected 404, got {get_deleted_response.status_code}")

        print("=" * 60)
        print("All Customer API tests completed!")


if __name__ == "__main__":
    asyncio.run(test_customers_api())

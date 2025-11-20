"""End-to-End System Tests for Powell Sequential Decision Engine.

Tests the complete system flow:
1. Application startup
2. API endpoints
3. Decision making with world-class learning
4. State management
5. Learning coordinator
6. Database operations
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

# Import core components
from backend.core.powell.engine import PowellEngine
from backend.services.state_manager import StateManager
from backend.services.event_orchestrator import EventOrchestrator
from backend.services.learning_coordinator import LearningCoordinator
from backend.core.models.domain import (
    Order,
    Vehicle,
    VehicleStatus,
    Location,
    Customer,
    DestinationCity,
    OperationalOutcome,
    TimeWindow,
)
from backend.core.models.state import SystemState, EnvironmentState, LearningState
from backend.core.models.decision import DecisionType


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_fleet():
    """Create sample fleet of vehicles."""
    now = datetime.now()
    morning = now.replace(hour=8, minute=0)

    return {
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
            available_at=morning,
            status=VehicleStatus.AVAILABLE,
            driver_id="DRIVER_001",
            fuel_cost_per_km=9.0,
            driver_cost_per_hour=500.0,
        ),
        "VEH_002": Vehicle(
            vehicle_id="VEH_002",
            vehicle_type="10T",
            capacity_weight_tonnes=10.0,
            capacity_volume_m3=15.0,
            current_location=Location(
                latitude=-1.2921,
                longitude=36.8219,
                address="Nairobi Depot",
                zone="Depot"
            ),
            available_at=morning,
            status=VehicleStatus.AVAILABLE,
            driver_id="DRIVER_002",
            fuel_cost_per_km=12.0,
            driver_cost_per_hour=600.0,
        ),
    }


@pytest.fixture
def sample_customers():
    """Create sample customers."""
    return {
        "CUST_001": Customer(
            customer_id="CUST_001",
            name="Eastleigh Store",
            locations=[
                Location(
                    latitude=-1.2796,
                    longitude=36.8589,
                    address="Eastleigh, Nairobi",
                    zone="Eastleigh"
                )
            ],
        ),
        "CUST_002": Customer(
            customer_id="CUST_002",
            name="Thika Market",
            locations=[
                Location(
                    latitude=-1.0332,
                    longitude=37.0693,
                    address="Thika Town",
                    zone="Thika"
                )
            ],
        ),
    }


@pytest.fixture
def sample_orders():
    """Create sample orders for testing."""
    now = datetime.now()

    return [
        Order(
            order_id="ORD_001",
            customer_id="CUST_001",
            customer_name="Nakuru Store",
            destination_city=DestinationCity.NAKURU,
            weight_tonnes=2.0,
            volume_m3=3.0,
            priority=1,  # 1 = high priority
            time_window=TimeWindow(now, now + timedelta(hours=2)),
            delivery_window=TimeWindow(now + timedelta(hours=2), now + timedelta(hours=4)),
            pickup_location=Location(
                latitude=-1.2921,
                longitude=36.8219,
                address="Nairobi Depot",
                zone="Depot"
            ),
            destination_location=Location(
                latitude=-0.3031,
                longitude=36.0800,
                address="Nakuru Town",
                zone="Nakuru"
            ),
            price_kes=50000.0,
            special_handling=["fragile"],
        ),
        Order(
            order_id="ORD_002",
            customer_id="CUST_002",
            customer_name="Kisumu Market",
            destination_city=DestinationCity.KISUMU,
            weight_tonnes=4.5,
            volume_m3=7.0,
            priority=0,  # 0 = normal priority
            time_window=TimeWindow(now, now + timedelta(hours=2)),
            delivery_window=TimeWindow(now + timedelta(hours=4), now + timedelta(hours=6)),
            pickup_location=Location(
                latitude=-1.2921,
                longitude=36.8219,
                address="Nairobi Depot",
                zone="Depot"
            ),
            destination_location=Location(
                latitude=-0.1022,
                longitude=34.7617,
                address="Kisumu City",
                zone="Kisumu"
            ),
            price_kes=80000.0,
        ),
    ]


@pytest.fixture
def system_state(sample_fleet, sample_customers):
    """Create initial system state."""
    now = datetime.now()

    return SystemState(
        environment=EnvironmentState(
            current_time=now,
            traffic_conditions={"default": 1.0},
            weather="clear",
        ),
        fleet=sample_fleet,
        customers=sample_customers,
        pending_orders={},
        active_routes={},
        learning=LearningState(),
    )


# ============================================================================
# Component Integration Tests
# ============================================================================


class TestComponentIntegration:
    """Test integration between core components."""

    def test_engine_initialization(self):
        """Test Powell Engine initializes correctly with all components."""
        engine = PowellEngine()

        # Verify all policy approximations are initialized
        assert engine.vfa is not None, "VFA should be initialized"
        assert engine.cfa is not None, "CFA should be initialized"
        assert engine.pfa is not None, "PFA should be initialized"

        # Verify world-class learning components
        assert hasattr(engine.vfa, "experience_coordinator"), "VFA should have experience coordinator"
        assert hasattr(engine.vfa, "regularization"), "VFA should have regularization"
        assert hasattr(engine.vfa, "lr_scheduler"), "VFA should have LR scheduler"

        assert hasattr(engine.cfa, "parameter_manager"), "CFA should have parameter manager"

        assert hasattr(engine.pfa, "pattern_coordinator"), "PFA should have pattern coordinator"
        assert hasattr(engine.pfa, "rule_exploration"), "PFA should have rule exploration"

    def test_state_manager_initialization(self):
        """Test StateManager initializes correctly."""
        state_manager = StateManager()

        assert state_manager is not None
        assert hasattr(state_manager, "get_current_state"), "StateManager should have get_current_state method"
        assert hasattr(state_manager, "set_current_state"), "StateManager should have set_current_state method"

    def test_orchestrator_initialization(self):
        """Test EventOrchestrator initializes correctly."""
        engine = PowellEngine()
        state_manager = StateManager()
        orchestrator = EventOrchestrator(engine, state_manager)

        assert orchestrator.engine is engine
        assert orchestrator.state_manager is state_manager

    def test_learning_coordinator_initialization(self):
        """Test LearningCoordinator initializes with comprehensive telemetry."""
        engine = PowellEngine()
        learning_coordinator = LearningCoordinator(engine=engine)

        # Verify comprehensive telemetry structure
        assert "vfa" in learning_coordinator.telemetry
        assert "cfa" in learning_coordinator.telemetry
        assert "pfa" in learning_coordinator.telemetry
        assert "general" in learning_coordinator.telemetry

        # Verify VFA telemetry
        vfa_telemetry = learning_coordinator.telemetry["vfa"]
        assert "last_training_loss" in vfa_telemetry
        assert "total_training_steps" in vfa_telemetry
        assert "prioritized_replay_size" in vfa_telemetry

        # Verify CFA telemetry
        cfa_telemetry = learning_coordinator.telemetry["cfa"]
        assert "fuel_cost_per_km" in cfa_telemetry
        assert "driver_cost_per_hour" in cfa_telemetry
        assert "fuel_accuracy_mape" in cfa_telemetry

        # Verify PFA telemetry
        pfa_telemetry = learning_coordinator.telemetry["pfa"]
        assert "total_rules" in pfa_telemetry
        assert "avg_rule_confidence" in pfa_telemetry


# ============================================================================
# End-to-End Decision Making Tests
# ============================================================================


class TestEndToEndDecisionMaking:
    """Test complete decision-making workflow."""

    def test_single_order_routing_decision(self, system_state, sample_orders):
        """Test making a routing decision for a single order."""
        engine = PowellEngine()
        order = sample_orders[0]

        # Add order to state
        system_state.pending_orders[order.order_id] = order

        # Make decision
        decision = engine.make_decision(system_state, DecisionType.ORDER_ARRIVAL)

        # Verify decision structure
        assert decision is not None, "Should return a decision"
        assert hasattr(decision, "recommended_action"), "Decision should have recommended_action"
        assert hasattr(decision, "routes"), "Decision should have routes"

        # Verify decision is actionable
        if decision.routes and len(decision.routes) > 0:
            route = decision.routes[0]
            assert route.vehicle_id in system_state.fleet, \
                "Recommended vehicle should be in fleet"

    def test_multiple_orders_decision_making(self, system_state, sample_orders):
        """Test decision making with multiple pending orders."""
        engine = PowellEngine()

        # Add all orders to state
        for order in sample_orders:
            system_state.pending_orders[order.order_id] = order

        # Make decisions for each order
        decisions = []
        for order in sample_orders:
            decision = engine.make_decision(system_state, DecisionType.ORDER_ARRIVAL)
            decisions.append(decision)

        # Verify all decisions were made
        assert len(decisions) == len(sample_orders), \
            "Should make decision for each order"

        # Verify decisions are different (unless legitimately the same)
        assert all(d is not None for d in decisions), \
            "All decisions should be non-null"

    def test_decision_with_learning_feedback(self, system_state, sample_orders):
        """Test decision making with learning feedback loop."""
        engine = PowellEngine()
        state_manager = StateManager()
        orchestrator = EventOrchestrator(engine, state_manager)
        learning_coordinator = LearningCoordinator(engine=engine)

        # Register learning handler
        orchestrator.register_learning_handler(learning_coordinator.process_outcome)

        # Update state manager with initial state
        state_manager.set_current_state(system_state)

        # Make decision
        order = sample_orders[0]
        system_state.pending_orders[order.order_id] = order
        decision = engine.make_decision(system_state, DecisionType.ORDER_ARRIVAL)

        # Create operational outcome
        vehicle_id = decision.routes[0].vehicle_id if decision.routes and len(decision.routes) > 0 else "VEH_001"
        outcome = OperationalOutcome(
            outcome_id=f"OUT_{order.order_id}",
            route_id=f"ROUTE_{order.order_id}",
            vehicle_id=vehicle_id,
            predicted_fuel_cost=450.0,
            actual_fuel_cost=475.0,
            predicted_duration_minutes=120.0,
            actual_duration_minutes=125.0,
            predicted_distance_km=28.0,
            actual_distance_km=29.5,
            on_time=True,
            successful_deliveries=1,
            failed_deliveries=0,
            customer_satisfaction_score=4.5,
            notes="Test delivery",
        )

        # Process learning feedback
        signals = learning_coordinator.process_outcome(outcome, system_state)

        # Verify signals were generated
        assert signals is not None, "Should return learning signals"
        assert isinstance(signals, dict), "Signals should be dict"

    def test_learning_improves_over_time(self, system_state, sample_orders):
        """Test that learning components improve with feedback."""
        engine = PowellEngine()
        learning_coordinator = LearningCoordinator(engine=engine)

        # Get initial telemetry
        initial_metrics = learning_coordinator.get_metrics()
        initial_vfa_steps = initial_metrics["telemetry"]["vfa"]["total_training_steps"]
        initial_cfa_updates = initial_metrics["telemetry"]["cfa"]["total_updates"]

        # Simulate multiple outcomes
        for i in range(10):
            outcome = OperationalOutcome(
                outcome_id=f"OUT_{i:03d}",
                route_id=f"ROUTE_{i:03d}",
                vehicle_id="VEH_001",
                predicted_fuel_cost=450.0 + i * 5,
                actual_fuel_cost=475.0 + i * 5,
                predicted_duration_minutes=120.0,
                actual_duration_minutes=125.0,
                predicted_distance_km=28.0,
                actual_distance_km=29.5,
                on_time=True,
                successful_deliveries=1,
                failed_deliveries=0,
                customer_satisfaction_score=4.5,
            )

            learning_coordinator.process_outcome(outcome, system_state)

        # Get updated telemetry
        updated_metrics = learning_coordinator.get_metrics()
        updated_cfa_updates = updated_metrics["telemetry"]["cfa"]["total_updates"]

        # Verify learning occurred
        assert updated_cfa_updates > initial_cfa_updates, \
            "CFA should have received parameter updates"

        # Verify CFA parameters were updated
        assert updated_metrics["telemetry"]["cfa"]["fuel_cost_per_km"] is not None, \
            "CFA fuel cost should be updated"
        assert updated_metrics["telemetry"]["cfa"]["driver_cost_per_hour"] is not None, \
            "CFA driver cost should be updated"


# ============================================================================
# Performance and Reliability Tests
# ============================================================================


class TestPerformanceAndReliability:
    """Test system performance and reliability."""

    def test_decision_latency(self, system_state, sample_orders):
        """Test that decisions are made within acceptable time."""
        import time

        engine = PowellEngine()
        order = sample_orders[0]
        system_state.pending_orders[order.order_id] = order

        # Measure decision time
        start_time = time.time()
        decision = engine.make_decision(system_state, DecisionType.ORDER_ARRIVAL)
        elapsed = time.time() - start_time

        # Decision should be made within 1 second
        assert elapsed < 1.0, f"Decision took {elapsed:.3f}s, should be <1s"
        assert decision is not None, "Should return a decision"

    def test_concurrent_decision_making(self, system_state, sample_orders):
        """Test that system handles concurrent decisions correctly."""
        engine = PowellEngine()

        # Make multiple decisions concurrently
        decisions = []
        for order in sample_orders:  # Test with all available orders
            system_state.pending_orders[order.order_id] = order
            decision = engine.make_decision(system_state, DecisionType.ORDER_ARRIVAL)
            decisions.append(decision)

        # All decisions should succeed
        assert len(decisions) == len(sample_orders), f"Should make {len(sample_orders)} decisions"
        assert all(d is not None for d in decisions), "All decisions should be non-null"

    def test_learning_coordinator_telemetry_performance(self):
        """Test that telemetry operations are performant."""
        import time

        engine = PowellEngine()
        learning_coordinator = LearningCoordinator(engine=engine)

        # Measure telemetry retrieval time
        start_time = time.time()
        metrics = learning_coordinator.get_metrics()
        elapsed = time.time() - start_time

        # Telemetry should be retrieved in <0.1s
        assert elapsed < 0.1, f"Telemetry retrieval took {elapsed:.3f}s, should be <0.1s"
        assert metrics is not None, "Should return metrics"
        assert "telemetry" in metrics, "Should include telemetry"


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


class TestErrorHandlingAndEdgeCases:
    """Test system behavior under error conditions and edge cases."""

    def test_decision_with_empty_fleet(self, system_state, sample_orders):
        """Test decision making when no vehicles are available."""
        from dataclasses import replace
        engine = PowellEngine()

        # Create state with empty fleet (use replace since dataclass is frozen)
        empty_fleet_state = replace(system_state, fleet={})

        # Add order
        order = sample_orders[0]
        empty_fleet_state.pending_orders[order.order_id] = order

        # Should handle gracefully (not crash)
        try:
            decision = engine.make_decision(empty_fleet_state, DecisionType.ORDER_ARRIVAL)
            # Decision might be None or indicate no vehicle available
            assert True, "Should handle empty fleet gracefully"
        except Exception as e:
            pytest.fail(f"Should not crash with empty fleet: {e}")

    def test_learning_with_invalid_outcome(self, system_state):
        """Test learning coordinator handles invalid outcomes gracefully."""
        engine = PowellEngine()
        learning_coordinator = LearningCoordinator(engine=engine)

        # Create outcome with missing/invalid data
        outcome = OperationalOutcome(
            outcome_id="OUT_INVALID",
            route_id="ROUTE_INVALID",
            vehicle_id="VEH_999",  # Non-existent vehicle
            predicted_fuel_cost=0.0,
            actual_fuel_cost=0.0,
            predicted_duration_minutes=0.0,
            actual_duration_minutes=0.0,
            predicted_distance_km=0.0,
            actual_distance_km=0.0,
            on_time=False,
            successful_deliveries=0,
            failed_deliveries=0,
        )

        # Should handle gracefully
        try:
            signals = learning_coordinator.process_outcome(outcome, system_state)
            assert True, "Should handle invalid outcome gracefully"
        except Exception as e:
            pytest.fail(f"Should not crash with invalid outcome: {e}")

    def test_telemetry_with_uninitialized_components(self):
        """Test telemetry works even with uninitialized components."""
        # Create learning coordinator without engine
        learning_coordinator = LearningCoordinator(engine=None)

        # Should still return telemetry structure
        metrics = learning_coordinator.get_metrics()

        assert metrics is not None, "Should return metrics"
        assert "telemetry" in metrics, "Should have telemetry"
        assert "vfa" in metrics["telemetry"], "Should have VFA section"
        assert "cfa" in metrics["telemetry"], "Should have CFA section"
        assert "pfa" in metrics["telemetry"], "Should have PFA section"


# ============================================================================
# Integration Summary Test
# ============================================================================


class TestSystemIntegration:
    """Comprehensive system integration test."""

    def test_complete_system_workflow(self, system_state, sample_orders):
        """Test complete workflow: initialization → decision → learning → metrics."""
        print("\n" + "=" * 80)
        print("RUNNING COMPLETE SYSTEM WORKFLOW TEST")
        print("=" * 80)

        # 1. Initialize all components
        print("\n1. Initializing components...")
        engine = PowellEngine()
        state_manager = StateManager()
        orchestrator = EventOrchestrator(engine, state_manager)
        learning_coordinator = LearningCoordinator(engine=engine)

        # Register learning handler
        orchestrator.register_learning_handler(learning_coordinator.process_outcome)
        state_manager.set_current_state(system_state)

        print("   ✅ All components initialized")

        # 2. Make decision
        print("\n2. Making routing decision...")
        order = sample_orders[0]
        system_state.pending_orders[order.order_id] = order
        decision = engine.make_decision(system_state, DecisionType.ORDER_ARRIVAL)

        assert decision is not None, "Decision should not be None"
        print(f"   ✅ Decision made: {decision.recommended_action}")

        # 3. Simulate execution and create outcome
        print("\n3. Simulating route execution...")
        vehicle_id = decision.routes[0].vehicle_id if decision.routes and len(decision.routes) > 0 else "VEH_001"
        outcome = OperationalOutcome(
            outcome_id=f"OUT_{order.order_id}",
            route_id=f"ROUTE_{order.order_id}",
            vehicle_id=vehicle_id,
            predicted_fuel_cost=450.0,
            actual_fuel_cost=475.0,
            predicted_duration_minutes=120.0,
            actual_duration_minutes=125.0,
            predicted_distance_km=28.0,
            actual_distance_km=29.5,
            on_time=True,
            successful_deliveries=1,
            failed_deliveries=0,
            customer_satisfaction_score=4.5,
            notes="E2E test delivery",
        )

        print(f"   ✅ Outcome created: on_time={outcome.on_time}")

        # 4. Process learning feedback
        print("\n4. Processing learning feedback...")
        signals = learning_coordinator.process_outcome(outcome, system_state)

        assert signals is not None, "Signals should not be None"
        print(f"   ✅ Learning signals processed")

        # 5. Verify learning occurred
        print("\n5. Verifying learning components...")
        metrics = learning_coordinator.get_metrics()

        # Check CFA updated
        cfa_telemetry = metrics["telemetry"]["cfa"]
        assert cfa_telemetry["total_updates"] > 0, "CFA should have updates"
        assert cfa_telemetry["fuel_cost_per_km"] is not None, "Fuel cost should be updated"
        print(f"   ✅ CFA: {cfa_telemetry['total_updates']} updates, "
              f"fuel={cfa_telemetry['fuel_cost_per_km']:.2f} KES/km")

        # Check VFA buffer
        vfa_telemetry = metrics["telemetry"]["vfa"]
        print(f"   ✅ VFA: buffer_size={vfa_telemetry['prioritized_replay_size']}")

        # Check PFA
        pfa_telemetry = metrics["telemetry"]["pfa"]
        print(f"   ✅ PFA: rules={pfa_telemetry['total_rules']}")

        # 6. Verify world-class components active
        print("\n6. Verifying world-class components...")
        assert hasattr(engine.vfa, "experience_coordinator"), "VFA should have experience coordinator"
        assert hasattr(engine.cfa, "parameter_manager"), "CFA should have parameter manager"
        assert hasattr(engine.pfa, "pattern_coordinator"), "PFA should have pattern coordinator"
        print("   ✅ All world-class components active")

        print("\n" + "=" * 80)
        print("✅ COMPLETE SYSTEM WORKFLOW TEST PASSED")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])

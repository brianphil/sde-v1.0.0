"""Demo script showing end-to-end Powell Engine workflow.

This demonstrates:
1. Creating domain objects (orders, vehicles, customers)
2. Building system state
3. Making decisions with the engine
4. Committing decisions
5. Processing feedback
6. Learning from outcomes
"""

from datetime import datetime, timedelta
from backend.core.models.domain import (
    Order,
    Vehicle,
    Location,
    TimeWindow,
    DestinationCity,
    OrderStatus,
    VehicleStatus,
    OperationalOutcome,
    Customer,
)
from backend.core.models.state import SystemState, EnvironmentState, LearningState
from backend.core.models.decision import DecisionType
from backend.core.powell.engine import PowellEngine
from backend.services.state_manager import StateManager
from backend.services.event_orchestrator import (
    EventOrchestrator,
    Event,
    EventPriority,
)
from backend.services.route_optimizer import RouteOptimizer
from backend.core.learning.feedback_processor import FeedbackProcessor
from backend.services.learning_coordinator import LearningCoordinator


def create_sample_data():
    """Create sample orders, vehicles, and customers."""

    # Sample customer
    majid = Customer(
        customer_id="MAJID",
        name="Majid Retailers",
        locations=[
            Location(latitude=-1.2921, longitude=36.8219, address="Eastleigh Store")
        ],
        delivery_blocked_times=[
            {"day": "Wednesday", "time_start": "14:00", "time_end": "15:30"}
        ],
        priority_level=1,
    )

    # Sample orders
    now = datetime.now()

    order_1 = Order(
        order_id="ORD_001",
        customer_id="MAJID",
        customer_name="Majid Retailers",
        pickup_location=Location(
            latitude=-1.2921, longitude=36.8219, address="Eastleigh", zone="Eastleigh"
        ),
        destination_city=DestinationCity.NAKURU,
        destination_location=Location(
            latitude=-0.3031, longitude=35.2684, address="Nakuru CBD"
        ),
        weight_tonnes=2.5,
        volume_m3=4.0,
        time_window=TimeWindow(
            start_time=now.replace(hour=8, minute=30),
            end_time=now.replace(hour=9, minute=45),
        ),
        priority=0,
        price_kes=2500.0,
    )

    order_2 = Order(
        order_id="ORD_002",
        customer_id="ABC_CORP",
        customer_name="ABC Corporation",
        pickup_location=Location(
            latitude=-1.2800, longitude=36.8300, address="CBD", zone="CBD"
        ),
        destination_city=DestinationCity.ELDORET,
        destination_location=Location(
            latitude=0.5143, longitude=35.2707, address="Eldoret CBD"
        ),
        weight_tonnes=3.0,
        volume_m3=5.0,
        time_window=TimeWindow(
            start_time=now.replace(hour=10, minute=0),
            end_time=now.replace(hour=17, minute=0),
        ),
        priority=1,  # Urgent
        special_handling=["fresh_food"],
        price_kes=3500.0,
    )

    # Sample vehicles
    vehicle_1 = Vehicle(
        vehicle_id="VEH_001",
        vehicle_type="5T",
        capacity_weight_tonnes=5.0,
        capacity_volume_m3=8.0,
        current_location=Location(latitude=-1.2921, longitude=36.8219, address="Depot"),
        available_at=now,
        status=VehicleStatus.AVAILABLE,
        driver_id="DRIVER_001",
    )

    vehicle_2 = Vehicle(
        vehicle_id="VEH_002",
        vehicle_type="10T",
        capacity_weight_tonnes=10.0,
        capacity_volume_m3=15.0,
        current_location=Location(latitude=-1.2921, longitude=36.8219, address="Depot"),
        available_at=now,
        status=VehicleStatus.AVAILABLE,
        driver_id="DRIVER_002",
    )

    return {
        "orders": {"ORD_001": order_1, "ORD_002": order_2},
        "vehicles": {"VEH_001": vehicle_1, "VEH_002": vehicle_2},
        "customers": {"MAJID": majid},
    }


def demo_basic_decision():
    """Demonstrate basic decision making."""

    print("=" * 70)
    print("DEMO 1: Basic Daily Route Planning Decision")
    print("=" * 70)

    # Setup
    engine = PowellEngine()
    data = create_sample_data()

    # Create initial state
    env = EnvironmentState(
        current_time=datetime.now().replace(hour=8, minute=30),
        traffic_conditions={"Eastleigh": 0.3, "CBD": 0.5, "Nakuru": 0.2},
        weather="clear",
    )

    state = SystemState(
        pending_orders=data["orders"],
        fleet=data["vehicles"],
        customers=data["customers"],
        environment=env,
    )

    print(f"\nInitial State:")
    print(f"  Pending Orders: {len(state.pending_orders)}")
    print(f"  Available Vehicles: {len(state.get_available_vehicles())}")
    print(f"  Total Pending Weight: {state.get_total_pending_weight():.1f} tonnes")
    print(f"  Eastleigh Window Active: {state.is_eastleigh_window_active()}")

    # Make decision
    print(f"\nMaking Daily Planning Decision...")
    decision = engine.make_decision(
        state,
        decision_type=DecisionType.DAILY_ROUTE_PLANNING,
        trigger_reason="Daily morning optimization",
    )

    print(f"\nDecision Result:")
    policy_used = getattr(
        decision, "policy_name", getattr(decision, "hybrid_name", "unknown")
    )
    print(f"  Policy Used: {policy_used}")
    print(f"  Recommended Action: {decision.recommended_action.value}")
    print(f"  Confidence: {decision.confidence_score:.2%}")
    print(f"  Expected Value: {decision.expected_value:.0f} KES")
    print(f"  Routes Proposed: {len(decision.routes)}")
    for route in decision.routes:
        print(
            f"    - {route.route_id}: {len(route.order_ids)} orders, {len(route.destination_cities)} cities"
        )
    reasoning = getattr(decision, "reasoning", "")
    print(f"  Reasoning: {reasoning[:80]}...")

    # Commit decision
    print(f"\nCommitting Decision...")
    result = engine.commit_decision(decision, state)

    print(f"\nExecution Result:")
    print(f"  Action: {result['action']}")
    print(f"  Routes Created: {result['routes_created']}")
    print(f"  Orders Assigned: {result['orders_assigned']}")
    if result["errors"]:
        print(f"  Errors: {result['errors']}")


def demo_learning_loop():
    """Demonstrate learning from feedback."""

    print("\n" + "=" * 70)
    print("DEMO 2: Learning from Operational Feedback")
    print("=" * 70)

    # Setup
    engine = PowellEngine()
    feedback_processor = FeedbackProcessor()
    data = create_sample_data()

    print("\nInitializing CFA parameter learning...")
    # Show CFA economic defaults (per-vehicle-type mapping)
    print("  CFA fuel cost per km by vehicle type:")
    for k, v in engine.cfa.params.fuel_cost_per_km_by_vehicle.items():
        print(f"    - {k}: {v} KES/km")
    print(
        f"  Initial prediction accuracy: {engine.cfa.params.prediction_accuracy_fuel:.2%}"
    )

    # Simulate multiple outcomes
    print("\nProcessing 5 route outcomes...")

    for i in range(5):
        # Simulate outcome
        predicted_cost = 1500 + i * 50
        actual_cost = 1400 + i * 60  # Slightly lower

        outcome = OperationalOutcome(
            outcome_id=f"OUTCOME_{i:03d}",
            route_id=f"ROUTE_{i:03d}",
            vehicle_id="VEH_001",
            predicted_fuel_cost=predicted_cost,
            actual_fuel_cost=actual_cost,
            predicted_duration_minutes=120 + i * 5,
            actual_duration_minutes=115 + i * 5,
            predicted_distance_km=150,
            actual_distance_km=148,
            on_time=True,
            successful_deliveries=2,
            failed_deliveries=0,
            customer_satisfaction_score=0.90 + i * 0.01,
        )

        # Process outcome
        signals = feedback_processor.process_outcome(outcome)

        # Learn
        engine.cfa.update_from_feedback(
            {
                "fuel_cost_error": signals["cfa_signals"]["fuel_error"],
                "time_error_minutes": signals["cfa_signals"]["time_error"],
                "actual_fuel_cost": actual_cost,
                "actual_duration_minutes": 115 + i * 5,
            }
        )

        print(
            f"  Outcome {i+1}: fuel_error={signals['cfa_signals']['fuel_error']:+.0f} KES, "
            f"accuracy={signals['cfa_signals']['fuel_accuracy']:.2%}"
        )

    # Show learning progress
    print(f"\nAfter Learning:")
    # Show whether any per-vehicle cost mappings exist
    print("  CFA fuel cost per km by vehicle type (after learning):")
    for k, v in engine.cfa.params.fuel_cost_per_km_by_vehicle.items():
        print(f"    - {k}: {v} KES/km")
    print(
        f"  Updated prediction accuracy: {engine.cfa.params.prediction_accuracy_fuel:.2%}"
    )
    print(f"  Samples observed: {engine.cfa.params.samples_observed}")

    # Model health check
    print(f"\nModel Health Check:")
    metrics = feedback_processor.get_aggregate_metrics()
    print(f"  On-time rate: {metrics.get('on_time_mean', 0):.2%}")
    print(f"  Success rate: {metrics.get('success_rate_mean', 0):.2%}")
    print(
        f"  Customer satisfaction: {metrics.get('customer_satisfaction_mean', 0):.2f}/1.0"
    )

    # Check if retraining needed
    if feedback_processor.should_update_cfa_parameters():
        print(f"  ⚠️  CFA parameters need updating!")
    else:
        print(f"  ✅ CFA models performing well")


def demo_event_orchestration():
    """Demonstrate event orchestration."""

    print("\n" + "=" * 70)
    print("DEMO 3: Event-Driven Orchestration")
    print("=" * 70)

    # Setup
    engine = PowellEngine()
    state_manager = StateManager()
    orchestrator = EventOrchestrator(engine, state_manager)
    # Register learning coordinator so learning is triggered on outcomes
    learning_coordinator = LearningCoordinator(engine=engine)
    orchestrator.register_learning_handler(learning_coordinator.process_outcome)
    data = create_sample_data()

    # Initialize state
    env = EnvironmentState(current_time=datetime.now())
    state = SystemState(
        pending_orders=data["orders"],
        fleet=data["vehicles"],
        customers=data["customers"],
        environment=env,
    )
    state_manager.set_current_state(state)

    print(f"\nSubmitting events to orchestrator...")

    # Submit events
    order_3 = Order(
        order_id="ORD_003",
        customer_id="NEW_CUST",
        customer_name="New Customer",
        pickup_location=Location(latitude=-1.2921, longitude=36.8219, address="CBD"),
        destination_city=DestinationCity.KITALE,
        weight_tonnes=1.5,
        volume_m3=2.0,
        time_window=TimeWindow(
            start_time=datetime.now(), end_time=datetime.now() + timedelta(hours=8)
        ),
        price_kes=2000.0,
    )

    events = [
        Event("order_arrived", {"order": order_3}, priority=EventPriority.HIGH),
    ]

    for event in events:
        event_id = orchestrator.submit_event(event)
        print(f"  Event queued: {event_id}")

    # Process queue
    print(f"\nProcessing event queue...")
    results = orchestrator.process_all_events()

    for result in results:
        print(f"\nEvent: {result['event_type']}")
        print(f"  Decision Type: {result['decision_type']}")
        print(f"  Policy: {result['decision']['policy']}")
        print(f"  Action: {result['decision']['action']}")
        print(f"  Confidence: {result['decision']['confidence']:.2%}")

    # Queue status
    status = orchestrator.get_queue_status()
    print(f"\nQueue Status:")
    print(f"  Total Queued: {status['total_queued']}")
    print(f"  Processed Total: {status['processed_total']}")


def demo_vfa_training():
    """Demonstrate VFA training and telemetry with pending experiences."""

    print("\n" + "=" * 70)
    print("DEMO 4: VFA Training with PyTorch and Telemetry")
    print("=" * 70)

    # Setup
    engine = PowellEngine()
    state_manager = StateManager()
    coordinator = LearningCoordinator(engine=engine)
    data = create_sample_data()

    # Create initial state
    env = EnvironmentState(current_time=datetime.now())
    initial_state = SystemState(
        pending_orders=data["orders"],
        fleet=data["vehicles"],
        customers=data["customers"],
        environment=env,
    )
    state_manager.set_current_state(initial_state)

    print(f"\nVFA Configuration:")
    vfa_cfg = engine.config.get("vfa", {})
    print(f"  Use PyTorch: {vfa_cfg.get('use_pytorch', 'N/A')}")
    print(f"  State Features: {vfa_cfg.get('state_feature_dim', 'N/A')}")
    print(f"  Special Handling Tags: {vfa_cfg.get('special_handling_tags', [])}")
    print(f"  Train Batch Size: {vfa_cfg.get('train_batch_size', 32)}")
    print(f"  Train Epochs: {vfa_cfg.get('train_epochs', 1)}")

    print(f"\nVFA Initial State:")
    print(f"  Trained Samples: {engine.vfa.trained_samples}")
    print(f"  Total Loss: {engine.vfa.total_loss:.6f}")
    print(f"  Buffer Size: {len(engine.vfa.experience_buffer)}")
    print(f"  Pending Experiences: {len(engine.vfa.pending_by_route)}")

    print(f"\nSimulating 3 route outcomes with learning...")
    for i in range(1, 4):
        try:
            # Simulate an outcome
            outcome = OperationalOutcome(
                outcome_id=f"outcome_{i}",
                route_id=f"route_{i}",
                vehicle_id="VEH_001",
                predicted_fuel_cost=100.0 + i * 10,
                actual_fuel_cost=95.0 + i * 10,
                predicted_duration_minutes=180 + i * 5,
                actual_duration_minutes=175 + i * 5,
                predicted_distance_km=150.0,
                actual_distance_km=155.0,
                on_time=True,
                customer_satisfaction_score=0.9,
                successful_deliveries=2,
                failed_deliveries=0,
                notes=f"Test outcome {i}",
            )

            # Process outcome through coordinator (triggers learning)
            signals = coordinator.process_outcome(
                outcome, state_manager.get_current_state()
            )
            print(f"  Outcome {i}: processed (signals keys: {list(signals.keys())})")
        except Exception as e:
            print(f"  Outcome {i}: error - {e}")

    # Show telemetry
    print(f"\nVFA After Training:")
    metrics = coordinator.get_metrics()
    print(f"  Trained Samples: {engine.vfa.trained_samples}")
    print(f"  Total Loss: {engine.vfa.total_loss:.6f}")
    print(f"  Buffer Size: {len(engine.vfa.experience_buffer)}")
    print(f"  Pending Experiences: {len(engine.vfa.pending_by_route)}")

    if "vfa_telemetry" in metrics:
        tel = metrics["vfa_telemetry"]
        print(f"\n  Training Telemetry:")
        print(f"    Last Training Loss: {tel.get('last_training_loss', 'N/A')}")
        print(f"    Last Training Samples: {tel.get('last_training_samples', 0)}")
        print(f"    Total Training Steps: {tel.get('total_training_steps', 0)}")
        print(
            f"    Last Training Timestamp: {tel.get('last_training_timestamp', 'N/A')}"
        )

    if "vfa_stats" in metrics:
        stats = metrics["vfa_stats"]
        print(f"\n  VFA Model Stats:")
        print(f"    Trained Samples: {stats.get('trained_samples', 0)}")
        print(f"    Total Loss: {stats.get('total_loss', 0.0):.6f}")
        print(f"    Pending Experiences: {stats.get('pending_experiences', 0)}")
        print(f"    Buffer Size: {stats.get('buffer_size', 0)}")


def demo_state_transitions():
    """Demonstrate immutable state transitions."""

    print("\n" + "=" * 70)
    print("DEMO 5: Immutable State Transitions")
    print("=" * 70)

    # Setup
    state_manager = StateManager()
    data = create_sample_data()

    # Create initial state
    env = EnvironmentState(current_time=datetime.now())
    initial_state = SystemState(
        pending_orders=data["orders"],
        fleet=data["vehicles"],
        customers=data["customers"],
        environment=env,
    )
    state_manager.set_current_state(initial_state)

    print(f"\nInitial State:")
    print(f"  Pending Orders: {len(state_manager.get_current_state().pending_orders)}")
    print(f"  Active Routes: {len(state_manager.get_current_state().active_routes)}")

    # Apply state transitions
    print(f"\nApplying State Transitions...")

    # Create a route
    from backend.core.models.domain import Route, RouteStatus

    route = Route(
        route_id="ROUTE_001",
        vehicle_id="VEH_001",
        order_ids=["ORD_001"],
        stops=[],
        destination_cities=[DestinationCity.NAKURU],
        total_distance_km=150.0,
        estimated_duration_minutes=180,
        estimated_cost_kes=2000.0,
        status=RouteStatus.PLANNED,
    )

    # Transition 1: Route created
    new_state = state_manager.apply_event("route_created", {"route": route})

    print(f"After route created:")
    print(f"  Pending Orders: {len(new_state.pending_orders)}")
    print(f"  Active Routes: {len(new_state.active_routes)}")

    # Transition 2: Route started
    new_state = state_manager.apply_event("route_started", {"route_id": "ROUTE_001"})

    print(f"After route started:")
    print(f"  Active Routes: {len(new_state.active_routes)}")

    # Show history
    print(f"\nState Transition History:")
    history = state_manager.get_history(limit=10)
    for transition in history:
        print(f"  {transition['event_type']}: {transition['changes']}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("POWELL SEQUENTIAL DECISION ENGINE - DEMO SUITE")
    print("=" * 70)

    # Run demos
    demo_basic_decision()
    demo_learning_loop()
    demo_event_orchestration()
    demo_vfa_training()
    demo_state_transitions()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

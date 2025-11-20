"""Direct Lookahead Approximation - Multi-period optimization."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from ..models.state import SystemState
from ..models.domain import Order, Vehicle, Route, RouteStatus
from ..models.decision import PolicyDecision, DecisionContext, DecisionType, ActionType
from ..models.domain import Location, TimeWindow, DestinationCity
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ForecastScenario:
    """Single scenario for future period."""

    scenario_id: str
    day_offset: int  # 0=today, 1=tomorrow, etc.
    expected_orders: int
    expected_demand_weight_tonnes: float
    expected_demand_volume_m3: float
    probability: float  # Likelihood of this scenario


@dataclass
class DLAPeriod:
    """Single period in lookahead horizon."""

    period_id: str
    start_time: datetime
    end_time: datetime

    # Forecast
    scenarios: List[ForecastScenario] = field(default_factory=list)
    expected_orders: int = 0
    expected_demand_weight_tonnes: float = 0.0
    expected_demand_volume_m3: float = 0.0
    expected_revenue: float = 0.0
    day_offset: int = 0

    # Planning
    planned_routes: List[Route] = field(default_factory=list)
    planned_costs: float = 0.0
    planned_profit: float = 0.0


class DirectLookaheadApproximation:
    """Multi-period optimization considering future demand.

    DLA decides current actions by simulating future periods (7-day horizon typical).
    Two variants:

    1. Deterministic: Assume expected demand, optimize for best profit
       - Simpler, faster
       - Good for stable demand

    2. Stochastic: Consider multiple demand scenarios, optimize for robustness
       - More complex, slower
       - Better for volatile demand

    DLA can use terminal value from VFA at horizon end.
    """

    def __init__(self, horizon_days: int = 7, deterministic: bool = True, vfa=None):
        """Initialize DLA.

        Args:
            horizon_days: Lookahead planning horizon (default 7 days)
            deterministic: Use deterministic vs stochastic optimization
            vfa: Optional VFA instance for terminal value
        """
        self.horizon_days = horizon_days
        self.deterministic = deterministic
        self.vfa = vfa

        # Parameters
        self.consolidation_threshold = 0.8  # Consolidate if forecast accuracy >= 80%
        self.demand_forecast_accuracy = 0.75  # Historical accuracy
        self.use_vfa_terminal_value = vfa is not None

    def evaluate(self, state: SystemState, context: DecisionContext) -> PolicyDecision:
        """Apply DLA to make multi-period optimized decision.

        Workflow:
        1. Build forecast for horizon
        2. Simulate each period
        3. Optimize route plans across horizon
        4. Use terminal value from VFA if available
        5. Recommend current period action
        """

        # Build forecast
        forecast_periods = self._build_forecast(state)

        if not forecast_periods:
            # No forecast data - fallback to immediate decision
            return PolicyDecision(
                policy_name="DLA",
                decision_type=context.decision_type,
                recommended_action=ActionType.DEFER_ORDER,
                routes=[],
                confidence_score=0.0,
                expected_value=0.0,
                reasoning="No forecast available",
            )

        # Optimize across horizon
        if self.deterministic:
            optimal_routes = self._optimize_deterministic(
                state, context, forecast_periods
            )
        else:
            optimal_routes = self._optimize_stochastic(state, context, forecast_periods)

        # Extract current period decision
        current_routes = [
            r for r in optimal_routes if r.route_id.startswith("route_today_")
        ]

        # Calculate expected value (multi-period profit)
        expected_value = self._calculate_expected_value(
            forecast_periods, optimal_routes
        )

        # Confidence depends on forecast accuracy
        confidence = self.demand_forecast_accuracy

        return PolicyDecision(
            policy_name="DLA",
            decision_type=context.decision_type,
            recommended_action=(
                ActionType.CREATE_ROUTE if current_routes else ActionType.DEFER_ORDER
            ),
            routes=current_routes,
            confidence_score=confidence,
            expected_value=expected_value,
            reasoning=f"Multi-period optimized: {self.horizon_days}-day horizon, expected profit {expected_value:.0f} KES",
            considered_alternatives=len(forecast_periods),
            is_deterministic=self.deterministic,
            policy_parameters={
                "horizon_days": self.horizon_days,
                "deterministic": self.deterministic,
                "periods": len(forecast_periods),
                "forecast_accuracy": self.demand_forecast_accuracy,
            },
        )

    def _build_forecast(self, state: SystemState) -> List[DLAPeriod]:
        """Build forecast for lookahead horizon."""
        periods = []
        current_time = state.environment.current_time

        for day_offset in range(self.horizon_days):
            period_start = current_time + timedelta(days=day_offset)
            period_start = period_start.replace(
                hour=6, minute=0, second=0
            )  # Start at 6 AM
            period_end = period_start + timedelta(days=1)

            period = DLAPeriod(
                period_id=f"period_day{day_offset}",
                start_time=period_start,
                end_time=period_end,
            )
            period.day_offset = day_offset
            # Prefer learned forecasts from LearningState if available
            learned_forecast = getattr(state.learning, "dla_demand_forecast", None)

            if learned_forecast and f"day_{day_offset}" in learned_forecast:
                info = learned_forecast.get(f"day_{day_offset}", {})
                expected_orders = int(info.get("expected_orders", 0))
                expected_weight = float(info.get("expected_weight", 0.0))
                expected_volume = float(info.get("expected_volume", 0.0))
                prob = float(info.get("probability", 1.0))

                scenarios = [
                    ForecastScenario(
                        scenario_id=f"learned_day{day_offset}",
                        day_offset=day_offset,
                        expected_orders=expected_orders,
                        expected_demand_weight_tonnes=expected_weight,
                        expected_demand_volume_m3=expected_volume,
                        probability=prob,
                    )
                ]
            else:
                # Fallback to heuristic scenarios
                scenarios = self._forecast_scenarios(state, day_offset)

            period.scenarios = scenarios

            # Aggregate to expected values
            for scenario in scenarios:
                period.expected_orders += int(
                    scenario.expected_orders * scenario.probability
                )
                period.expected_revenue += (
                    scenario.expected_orders * 150.0 * scenario.probability
                )
                period.expected_demand_weight_tonnes += (
                    scenario.expected_demand_weight_tonnes * scenario.probability
                )
                period.expected_demand_volume_m3 += (
                    scenario.expected_demand_volume_m3 * scenario.probability
                )

            periods.append(period)

        return periods

    def _forecast_scenarios(
        self, state: SystemState, day_offset: int
    ) -> List[ForecastScenario]:
        """Generate demand scenarios for a period."""
        scenarios = []

        # Base scenario: use historical average
        base_orders = 10 + (5 if day_offset < 2 else 0)  # More orders early in week
        base_weight = base_orders * 0.5
        base_volume = base_orders * 0.3

        # Scenario 1: High demand (20% probability)
        scenarios.append(
            ForecastScenario(
                scenario_id=f"scenario_high_day{day_offset}",
                day_offset=day_offset,
                expected_orders=int(base_orders * 1.5),
                expected_demand_weight_tonnes=base_weight * 1.5,
                expected_demand_volume_m3=base_volume * 1.5,
                probability=0.2,
            )
        )

        # Scenario 2: Normal demand (60% probability)
        scenarios.append(
            ForecastScenario(
                scenario_id=f"scenario_normal_day{day_offset}",
                day_offset=day_offset,
                expected_orders=base_orders,
                expected_demand_weight_tonnes=base_weight,
                expected_demand_volume_m3=base_volume,
                probability=0.6,
            )
        )

        # Scenario 3: Low demand (20% probability)
        scenarios.append(
            ForecastScenario(
                scenario_id=f"scenario_low_day{day_offset}",
                day_offset=day_offset,
                expected_orders=int(base_orders * 0.5),
                expected_demand_weight_tonnes=base_weight * 0.5,
                expected_demand_volume_m3=base_volume * 0.5,
                probability=0.2,
            )
        )

        return scenarios

    def _optimize_deterministic(
        self, state: SystemState, context: DecisionContext, periods: List[DLAPeriod]
    ) -> List[Route]:
        """Deterministic optimization: plan assuming expected demand."""
        all_routes = []

        for period in periods:
            # Create a "synthetic" set of orders based on forecast
            if period == periods[0]:
                # Today: use actual pending orders
                orders_to_route = context.orders_to_consider
            else:
                # Future: create synthetic orders from forecast
                orders_to_route = self._generate_synthetic_orders(period)

            # Optimize routing for this period
            period_routes = self._optimize_period(
                state, orders_to_route, context.vehicles_available, period
            )
            all_routes.extend(period_routes)

            # Track period plan
            period.planned_routes = period_routes
            period.planned_costs = sum(r.estimated_cost_kes for r in period_routes)
            period.planned_profit = (
                sum(state.get_estimated_route_value(r) for r in period_routes)
                - period.planned_costs
            )

        return all_routes

    def _optimize_stochastic(
        self, state: SystemState, context: DecisionContext, periods: List[DLAPeriod]
    ) -> List[Route]:
        """Stochastic optimization: consider multiple scenarios, optimize for robustness."""

        # Simplified: average across scenarios (in production, use robust optimization)
        # Find routes that perform well across multiple scenarios

        all_routes = []

        for period in periods:
            # Evaluate each scenario
            scenario_solutions = []

            for scenario in period.scenarios:
                # Create orders for this scenario
                scenario_orders = self._generate_orders_for_scenario(scenario, state)

                # Optimize for scenario
                sol = self._optimize_period(
                    state, scenario_orders, context.vehicles_available, period
                )
                scenario_solutions.append((scenario, sol))

            # Choose robust solution (good across scenarios)
            best_routes = max(
                scenario_solutions,
                key=lambda x: sum(state.get_route_profitability(r) for r in x[1]),
            )[1]

            all_routes.extend(best_routes)
            period.planned_routes = best_routes

        return all_routes

    def _optimize_period(
        self,
        state: SystemState,
        orders: Dict[str, Order],
        vehicles: Dict[str, Vehicle],
        period: DLAPeriod,
    ) -> List[Route]:
        """Optimize routing for single period (simple greedy)."""
        routes = []

        if not orders or not vehicles:
            return routes

        # Simple greedy: assign orders to first-fit vehicles
        for vehicle in vehicles.values():
            vehicle_orders = {}
            remaining_weight = vehicle.capacity_weight_tonnes
            remaining_volume = vehicle.capacity_volume_m3

            for order in orders.values():
                if (
                    order.weight_tonnes <= remaining_weight
                    and order.volume_m3 <= remaining_volume
                ):
                    vehicle_orders[order.order_id] = order
                    remaining_weight -= order.weight_tonnes
                    remaining_volume -= order.volume_m3

            if vehicle_orders:
                route = self._create_route_from_orders(
                    state, vehicle_orders, vehicle, period
                )
                routes.append(route)

        return routes

    def _generate_synthetic_orders(self, period: DLAPeriod) -> Dict[str, Order]:
        """Generate synthetic orders for future period based on forecast."""
        orders: Dict[str, Order] = {}

        num = max(1, int(period.expected_orders))
        avg_weight = (period.expected_demand_weight_tonnes / num) if num > 0 else 0.5
        avg_volume = (period.expected_demand_volume_m3 / num) if num > 0 else 0.3

        for i in range(num):
            oid = f"synth_{period.period_id}_{i}_{uuid.uuid4().hex[:6]}"
            loc = Location(
                latitude=0.0 + i * 0.001, longitude=0.0 + i * 0.001, address="synthetic"
            )
            tw_start = period.start_time
            tw_end = period.start_time.replace(hour=18, minute=0)
            tw = TimeWindow(start_time=tw_start, end_time=tw_end)

            order = Order(
                order_id=oid,
                customer_id=f"cust_{i}",
                customer_name="synthetic",
                pickup_location=loc,
                destination_city=DestinationCity.NAKURU,
                destination_location=loc,
                weight_tonnes=avg_weight,
                volume_m3=avg_volume,
                time_window=tw,
                delivery_window=None,
                priority=0,
                special_handling=[],
                price_kes=150.0,
            )

            orders[oid] = order

        return orders

    def _generate_orders_for_scenario(
        self, scenario: ForecastScenario, state: SystemState
    ) -> Dict[str, Order]:
        """Generate orders matching scenario characteristics."""
        orders: Dict[str, Order] = {}

        count = max(1, int(scenario.expected_orders))
        avg_weight = scenario.expected_demand_weight_tonnes / max(1, count)
        avg_volume = scenario.expected_demand_volume_m3 / max(1, count)

        for i in range(count):
            oid = f"sc_{scenario.scenario_id}_{i}_{uuid.uuid4().hex[:6]}"
            loc = Location(
                latitude=0.0 + i * 0.001, longitude=0.0 + i * 0.001, address="scenario"
            )
            now = datetime.now()
            tw = TimeWindow(start_time=now, end_time=now + timedelta(hours=8))

            order = Order(
                order_id=oid,
                customer_id=f"cust_s_{i}",
                customer_name="scenario",
                pickup_location=loc,
                destination_city=DestinationCity.NAKURU,
                destination_location=loc,
                weight_tonnes=avg_weight,
                volume_m3=avg_volume,
                time_window=tw,
                delivery_window=None,
                priority=0,
                special_handling=[],
                price_kes=150.0,
            )

            orders[oid] = order

        return orders

    def _create_route_from_orders(
        self,
        state: SystemState,
        orders: Dict[str, Order],
        vehicle: Vehicle,
        period: DLAPeriod,
    ) -> Route:
        """Create route from orders for period."""
        route_id = f"route_dla_{uuid.uuid4().hex[:12]}"

        total_weight = sum(o.weight_tonnes for o in orders.values())
        total_volume = sum(o.volume_m3 for o in orders.values())

        route = Route(
            route_id=route_id,
            vehicle_id=vehicle.vehicle_id,
            order_ids=list(orders.keys()),
            stops=[],
            destination_cities=list(set(o.destination_city for o in orders.values())),
            total_distance_km=0.0,
            estimated_duration_minutes=0,
            estimated_cost_kes=total_weight * 50.0,  # Rough estimate
        )

        return route

    def _calculate_expected_value(
        self, periods: List[DLAPeriod], routes: List[Route]
    ) -> float:
        """Calculate expected multi-period profit."""
        total_profit = 0.0

        for period in periods:
            revenue = period.expected_revenue
            costs = period.planned_costs
            profit = revenue - costs
            # Discount future profit using standard exponential discounting by day offset
            discount_factor = 0.95 ** getattr(period, "day_offset", 0)
            total_profit += profit * discount_factor

        # Add terminal value from VFA if available
        if self.use_vfa_terminal_value and self.vfa:
            # In production: compute VFA value at horizon end
            terminal_value = 1000.0  # Placeholder
            total_profit += 0.95**self.horizon_days * terminal_value

        return total_profit

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DLA state."""
        return {
            "horizon_days": self.horizon_days,
            "deterministic": self.deterministic,
            "consolidation_threshold": self.consolidation_threshold,
            "demand_forecast_accuracy": self.demand_forecast_accuracy,
            "use_vfa_terminal_value": self.use_vfa_terminal_value,
        }

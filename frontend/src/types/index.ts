// Core domain models
export interface Location {
  latitude: number;
  longitude: number;
  address: string;
  zone: string;
}

export interface TimeWindow {
  start_time: string;
  end_time: string;
}

export enum VehicleStatus {
  AVAILABLE = 'available',
  IN_TRANSIT = 'in_transit',
  MAINTENANCE = 'maintenance',
  OFF_DUTY = 'off_duty',
}

export interface Vehicle {
  vehicle_id: string;
  vehicle_type: string;
  capacity_weight_tonnes: number;
  capacity_volume_m3: number;
  current_location: Location;
  available_at: string;
  status: VehicleStatus;
  driver_id: string;
  fuel_cost_per_km: number;
  driver_cost_per_hour: number;
}

export enum DestinationCity {
  NAKURU = 'Nakuru',
  ELDORET = 'Eldoret',
  KISUMU = 'kisumu',
  KITALE = 'Kitale',
}

export enum OrderStatus {
  PENDING = 'pending',
  ASSIGNED = 'assigned',
  IN_TRANSIT = 'in_transit',
  DELIVERED = 'delivered',
  CANCELLED = 'cancelled',
}

export interface Order {
  order_id: string;
  customer_id: string;
  customer_name: string;
  destination_city: DestinationCity;
  weight_tonnes: number;
  volume_m3: number;
  priority: number;
  time_window: TimeWindow;
  delivery_window: TimeWindow;
  pickup_location: Location;
  destination_location: Location;
  price_kes: number;
  special_handling?: string[];
  status: OrderStatus;
  assigned_route_id?: string;
  customer_constraints?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Customer {
  customer_id: string;
  name: string;
  locations: Location[];
  delivery_blocked_times?: any[];
  priority_level?: number;
  fresh_food_customer?: boolean;
}

export enum RouteStatus {
  PLANNED = 'planned',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
}

export interface Route {
  route_id: string;
  vehicle_id: string;
  order_ids: string[];
  stops: Location[];
  destination_cities: DestinationCity[];
  total_distance_km: number;
  estimated_duration_minutes: number;
  estimated_cost_kes: number;
  status: RouteStatus;
  estimated_fuel_cost: number;
  estimated_time_cost: number;
  estimated_delay_penalty: number;
  actual_distance_km?: number;
  actual_duration_minutes?: number;
  actual_cost_kes?: number;
  actual_fuel_cost?: number;
  decision_id?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface OperationalOutcome {
  outcome_id: string;
  route_id: string;
  vehicle_id: string;
  predicted_fuel_cost: number;
  actual_fuel_cost: number;
  predicted_duration_minutes: number;
  actual_duration_minutes: number;
  predicted_distance_km: number;
  actual_distance_km: number;
  on_time: boolean;
  delay_minutes?: number;
  successful_deliveries?: number;
  failed_deliveries?: number;
  traffic_conditions?: Record<string, number>;
  weather?: string;
  day_of_week?: string;
  customer_satisfaction_score?: number;
  notes?: string;
  recorded_at: string;
}

// Learning Metrics
export interface TelemetryMetrics {
  vfa: {
    last_training_loss: number | null;
    last_training_samples: number;
    total_training_steps: number;
    last_training_timestamp: string | null;
    prioritized_replay_size: number;
    current_learning_rate: number | null;
    early_stopping_triggered: boolean;
  };
  cfa: {
    fuel_cost_per_km: number | null;
    driver_cost_per_hour: number | null;
    fuel_accuracy_mape: number | null;
    time_accuracy_mape: number | null;
    fuel_converged: boolean;
    time_converged: boolean;
    total_updates: number;
  };
  pfa: {
    total_rules: number;
    active_rules: number;
    patterns_mined: number;
    avg_rule_confidence: number | null;
    avg_rule_lift: number | null;
    exploration_rate: number | null;
    last_mining_timestamp: string | null;
  };
  general: {
    total_outcomes_processed: number;
    last_outcome_timestamp: string | null;
    learning_active: boolean;
  };
}

export interface LearningMetrics {
  aggregate_metrics: any;
  model_accuracies: any;
  telemetry: TelemetryMetrics;
  vfa_realtime?: any;
  cfa_realtime?: any;
  pfa_realtime?: any;
}

// API Response types
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}

export interface HealthResponse {
  status: string;
  components: Record<string, string>;
}

// Decision types
export enum DecisionType {
  DAILY_ROUTE_PLANNING = 'daily_route_planning',
  ORDER_ARRIVAL = 'order_arrival',
  REAL_TIME_ADJUSTMENT = 'real_time_adjustment',
  BACKHAUL_OPPORTUNITY = 'backhaul_opportunity',
}

export enum ActionType {
  CREATE_ROUTE = 'create_route',
  WAIT_FOR_CONSOLIDATION = 'wait_for_consolidation',
  MODIFY_ROUTE = 'modify_route',
  NO_ACTION = 'no_action',
}

export interface PolicyDecision {
  policy_name: string;
  decision_type: DecisionType;
  recommended_action: ActionType;
  routes: Route[];
  confidence_score: number;
  expected_value: number;
  reasoning: string;
  considered_alternatives: number;
  is_deterministic: boolean;
  decision_id: string;
  policy_parameters: Record<string, any>;
  created_at: string;
}

// Policy Comparison Types
export interface PolicyRecommendation {
  policy_name: string;
  recommended_action: ActionType;
  confidence_score: number;
  expected_value: number;
  routes: Route[];
  reasoning: string;
  policy_parameters: Record<string, any>;
}

export interface AgreementAnalysis {
  agreement_score: number; // 0.0 to 1.0
  consensus_action: string;
  consensus_count: number;
  total_policies: number;
  conflicts: Array<{
    policy: string;
    action: string;
    confidence: number;
  }>;
  avg_confidence: number;
  avg_expected_value: number;
}

export interface PolicyComparisonResponse {
  decision_type: DecisionType;
  timestamp: string;

  // Individual policy recommendations
  pfa: PolicyRecommendation;
  vfa: PolicyRecommendation;
  cfa: PolicyRecommendation;
  dla: PolicyRecommendation;

  // Engine's recommended policy
  recommended: PolicyRecommendation;

  // Agreement analysis
  agreement_analysis: AgreementAnalysis;

  // Metadata
  computation_time_ms: number;
  trigger_reason: string;
}

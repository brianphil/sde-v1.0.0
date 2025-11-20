import axios from 'axios';
import type {
  Order,
  Vehicle,
  Customer,
  Route,
  PolicyDecision,
  LearningMetrics,
  HealthResponse,
  OperationalOutcome,
  DecisionType,
  PolicyComparisonResponse,
} from '../types';

const API_BASE_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health
export const healthAPI = {
  check: () => api.get<HealthResponse>('/health'),
};

// Orders
export const ordersAPI = {
  list: () => api.get<Order[]>('/orders'),
  get: (orderId: string) => api.get<Order>(`/orders/${orderId}`),
  create: (order: Partial<Order>) => api.post<Order>('/orders', order),
  update: (orderId: string, order: Partial<Order>) => 
    api.put<Order>(`/orders/${orderId}`, order),
  delete: (orderId: string) => api.delete(`/orders/${orderId}`),
};

// Vehicles
export const vehiclesAPI = {
  list: () => api.get<Vehicle[]>('/vehicles'),
  get: (vehicleId: string) => api.get<Vehicle>(`/vehicles/${vehicleId}`),
  update: (vehicleId: string, vehicle: Partial<Vehicle>) => 
    api.put<Vehicle>(`/vehicles/${vehicleId}`, vehicle),
};

// Customers
export const customersAPI = {
  list: () => api.get<Customer[]>('/customers'),
  get: (customerId: string) => api.get<Customer>(`/customers/${customerId}`),
  create: (customer: Partial<Customer>) => api.post<Customer>('/customers', customer),
};

// Routes
export const routesAPI = {
  list: () => api.get<Route[]>('/routes'),
  get: (routeId: string) => api.get<Route>(`/routes/${routeId}`),
  active: () => api.get<Route[]>('/routes/active'),
  completed: () => api.get<Route[]>('/routes/completed'),
  start: (routeId: string) => api.post<Route>(`/routes/${routeId}/start`),
  complete: (routeId: string, outcome: Partial<OperationalOutcome>) => 
    api.post<Route>(`/routes/${routeId}/complete`, outcome),
};

// Decisions
export const decisionsAPI = {
  makeDecision: (decisionType: DecisionType, orderIds?: string[]) =>
    api.post<PolicyDecision>('/decisions/make', {
      decision_type: decisionType,
      order_ids: orderIds
    }),
  comparePolicies: (decisionType: DecisionType, triggerReason?: string) =>
    api.post<PolicyComparisonResponse>('/decisions/compare-policies', {
      decision_type: decisionType,
      trigger_reason: triggerReason || 'Manual policy comparison from UI',
    }),
  history: (limit?: number) => api.get<PolicyDecision[]>('/decisions/history', {
    params: { limit },
  }),
  commit: (decisionId: string) => api.post(`/decisions/${decisionId}/commit`),
};

// Learning Metrics
export const metricsAPI = {
  get: () => api.get<LearningMetrics>('/metrics'),
  telemetry: () => api.get<any>('/metrics/telemetry'),
};

export default api;

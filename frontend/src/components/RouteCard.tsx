import { Route, Order, Vehicle } from '../types';
import { Package, Truck, MapPin, TrendingUp, Clock, DollarSign } from 'lucide-react';

interface RouteCardProps {
  route: Route;
  orders: Order[];
  vehicle?: Vehicle;
  showConsolidation?: boolean;
  detailed?: boolean;
}

export default function RouteCard({ route, orders, vehicle, showConsolidation = true, detailed = false }: RouteCardProps) {
  // Get orders for this route
  const routeOrders = orders.filter(o => route.order_ids.includes(o.order_id));

  // Calculate consolidation metrics
  const totalWeight = routeOrders.reduce((sum, o) => sum + o.weight_tonnes, 0);
  const totalVolume = routeOrders.reduce((sum, o) => sum + o.volume_m3, 0);
  const totalRevenue = routeOrders.reduce((sum, o) => sum + o.price_kes, 0);

  // Vehicle capacity (fallback if vehicle not provided)
  const vehicleCapacity = vehicle?.capacity_tonnes || 10;
  const vehicleVolumeCapacity = vehicle?.capacity_m3 || 40;

  const weightUtilization = (totalWeight / vehicleCapacity) * 100;
  const volumeUtilization = (totalVolume / vehicleVolumeCapacity) * 100;
  const maxUtilization = Math.max(weightUtilization, volumeUtilization);

  // Group orders by destination
  const ordersByDestination = routeOrders.reduce((acc, order) => {
    const dest = order.destination_city;
    if (!acc[dest]) acc[dest] = [];
    acc[dest].push(order);
    return acc;
  }, {} as Record<string, Order[]>);

  // Calculate cost per KES revenue
  const costPerRevenue = totalRevenue > 0 ? (route.estimated_cost_kes / totalRevenue) * 100 : 0;
  const profitMargin = totalRevenue - route.estimated_cost_kes;

  return (
    <div className="border rounded-lg bg-white overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-primary-50 to-primary-100 px-4 py-3 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Truck className="h-5 w-5 text-primary-600" />
            <div>
              <h4 className="font-bold text-gray-900">{route.vehicle_id}</h4>
              <p className="text-xs text-gray-600">{route.route_id}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm font-medium text-gray-700">{routeOrders.length} orders</p>
            <p className="text-xs text-gray-500">{route.destination_cities.join(', ')}</p>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Consolidation Insights */}
        {showConsolidation && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <h5 className="font-semibold text-sm mb-2 text-blue-900">Consolidation Insights</h5>
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div>
                <p className="text-gray-600">Orders Grouped</p>
                <p className="font-bold text-blue-900">{routeOrders.length} orders</p>
                <p className="text-gray-500">{Object.keys(ordersByDestination).length} destinations</p>
              </div>
              <div>
                <p className="text-gray-600">Revenue</p>
                <p className="font-bold text-green-700">{totalRevenue.toFixed(0)} KES</p>
                <p className="text-gray-500">Profit: {profitMargin.toFixed(0)} KES</p>
              </div>
              <div>
                <p className="text-gray-600">Cost Efficiency</p>
                <p className="font-bold text-gray-900">{costPerRevenue.toFixed(1)}%</p>
                <p className="text-gray-500">cost per revenue</p>
              </div>
            </div>
          </div>
        )}

        {/* Vehicle Capacity Utilization */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Capacity Utilization</span>
            <span className="text-sm font-bold text-gray-900">{maxUtilization.toFixed(0)}%</span>
          </div>

          {/* Weight Bar */}
          <div className="mb-2">
            <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
              <span>Weight</span>
              <span>{totalWeight.toFixed(1)}T / {vehicleCapacity}T</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  weightUtilization > 90 ? 'bg-red-500' :
                  weightUtilization > 75 ? 'bg-yellow-500' :
                  'bg-green-500'
                }`}
                style={{ width: `${Math.min(weightUtilization, 100)}%` }}
              />
            </div>
          </div>

          {/* Volume Bar */}
          <div>
            <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
              <span>Volume</span>
              <span>{totalVolume.toFixed(1)}m³ / {vehicleVolumeCapacity}m³</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  volumeUtilization > 90 ? 'bg-red-500' :
                  volumeUtilization > 75 ? 'bg-yellow-500' :
                  'bg-green-500'
                }`}
                style={{ width: `${Math.min(volumeUtilization, 100)}%` }}
              />
            </div>
          </div>
        </div>

        {/* Route Metrics Grid */}
        <div className="grid grid-cols-3 gap-3 text-xs">
          <div className="bg-gray-50 rounded p-2">
            <div className="flex items-center gap-1 text-gray-600 mb-1">
              <MapPin className="h-3 w-3" />
              <span>Distance</span>
            </div>
            <p className="font-bold text-gray-900">{route.total_distance_km.toFixed(0)} km</p>
          </div>
          <div className="bg-gray-50 rounded p-2">
            <div className="flex items-center gap-1 text-gray-600 mb-1">
              <Clock className="h-3 w-3" />
              <span>Duration</span>
            </div>
            <p className="font-bold text-gray-900">{Math.floor(route.estimated_duration_minutes / 60)}h {route.estimated_duration_minutes % 60}m</p>
          </div>
          <div className="bg-gray-50 rounded p-2">
            <div className="flex items-center gap-1 text-gray-600 mb-1">
              <DollarSign className="h-3 w-3" />
              <span>Cost</span>
            </div>
            <p className="font-bold text-gray-900">{route.estimated_cost_kes.toFixed(0)} KES</p>
          </div>
        </div>

        {/* Stop Sequencing */}
        {detailed && (
          <div className="border-t pt-3">
            <h5 className="font-semibold text-sm mb-3 flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              Route Sequence ({Object.keys(ordersByDestination).length} stops)
            </h5>
            <div className="space-y-2">
              {/* Depot Start */}
              <div className="flex items-start gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-6 h-6 rounded-full bg-gray-300 flex items-center justify-center text-xs font-bold">
                    0
                  </div>
                  <div className="w-0.5 h-8 bg-gray-300"></div>
                </div>
                <div className="flex-1 pb-2">
                  <p className="font-medium text-sm">Depot (Eastleigh)</p>
                  <p className="text-xs text-gray-500">Start - Load {routeOrders.length} orders</p>
                </div>
              </div>

              {/* Destination Stops */}
              {Object.entries(ordersByDestination).map(([destination, destOrders], idx) => {
                const stopNumber = idx + 1;
                const isLastStop = idx === Object.keys(ordersByDestination).length - 1;

                return (
                  <div key={destination} className="flex items-start gap-3">
                    <div className="flex flex-col items-center">
                      <div className="w-6 h-6 rounded-full bg-primary-500 flex items-center justify-center text-xs font-bold text-white">
                        {stopNumber}
                      </div>
                      {!isLastStop && <div className="w-0.5 h-8 bg-gray-300"></div>}
                    </div>
                    <div className="flex-1 pb-2">
                      <p className="font-medium text-sm">{destination}</p>
                      <p className="text-xs text-gray-600 mb-1">
                        Deliver {destOrders.length} order{destOrders.length > 1 ? 's' : ''} • {destOrders.reduce((sum, o) => sum + o.weight_tonnes, 0).toFixed(1)}T
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {destOrders.map(order => (
                          <div key={order.order_id} className="inline-flex items-center gap-1 bg-blue-100 px-2 py-0.5 rounded text-xs">
                            <Package className="h-3 w-3" />
                            <span className="font-mono">{order.order_id.slice(-6)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Return to Depot */}
              <div className="flex items-start gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-6 h-6 rounded-full bg-gray-300 flex items-center justify-center text-xs font-bold">
                    {Object.keys(ordersByDestination).length + 1}
                  </div>
                </div>
                <div className="flex-1">
                  <p className="font-medium text-sm">Return to Depot</p>
                  <p className="text-xs text-gray-500">End route</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Order List (if not detailed) */}
        {!detailed && (
          <div className="border-t pt-3">
            <h5 className="font-semibold text-sm mb-2">Orders on Route</h5>
            <div className="flex flex-wrap gap-1">
              {routeOrders.map(order => (
                <div key={order.order_id} className="inline-flex items-center gap-1 bg-gray-100 px-2 py-1 rounded text-xs">
                  <Package className="h-3 w-3" />
                  <span className="font-mono">{order.order_id.slice(-6)}</span>
                  <span className="text-gray-500">→ {order.destination_city}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
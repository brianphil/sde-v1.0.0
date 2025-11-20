import { useQuery } from '@tanstack/react-query'
import { routesAPI } from '../services/api'

export default function RoutesPage() {
  const { data: routes, isLoading } = useQuery({
    queryKey: ['routes'],
    queryFn: async () => {
      const response = await routesAPI.list()
      return response.data
    },
  })

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Routes</h1>
      <div className="bg-white rounded-lg shadow p-6">
        {isLoading ? (
          <p className="text-gray-500">Loading routes...</p>
        ) : Array.isArray(routes) && routes.length > 0 ? (
          <div className="space-y-4">
            {routes.map(route => (
              <div key={route.route_id} className="border rounded-lg p-4">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-semibold">{route.route_id}</h3>
                    <p className="text-sm text-gray-600">Vehicle: {route.vehicle_id}</p>
                    <p className="text-sm text-gray-600">Orders: {route.order_ids.join(', ')}</p>
                    <p className="text-sm text-gray-600">Distance: {route.total_distance_km} km</p>
                    <p className="text-sm text-gray-600">Duration: {route.estimated_duration_minutes} min</p>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    route.status === 'completed' ? 'bg-green-100 text-green-800' :
                    route.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {route.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No routes found</p>
        )}
      </div>
    </div>
  )
}

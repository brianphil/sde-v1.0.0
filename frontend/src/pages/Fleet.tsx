import { useQuery } from '@tanstack/react-query'
import { vehiclesAPI } from '../services/api'

export default function Fleet() {
  const { data: vehicles, isLoading } = useQuery({
    queryKey: ['vehicles'],
    queryFn: async () => {
      const response = await vehiclesAPI.list()
      return response.data
    },
  })

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Fleet Management</h1>
      <div className="bg-white rounded-lg shadow p-6">
        {isLoading ? (
          <p className="text-gray-500">Loading vehicles...</p>
        ) : Array.isArray(vehicles) && vehicles.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {vehicles.map(vehicle => (
              <div key={vehicle.vehicle_id} className="border rounded-lg p-4">
                <h3 className="font-semibold">{vehicle.vehicle_id}</h3>
                <p className="text-sm text-gray-600">{vehicle.vehicle_type} - {vehicle.capacity_weight_tonnes}T</p>
                <p className="text-sm text-gray-600">Driver: {vehicle.driver_id}</p>
                <span className={`inline-block px-2 py-1 text-xs rounded-full ${
                  vehicle.status === 'available' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                }`}>
                  {vehicle.status}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No vehicles found</p>
        )}
      </div>
    </div>
  )
}

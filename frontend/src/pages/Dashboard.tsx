import { useQuery } from "@tanstack/react-query";
import { Activity, Truck, Package, CheckCircle } from "lucide-react";
import { metricsAPI, ordersAPI, vehiclesAPI, routesAPI } from "../services/api";
import type { LearningMetrics } from "../types";

export default function Dashboard() {
  const { data: metrics } = useQuery<LearningMetrics>({
    queryKey: ["metrics"],
    queryFn: async () => {
      const response = await metricsAPI.get();
      return response.data;
    },
    refetchInterval: 5000,
  });

  const { data: orders } = useQuery({
    queryKey: ["orders"],
    queryFn: async () => {
      const response = await ordersAPI.list();
      return response.data;
    },
  });

  const { data: vehicles } = useQuery({
    queryKey: ["vehicles"],
    queryFn: async () => {
      const response = await vehiclesAPI.list();
      return response.data;
    },
  });

  const { data: activeRoutes } = useQuery({
    queryKey: ["routes", "active"],
    queryFn: async () => {
      const response = await routesAPI.active();
      return response.data;
    },
  });

  const pendingOrders = Array.isArray(orders) ? orders.length : 0;
  const availableVehicles = Array.isArray(vehicles)
    ? vehicles.filter((v) => v.status === "available").length
    : 0;
  const activeRoutesCount = Array.isArray(activeRoutes)
    ? activeRoutes.length
    : 0;
  const totalOutcomes =
    metrics?.telemetry?.general?.total_outcomes_processed || 0;

  const fuelCost = metrics?.telemetry?.cfa?.fuel_cost_per_km;
  const timeCost = metrics?.telemetry?.cfa?.driver_cost_per_hour;
  const cfaUpdates = metrics?.telemetry?.cfa?.total_updates || 0;

  const vfaSteps = metrics?.telemetry?.vfa?.total_training_steps || 0;
  const bufferSize = metrics?.telemetry?.vfa?.prioritized_replay_size || 0;
  const learningRate = metrics?.telemetry?.vfa?.current_learning_rate;

  const totalRules = metrics?.telemetry?.pfa?.total_rules || 0;
  const ruleConfidence = metrics?.telemetry?.pfa?.avg_rule_confidence || 0;
  const explorationRate = metrics?.telemetry?.pfa?.exploration_rate || 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Senga Sequential Decision Engine - Real-time Overview
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Pending Orders"
          value={pendingOrders}
          icon={Package}
          color="blue"
        />
        <StatCard
          title="Available Vehicles"
          value={availableVehicles}
          icon={Truck}
          color="green"
        />
        <StatCard
          title="Active Routes"
          value={activeRoutesCount}
          icon={Activity}
          color="yellow"
        />
        <StatCard
          title="Outcomes Processed"
          value={totalOutcomes}
          icon={CheckCircle}
          color="purple"
        />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Cost Function Approximation
          </h2>
          <div className="space-y-3">
            <MetricRow
              label="Fuel Cost"
              value={fuelCost ? `${fuelCost.toFixed(2)} KES/km` : "N/A"}
            />
            <MetricRow
              label="Time Cost"
              value={timeCost ? `${timeCost.toFixed(2)} KES/hr` : "N/A"}
            />
            <MetricRow label="Total Updates" value={cfaUpdates} />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Value Function Approximation
          </h2>
          <div className="space-y-3">
            <MetricRow label="Training Steps" value={vfaSteps} />
            <MetricRow
              label="Experience Buffer"
              value={`${bufferSize} samples`}
            />
            <MetricRow
              label="Learning Rate"
              value={learningRate ? learningRate.toFixed(6) : "N/A"}
            />
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Policy Function Approximation
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-gray-500">Total Rules</p>
            <p className="text-2xl font-bold text-gray-900">{totalRules}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Avg Confidence</p>
            <p className="text-2xl font-bold text-gray-900">
              {(ruleConfidence * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Exploration Rate</p>
            <p className="text-2xl font-bold text-gray-900">
              {(explorationRate * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: number | string;
  icon: React.ElementType;
  color: "blue" | "green" | "yellow" | "purple";
}

function StatCard({ title, value, icon: Icon, color }: StatCardProps) {
  const colorClasses = {
    blue: "bg-blue-50 text-blue-600",
    green: "bg-green-50 text-green-600",
    yellow: "bg-yellow-50 text-yellow-600",
    purple: "bg-purple-50 text-purple-600",
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  );
}

function MetricRow({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-gray-600">{label}</span>
      <span className="text-sm font-medium text-gray-900">{value}</span>
    </div>
  );
}

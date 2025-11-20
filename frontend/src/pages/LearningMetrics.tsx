import { useQuery } from '@tanstack/react-query'
import { BrainCircuit, TrendingUp, Zap } from 'lucide-react'
import { metricsAPI } from '../services/api'
import type { LearningMetrics as LearningMetricsType } from '../types'

export default function LearningMetrics() {
  const { data: metrics, isLoading } = useQuery<LearningMetricsType>({
    queryKey: ['metrics'],
    queryFn: async () => {
      const response = await metricsAPI.get()
      return response.data
    },
    refetchInterval: 3000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Loading metrics...</p>
      </div>
    )
  }

  const telemetry = metrics?.telemetry

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Learning Metrics</h1>
        <p className="mt-1 text-sm text-gray-500">
          World-Class Learning Telemetry - VFA, CFA, PFA
        </p>
      </div>

      {/* VFA Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center mb-6">
          <BrainCircuit className="h-6 w-6 text-blue-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">
            Value Function Approximation (VFA)
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            label="Training Steps"
            value={telemetry?.vfa.total_training_steps || 0}
          />
          <MetricCard
            label="Last Loss"
            value={telemetry?.vfa.last_training_loss?.toFixed(4) || 'N/A'}
          />
          <MetricCard
            label="Experience Buffer"
            value={`${telemetry?.vfa.prioritized_replay_size || 0} samples`}
          />
          <MetricCard
            label="Learning Rate"
            value={telemetry?.vfa.current_learning_rate?.toFixed(6) || 'N/A'}
          />
        </div>
        {telemetry?.vfa.early_stopping_triggered && (
          <div className="mt-4 p-3 bg-yellow-50 rounded-md">
            <p className="text-sm text-yellow-800">Early stopping triggered</p>
          </div>
        )}
      </div>

      {/* CFA Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center mb-6">
          <TrendingUp className="h-6 w-6 text-green-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">
            Cost Function Approximation (CFA)
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            label="Fuel Cost"
            value={`${telemetry?.cfa.fuel_cost_per_km?.toFixed(2) || 'N/A'} KES/km`}
            badge={telemetry?.cfa.fuel_converged ? 'Converged' : 'Learning'}
            badgeColor={telemetry?.cfa.fuel_converged ? 'green' : 'yellow'}
          />
          <MetricCard
            label="Driver Cost"
            value={`${telemetry?.cfa.driver_cost_per_hour?.toFixed(2) || 'N/A'} KES/hr`}
            badge={telemetry?.cfa.time_converged ? 'Converged' : 'Learning'}
            badgeColor={telemetry?.cfa.time_converged ? 'green' : 'yellow'}
          />
          <MetricCard
            label="Fuel Accuracy (MAPE)"
            value={telemetry?.cfa.fuel_accuracy_mape
              ? `${(telemetry.cfa.fuel_accuracy_mape * 100).toFixed(2)}%`
              : 'N/A'}
          />
          <MetricCard
            label="Total Updates"
            value={telemetry?.cfa.total_updates || 0}
          />
        </div>
      </div>

      {/* PFA Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center mb-6">
          <Zap className="h-6 w-6 text-purple-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">
            Policy Function Approximation (PFA)
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            label="Total Rules"
            value={telemetry?.pfa.total_rules || 0}
          />
          <MetricCard
            label="Active Rules"
            value={telemetry?.pfa.active_rules || 0}
          />
          <MetricCard
            label="Avg Confidence"
            value={`${((telemetry?.pfa.avg_rule_confidence || 0) * 100).toFixed(1)}%`}
          />
          <MetricCard
            label="Avg Lift"
            value={telemetry?.pfa.avg_rule_lift?.toFixed(2) || 'N/A'}
          />
        </div>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
          <MetricCard
            label="Patterns Mined"
            value={telemetry?.pfa.patterns_mined || 0}
          />
          <MetricCard
            label="Exploration Rate (µ)"
            value={`${((telemetry?.pfa.exploration_rate || 0) * 100).toFixed(1)}%`}
          />
        </div>
      </div>

      {/* General System Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">System Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <MetricCard
            label="Total Outcomes Processed"
            value={telemetry?.general.total_outcomes_processed || 0}
          />
          <MetricCard
            label="Learning Active"
            value={telemetry?.general.learning_active ? 'Yes' : 'No'}
            badge={telemetry?.general.learning_active ? 'Active' : 'Inactive'}
            badgeColor={telemetry?.general.learning_active ? 'green' : 'gray'}
          />
          <MetricCard
            label="Last Update"
            value={telemetry?.general.last_outcome_timestamp
              ? new Date(telemetry.general.last_outcome_timestamp).toLocaleTimeString()
              : 'Never'}
          />
        </div>
      </div>
    </div>
  )
}

interface MetricCardProps {
  label: string
  value: string | number
  badge?: string
  badgeColor?: 'green' | 'yellow' | 'gray'
}

function MetricCard({ label, value, badge, badgeColor = 'gray' }: MetricCardProps) {
  const badgeColors = {
    green: 'bg-green-100 text-green-800',
    yellow: 'bg-yellow-100 text-yellow-800',
    gray: 'bg-gray-100 text-gray-800',
  }

  return (
    <div className="border rounded-lg p-4">
      <p className="text-sm text-gray-500 mb-1">{label}</p>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      {badge && (
        <span className={`inline-block mt-2 px-2 py-1 text-xs rounded-full ${badgeColors[badgeColor]}`}>
          {badge}
        </span>
      )}
    </div>
  )
}

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { decisionsAPI } from '../services/api';
import { DecisionType, PolicyComparisonResponse, PolicyRecommendation } from '../types';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';

export default function PolicyComparison() {
  const [selectedType, setSelectedType] = useState<DecisionType>(DecisionType.ORDER_ARRIVAL);
  const [comparison, setComparison] = useState<PolicyComparisonResponse | null>(null);

  const comparePoliciesMutation = useMutation({
    mutationFn: () => decisionsAPI.comparePolicies(selectedType),
    onSuccess: (response) => {
      setComparison(response.data);
    },
  });

  const getAgreementColor = (score: number) => {
    if (score >= 0.75) return 'text-green-600 bg-green-50';
    if (score >= 0.5) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.7) return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (confidence >= 0.4) return <Minus className="h-4 w-4 text-yellow-600" />;
    return <TrendingDown className="h-4 w-4 text-red-600" />;
  };

  const PolicyCard = ({ policy, recommendation, isRecommended }: {
    policy: string;
    recommendation: PolicyRecommendation;
    isRecommended?: boolean;
  }) => (
    <div className={`border rounded-lg p-4 ${isRecommended ? 'ring-2 ring-primary-500 bg-primary-50' : 'bg-white'}`}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-bold text-lg">{policy}</h3>
          {isRecommended && (
            <span className="text-xs text-primary-600 font-medium">✓ Engine Recommended</span>
          )}
        </div>
        {getConfidenceIcon(recommendation.confidence_score)}
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Action:</span>
          <span className="font-medium">{recommendation.recommended_action.replace(/_/g, ' ')}</span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Confidence:</span>
          <span className="font-medium">{(recommendation.confidence_score * 100).toFixed(1)}%</span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Expected Value:</span>
          <span className="font-medium">{recommendation.expected_value.toFixed(0)} KES</span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Routes:</span>
          <span className="font-medium">{recommendation.routes.length}</span>
        </div>

        <div className="mt-3 pt-3 border-t">
          <p className="text-xs text-gray-600 font-medium mb-1">Reasoning:</p>
          <p className="text-xs text-gray-700">{recommendation.reasoning}</p>
        </div>

        {Object.keys(recommendation.policy_parameters).length > 0 && (
          <div className="mt-2">
            <p className="text-xs text-gray-600 font-medium mb-1">Parameters:</p>
            <div className="text-xs text-gray-600 space-y-1">
              {Object.entries(recommendation.policy_parameters).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span>{key}:</span>
                  <span className="font-mono">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">AI Policy Comparison</h2>
        <p className="text-gray-600 mt-1">
          Compare all 4 Powell policy classes to understand decision consensus and disagreement
        </p>
      </div>

      {/* Decision Type Selector */}
      <div className="bg-white border rounded-lg p-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Decision Type
        </label>
        <select
          value={selectedType}
          onChange={(e) => setSelectedType(e.target.value as DecisionType)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        >
          <option value={DecisionType.ORDER_ARRIVAL}>Order Arrival</option>
          <option value={DecisionType.DAILY_ROUTE_PLANNING}>Daily Route Planning</option>
          <option value={DecisionType.REAL_TIME_ADJUSTMENT}>Real-Time Adjustment</option>
          <option value={DecisionType.BACKHAUL_OPPORTUNITY}>Backhaul Opportunity</option>
        </select>

        <button
          onClick={() => comparePoliciesMutation.mutate()}
          disabled={comparePoliciesMutation.isPending}
          className="mt-4 w-full bg-primary-600 text-white py-2 px-4 rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {comparePoliciesMutation.isPending ? 'Comparing Policies...' : 'Compare All 4 Policies'}
        </button>
      </div>

      {/* Results */}
      {comparison && (
        <div className="space-y-6">
          {/* Agreement Analysis */}
          <div className={`border rounded-lg p-6 ${getAgreementColor(comparison.agreement_analysis.agreement_score)}`}>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h3 className="font-bold text-lg mb-2">Policy Agreement Analysis</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600">Agreement Score</p>
                    <p className="text-2xl font-bold">
                      {(comparison.agreement_analysis.agreement_score * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Consensus</p>
                    <p className="font-bold">{comparison.agreement_analysis.consensus_action.replace(/_/g, ' ')}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Agreeing Policies</p>
                    <p className="font-bold">{comparison.agreement_analysis.consensus_count}/{comparison.agreement_analysis.total_policies}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Avg Confidence</p>
                    <p className="font-bold">{(comparison.agreement_analysis.avg_confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>

                {comparison.agreement_analysis.conflicts.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-current/20">
                    <p className="font-medium mb-2 flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4" />
                      Policy Conflicts ({comparison.agreement_analysis.conflicts.length})
                    </p>
                    <div className="space-y-1">
                      {comparison.agreement_analysis.conflicts.map((conflict, idx) => (
                        <div key={idx} className="text-sm flex items-center gap-2">
                          <XCircle className="h-4 w-4" />
                          <span className="font-medium">{conflict.policy}:</span>
                          <span>{conflict.action.replace(/_/g, ' ')}</span>
                          <span className="text-xs">({(conflict.confidence * 100).toFixed(0)}% confidence)</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {comparison.agreement_analysis.conflicts.length === 0 && (
                  <div className="mt-4 pt-4 border-t border-current/20">
                    <p className="font-medium flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4" />
                      All policies agree - high confidence decision!
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Individual Policy Recommendations */}
          <div>
            <h3 className="font-bold text-lg mb-4">Individual Policy Recommendations</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <PolicyCard policy="PFA (Pattern-Based)" recommendation={comparison.pfa} />
              <PolicyCard policy="VFA (Value Learning)" recommendation={comparison.vfa} />
              <PolicyCard policy="CFA (Cost Learning)" recommendation={comparison.cfa} />
              <PolicyCard policy="DLA (Lookahead)" recommendation={comparison.dla} />
            </div>
          </div>

          {/* Engine Recommended Decision */}
          <div>
            <h3 className="font-bold text-lg mb-4">Engine's Final Recommendation</h3>
            <PolicyCard
              policy={`${comparison.recommended.policy_name} (Selected)`}
              recommendation={comparison.recommended}
              isRecommended={true}
            />
          </div>

          {/* Metadata */}
          <div className="text-sm text-gray-600 border-t pt-4">
            <p>Computation time: {comparison.computation_time_ms.toFixed(2)}ms</p>
            <p>Timestamp: {new Date(comparison.timestamp).toLocaleString()}</p>
          </div>
        </div>
      )}

      {comparePoliciesMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 font-medium">Error comparing policies</p>
          <p className="text-red-600 text-sm mt-1">
            {comparePoliciesMutation.error?.message || 'An unexpected error occurred'}
          </p>
        </div>
      )}
    </div>
  );
}
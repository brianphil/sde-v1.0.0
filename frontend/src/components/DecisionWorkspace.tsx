import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  decisionsAPI,
  ordersAPI,
  routesAPI,
  vehiclesAPI,
} from "../services/api";
import {
  DecisionType,
  PolicyComparisonResponse,
  Route,
  OrderStatus,
} from "../types";
import {
  CheckCircle2,
  AlertTriangle,
  TrendingUp,
  ArrowRight,
  XCircle,
  Play,
} from "lucide-react";
import RouteCard from "./RouteCard";

type WorkflowStep =
  | "select"
  | "compare"
  | "review"
  | "execute"
  | "track"
  | "outcome";

export default function DecisionWorkspace() {
  const queryClient = useQueryClient();
  const [currentStep, setCurrentStep] = useState<WorkflowStep>("select");
  const [selectedOrders, setSelectedOrders] = useState<string[]>([]);
  const [comparison, setComparison] = useState<PolicyComparisonResponse | null>(
    null
  );
  const [executedRoutes, setExecutedRoutes] = useState<Route[]>([]);

  // Fetch pending orders
  const { data: orders } = useQuery({
    queryKey: ["orders"],
    queryFn: async () => {
      const response = await ordersAPI.list();
      return response.data;
    },
  });

  // Fetch vehicles
  const { data: vehicles } = useQuery({
    queryKey: ["vehicles"],
    queryFn: async () => {
      const response = await vehiclesAPI.list();
      return response.data;
    },
  });

  // Compare policies mutation
  const comparePoliciesMutation = useMutation({
    mutationFn: () =>
      decisionsAPI.comparePolicies(
        DecisionType.ORDER_ARRIVAL,
        `Manual routing for ${selectedOrders.length} orders`
      ),
    onSuccess: (response) => {
      setComparison(response.data);
      setCurrentStep("compare");
    },
  });

  // Make decision mutation
  const makeDecisionMutation = useMutation({
    mutationFn: () =>
      decisionsAPI.makeDecision(DecisionType.ORDER_ARRIVAL, selectedOrders),
    onSuccess: (response) => {
      // Decision created, now need to commit it
      if (response.data.decision_id) {
        commitDecisionMutation.mutate(response.data.decision_id);
      }
    },
  });

  // Commit decision mutation
  const commitDecisionMutation = useMutation({
    mutationFn: (decisionId: string) => decisionsAPI.commit(decisionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["routes"] });
      queryClient.invalidateQueries({ queryKey: ["orders"] });
      setCurrentStep("track");
      // Fetch created routes
      setTimeout(() => {
        routesAPI.active().then((res) => setExecutedRoutes(res.data || []));
      }, 1000);
    },
  });

  const pendingOrders = Array.isArray(orders)
    ? orders.filter((o) => o.status === OrderStatus.PENDING)
    : [];
  const selectedOrdersData = pendingOrders.filter((o) =>
    selectedOrders.includes(o.order_id)
  );

  const toggleOrder = (orderId: string) => {
    setSelectedOrders((prev) =>
      prev.includes(orderId)
        ? prev.filter((id) => id !== orderId)
        : [...prev, orderId]
    );
  };

  const steps = [
    {
      id: "select" as WorkflowStep,
      label: "1. Select Orders",
      done: selectedOrders.length > 0,
    },
    {
      id: "compare" as WorkflowStep,
      label: "2. SDE Recommendation",
      done: comparison !== null,
    },
    { id: "review" as WorkflowStep, label: "3. Review & Decide", done: false },
    {
      id: "execute" as WorkflowStep,
      label: "4. Execute Routes",
      done: executedRoutes.length > 0,
    },
    { id: "track" as WorkflowStep, label: "5. Track & Record", done: false },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">
          End-to-End Decision Workflow
        </h2>
        <p className="text-gray-600 mt-1">
          Complete decision support: Select orders → Get SDE recommendation →
          Execute → Track → Learn
        </p>
      </div>

      {/* Progress Steps */}
      <div className="bg-white border rounded-lg p-4">
        <div className="flex items-center justify-between">
          {steps.map((step, idx) => (
            <div key={step.id} className="flex items-center">
              <button
                onClick={() => step.done && setCurrentStep(step.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                  currentStep === step.id
                    ? "bg-primary-100 text-primary-700 font-medium"
                    : step.done
                    ? "text-green-600 hover:bg-green-50"
                    : "text-gray-400"
                }`}
              >
                {step.done ? (
                  <CheckCircle2 className="h-5 w-5" />
                ) : (
                  <div className="h-5 w-5 rounded-full border-2 border-current" />
                )}
                <span className="text-sm">{step.label}</span>
              </button>
              {idx < steps.length - 1 && (
                <ArrowRight className="h-4 w-4 text-gray-300 mx-2" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: Select Orders */}
      {currentStep === "select" && (
        <div className="bg-white border rounded-lg p-6">
          <h3 className="text-lg font-bold mb-4">
            Step 1: Select Orders to Route
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Choose which pending orders you want to create routes for. SDE will
            analyze all 4 policies for these specific orders.
          </p>

          <div className="space-y-2">
            {pendingOrders.length > 0 ? (
              pendingOrders.map((order) => (
                <div
                  key={order.order_id}
                  onClick={() => toggleOrder(order.order_id)}
                  className={`border rounded-lg p-4 cursor-pointer transition-all ${
                    selectedOrders.includes(order.order_id)
                      ? "border-primary-500 bg-primary-50"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <input
                        type="checkbox"
                        checked={selectedOrders.includes(order.order_id)}
                        onChange={() => {}}
                        className="mt-1"
                      />
                      <div>
                        <p className="font-medium">{order.order_id}</p>
                        <p className="text-sm text-gray-600">
                          {order.customer_name} → {order.destination_city}
                        </p>
                        <p className="text-sm text-gray-500">
                          {order.weight_tonnes}T, {order.volume_m3}m³ |
                          Priority: {order.priority}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{order.price_kes} KES</p>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>No pending orders available</p>
                <p className="text-sm mt-1">
                  Create orders first to use the decision workflow
                </p>
              </div>
            )}
          </div>

          {selectedOrders.length > 0 && (
            <div className="mt-6 flex items-center justify-between bg-gray-50 p-4 rounded-lg">
              <div>
                <p className="font-medium">
                  {selectedOrders.length} orders selected
                </p>
                <p className="text-sm text-gray-600">
                  Total:{" "}
                  {selectedOrdersData
                    .reduce((sum, o) => sum + o.weight_tonnes, 0)
                    .toFixed(1)}
                  T,{" "}
                  {selectedOrdersData
                    .reduce((sum, o) => sum + o.price_kes, 0)
                    .toFixed(0)}{" "}
                  KES
                </p>
              </div>
              <button
                onClick={() => comparePoliciesMutation.mutate()}
                disabled={comparePoliciesMutation.isPending}
                className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 font-medium flex items-center gap-2"
              >
                {comparePoliciesMutation.isPending ? (
                  "Analyzing..."
                ) : (
                  <>
                    Get AI Recommendation <TrendingUp className="h-4 w-4" />
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Step 2: Compare Policies */}
      {currentStep === "compare" && comparison && (
        <div className="space-y-6">
          {/* Agreement Analysis */}
          <div
            className={`border rounded-lg p-6 ${
              comparison.agreement_analysis.agreement_score >= 0.75
                ? "bg-green-50 border-green-200"
                : comparison.agreement_analysis.agreement_score >= 0.5
                ? "bg-yellow-50 border-yellow-200"
                : "bg-red-50 border-red-200"
            }`}
          >
            <h3 className="text-lg font-bold mb-4">
              SDE Policy Agreement Analysis
            </h3>
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-600">Agreement</p>
                <p className="text-3xl font-bold">
                  {(
                    comparison.agreement_analysis.agreement_score * 100
                  ).toFixed(0)}
                  %
                </p>
              </div>
              <div>
                <p className="text-gray-600">Consensus</p>
                <p className="font-bold">
                  {comparison.agreement_analysis.consensus_action.replace(
                    /_/g,
                    " "
                  )}
                </p>
              </div>
              <div>
                <p className="text-gray-600">Policies Agree</p>
                <p className="font-bold">
                  {comparison.agreement_analysis.consensus_count}/
                  {comparison.agreement_analysis.total_policies}
                </p>
              </div>
              <div>
                <p className="text-gray-600">Avg Confidence</p>
                <p className="font-bold">
                  {(comparison.agreement_analysis.avg_confidence * 100).toFixed(
                    0
                  )}
                  %
                </p>
              </div>
            </div>

            {comparison.agreement_analysis.conflicts.length > 0 && (
              <div className="mt-4 pt-4 border-t border-current/20">
                <p className="font-medium mb-2 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Conflicts: {
                    comparison.agreement_analysis.conflicts.length
                  }{" "}
                  policies disagree
                </p>
                {comparison.agreement_analysis.conflicts.map((c, idx) => (
                  <p key={idx} className="text-sm">
                    • {c.policy}: {c.action.replace(/_/g, " ")} (
                    {(c.confidence * 100).toFixed(0)}%)
                  </p>
                ))}
              </div>
            )}
          </div>

          {/* Individual Policy Recommendations */}
          <div className="grid grid-cols-2 gap-4">
            {[
              { name: "PFA (Pattern)", data: comparison.pfa },
              { name: "VFA (Value)", data: comparison.vfa },
              { name: "CFA (Cost)", data: comparison.cfa },
              { name: "DLA (Lookahead)", data: comparison.dla },
            ].map(({ name, data }) => (
              <div key={name} className="border rounded-lg p-4 bg-white">
                <h4 className="font-bold mb-2">{name}</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Action:</span>
                    <span className="font-medium">
                      {data.recommended_action.replace(/_/g, " ")}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Confidence:</span>
                    <span className="font-medium">
                      {(data.confidence_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Value:</span>
                    <span className="font-medium">
                      {data.expected_value.toFixed(0)} KES
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Routes:</span>
                    <span className="font-medium">{data.routes.length}</span>
                  </div>
                  <div className="mt-2 pt-2 border-t">
                    <p className="text-xs text-gray-600">{data.reasoning}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Engine Recommendation */}
          <div className="border-2 border-primary-500 rounded-lg p-6 bg-primary-50">
            <h3 className="text-lg font-bold mb-2 flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-primary-600" />
              Engine's Final Recommendation:{" "}
              {comparison.recommended.policy_name}
            </h3>
            <div className="grid grid-cols-4 gap-4 text-sm mb-4">
              <div>
                <p className="text-gray-600">Action</p>
                <p className="font-bold">
                  {comparison.recommended.recommended_action.replace(/_/g, " ")}
                </p>
              </div>
              <div>
                <p className="text-gray-600">Confidence</p>
                <p className="font-bold">
                  {(comparison.recommended.confidence_score * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-gray-600">Expected Value</p>
                <p className="font-bold">
                  {comparison.recommended.expected_value.toFixed(0)} KES
                </p>
              </div>
              <div>
                <p className="text-gray-600">Routes</p>
                <p className="font-bold">
                  {comparison.recommended.routes.length}
                </p>
              </div>
            </div>
            <p className="text-sm text-gray-700">
              {comparison.recommended.reasoning}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={() => setCurrentStep("select")}
              className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              ← Back to Selection
            </button>
            <button
              onClick={() => makeDecisionMutation.mutate()}
              disabled={
                makeDecisionMutation.isPending ||
                commitDecisionMutation.isPending
              }
              className="flex-1 bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 font-medium flex items-center justify-center gap-2"
            >
              {makeDecisionMutation.isPending ||
              commitDecisionMutation.isPending ? (
                "Executing Decision..."
              ) : (
                <>
                  Execute Decision & Create Routes <Play className="h-4 w-4" />
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Track Execution */}
      {currentStep === "track" && (
        <div className="bg-white border rounded-lg p-6">
          <h3 className="text-lg font-bold mb-4">Routes Created & Tracking</h3>
          <p className="text-sm text-gray-600 mb-4">
            Routes have been created. In a production system, you would track
            their progress here and record outcomes when complete.
          </p>

          {executedRoutes.length > 0 ? (
            <div className="space-y-4">
              {executedRoutes.map((route) => {
                const vehicle = Array.isArray(vehicles)
                  ? vehicles.find((v) => v.vehicle_id === route.vehicle_id)
                  : undefined;
                return (
                  <RouteCard
                    key={route.route_id}
                    route={route}
                    orders={orders || []}
                    vehicle={vehicle}
                    showConsolidation={true}
                    detailed={true}
                  />
                );
              })}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-500">Loading created routes...</p>
            </div>
          )}

          <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-800">
              <strong>Next Steps (Future Enhancement):</strong>
              <br />
              • Mark routes as in-progress when execution starts
              <br />
              • Record actual fuel costs, time, delays
              <br />
              • Submit outcomes to trigger learning updates
              <br />• See how VFA, CFA, PFA, DLA improve based on feedback
            </p>
          </div>

          <button
            onClick={() => {
              setCurrentStep("select");
              setSelectedOrders([]);
              setComparison(null);
              setExecutedRoutes([]);
            }}
            className="mt-4 w-full bg-primary-600 text-white py-2 px-4 rounded-lg hover:bg-primary-700 font-medium"
          >
            Start New Decision Workflow
          </button>
        </div>
      )}

      {/* Errors */}
      {(comparePoliciesMutation.isError ||
        makeDecisionMutation.isError ||
        commitDecisionMutation.isError) && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 font-medium flex items-center gap-2">
            <XCircle className="h-5 w-5" />
            Error in workflow
          </p>
          <p className="text-red-600 text-sm mt-1">
            {comparePoliciesMutation.error?.message ||
              makeDecisionMutation.error?.message ||
              commitDecisionMutation.error?.message ||
              "An unexpected error occurred"}
          </p>
        </div>
      )}
    </div>
  );
}

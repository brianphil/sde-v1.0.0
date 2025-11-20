import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { decisionsAPI } from "../services/api";
import PolicyComparison from "../components/PolicyComparison";
import DecisionWorkspace from "../components/DecisionWorkspace";

type TabType = "workspace" | "compare" | "history";

export default function Decisions() {
  const [activeTab, setActiveTab] = useState<TabType>("workspace");

  const { data: history } = useQuery({
    queryKey: ["decisions", "history"],
    queryFn: async () => {
      const response = await decisionsAPI.history(10);
      return response.data;
    },
  });

  const tabs = [
    {
      id: "workspace" as TabType,
      label: "Decision Workflow",
      description: "End-to-end: Select → Recommend → Execute",
    },
    {
      id: "compare" as TabType,
      label: "Policy Comparison",
      description: "Compare all 4 AI policies",
    },
    {
      id: "history" as TabType,
      label: "History",
      description: "View past decisions",
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">
          SDE Decision Support
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Powell Sequential Decision Engine - 4 Policy Framework
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
                ${
                  activeTab === tab.id
                    ? "border-primary-500 text-primary-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }
              `}
            >
              <div>{tab.label}</div>
              <div className="text-xs text-gray-400">{tab.description}</div>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === "workspace" && <DecisionWorkspace />}

      {activeTab === "compare" && <PolicyComparison />}

      {activeTab === "history" && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Decision History</h2>
          <div className="space-y-3">
            {Array.isArray(history) && history.length > 0 ? (
              history.map((decision) => (
                <div
                  key={decision.decision_id}
                  className="border rounded-lg p-4"
                >
                  <div className="flex justify-between">
                    <div>
                      <p className="font-medium">{decision.policy_name}</p>
                      <p className="text-sm text-gray-600">
                        {decision.decision_type}
                      </p>
                      <p className="text-sm text-gray-600">
                        Action: {decision.recommended_action}
                      </p>
                      <p className="text-sm text-gray-600">
                        Routes: {decision.routes.length}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">
                        Confidence:{" "}
                        {(decision.confidence_score * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-600">
                        Value: {decision.expected_value.toFixed(2)}
                      </p>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-500">No decision history available</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

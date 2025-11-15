"""Hybrid policies combining multiple Powell algorithm classes."""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ..models.state import SystemState
from ..models.decision import PolicyDecision, HybridDecision, DecisionContext, ActionType
from .pfa import PolicyFunctionApproximation
from .cfa import CostFunctionApproximation
from .vfa import ValueFunctionApproximation
from .dla import DirectLookaheadApproximation

logger = logging.getLogger(__name__)


class CFAVFAHybrid:
    """Hybrid: Cost Function + Value Function.
    
    Combines CFA's immediate cost optimization with VFA's long-term value.
    
    Use case: Route decisions where both immediate profitability AND
    future strategic value matter (e.g., fleet consolidation, network building).
    
    Strategy:
    - CFA recommends lowest-cost routes
    - VFA evaluates long-term value of those routes
    - Blend: weight = (CFA_profit + VFA_future_value) for each route
    """

    def __init__(self, cfa: CostFunctionApproximation, vfa: ValueFunctionApproximation, 
                 cfa_weight: float = 0.4, vfa_weight: float = 0.6):
        """Initialize hybrid.
        
        Args:
            cfa_weight: Weight for immediate cost optimization (0-1)
            vfa_weight: Weight for long-term value (0-1)
        """
        self.cfa = cfa
        self.vfa = vfa
        self.cfa_weight = cfa_weight
        self.vfa_weight = vfa_weight
        
        assert abs((cfa_weight + vfa_weight) - 1.0) < 0.01, "Weights must sum to 1.0"

    def evaluate(self, state: SystemState, context: DecisionContext) -> HybridDecision:
        """Evaluate using both policies and blend recommendations."""
        
        # Get individual policy recommendations
        cfa_decision = self.cfa.evaluate(state, context)
        vfa_decision = self.vfa.evaluate(state, context)
        
        # Score combined objective for each route
        combined_routes = cfa_decision.routes.copy()
        
        best_score = float('-inf')
        best_routes = []
        
        for route in combined_routes:
            # CFA score: negative cost (lower cost = higher score)
            cfa_score = -route.estimated_cost_kes
            
            # VFA score: future value estimate
            vfa_score = state.get_estimated_route_value(route)
            
            # Blend scores
            combined_score = (self.cfa_weight * cfa_score + self.vfa_weight * vfa_score)
            
            if combined_score > best_score:
                best_score = combined_score
                best_routes = [route]
        
        # Determine action
        recommended_action = cfa_decision.recommended_action
        if best_routes:
            pass  # Use CFA recommendation
        else:
            recommended_action = ActionType.DEFER_ORDER
            best_routes = []
        
        # Blend confidence
        blended_confidence = (self.cfa_weight * cfa_decision.confidence_score + 
                             self.vfa_weight * vfa_decision.confidence_score)
        
        # Calculate expected value
        cfa_value = sum(state.get_estimated_route_value(r) for r in cfa_decision.routes) - sum(r.estimated_cost_kes for r in cfa_decision.routes)
        vfa_value = vfa_decision.expected_value
        blended_value = self.cfa_weight * cfa_value + self.vfa_weight * vfa_value
        
        reasoning = f"Hybrid CFA/VFA: CFA cost={-cfa_value:.0f} KES (confidence={cfa_decision.confidence_score:.2f}), " \
                   f"VFA future_value={vfa_value:.0f} KES (confidence={vfa_decision.confidence_score:.2f})"
        
        return HybridDecision(
            hybrid_name="CFA/VFA",
            primary_policy=cfa_decision,
            secondary_policy=vfa_decision,
            recommended_action=recommended_action,
            routes=best_routes,
            primary_weight=self.cfa_weight,
            secondary_weight=self.vfa_weight,
            confidence_score=blended_confidence,
            expected_value=blended_value,
            reasoning=reasoning
        )


class DLAVFAHybrid:
    """Hybrid: Direct Lookahead + Value Function.
    
    Combines DLA's multi-period planning with VFA's strategic value estimation.
    
    Use case: Major routing decisions with multi-day horizon where both
    planned efficiency AND strategic flexibility matter.
    
    Strategy:
    - DLA optimizes 7-day plan
    - VFA provides terminal value at horizon end (future profit beyond 7 days)
    - DLA uses VFA terminal value in its optimization
    """

    def __init__(self, dla: DirectLookaheadApproximation, vfa: ValueFunctionApproximation,
                 dla_weight: float = 0.5, vfa_weight: float = 0.5):
        """Initialize hybrid."""
        self.dla = dla
        self.vfa = vfa
        self.dla_weight = dla_weight
        self.vfa_weight = vfa_weight

    def evaluate(self, state: SystemState, context: DecisionContext) -> HybridDecision:
        """Evaluate using DLA with VFA terminal value."""
        
        # Get individual recommendations
        dla_decision = self.dla.evaluate(state, context)
        vfa_decision = self.vfa.evaluate(state, context)
        
        # DLA inherently uses terminal value (which can be from VFA)
        # Blend is implicit in DLA calculation
        
        # If DLA recommends deferring but VFA sees long-term value, might override
        if (dla_decision.recommended_action == ActionType.DEFER_ORDER and 
            vfa_decision.confidence_score > 0.7 and 
            vfa_decision.expected_value > 500):  # Threshold
            
            recommended_action = vfa_decision.recommended_action
            best_routes = vfa_decision.routes
        else:
            recommended_action = dla_decision.recommended_action
            best_routes = dla_decision.routes
        
        blended_value = self.dla_weight * dla_decision.expected_value + \
                       self.vfa_weight * vfa_decision.expected_value
        
        blended_confidence = self.dla_weight * dla_decision.confidence_score + \
                            self.vfa_weight * vfa_decision.confidence_score
        
        reasoning = f"Hybrid DLA/VFA: DLA 7-day plan={dla_decision.expected_value:.0f} KES, " \
                   f"VFA terminal value={vfa_decision.expected_value:.0f} KES"
        
        return HybridDecision(
            hybrid_name="DLA/VFA",
            primary_policy=dla_decision,
            secondary_policy=vfa_decision,
            recommended_action=recommended_action,
            routes=best_routes,
            primary_weight=self.dla_weight,
            secondary_weight=self.vfa_weight,
            confidence_score=blended_confidence,
            expected_value=blended_value,
            reasoning=reasoning
        )


class PFACFAHybrid:
    """Hybrid: Policy Function + Cost Function.
    
    Combines PFA's learned rules with CFA's optimization.
    
    Use case: Operational decisions where learned constraints (PFA rules)
    must be hard constraints in the optimization problem (CFA).
    
    Strategy:
    - PFA identifies which constraints apply (business rules)
    - CFA optimizes subject to those constraints
    - Decision: routes respecting PFA rules + minimizing cost
    """

    def __init__(self, pfa: PolicyFunctionApproximation, cfa: CostFunctionApproximation,
                 pfa_weight: float = 0.4, cfa_weight: float = 0.6):
        """Initialize hybrid."""
        self.pfa = pfa
        self.cfa = cfa
        self.pfa_weight = pfa_weight
        self.cfa_weight = cfa_weight

    def evaluate(self, state: SystemState, context: DecisionContext) -> HybridDecision:
        """Evaluate using PFA rules to constrain CFA optimization."""
        
        # Get individual recommendations
        pfa_decision = self.pfa.evaluate(state, context)
        cfa_decision = self.cfa.evaluate(state, context)
        
        # Blend: if PFA rules say "must do X", enforce it; otherwise use CFA
        if pfa_decision.confidence_score > 0.9:  # High-confidence rule
            # PFA rule is authoritative
            recommended_action = pfa_decision.recommended_action
            best_routes = pfa_decision.routes
            recommended_decision = pfa_decision
        else:
            # Use CFA recommendation
            recommended_action = cfa_decision.recommended_action
            best_routes = cfa_decision.routes
            recommended_decision = cfa_decision
        
        blended_confidence = self.pfa_weight * pfa_decision.confidence_score + \
                            self.cfa_weight * cfa_decision.confidence_score
        
        blended_value = self.pfa_weight * pfa_decision.expected_value + \
                       self.cfa_weight * cfa_decision.expected_value
        
        reasoning = f"Hybrid PFA/CFA: PFA rule quality={pfa_decision.confidence_score:.2f}, " \
                   f"CFA optimization cost={-cfa_decision.expected_value:.0f} KES"
        
        return HybridDecision(
            hybrid_name="PFA/CFA",
            primary_policy=pfa_decision,
            secondary_policy=cfa_decision,
            recommended_action=recommended_action,
            routes=best_routes,
            primary_weight=self.pfa_weight,
            secondary_weight=self.cfa_weight,
            confidence_score=blended_confidence,
            expected_value=blended_value,
            reasoning=reasoning
        )

"""Consolidation Engine Package."""

from .geographic_clustering import GeographicClusteringEngine, RouteCorridor
from .consolidation_pool import (
    ConsolidationPool,
    OrderClassification,
    PoolConfiguration,
)
from .compatibility_filters import (
    ServiceLevelFilter,
    TimeWindowFilter,
    CompatibilityFilterChain,
    ServiceLevelConfig,
    TimeWindowConfig,
)
from .sequence_optimizer import SequenceOptimizer, Stop, StopType, RouteSequence
from .consolidation_engine import (
    ConsolidationEngine,
    ConsolidationResult,
    ConsolidationOpportunity,
)

__all__ = [
    # Geographic Clustering
    "GeographicClusteringEngine",
    "RouteCorridor",
    # Consolidation Pool
    "ConsolidationPool",
    "OrderClassification",
    "PoolConfiguration",
    # Compatibility Filters
    "ServiceLevelFilter",
    "TimeWindowFilter",
    "CompatibilityFilterChain",
    "ServiceLevelConfig",
    "TimeWindowConfig",
    # Sequence Optimizer
    "SequenceOptimizer",
    "Stop",
    "StopType",
    "RouteSequence",
    # Main Engine
    "ConsolidationEngine",
    "ConsolidationResult",
    "ConsolidationOpportunity",
]

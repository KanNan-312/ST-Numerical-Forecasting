"""Graph neural network baselines."""
from .gnn_forecaster import (  # noqa: F401  (registers all GNN models)
    GCNTCNForecaster,
    GraphWaveNetForecaster,
    STGCNForecaster,
    STSGCNForecaster,
    STSGCNet,
    STLLMPlusForecaster,
)

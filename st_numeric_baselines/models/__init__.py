from .base import BaseForecaster
from .registry import get, available, register

# register built-ins
from .naive.ar_univariate import ARUnivariateForecaster  

# statistical + ML baselines
from .ml.rf import RandomForestPCAForecaster  
from .ml.xgb import XGBPCAForecaster 

# deep learning baselines
from .dl.rnn import RNNForecaster
from .dl.lstm import LSTMForecaster
from .dl.dlinear import DLinearForecaster
from .dl.timemixer import TimeMixerForecaster
from .dl.patchtst import PatchTSTForecaster
from .dl.informer import InformerForecaster
from .dl.autoformer import AutoformerForecaster
from .dl.fedformer import FEDformerForecaster
from .dl.itransformer import iTransformerForecaster
from .dl.gpt4ts import GPT4TSForecaster
from .dl.timellm import TimeLLMForecaster

# graph neural network baselines
from .gnn.gnn_forecaster import (  # noqa: F401
    GCNTCNForecaster,
    GraphWaveNetForecaster,
    STGCNForecaster,
    STSGCNForecaster,
    STLLMPlusForecaster,
)

# graph neural network baselines — traffic
from .gnn.dcrnn import DCRNNForecaster  # noqa: F401
from .gnn.stgformer import STGformerForecaster  # noqa: F401
from .gnn.d2stgnn import D2STGNNForecaster  # noqa: F401
from .gnn.staeformer import STAEformerForecaster  # noqa: F401

# graph neural network baselines — crime (simplified adaptations, see module docstrings)
from .gnn.aist import AISTForecaster  # noqa: F401
from .gnn.st_hhol import STHHOLForecaster  # noqa: F401

# graph neural network baselines — explainability / causal
from .gnn.cast import CaSTForecaster  # noqa: F401
from .gnn.stexplainer import STExplainerForecaster  # noqa: F401

# graph neural network baselines — no external adjacency needed at all
# (requires_graph = False; run on datasets with no graph.npz/dataset.graph.path)
from .gnn.stid import STIDForecaster  # noqa: F401
from .gnn.agcrn import AGCRNForecaster  # noqa: F401
from .gnn.mtgnn import MTGNNForecaster  # noqa: F401
from .gnn.testam import TESTAMForecaster  # noqa: F401

# fixed-weight ensemble over a configurable list of this registry's own models
from .ensemble_st import EnsembleSTForecaster  # noqa: F401

# frozen pretrained experts + trained graph-conditioned router
from .gc_moe import GCMoEForecaster  # noqa: F401

# optional foundation-model wrappers
from .foundation.timesfm import TimesFMZeroForecaster, TimesFMCalibratedForecaster, TimesFMFullFineTuneForecaster  
from .foundation.chronos import ChronosZeroForecaster, ChronosCalibratedForecaster, ChronosFullFineTuneForecaster 

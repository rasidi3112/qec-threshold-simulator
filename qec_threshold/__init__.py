"""Quantum Error Correction Threshold Simulator."""

__version__ = "1.0.0"

from .core.pauli import PauliOperator, QubitError
from .core.surface_code import SurfaceCode
from .decoders.mwpm import MWPMDecoder, UnionFind
from .noise_models.depolarizing import DepolarizingChannel
from .noise_models.biased import BiasedNoiseChannel
from .noise_models.correlated import CorrelatedNoiseChannel
from .analysis.threshold_estimator import ThresholdEstimator, ThresholdResult
from .analysis.finite_size_scaling import FiniteSizeScalingAnalyzer, FiniteSizeScalingResult
from .visualization.threshold_plots import ThresholdVisualizer
from .visualization.advanced_plots import AdvancedVisualizer

__all__ = [
    "PauliOperator",
    "QubitError", 
    "SurfaceCode",
    "MWPMDecoder",
    "UnionFind",
    "DepolarizingChannel",
    "BiasedNoiseChannel",
    "CorrelatedNoiseChannel",
    "ThresholdEstimator",
    "ThresholdResult",
    "FiniteSizeScalingAnalyzer",
    "FiniteSizeScalingResult",
    "ThresholdVisualizer",
    "AdvancedVisualizer",
]

"""Surprise-modulated predictive plasticity toy package."""

from .config import ExperimentConfig, ModelConfig
from .data import generate_switching_sequence, make_features
from .model import PredictiveCircuit, StepStats

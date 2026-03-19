from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    input_dim: int = 6
    lr_out: float = 0.03
    lr_inh: float = 0.012
    lr_mem: float = 0.012
    lr_alpha: float = 0.01
    lr_beta: float = 0.01
    gate_strength: float = 2.5
    gate_baseline: float = 0.65
    gate_cap: float = 2.0
    novelty_margin: float = 0.015
    err_fast_decay: float = 0.85
    err_slow_decay: float = 0.985
    use_gate: bool = True
    seed: int = 0


@dataclass
class ExperimentConfig:
    T: int = 1600
    k: int = 6
    switch_points: Tuple[int, ...] = (350, 700, 1050, 1350)
    noise: float = 0.08
    seed: int = 0
    post_switch_window: int = 50
    out_prefix: str = "outputs/surprise_toy"
    model_plain: ModelConfig = field(default_factory=lambda: ModelConfig(use_gate=False))
    model_gated: ModelConfig = field(default_factory=lambda: ModelConfig(use_gate=True))

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def tanh(z: np.ndarray | float) -> np.ndarray | float:
    return np.tanh(np.clip(z, -10.0, 10.0))


@dataclass
class StepStats:
    p: float
    target: float
    loss: float
    err: float
    exc_drive: float
    inh_drive: float
    mem_state: float
    alpha: float
    beta: float
    gate: float
    novelty: float
    err_fast: float
    err_slow: float
    ei_imbalance: float


class PredictiveCircuit:
    """
    Small predictive circuit with explicit excitatory, inhibitory and memory pathways.

    The gate is based on a short-vs-long timescale mismatch in absolute error.
    This acts like a crude change detector and is less sensitive to one-step noise
    than a raw absolute-error gate.
    """

    def __init__(
        self,
        input_dim: int,
        lr_out: float = 0.03,
        lr_inh: float = 0.012,
        lr_mem: float = 0.012,
        lr_alpha: float = 0.01,
        lr_beta: float = 0.01,
        gate_strength: float = 2.5,
        gate_baseline: float = 0.65,
        gate_cap: float = 2.0,
        novelty_margin: float = 0.015,
        err_fast_decay: float = 0.85,
        err_slow_decay: float = 0.985,
        use_gate: bool = True,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)

        self.w_exc = np.abs(0.15 * rng.standard_normal(self.input_dim))
        self.w_inh = np.abs(0.15 * rng.standard_normal(self.input_dim))
        self.w_mem = 0.15 * rng.standard_normal(self.input_dim)

        self.b = 0.0
        self.b_inh = 0.0
        self.b_mem = 0.0

        self.beta = 0.50
        self.kappa = 0.45
        self.alpha = 0.15
        self.a_mem = 0.20

        self.y_prev = 0.0
        self.mem_prev = 0.0
        self.err_fast = 0.25
        self.err_slow = 0.25

        self.lr_out = float(lr_out)
        self.lr_inh = float(lr_inh)
        self.lr_mem = float(lr_mem)
        self.lr_alpha = float(lr_alpha)
        self.lr_beta = float(lr_beta)

        self.gate_strength = float(gate_strength)
        self.gate_baseline = float(gate_baseline)
        self.gate_cap = float(gate_cap)
        self.novelty_margin = float(novelty_margin)
        self.err_fast_decay = float(err_fast_decay)
        self.err_slow_decay = float(err_slow_decay)
        self.use_gate = bool(use_gate)

    def forward(self, x: np.ndarray) -> Dict[str, float]:
        h_inh = float(sigmoid(np.dot(self.w_inh, x) + self.b_inh))
        mem = float(tanh(np.dot(self.w_mem, x) + self.a_mem * self.mem_prev + self.b_mem))
        exc_drive = float(np.dot(self.w_exc, x))
        inh_drive = float(self.beta * h_inh)
        v = exc_drive + self.kappa * mem - inh_drive + self.alpha * self.y_prev + self.b
        p = float(sigmoid(v))
        return {
            "h_inh": h_inh,
            "mem": mem,
            "exc_drive": exc_drive,
            "inh_drive": inh_drive,
            "v": float(v),
            "p": p,
        }

    def compute_gate(self, abs_err: float) -> tuple[float, float]:
        err_fast_next = self.err_fast_decay * self.err_fast + (1.0 - self.err_fast_decay) * abs_err
        err_slow_next = self.err_slow_decay * self.err_slow + (1.0 - self.err_slow_decay) * abs_err
        novelty = max(0.0, err_fast_next - err_slow_next - self.novelty_margin) / (err_slow_next + 1e-8)
        if self.use_gate:
            gate = self.gate_baseline + self.gate_strength * min(self.gate_cap, novelty)
        else:
            gate = 1.0
        return float(gate), float(novelty)

    def step(self, x: np.ndarray, target: float) -> StepStats:
        out = self.forward(x)
        p = out["p"]
        err = float(target - p)
        abs_err = abs(err)

        gate, novelty = self.compute_gate(abs_err)

        # Excitatory path.
        self.w_exc += self.lr_out * gate * err * x
        self.w_exc = np.clip(self.w_exc, 0.0, None)

        # Inhibitory path.
        h_inh = out["h_inh"]
        dh_inh = h_inh * (1.0 - h_inh)
        self.beta += self.lr_beta * gate * err * (-h_inh)
        self.beta = float(max(0.0, self.beta))
        self.w_inh += self.lr_inh * gate * err * (-self.beta * dh_inh * x)
        self.w_inh = np.clip(self.w_inh, 0.0, None)
        self.b_inh += self.lr_inh * gate * err * (-self.beta * dh_inh)

        # Memory path.
        mem = out["mem"]
        dmem = 1.0 - mem * mem
        self.w_mem += self.lr_mem * gate * err * (self.kappa * dmem * x)
        self.a_mem += self.lr_mem * gate * err * (self.kappa * dmem * self.mem_prev)
        self.b_mem += self.lr_mem * gate * err * (self.kappa * dmem)
        self.a_mem = float(np.clip(self.a_mem, -2.0, 2.0))

        # Autoregressive self-term alpha: yes, it is learned.
        self.alpha += self.lr_alpha * gate * err * self.y_prev
        self.alpha = float(np.clip(self.alpha, -2.0, 2.0))

        self.b += self.lr_out * gate * err

        # Update internal state after learning.
        self.y_prev = float(target)
        self.mem_prev = float(mem)
        self.err_fast = self.err_fast_decay * self.err_fast + (1.0 - self.err_fast_decay) * abs_err
        self.err_slow = self.err_slow_decay * self.err_slow + (1.0 - self.err_slow_decay) * abs_err

        ei_imbalance = abs(out["exc_drive"] - out["inh_drive"]) / (out["exc_drive"] + out["inh_drive"] + 1e-8)
        loss = -(target * np.log(p + 1e-8) + (1.0 - target) * np.log(1.0 - p + 1e-8))

        return StepStats(
            p=p,
            target=float(target),
            loss=float(loss),
            err=err,
            exc_drive=float(out["exc_drive"]),
            inh_drive=float(out["inh_drive"]),
            mem_state=float(mem),
            alpha=float(self.alpha),
            beta=float(self.beta),
            gate=float(gate),
            novelty=float(novelty),
            err_fast=float(self.err_fast),
            err_slow=float(self.err_slow),
            ei_imbalance=float(ei_imbalance),
        )

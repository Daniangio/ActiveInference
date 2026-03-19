from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from .config import ExperimentConfig, ModelConfig
from .data import aligned_switch_indices, generate_switching_sequence, make_features
from .model import PredictiveCircuit, StepStats
from .plotting import plot_experiment_results


def moving_average(x: np.ndarray, w: int = 31) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")


def _stack_history(history: List[StepStats]) -> Dict[str, np.ndarray]:
    return {
        "loss": np.array([h.loss for h in history], dtype=float),
        "gate": np.array([h.gate for h in history], dtype=float),
        "novelty": np.array([h.novelty for h in history], dtype=float),
        "err": np.array([h.err for h in history], dtype=float),
        "p": np.array([h.p for h in history], dtype=float),
        "target": np.array([h.target for h in history], dtype=float),
        "exc": np.array([h.exc_drive for h in history], dtype=float),
        "inh": np.array([h.inh_drive for h in history], dtype=float),
        "mem": np.array([h.mem_state for h in history], dtype=float),
        "alpha": np.array([h.alpha for h in history], dtype=float),
        "beta": np.array([h.beta for h in history], dtype=float),
        "err_fast": np.array([h.err_fast for h in history], dtype=float),
        "err_slow": np.array([h.err_slow for h in history], dtype=float),
        "ei_imbalance": np.array([h.ei_imbalance for h in history], dtype=float),
    }


def evaluate_post_switch(losses: np.ndarray, switch_idx: List[int], window: int = 50) -> Dict[str, float | list[float]]:
    vals = []
    for s in switch_idx:
        end = min(len(losses), s + window)
        if end > s:
            vals.append(float(np.mean(losses[s:end])))
    return {
        "post_switch_mean": float(np.mean(vals)) if vals else float("nan"),
        "post_switch_std": float(np.std(vals)) if vals else float("nan"),
        "windows": vals,
    }


def run_experiment(config: ExperimentConfig) -> Dict:
    seq, raw_switches, regime = generate_switching_sequence(
        T=config.T,
        switch_points=config.switch_points,
        noise=config.noise,
        seed=config.seed,
    )
    X, Y = make_features(seq, k=config.k)
    switches = aligned_switch_indices(raw_switches, k=config.k)

    mp = asdict(config.model_plain)
    mg = asdict(config.model_gated)
    mp["input_dim"] = config.k
    mg["input_dim"] = config.k

    plain = PredictiveCircuit(**mp)
    gated = PredictiveCircuit(**mg)

    hist_plain: List[StepStats] = []
    hist_gated: List[StepStats] = []

    for x, y in zip(X, Y):
        hist_plain.append(plain.step(x, y))
        hist_gated.append(gated.step(x, y))

    plain_d = _stack_history(hist_plain)
    gated_d = _stack_history(hist_gated)
    metrics_plain = evaluate_post_switch(plain_d["loss"], switches, window=config.post_switch_window)
    metrics_gated = evaluate_post_switch(gated_d["loss"], switches, window=config.post_switch_window)

    return {
        "config": config,
        "seq": seq,
        "X": X,
        "Y": Y,
        "raw_switches": raw_switches,
        "switches": switches,
        "regime": regime[config.k + 1 :],
        "plain": plain_d,
        "gated": gated_d,
        "metrics_plain": metrics_plain,
        "metrics_gated": metrics_gated,
    }


def main_experiment_cli() -> None:
    parser = argparse.ArgumentParser(description="Run the surprise-modulated predictive plasticity toy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", type=str, default="outputs/surprise_toy")
    parser.add_argument("--gate-strength", type=float, default=2.5)
    parser.add_argument("--gate-baseline", type=float, default=0.65)
    parser.add_argument("--gate-cap", type=float, default=2.0)
    parser.add_argument("--novelty-margin", type=float, default=0.015)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed,
        noise=args.noise,
        out_prefix=args.out_prefix,
        post_switch_window=args.window,
        model_plain=ModelConfig(use_gate=False, seed=args.seed),
        model_gated=ModelConfig(
            use_gate=True,
            seed=args.seed,
            gate_strength=args.gate_strength,
            gate_baseline=args.gate_baseline,
            gate_cap=args.gate_cap,
            novelty_margin=args.novelty_margin,
        ),
    )

    results = run_experiment(cfg)
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    out_paths = plot_experiment_results(results, out_prefix=args.out_prefix)

    print("Post-switch mean loss (plain):", results["metrics_plain"]["post_switch_mean"])
    print("Post-switch mean loss (gated):", results["metrics_gated"]["post_switch_mean"])
    print("Per-switch windows plain:", [round(v, 4) for v in results["metrics_plain"]["windows"]])
    print("Per-switch windows gated:", [round(v, 4) for v in results["metrics_gated"]["windows"]])
    for p in out_paths:
        print("Saved:", p)

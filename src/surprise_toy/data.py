from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def generate_switching_sequence(
    T: int = 1600,
    switch_points: Sequence[int] = (350, 700, 1050, 1350),
    noise: float = 0.08,
    seed: int = 0,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Binary sequence with hidden rule switches.

    Rule 0: alternation tendency
    Rule 1: repetition tendency
    """
    rng = np.random.default_rng(seed)
    seq = np.zeros(T, dtype=float)
    seq[0] = rng.integers(0, 2)
    regime = np.zeros(T, dtype=int)

    current_rule = 0
    switch_set = set(int(s) for s in switch_points)
    for t in range(1, T):
        if t in switch_set:
            current_rule = 1 - current_rule
        regime[t] = current_rule

        prev = seq[t - 1]
        target = (1.0 - prev) if current_rule == 0 else prev
        if rng.random() < noise:
            target = 1.0 - target
        seq[t] = target

    return seq, list(switch_points), regime


def make_features(seq: np.ndarray, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Use the last k bits to predict the next bit."""
    X, Y = [], []
    for t in range(k, len(seq) - 1):
        X.append(seq[t - k : t])
        Y.append(seq[t + 1])
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)


def aligned_switch_indices(raw_switches: Sequence[int], k: int) -> List[int]:
    return [int(s - k) for s in raw_switches if (s - k) >= 0]

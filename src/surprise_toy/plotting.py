from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def moving_average(x: np.ndarray, w: int = 31) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")


def plot_experiment_results(results: Dict, out_prefix: str = "outputs/surprise_toy") -> Tuple[str, str]:
    switches = results["switches"]

    fig1 = plt.figure(figsize=(11, 4.2))
    plt.plot(moving_average(results["plain"]["loss"], 35), label="plain predictive learning")
    plt.plot(moving_average(results["gated"]["loss"], 35), label="predictive learning + surprise gate")
    for s in switches:
        plt.axvline(s, linestyle="--", alpha=0.5)
    plt.title("Adaptation after rule switches")
    plt.xlabel("time")
    plt.ylabel("binary cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    path1 = f"{out_prefix}_loss.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(11, 8.0))
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(moving_average(results["gated"]["gate"], 25), label="gate")
    ax1.plot(moving_average(results["gated"]["novelty"], 25), label="novelty", alpha=0.85)
    for s in switches:
        ax1.axvline(s, linestyle="--", alpha=0.5)
    ax1.set_ylabel("gate")
    ax1.legend(loc="upper right")
    ax1.set_title("Internal variables of the gated model")

    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(moving_average(results["gated"]["exc"], 25), label="exc drive")
    ax2.plot(moving_average(results["gated"]["inh"], 25), label="inh drive")
    for s in switches:
        ax2.axvline(s, linestyle="--", alpha=0.5)
    ax2.set_ylabel("drive")
    ax2.legend(loc="upper right")

    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(moving_average(results["gated"]["alpha"], 25), label="alpha")
    ax3.plot(moving_average(results["gated"]["beta"], 25), label="beta")
    ax3.plot(moving_average(results["gated"]["mem"], 25), label="memory")
    for s in switches:
        ax3.axvline(s, linestyle="--", alpha=0.5)
    ax3.set_ylabel("state")
    ax3.legend(loc="upper right")

    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(moving_average(results["gated"]["err_fast"], 25), label="fast |err|")
    ax4.plot(moving_average(results["gated"]["err_slow"], 25), label="slow |err|")
    ax4.plot(moving_average(results["gated"]["ei_imbalance"], 25), label="E/I imbalance")
    for s in switches:
        ax4.axvline(s, linestyle="--", alpha=0.5)
    ax4.set_ylabel("timescales")
    ax4.set_xlabel("time")
    ax4.legend(loc="upper right")

    plt.tight_layout()
    path2 = f"{out_prefix}_internals.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    return path1, path2

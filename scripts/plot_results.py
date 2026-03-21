"""
Plot the training curve from a history.json file produced by run_experiment.py.

Usage
-----
    python scripts/plot_results.py outputs/olmoe_router_reward/history.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(history_path: str) -> None:
    with open(history_path) as f:
        history = json.load(f)

    baseline = history["baseline_fitness"]
    gens = history["generations"]

    x          = [g["generation"] for g in gens]
    mean_reward = [g["mean_fitness"] for g in gens]
    max_reward  = [g["max_fitness"]  for g in gens]
    best_ever   = [g["best_ever"]    for g in gens]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1.2, label=f"Baseline ({baseline:.3f})")
    ax.plot(x, mean_reward, label="Population mean reward", linewidth=1.5, alpha=0.8)
    ax.plot(x, max_reward,  label="Generation best reward", linewidth=1.0, alpha=0.6, linestyle=":")
    ax.plot(x, best_ever,   label="Best ever reward",       linewidth=2.0, color="tab:red")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean reward (GRM-3B)")
    ax.set_title("Evolutionary Router Search — Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(history_path).parent / "training_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_results.py <path/to/history.json>")
        sys.exit(1)
    plot(sys.argv[1])

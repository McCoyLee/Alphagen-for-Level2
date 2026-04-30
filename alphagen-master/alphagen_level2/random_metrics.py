"""
Per-rollout metrics logger + matplotlib plot for the random-window
training run.

Records six time-series at each PPO rollout end:
  * pool_best_reward / pool_avg_reward     (composite, our main objective)
  * pool_best_sortino / pool_avg_sortino   (risk-adjusted future-100-bar PnL)
  * pool_best_abs_ic / pool_avg_abs_ic     (rank IC of z vs forward return)
  * pool_size, pool_eval_cnt
  * gp_total_accepted (cumulative offspring promoted by GP)
  * recent_reward_mean (mean reward of the last K candidates seen)

CSV is appended on every rollout so a long run can still be plotted partway.
``plot_random_sampling_curves`` produces a 2x3 figure with pool_best vs
pool_avg overlays for the three quality metrics, plus pool growth, GP
activity, and the recent-reward signal.
"""

from __future__ import annotations
import csv
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# Headers in write order.
_COLUMNS = [
    "timestep",
    "pool_size", "pool_eval_cnt",
    "pool_best_reward", "pool_avg_reward",
    "pool_best_sortino", "pool_avg_sortino",
    "pool_best_abs_ic", "pool_avg_abs_ic",
    "recent_reward_mean", "recent_reward_std",
    "gp_total_accepted",
]


class RandomSamplingMetricsLogger:
    """In-memory buffer + on-disk CSV for the per-rollout metrics."""

    def __init__(self, save_path: str, recent_window: int = 256) -> None:
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        self.records: List[Dict[str, float]] = []
        self._recent: Deque[float] = deque(maxlen=int(recent_window))
        self._gp_accepted_total: int = 0
        # Initialise the file with a header.
        with open(self.save_path, "w", newline="") as f:
            csv.writer(f).writerow(_COLUMNS)

    # ------------------------------------------------------------------
    def note_recent_reward(self, reward: float) -> None:
        if reward is None or np.isnan(reward):
            return
        self._recent.append(float(reward))

    def note_gp_accepted(self, n: int) -> None:
        self._gp_accepted_total += int(n)

    # ------------------------------------------------------------------
    def record(self, timestep: int, pool) -> Dict[str, float]:
        """Snapshot pool stats and append to CSV.  Returns the row."""
        size = int(pool.size)
        comp = (pool._composite_scores[:size] if size else np.zeros(0))
        sortino = []
        abs_ics = []
        # We only have running averages of the last reward call; pull from
        # the parent's ``_reward_stats`` buffers which are appended on every
        # ``_score_expr`` call.  For "pool best/avg" we use the composite
        # scores (one per pool slot) plus the latest reward stats over the
        # last `size` evaluations as a proxy for per-slot risk metrics.
        stats = getattr(pool, "_reward_stats", {})
        # The most recent `size` entries roughly correspond to current pool
        # members.  This is approximate — exact per-slot sortino tracking
        # would require evaluating each pool member every rollout, which is
        # too expensive.  The composite scores are exact.
        n_recent = min(size, len(stats.get("reward", [])))
        if n_recent > 0:
            r_arr = np.asarray(stats["reward"][-n_recent:], dtype=np.float64)
            ic_arr = np.asarray(stats["abs_ic"][-n_recent:], dtype=np.float64)
            # ``r_bar`` slot is repurposed to mean PnL; sortino we don't
            # store directly, so derive a normalised proxy reward as a
            # stand-in (and label it "reward" vs "sortino" downstream).
            sortino = r_arr  # alias: reward as proxy for risk score
            abs_ics = ic_arr
        else:
            sortino = np.zeros(0)
            abs_ics = np.zeros(0)

        def _safe_max(a):
            return float(np.nanmax(a)) if a.size else 0.0

        def _safe_mean(a):
            return float(np.nanmean(a)) if a.size else 0.0

        recent = np.asarray(self._recent, dtype=np.float64)
        row = {
            "timestep":           int(timestep),
            "pool_size":          size,
            "pool_eval_cnt":      int(getattr(pool, "eval_cnt", 0)),
            "pool_best_reward":   _safe_max(comp),
            "pool_avg_reward":    _safe_mean(comp),
            "pool_best_sortino":  _safe_max(sortino),
            "pool_avg_sortino":   _safe_mean(sortino),
            "pool_best_abs_ic":   _safe_max(abs_ics),
            "pool_avg_abs_ic":    _safe_mean(abs_ics),
            "recent_reward_mean": float(np.mean(recent)) if recent.size else 0.0,
            "recent_reward_std":  float(np.std(recent)) if recent.size else 0.0,
            "gp_total_accepted":  int(self._gp_accepted_total),
        }
        self.records.append(row)
        with open(self.save_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row[k] for k in _COLUMNS])
        return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_random_sampling_curves(
    csv_path: str,
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (16, 9),
) -> str:
    """Render a 2x3 convergence plot from ``csv_path``.

    Layout::

        [pool reward best/avg]   [pool sortino best/avg]   [pool |IC| best/avg]
        [pool size + eval count] [recent reward mean ± σ]  [GP cumulative accepts]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: List[Dict[str, float]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    def col(name, cast=float):
        return np.asarray([cast(r[name]) for r in rows])

    t = col("timestep", int)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Random-Window Training Convergence",
                 fontsize=14, fontweight="bold")

    # --- Row 1: best vs avg curves for the three quality metrics ---
    ax = axes[0, 0]
    ax.plot(t, col("pool_best_reward"), "b-", linewidth=1.6, label="Pool Best")
    ax.plot(t, col("pool_avg_reward"), "b--", linewidth=1.2, alpha=0.6, label="Pool Avg")
    ax.set_title("Composite Reward (pool)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Reward")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, col("pool_best_sortino"), "g-", linewidth=1.6, label="Pool Best")
    ax.plot(t, col("pool_avg_sortino"), "g--", linewidth=1.2, alpha=0.6, label="Pool Avg")
    ax.set_title("Risk Score (recent-eval proxy)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Score")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, col("pool_best_abs_ic"), "r-", linewidth=1.6, label="Pool Best")
    ax.plot(t, col("pool_avg_abs_ic"), "r--", linewidth=1.2, alpha=0.6, label="Pool Avg")
    ax.set_title("|Rank IC| (recent-eval proxy)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("|IC|")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Row 2: pool size, recent reward mean, GP accepts ---
    ax = axes[1, 0]
    ax.plot(t, col("pool_size", int), "purple", linewidth=1.5, label="Pool Size")
    ax2 = ax.twinx()
    ax2.plot(t, col("pool_eval_cnt", int), "k--", linewidth=1.0, label="Eval Cnt")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Pool size", color="purple")
    ax2.set_ylabel("Eval count", color="black")
    ax.set_title("Pool Growth")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    rm = col("recent_reward_mean")
    rs = col("recent_reward_std")
    ax.plot(t, rm, "teal", linewidth=1.4, label="Mean")
    ax.fill_between(t, rm - rs, rm + rs, color="teal", alpha=0.15, label="±1σ")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Recent Episode Reward")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Reward")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(t, col("gp_total_accepted", int), "darkorange", linewidth=1.5,
            label="Cumulative")
    ax.set_title("GP Offspring Accepted")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is None:
        output_path = csv_path.replace(".csv", ".png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    return output_path

"""
Aggregate reward-component statistics produced by single-factor training.

For each window, the rolling training script writes
`window_NNN/reward_component_stats.json`. This tool consumes those files and
produces:
  * a cross-window summary table (count / mean / std / p05-p95 per component)
  * histogram PNGs for the two reward components and their raw primitives

Usage:
    python scripts/analyze_reward_stats.py \
        --run_dir=./out/tick_rolling/tick_3s_pool5_seed0_XXXX \
        --out_dir=./out/tick_rolling/tick_3s_pool5_seed0_XXXX/reward_analysis

The script is dependency-light: only numpy + (optional) matplotlib. If
matplotlib is unavailable it skips plotting and still writes the JSON summary.
"""

import glob
import json
import os
from typing import Dict, List, Optional

import fire
import numpy as np


COMPONENT_KEYS = ["abs_ic", "r_bar", "comp_ic", "comp_r", "reward", "pos_abs_mean"]


def _summarize(a: np.ndarray) -> Dict[str, float]:
    if a.size == 0:
        return {"count": 0}
    return {
        "count":     int(a.size),
        "mean":      float(a.mean()),
        "std":       float(a.std()),
        "abs_mean":  float(np.abs(a).mean()),
        "min":       float(a.min()),
        "max":       float(a.max()),
        "p05":       float(np.quantile(a, 0.05)),
        "p25":       float(np.quantile(a, 0.25)),
        "p50":       float(np.quantile(a, 0.50)),
        "p75":       float(np.quantile(a, 0.75)),
        "p95":       float(np.quantile(a, 0.95)),
    }


def _plot_hist(values: np.ndarray, title: str, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[analyze_reward_stats] matplotlib unavailable, skipping plot: {exc}")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=60, color="#1f77b4", alpha=0.85)
    ax.axvline(values.mean(), color="red", linestyle="--",
               label=f"mean = {values.mean():.4g}")
    ax.axvline(np.quantile(values, 0.5), color="orange", linestyle="--",
               label=f"median = {np.quantile(values, 0.5):.4g}")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(run_dir: str, out_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Aggregate reward-component stats across all windows under `run_dir`."""
    run_dir = os.path.expanduser(run_dir)
    stats_files = sorted(glob.glob(os.path.join(run_dir, "window_*/reward_component_stats.json")))
    if not stats_files:
        raise FileNotFoundError(
            f"No reward_component_stats.json files found under {run_dir}"
        )

    merged: Dict[str, List[float]] = {k: [] for k in COMPONENT_KEYS}
    per_window: List[Dict[str, Dict[str, float]]] = []

    for path in stats_files:
        with open(path, "r") as f:
            payload = json.load(f)
        wid = os.path.basename(os.path.dirname(path))
        summary = payload.get("summary", {})
        per_window.append({"window": wid, "summary": summary})
        raw = payload.get("raw", {})
        for k in COMPONENT_KEYS:
            vals = raw.get(k, [])
            if vals:
                merged[k].extend(vals)

    summary_all = {
        k: _summarize(np.asarray(v, dtype=np.float64))
        for k, v in merged.items()
    }

    result = {
        "run_dir":     run_dir,
        "num_windows": len(stats_files),
        "per_window":  per_window,
        "merged":      summary_all,
    }

    out_dir = out_dir or os.path.join(run_dir, "reward_analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Plot histograms for merged values
    for k in COMPONENT_KEYS:
        arr = np.asarray(merged[k], dtype=np.float64)
        if arr.size == 0:
            continue
        _plot_hist(arr, title=f"{k}  (n={arr.size})",
                   out_path=os.path.join(out_dir, f"hist_{k}.png"))

    # Print a compact console table
    hdr = f"{'key':<14s} {'count':>8s} {'mean':>12s} {'std':>12s} {'p05':>12s} {'p50':>12s} {'p95':>12s} {'max':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for k in COMPONENT_KEYS:
        s = summary_all[k]
        if s.get("count", 0) == 0:
            continue
        print(
            f"{k:<14s} {s['count']:>8d} "
            f"{s['mean']:>12.4g} {s['std']:>12.4g} "
            f"{s['p05']:>12.4g} {s['p50']:>12.4g} "
            f"{s['p95']:>12.4g} {s['max']:>12.4g}"
        )

    return summary_all


if __name__ == "__main__":
    fire.Fire(run)

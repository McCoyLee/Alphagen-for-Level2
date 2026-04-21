"""
Plot a full-period tick-feature overview for one stock.

Example:
    python scripts/plot_tick_overview.py \
      --data_root=/root/data/subset_data \
      --instrument=159845.sz \
      --start=2024-01-01 \
      --end=2026-02-28 \
      --out=tick_overview_159845.png
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from alphagen_level2.stock_data_tick import TickStockData, TickFeatureType
from alphagen_level2.config_tick import TICK_FEATURES


def _feature_series(data: TickStockData, stock_idx: int = 0) -> Dict[str, pd.Series]:
    """
    Return usable feature series (excluding backtrack/future margins).
    """
    start = data.max_backtrack_days
    stop = data.data.shape[0] - data.max_future_days
    idx = data._dates[start:stop]  # noqa: SLF001 - plotting utility script

    vals = data.data[start:stop, :, stock_idx].detach().cpu().numpy()
    s = lambda ft: pd.Series(vals[:, int(ft)], index=idx, dtype=np.float64).ffill()

    return {
        "mid": s(TickFeatureType.MID),
        "close": s(TickFeatureType.CLOSE),
        "volume": s(TickFeatureType.VOLUME).fillna(0.0),
        "turnover": s(TickFeatureType.TURNOVER).fillna(0.0),
        "spread": s(TickFeatureType.SPREAD),
        "ret": s(TickFeatureType.RET).fillna(0.0),
        "imbalance1": s(TickFeatureType.IMBALANCE_1).fillna(0.0),
        "imbalance_total": s(TickFeatureType.IMBALANCE_TOTAL).fillna(0.0),
    }


def plot_tick_overview(
    data_root: str,
    instrument: str,
    start: str,
    end: str,
    out: str,
    bar_size_sec: int = 3,
    max_workers: int = 8,
    cache_dir: str = "./out/tick_cache",
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = TickStockData(
        instrument=[instrument],
        start_time=start,
        end_time=end,
        max_backtrack_days=1200,
        max_future_days=100,
        features=TICK_FEATURES,
        device=device,
        data_root=data_root,
        cache_dir=cache_dir,
        max_workers=max_workers,
        bar_size_sec=bar_size_sec,
    )

    series = _feature_series(data, stock_idx=0)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Tick Overview | {instrument} | {start} ~ {end}", fontsize=14, fontweight="bold")

    # Price
    ax = axes[0]
    ax.plot(series["mid"], label="MID", color="tab:blue", linewidth=1.1)
    ax.plot(series["close"], label="CLOSE", color="tab:orange", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Price")
    ax.set_title("Price (MID / CLOSE)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Volume & Turnover
    ax = axes[1]
    ax.plot(series["volume"], label="Volume", color="tab:green", linewidth=0.8)
    ax2 = ax.twinx()
    ax2.plot(series["turnover"], label="Turnover", color="tab:red", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Volume")
    ax2.set_ylabel("Turnover")
    ax.set_title("Volume & Turnover")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Spread
    ax = axes[2]
    ax.plot(series["spread"], label="Spread", color="tab:purple", linewidth=0.9)
    ax.set_ylabel("Spread")
    ax.set_title("Bid-Ask Spread")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Return & Imbalance
    ax = axes[3]
    ax.plot(series["ret"], label="Log Return", color="tab:gray", linewidth=0.7, alpha=0.8)
    ax.plot(series["imbalance1"], label="Imbalance L1", color="tab:cyan", linewidth=0.8)
    ax.plot(series["imbalance_total"], label="Imbalance Total", color="tab:brown", linewidth=0.8)
    ax.axhline(0.0, linestyle=":", color="black", linewidth=0.8)
    ax.set_ylabel("Value")
    ax.set_title("Return & Order Imbalance")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Datetime")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot full-period tick feature overview for one stock.")
    ap.add_argument("--data_root", required=True, help="Root directory of Level2 HDF5 data.")
    ap.add_argument("--instrument", required=True, help="Stock code, e.g. 159845.sz")
    ap.add_argument("--start", required=True, help="Start date, e.g. 2024-01-01")
    ap.add_argument("--end", required=True, help="End date, e.g. 2026-02-28")
    ap.add_argument("--out", default="tick_overview.png", help="Output PNG path.")
    ap.add_argument("--bar_size_sec", type=int, default=3)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--cache_dir", default="./out/tick_cache")
    args = ap.parse_args()

    plot_tick_overview(
        data_root=args.data_root,
        instrument=args.instrument,
        start=args.start,
        end=args.end,
        out=args.out,
        bar_size_sec=args.bar_size_sec,
        max_workers=args.max_workers,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()

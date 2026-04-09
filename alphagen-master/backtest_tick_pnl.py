"""
Tick-level P&L backtest for specific stable factors.

For each factor, simulates a long-short strategy on the test period:
  - Signal = z-score of the factor value (computed bar-by-bar)
  - Position = clipped z-score in [-1, 1]
  - Bar P&L = position[t] * next_bar_log_return[t]
  - Transaction cost applied only when position changes

Outputs:
  - Printed summary: Sharpe, Ann.Return, MaxDD, IC
  - tick_pnl_backtest.png: P&L curves

Usage:
    cd alphagen-master
    python backtest_tick_pnl.py \
        --data_root ~/EquityLevel2/stock \
        --instrument 510300.sh \
        --start 2024-01-01 \
        --end 2024-06-30
"""

import argparse
import sys
import os
import numpy as np
import torch
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphagen_level2.stock_data_tick import TickStockData, TickFeatureType
from alphagen_level2.config_tick import TICK_FEATURES
from alphagen.data.expression import Feature, Ref, Sub, Div as DivExpr
from alphagen.data.parser import ExpressionParser
from alphagen.data.expression import *
from alphagen_level2.config_tick import OPERATORS as TICK_OPERATORS


# ── Factors to backtest ───────────────────────────────────────────────────────
# The 3 stable factors from your LLM+RL run:
FACTORS = [
    {
        "name": "Open/Mid Ratio",
        "expr": "Div($open,$mid)",
        "mean_w": -0.05130,   # sign from stable pool (negative → short when ratio high)
    },
    {
        "name": "SignedVol MAD(100)",
        "expr": "Mad(Mul(1.0,$signed_volume),100d)",
        "mean_w": +0.07062,
    },
    {
        "name": "Turnover WMA-GT(100)",
        "expr": "Greater(-2.0,WMA(Greater($turnover,-0.5),100d))",
        "mean_w": -0.04832,
    },
]

# ── Backtest parameters ───────────────────────────────────────────────────────
COST_BPS = 0.3          # round-trip transaction cost (bps), applied on position change
ZSCORE_CLIP = 2.0       # clip z-score to [-clip, clip] for position sizing
ZSCORE_WINDOW = 600     # rolling window (bars) for z-score normalization (≈30min)
TRADE_EVERY_N = 20      # only rebalance every N bars (1 = 3s, 20 ≈ 1min)
ANNUAL_BARS = 4800 * 244  # bars per year (4800 bars/day × 244 trading days)


# ── Parser ────────────────────────────────────────────────────────────────────
def build_parser() -> ExpressionParser:
    parser = ExpressionParser(
        TICK_OPERATORS,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={"Max": [Greater], "Min": [Less], "Delta": [Sub]},
    )
    parser._features = {f.name.lower(): f for f in TickFeatureType}
    return parser


# ── Signal processing ─────────────────────────────────────────────────────────
def rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling z-score; NaN for first `window` bars."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        seg = x[i - window: i]
        mu = np.nanmean(seg)
        sig = np.nanstd(seg)
        if sig > 1e-9:
            out[i] = (x[i] - mu) / sig
    return out


def _rolling_zscore_fast(x: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling z-score using pandas (much faster)."""
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=window).mean()
    sigma = s.rolling(window, min_periods=window).std(ddof=0)
    z = (s - mu) / sigma.clip(lower=1e-9)
    z[sigma < 1e-9] = np.nan
    return z.to_numpy(dtype=np.float64)


# ── Core P&L simulation ───────────────────────────────────────────────────────
def simulate_pnl(
    signal_raw: np.ndarray,   # [n_bars], raw factor values
    log_ret: np.ndarray,       # [n_bars], log return at each bar
    direction: float,          # +1 or -1 (from mean_w sign)
    cost_bps: float = COST_BPS,
    zscore_window: int = ZSCORE_WINDOW,
    zscore_clip: float = ZSCORE_CLIP,
    trade_every_n: int = TRADE_EVERY_N,
) -> dict:
    n = len(signal_raw)

    # 1. Normalize signal to position via rolling z-score
    z = _rolling_zscore_fast(signal_raw, zscore_window)
    z = np.nan_to_num(z, nan=0.0)
    pos_raw = np.clip(z * direction, -zscore_clip, zscore_clip) / zscore_clip  # in [-1,1]

    # 2. Only rebalance every N bars to reduce turnover
    pos = np.zeros(n, dtype=np.float64)
    current_pos = 0.0
    for i in range(0, n, trade_every_n):
        current_pos = pos_raw[i] if not np.isnan(pos_raw[i]) else 0.0
        end = min(i + trade_every_n, n)
        pos[i:end] = current_pos

    # 3. Compute P&L bar-by-bar
    cost_per_unit = cost_bps * 1e-4  # convert bps to fraction
    bar_pnl = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta_pos = abs(pos[i] - pos[i - 1])
        trade_cost = delta_pos * cost_per_unit
        bar_pnl[i] = pos[i - 1] * log_ret[i] - trade_cost

    # 4. Aggregate to daily P&L (4800 bars/day for 3s bars)
    bars_per_day = 4800
    n_full_days = n // bars_per_day
    daily_pnl = np.array([
        bar_pnl[d * bars_per_day: (d + 1) * bars_per_day].sum()
        for d in range(n_full_days)
    ])

    cum_pnl = np.cumsum(daily_pnl)

    # 5. Metrics
    ann_factor = 244  # trading days per year
    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std(ddof=1) + 1e-12
    sharpe = mean_daily / std_daily * np.sqrt(ann_factor)
    ann_return = mean_daily * ann_factor

    rolling_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - rolling_max
    max_dd = drawdown.min()

    # IC: correlation of position with next-bar return (bar level)
    valid = ~np.isnan(pos[:-1]) & ~np.isnan(log_ret[1:]) & (pos[:-1] != 0)
    if valid.sum() > 10:
        ic = float(np.corrcoef(pos[:-1][valid], log_ret[1:][valid])[0, 1])
    else:
        ic = float("nan")

    return {
        "bar_pnl": bar_pnl,
        "daily_pnl": daily_pnl,
        "cum_pnl": cum_pnl,
        "sharpe": sharpe,
        "ann_return": ann_return,
        "max_dd": max_dd,
        "ic": ic,
        "n_full_days": n_full_days,
        "total_bars": n,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: list, out_path: str = "tick_pnl_backtest.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n = len(results)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 4 * (n + 1)))

    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    # Individual panels
    for i, r in enumerate(results):
        ax = axes[i]
        days = np.arange(r["n_full_days"])
        ax.plot(days, r["cum_pnl"] * 100, color=colors[i], linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(days, 0, r["cum_pnl"] * 100,
                        where=r["cum_pnl"] >= 0, alpha=0.15, color=colors[i])
        ax.fill_between(days, 0, r["cum_pnl"] * 100,
                        where=r["cum_pnl"] < 0, alpha=0.15, color="red")
        title = (
            f"{r['name']}  |  "
            f"Sharpe: {r['sharpe']:.2f}  |  "
            f"AnnRet: {r['ann_return']*100:.1f}%  |  "
            f"MaxDD: {r['max_dd']*100:.1f}%  |  "
            f"IC: {r['ic']:.4f}"
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Cum. Log-Return (%)")
        ax.grid(True, alpha=0.3)

    # Comparison panel
    ax_cmp = axes[n]
    for i, r in enumerate(results):
        days = np.arange(r["n_full_days"])
        ax_cmp.plot(days, r["cum_pnl"] * 100, color=colors[i],
                    label=r["name"], linewidth=1.5)
    ax_cmp.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_cmp.set_title("All Factors — Comparison", fontsize=10)
    ax_cmp.set_ylabel("Cum. Log-Return (%)")
    ax_cmp.set_xlabel("Trading Days")
    ax_cmp.legend(loc="upper left", fontsize=9)
    ax_cmp.grid(True, alpha=0.3)

    plt.suptitle(
        f"Tick-Level Factor P&L  |  cost={COST_BPS}bps RT  |  "
        f"rebalance every {TRADE_EVERY_N} bars ({TRADE_EVERY_N*3}s)",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="~/EquityLevel2/stock")
    ap.add_argument("--instrument", default="510300.sh")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-06-30")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="tick_pnl_backtest.png")
    ap.add_argument("--cost_bps", type=float, default=COST_BPS)
    ap.add_argument("--trade_every_n", type=int, default=TRADE_EVERY_N)
    ap.add_argument("--zscore_window", type=int, default=ZSCORE_WINDOW)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load tick data ────────────────────────────────────────────────────────
    print(f"\nLoading tick data: {args.instrument}  {args.start} → {args.end}")
    data = TickStockData(
        instrument=[args.instrument],
        start_time=args.start,
        end_time=args.end,
        features=TICK_FEATURES,
        data_root=args.data_root,
        device=device,
        max_backtrack_days=1200,   # enough for 100-bar factors + buffer
        max_future_days=0,         # no lookahead needed for P&L
    )
    print(f"  Loaded: {data.n_days} usable bars  ({data.n_days // 4800:.0f} days approx)")

    # ── Compute next-bar log return ───────────────────────────────────────────
    # close[t] from the data tensor; ret[t] = log(close[t]) - log(close[t-1])
    # We need the raw RET feature (already log return in TickFeatureType)
    ret_feat = Feature(TickFeatureType.RET)
    ret_tensor = ret_feat.evaluate(data)          # [n_bars, 1]
    log_ret = ret_tensor.squeeze(-1).cpu().numpy().astype(np.float64)  # [n_bars]
    log_ret = np.nan_to_num(log_ret, nan=0.0)

    # ── Parse and evaluate each factor ───────────────────────────────────────
    parser = build_parser()
    all_results = []

    print("\n" + "=" * 70)
    print(f"{'Factor':<28} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>8} {'IC':>8}")
    print("=" * 70)

    for fdef in FACTORS:
        name = fdef["name"]
        expr_str = fdef["expr"]
        direction = np.sign(fdef["mean_w"]) if fdef["mean_w"] != 0 else 1.0

        try:
            expr = parser.parse(expr_str)
        except Exception as e:
            print(f"  [{name}] PARSE ERROR: {e}")
            continue

        try:
            alpha_tensor = expr.evaluate(data)    # [n_bars, 1]
        except Exception as e:
            print(f"  [{name}] EVAL ERROR: {e}")
            continue

        signal = alpha_tensor.squeeze(-1).cpu().numpy().astype(np.float64)
        signal = np.nan_to_num(signal, nan=0.0)

        res = simulate_pnl(
            signal_raw=signal,
            log_ret=log_ret,
            direction=direction,
            cost_bps=args.cost_bps,
            zscore_window=args.zscore_window,
            zscore_clip=ZSCORE_CLIP,
            trade_every_n=args.trade_every_n,
        )
        res["name"] = name
        res["expr"] = expr_str
        all_results.append(res)

        print(
            f"  {name:<26} "
            f"{res['sharpe']:>7.2f} "
            f"{res['ann_return']*100:>8.1f} "
            f"{res['max_dd']*100:>8.1f} "
            f"{res['ic']:>8.4f}"
        )

    print("=" * 70)

    if not all_results:
        print("No factors evaluated successfully.")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(all_results, out_path=args.out)

    # ── Also save daily P&L to CSV ────────────────────────────────────────────
    csv_path = args.out.replace(".png", "_daily.csv")
    n_days = min(r["n_full_days"] for r in all_results)
    df = pd.DataFrame({r["name"]: r["daily_pnl"][:n_days] for r in all_results})
    df.index.name = "day"
    df.to_csv(csv_path)
    print(f"Daily P&L CSV saved → {csv_path}")


if __name__ == "__main__":
    main()

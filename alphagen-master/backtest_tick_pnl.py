"""
Tick-level P&L backtest for specific stable factors.
 
Key design:
  - Uses HOLDING-PERIOD returns that MATCH the training target horizon
    (default 100 bars = 5 minutes for 3s bars).
  - Non-overlapping trades: enter at signal, hold for `holding_bars`,
    exit, then re-enter at next signal → clean P&L accounting.
  - Positions flattened at day boundaries (intraday only).
  - Direction from `mean_w` sign (calibrated on same 1-hour target).
 
Outputs:
  - Printed summary: Sharpe, Ann.Return, MaxDD, IC(factor), IC(strategy)
  - tick_pnl_backtest.png: P&L curves
  - tick_pnl_backtest_trades.csv: per-trade log
 
Usage:
    cd alphagen-master
    python backtest_tick_pnl.py \\
        --data_root ~/EquityLevel2/stock \\
        --instrument 510300.sh \\
        --start 2024-01-01 \\
        --end 2024-06-30
 
    # Override holding period (default = 100 bars ≈ 5 min):
    python backtest_tick_pnl.py --holding_bars 600   # 30 min
 
    # Auto-detect direction (ignore mean_w sign, use IC on first 20% of data):
    python backtest_tick_pnl.py --auto_direction
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
from alphagen_level2.calculator_tick import TickCalculator
from alphagen_level2.config_tick import TICK_FEATURES
from alphagen.data.expression import Feature, Ref
from alphagen.data.parser import ExpressionParser
from alphagen.data.expression import *
from alphagen_level2.config_tick import OPERATORS as TICK_OPERATORS
 
 
# ── Factors to backtest ───────────────────────────────────────────────────────
FACTORS = [
    {
        "name": "Open/Mid Ratio",
        "expr": "Div($open,$mid)",
        "mean_w": -0.05130,
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
DEFAULT_HOLDING_BARS = 100  # match training target: Ref(mid_prc,-100)/mid_prc-1
COST_BPS = 5.0               # round-trip cost per trade (ETF single-side 2.5 bps)
 
 
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
 
 
# ── Core P&L: non-overlapping fixed-holding trades ───────────────────────────
def simulate_pnl(
    signal: np.ndarray,        # [n_bars], raw factor values
    mid_prc: np.ndarray,       # [n_bars], mid prices ((ask1+bid1)/2)
    bars_per_day: int,
    direction: float,          # +1 or -1
    holding_bars: int = DEFAULT_HOLDING_BARS,
    cost_bps: float = COST_BPS,
    execution_delay: int = 1,
) -> dict:
    """
    Non-overlapping trades aligned to `holding_bars` horizon.
 
    At each trade entry `t`:
      - signal_zscore = rolling zscore of signal up to t (lookback = holding_bars)
      - position = clamp(zscore * direction, -1, 1)
      - hold until t + holding_bars (or end of day, whichever first)
      - trade_ret = mid_prc[exit] / mid_prc[entry] - 1
      - trade_pnl = position * trade_ret - |position| * cost_bps * 1e-4
    """
    n = len(signal)
    cost_frac = cost_bps * 1e-4
 
    trades = []     # list of {entry, exit, pos, ret, pnl}
 
    # ── Walk through days, doing non-overlapping intraday trades ─────────
    day_starts = list(range(0, n, bars_per_day))
 
    # ── Pre-process signal: forward-fill NaN then fill remaining with 0 ──
    # Rolling operators (Mad, WMA, etc.) propagate NaN across windows when
    # the underlying feature has missing bars (no ticks). Forward-filling
    # preserves the last valid signal value, which is more appropriate than
    # substituting 0 for a momentum/MAD-based signal.
    signal_series = pd.Series(signal, dtype=np.float64)
    signal_series = signal_series.ffill().fillna(0.0)
    signal_clean = signal_series.to_numpy()
 
    # Rolling z-score state
    zscore_window = 600  # fixed z-score lookback per training setup
    z_mean = signal_series.rolling(
        zscore_window, min_periods=max(zscore_window // 2, 1)
    ).mean().to_numpy()
    z_std = signal_series.rolling(
        zscore_window, min_periods=max(zscore_window // 2, 1)
    ).std(ddof=0).to_numpy()
 
    for day_idx, day_start in enumerate(day_starts):
        day_end = min(day_start + bars_per_day, n)
 
        t = day_start + zscore_window  # skip warm-up within day
        while t + execution_delay + holding_bars <= day_end:
            # ── Compute position using signal z-score at entry ────────
            std_val = z_std[t]
            mean_val = z_mean[t]
            # Guard both std and mean against NaN/degenerate values
            if (np.isnan(std_val) or std_val < 1e-12 or np.isnan(mean_val)):
                t += holding_bars
                continue
            z = (signal_clean[t] - mean_val) / std_val
            if np.isnan(z):  # should not happen after above guards, but be safe
                t += holding_bars
                continue
            z = np.clip(z, -3.0, 3.0) / 3.0  # normalize to [-1, 1]
            pos = z * direction
 
            if abs(pos) < 0.5:     # dead zone: skip negligible positions
                t += holding_bars
                continue
 
            # ── Compute return over holding period ────────────────────
            entry_bar = t + execution_delay
            exit_bar = entry_bar + holding_bars
            # Clamp exit to end-of-day (flatten before close)
            exit_bar = min(exit_bar, day_end - 1)
 
            if exit_bar <= entry_bar:
                t += holding_bars
                continue
 
            entry_px = mid_prc[entry_bar]
            exit_px = mid_prc[exit_bar]
            if np.isnan(entry_px) or np.isnan(exit_px) or entry_px <= 0:
                t += holding_bars
                continue
 
            trade_ret = (exit_px / entry_px) - 1.0
            trade_pnl = pos * trade_ret - abs(pos) * cost_frac
 
            trades.append({
                "entry": entry_bar,
                "exit": exit_bar,
                "day": day_idx,
                "position": pos,
                "ret": trade_ret,
                "pnl": trade_pnl,
            })
 
            t = exit_bar - execution_delay  # non-overlapping: next trade starts after exit
 
    if not trades:
        return {
            "trades": [],
            "daily_pnl": np.array([]),
            "cum_pnl": np.array([]),
            "sharpe": float("nan"),
            "ann_return": float("nan"),
            "max_dd": float("nan"),
            "strategy_ic": float("nan"),
            "n_trades": 0,
            "n_full_days": 0,
            "win_rate": float("nan"),
        }
 
    # ── Aggregate to daily P&L ────────────────────────────────────────────
    n_calendar_days = len(day_starts)
    daily_pnl = np.zeros(n_calendar_days, dtype=np.float64)
    for tr in trades:
        pnl_val = tr["pnl"]
        if not np.isnan(pnl_val):   # guard: skip any trade with NaN pnl
            daily_pnl[tr["day"]] += pnl_val
 
    # ── Trim trailing zero-days (no data) ─────────────────────────────────
    last_trade_day = max(tr["day"] for tr in trades)
    daily_pnl = daily_pnl[: last_trade_day + 1]
    cum_pnl = np.cumsum(daily_pnl)
 
    # ── Metrics (robust to small samples) ────────────────────────────────
    ann_factor = 244
    valid_days = daily_pnl[daily_pnl != 0]   # days with actual trades
    if len(valid_days) < 2:
        mean_daily = float("nan")
        std_daily = float("nan")
        sharpe = float("nan")
        ann_return = float("nan")
    else:
        mean_daily = valid_days.mean()
        std_daily = valid_days.std(ddof=1)
        sharpe = (mean_daily / std_daily * np.sqrt(ann_factor)
                  if std_daily > 1e-12 else float("nan"))
        ann_return = mean_daily * ann_factor
 
    rolling_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - rolling_max
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
 
    trade_pnls = np.array([t["pnl"] for t in trades if not np.isnan(t["pnl"])])
    win_rate = (trade_pnls > 0).mean() if len(trade_pnls) > 0 else float("nan")
 
    # ── Strategy IC: correlation of executed position with realized return ─
    # NOTE: This is a strategy-level metric (contains direction + sizing),
    # not pure factor IC.
    positions = np.array([t["position"] for t in trades])
    returns = np.array([t["ret"] for t in trades])
    valid = ~np.isnan(positions) & ~np.isnan(returns) & (positions != 0)
    strategy_ic = safe_corr(positions[valid], returns[valid], min_n=10)
 
    return {
        "trades": trades,
        "daily_pnl": daily_pnl,
        "cum_pnl": cum_pnl,
        "sharpe": sharpe,
        "ann_return": ann_return,
        "max_dd": max_dd,
        "strategy_ic": strategy_ic,
        "n_trades": len(trades),
        "n_full_days": len(daily_pnl),
        "win_rate": win_rate,
    }


def calc_factor_ic(
    signal: np.ndarray,
    mid_prc: np.ndarray,
    bars_per_day: int,
    holding_bars: int,
    execution_delay: int = 1,
):
    """
    Pure factor IC on all *eligible* decision points (no direction/position/deadzone).
    Eligibility: t, t+delay, t+delay+holding all inside same trading day.
    """
    n = len(signal)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    day_starts = list(range(0, n, bars_per_day))

    for day_start in day_starts:
        day_end = min(day_start + bars_per_day, n)
        last_t = day_end - execution_delay - holding_bars
        if last_t <= day_start:
            continue
        for t in range(day_start, last_t):
            entry = t + execution_delay
            exit_ = entry + holding_bars
            p0 = mid_prc[entry]
            p1 = mid_prc[exit_]
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
                continue
            fwd_ret[t] = p1 / p0 - 1.0

    valid = ~np.isnan(signal) & ~np.isnan(fwd_ret)
    n_valid = int(valid.sum())
    ic = safe_corr(signal[valid], fwd_ret[valid], min_n=30)
    return ic, n_valid


def safe_corr(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> float:
    """NaN-safe Pearson correlation with std guard (prevents numpy warnings)."""
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if x.size < min_n:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def auto_detect_direction(
    signal: np.ndarray,
    mid_prc: np.ndarray,
    holding_bars: int,
    calibration_frac: float = 0.2,
    execution_delay: int = 1,
) -> float:
    """
    Use the first `calibration_frac` of data to detect the sign of the
    factor's correlation with forward returns at the target horizon,
    accounting for execution_delay (signal at t → entry at t+delay).
    Returns +1 or -1.
    """
    n_cal = int(len(signal) * calibration_frac)
    if n_cal < holding_bars * 3:
        return 1.0
 
    # Forward return from entry (t+delay) to exit (t+delay+holding)
    fwd_ret = np.full(n_cal, np.nan, dtype=np.float64)
    for t in range(n_cal - execution_delay - holding_bars):
        entry = t + execution_delay
        exit_ = entry + holding_bars
        if (mid_prc[entry] > 0 and not np.isnan(mid_prc[entry])
                and not np.isnan(mid_prc[exit_])):
            fwd_ret[t] = mid_prc[exit_] / mid_prc[entry] - 1.0
 
    valid = ~np.isnan(signal[:n_cal]) & ~np.isnan(fwd_ret)
    if valid.sum() < 30:
        return 1.0
    corr = np.corrcoef(signal[:n_cal][valid], fwd_ret[valid])[0, 1]
    return 1.0 if corr >= 0 else -1.0
 
 
# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(
    results: list,
    holding_bars: int,
    cost_bps: float,
    out_path: str = "tick_pnl_backtest.png",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
 
    n = len(results)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 4 * (n + 1)))
    if n == 0:
        return
 
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
 
    for i, r in enumerate(results):
        ax = axes[i]
        days = np.arange(r["n_full_days"])
        ax.plot(days, r["cum_pnl"] * 100, color=colors[i % len(colors)], linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(
            days, 0, r["cum_pnl"] * 100,
            where=r["cum_pnl"] >= 0, alpha=0.15, color=colors[i % len(colors)],
        )
        ax.fill_between(
            days, 0, r["cum_pnl"] * 100,
            where=r["cum_pnl"] < 0, alpha=0.15, color="red",
        )
        title = (
            f"{r['name']}  |  "
            f"Sharpe: {r['sharpe']:.2f}  |  "
            f"AnnRet: {r['ann_return']*100:.1f}%  |  "
            f"MaxDD: {r['max_dd']*100:.1f}%  |  "
            f"IC(f): {r['factor_ic']:.4f}  |  "
            f"IC(s): {r.get('strategy_ic', float('nan')):.4f}  |  "
            f"Trades: {r['n_trades']}  WinRate: {r['win_rate']*100:.0f}%"
        )
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Cum. Return (%)")
        ax.grid(True, alpha=0.3)
 
    ax_cmp = axes[n]
    for i, r in enumerate(results):
        days = np.arange(r["n_full_days"])
        ax_cmp.plot(
            days, r["cum_pnl"] * 100,
            color=colors[i % len(colors)], label=r["name"], linewidth=1.5,
        )
    ax_cmp.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_cmp.set_title("All Factors — Comparison", fontsize=10)
    ax_cmp.set_ylabel("Cum. Return (%)")
    ax_cmp.set_xlabel("Trading Days")
    ax_cmp.legend(loc="upper left", fontsize=9)
    ax_cmp.grid(True, alpha=0.3)
 
    hold_min = holding_bars * 3 / 60
    plt.suptitle(
        f"Tick-Level Factor P&L  |  holding={holding_bars} bars ({hold_min:.0f}min)"
        f"  |  cost={cost_bps}bps RT",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved -> {out_path}")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Tick-level P&L backtest for stable factors")
    ap.add_argument("--data_root", default="/root/data/subset_data")
    ap.add_argument("--instrument", default="159845.sz")
    ap.add_argument("--start", default="2025-7-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="tick_pnl_backtest.png")
    ap.add_argument("--holding_bars", type=int, default=DEFAULT_HOLDING_BARS,
                    help="Holding period in bars (default=100≈5min, matching training target)")
    ap.add_argument("--cost_bps", type=float, default=COST_BPS,
                    help="Round-trip transaction cost in bps (default=5.0; single-side=2.5)")
    ap.add_argument("--auto_direction", action="store_true",
                    help="Auto-detect factor direction from first 20%% of data "
                         "(overrides mean_w sign)")
    ap.add_argument("--max_backtrack", type=int, default=1200,
                    help="Max backtrack bars for factor lookback buffer")
    ap.add_argument("--execution_delay", type=int, default=1,
                        help="Bars between signal observation and trade entry "
                            "(default=1: realistic; 0: same-bar fill, inflates perf ")
    ap.add_argument(
        "--factor_ic_mode",
        choices=["intraday", "train_compatible"],
        default="intraday",
        help=(
            "How to compute IC(f): "
            "'intraday' uses tradable intraday points in backtest; "
            "'train_compatible' uses TickCalculator target IC (same style as training)."
        ),
    )
    args = ap.parse_args()
 
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Holding period: {args.holding_bars} bars "
          f"({args.holding_bars * 3 / 60:.0f} min)")
    print(f"Transaction cost: {args.cost_bps} bps RT")
    print(f"Factor IC mode: {args.factor_ic_mode}")
 
    # ── Load tick data ────────────────────────────────────────────────────
    # Need max_future_days >= holding_bars for train-compatible IC target
    # and auto_direction forward-return calibration.
    # For P&L itself, we still stop trading `holding_bars` before day end anyway.
    print(f"\nLoading tick data: {args.instrument}  {args.start} -> {args.end}")
    max_future = max(args.holding_bars + args.execution_delay, 0)
    data = TickStockData(
        instrument=[args.instrument],
        start_time=args.start,
        end_time=args.end,
        features=TICK_FEATURES,
        data_root=args.data_root,
        device=device,
        max_backtrack_days=args.max_backtrack,
        max_future_days=max_future,
    )
 
    bpd = data.bars_per_day
    n_bars = data.n_days
    n_days_approx = n_bars // bpd
    print(f"  bars_per_day: {bpd}")
    print(f"  usable bars: {n_bars}  (~{n_days_approx} days)")
 
    # ── Extract mid prices (raw, for return computation) ─────────────────
    mid_feat = Feature(TickFeatureType.MID)
    mid_tensor = mid_feat.evaluate(data)  # [n_bars, 1]
    mid_prc = mid_tensor.squeeze(-1).cpu().numpy().astype(np.float64)
 
    # ── Parse and evaluate each factor ────────────────────────────────────
    parser = build_parser()
    # Training-compatible IC calculator (same IC definition path as training code)
    ic_calculator = None
    if args.factor_ic_mode == "train_compatible":
        mid_expr = Feature(TickFeatureType.MID)
        train_style_target = Ref(mid_expr, -args.holding_bars) / mid_expr - 1
        ic_calculator = TickCalculator(data, train_style_target)
    all_results = []
 
    header = (
        f"{'Factor':<28} {'Dir':>4} {'Sharpe':>7} {'AnnRet%':>8} "
        f"{'MaxDD%':>8} {'IC(f)':>8} {'IC(s)':>8} {'#Trades':>8} {'WinRate':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
 
    for fdef in FACTORS:
        name = fdef["name"]
        expr_str = fdef["expr"]
 
        try:
            expr = parser.parse(expr_str)
        except Exception as e:
            print(f"  [{name}] PARSE ERROR: {e}")
            continue
 
        try:
            alpha_tensor = expr.evaluate(data)  # [n_bars, 1]
        except Exception as e:
            print(f"  [{name}] EVAL ERROR: {e}")
            continue
 
        signal = alpha_tensor.squeeze(-1).cpu().numpy().astype(np.float64)
        signal_clean = pd.Series(signal, dtype=np.float64).ffill().fillna(0.0).to_numpy()
        if args.factor_ic_mode == "train_compatible":
            assert ic_calculator is not None
            factor_ic = float(ic_calculator.calc_single_IC_ret(expr))
            ic_n = int(data.n_days)
        else:
            factor_ic, ic_n = calc_factor_ic(
                signal_clean,
                mid_prc,
                bars_per_day=bpd,
                holding_bars=args.holding_bars,
                execution_delay=args.execution_delay,
            )
 
        # ── Direction: from mean_w or auto-detect ─────────────────────
        if args.auto_direction:
            direction = auto_detect_direction(
                signal, mid_prc, args.holding_bars, calibration_frac=0.2, execution_delay=args.execution_delay,
            )
            dir_label = f"auto({'+'if direction>0 else '-'})"
        else:
            direction = np.sign(fdef["mean_w"]) if fdef["mean_w"] != 0 else 1.0
            dir_label = f"{'+'if direction>0 else '-'}(w)"
 
        # ── Simulate P&L ──────────────────────────────────────────────
        res = simulate_pnl(
            signal=signal,
            mid_prc=mid_prc,
            bars_per_day=bpd,
            direction=direction,
            holding_bars=args.holding_bars,
            cost_bps=args.cost_bps,
            execution_delay=args.execution_delay,
        )
        res["name"] = name
        res["expr"] = expr_str
        res["direction"] = direction
        res.setdefault("strategy_ic", float("nan"))
        res["factor_ic"] = factor_ic
        res["factor_ic_n"] = ic_n
        all_results.append(res)
 
        print(
            f"  {name:<26} {dir_label:>4} "
            f"{res['sharpe']:>7.2f} "
            f"{res['ann_return']*100:>8.2f} "
            f"{res['max_dd']*100:>8.2f} "
            f"{res['factor_ic']:>8.4f} "
            f"{res.get('strategy_ic', float('nan')):>8.4f} "
            f"{res['n_trades']:>8d} "
            f"{res['win_rate']*100:>7.1f}%"
        )
 
    print("=" * len(header))
 
    if not all_results:
        print("No factors evaluated successfully.")
        return
 
    # ── Plot ──────────────────────────────────────────────────────────────
    plot_results(all_results, args.holding_bars, args.cost_bps, out_path=args.out)
 
    # ── Save trade log ────────────────────────────────────────────────────
    csv_path = args.out.replace(".png", "_trades.csv")
    rows = []
    for r in all_results:
        for tr in r["trades"]:
            rows.append({
                "factor": r["name"],
                "entry_bar": tr["entry"],
                "exit_bar": tr["exit"],
                "day": tr["day"],
                "position": tr["position"],
                "ret": tr["ret"],
                "pnl": tr["pnl"],
            })
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Trade log saved -> {csv_path}")
 
    # ── Save daily P&L ────────────────────────────────────────────────────
    daily_csv = args.out.replace(".png", "_daily.csv")
    max_days = max(r["n_full_days"] for r in all_results)
    daily_dict = {}
    for r in all_results:
        padded = np.zeros(max_days)
        padded[: len(r["daily_pnl"])] = r["daily_pnl"]
        daily_dict[r["name"]] = padded
    pd.DataFrame(daily_dict).to_csv(daily_csv, index_label="day")
    print(f"Daily P&L saved -> {daily_csv}")
 
 
if __name__ == "__main__":
    main()

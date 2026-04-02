"""
Rolling-window (walk-forward) RL alpha training on Level 2 data.

Designed for:
  - 3-second bars (bar_size_sec=3)
  - Single ETF instrument
  - Time range 2022-2026
  - Walk-forward windows: 1 year train, 6 months valid, 6 months test

Default schedule:
  Window 0: train 2022-01-01~2022-12-31, valid 2023-01-01~2023-06-30, test 2023-07-01~2023-12-31
  Window 1: train 2022-07-01~2023-06-30, valid 2023-07-01~2023-12-31, test 2024-01-01~2024-06-30
  Window 2: train 2023-01-01~2023-12-31, valid 2024-01-01~2024-06-30, test 2024-07-01~2024-12-31
  ...rolling forward in 6-month steps...

Each window:
  1. Loads train/valid/test data from Level 2 HDF5 files
  2. Trains an RL agent to discover alpha factors
  3. Carries the best factors (alpha pool) forward as warm-start for next window
  4. Logs per-window and cross-window metrics

Usage:
    # Single ETF, 3s bars, default 2022-2026 schedule:
    python scripts/rl_level2_rolling.py \\
        --data_root=~/EquityLevel2/stock \\
        --instruments='["510300.sh"]' \\
        --bar_size_sec=3

    # With Level 2 features and regime features:
    python scripts/rl_level2_rolling.py \\
        --data_root=~/EquityLevel2/stock \\
        --instruments='["510300.sh"]' \\
        --bar_size_sec=3 \\
        --use_level2_features \\
        --use_regime_features

    # Custom window parameters:
    python scripts/rl_level2_rolling.py \\
        --data_root=~/EquityLevel2/stock \\
        --instruments='["510300.sh"]' \\
        --bar_size_sec=3 \\
        --train_months=12 --valid_months=6 --test_months=6 --step_months=6
"""

import json
import os
from typing import Optional, List, Union, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import torch
import fire
from sb3_contrib.ppo_mask import MaskablePPO

from alphagen.data.expression import Feature, Ref
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_level2.stock_data import Level2StockData, Level2FeatureType
from alphagen_level2.calculator import Level2Calculator
from alphagen_level2.env_wrapper import Level2AlphaEnv
from alphagen_level2.config import (
    BASIC_FEATURES, LEVEL2_FEATURES,
    DELTA_TIMES_3S, DELTA_TIMES_3MIN,
)
from alphagen_level2.convergence_logger import ConvergenceLogger, plot_convergence
from alphagen_level2.diversity_pool import DiversityMseAlphaPool

# Import the callback from the base script
from scripts.rl_level2 import Level2Callback


# ---------------------------------------------------------------------------
# Rolling window schedule
# ---------------------------------------------------------------------------
def build_rolling_schedule(
    global_start: str = "2022-01-01",
    global_end: str = "2026-12-31",
    train_months: int = 12,
    valid_months: int = 6,
    test_months: int = 6,
    step_months: int = 6,
) -> List[dict]:
    """
    Generate a list of walk-forward windows.

    Each window dict has keys: train_start, train_end, valid_start, valid_end,
    test_start, test_end, window_id.
    """
    windows = []
    start = datetime.strptime(global_start, "%Y-%m-%d")
    end = datetime.strptime(global_end, "%Y-%m-%d")
    window_id = 0

    while True:
        train_start = start + relativedelta(months=step_months * window_id)
        train_end = train_start + relativedelta(months=train_months) - relativedelta(days=1)
        valid_start = train_end + relativedelta(days=1)
        valid_end = valid_start + relativedelta(months=valid_months) - relativedelta(days=1)
        test_start = valid_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)

        if test_end > end:
            break

        windows.append({
            "window_id": window_id,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "valid_start": valid_start.strftime("%Y-%m-%d"),
            "valid_end": valid_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        window_id += 1

    return windows


# ---------------------------------------------------------------------------
# Per-window training
# ---------------------------------------------------------------------------
def train_one_window(
    window: dict,
    instruments: Union[str, List[str]],
    features: List[Level2FeatureType],
    bar_size_sec: int,
    max_backtrack_bars: int,
    max_future_bars: int,
    pool_capacity: int,
    steps: int,
    data_root: str,
    cache_dir: Optional[str],
    max_workers: int,
    device: torch.device,
    save_root: str,
    seed: int,
    prev_pool_path: Optional[str] = None,
    ic_mut_threshold: float = 0.99,
    diversity_bonus: float = 0.0,
) -> dict:
    """
    Train one walk-forward window.

    Returns dict with window results including best IC, pool path, etc.
    """
    wid = window["window_id"]
    print(f"\n{'='*70}")
    print(f"[Rolling] Window {wid}: "
          f"train={window['train_start']}~{window['train_end']}, "
          f"valid={window['valid_start']}~{window['valid_end']}, "
          f"test={window['test_start']}~{window['test_end']}")
    print(f"{'='*70}")

    window_dir = os.path.join(save_root, f"window_{wid:03d}")
    os.makedirs(window_dir, exist_ok=True)

    # Save window metadata
    with open(os.path.join(window_dir, "window_meta.json"), "w") as f:
        json.dump(window, f, indent=2)

    close = Feature(Level2FeatureType.CLOSE)
    target = Ref(close, -max_future_bars) / close - 1

    def get_dataset(start: str, end: str) -> Level2StockData:
        return Level2StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            max_backtrack_days=max_backtrack_bars,
            max_future_days=max_future_bars,
            features=features,
            device=device,
            data_root=data_root,
            cache_dir=cache_dir,
            max_workers=max_workers,
            bar_size_sec=bar_size_sec,
        )

    print(f"[Window {wid}] Loading datasets...")
    segments = [
        (window["train_start"], window["train_end"]),
        (window["valid_start"], window["valid_end"]),
        (window["test_start"], window["test_end"]),
    ]
    datasets = [get_dataset(*s) for s in segments]
    for i, (seg, ds) in enumerate(zip(segments, datasets)):
        label = ["train", "valid", "test"][i]
        print(f"  {label}: {seg[0]}~{seg[1]}, {ds.n_days} bars, "
              f"{ds.n_stocks} stocks, {ds.n_features} features")

    calculators = [Level2Calculator(d, target) for d in datasets]

    # Build pool (optionally warm-start from previous window)
    use_diversity = diversity_bonus > 0 or ic_mut_threshold < 0.99
    if use_diversity:
        pool = DiversityMseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device,
            ic_mut_threshold=ic_mut_threshold,
            diversity_bonus=diversity_bonus,
        )
    else:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device,
        )

    # Warm-start: load expressions from previous window's pool
    if prev_pool_path is not None and os.path.exists(prev_pool_path):
        print(f"[Window {wid}] Warm-starting from {prev_pool_path}")
        try:
            with open(prev_pool_path, "r") as f:
                prev_state = json.load(f)
            if "exprs" in prev_state and len(prev_state["exprs"]) > 0:
                from alphagen.data.parser import ExpressionParser
                from alphagen_level2.config import OPERATORS as L2_OPERATORS
                parser = ExpressionParser(
                    L2_OPERATORS, ignore_case=True,
                    non_positive_time_deltas_allowed=False,
                )
                loaded_exprs = []
                for expr_str in prev_state["exprs"]:
                    try:
                        loaded_exprs.append(parser.parse(expr_str))
                    except Exception:
                        pass
                if loaded_exprs:
                    pool.force_load_exprs(loaded_exprs)
                    print(f"  Loaded {len(loaded_exprs)} expressions from previous window")
        except Exception as e:
            print(f"  Warm-start failed: {e}")

    env = Level2AlphaEnv(
        pool=pool,
        use_level2_features=(len(features) > 6),
        device=device,
        print_expr=True,
    )

    conv_logger = ConvergenceLogger(save_dir=window_dir)
    callback = Level2Callback(
        save_path=window_dir,
        valid_calculator=calculators[1],
        test_calculators=calculators[2:],
        verbose=1,
        convergence_logger=conv_logger,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        n_steps=2048,
        tensorboard_log=os.path.join(save_root, "tensorboard"),
        device=device,
        verbose=1,
    )

    print(f"[Window {wid}] Training for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=callback,
        tb_log_name=f"window_{wid:03d}",
    )

    # Save final pool
    final_pool_path = os.path.join(window_dir, "final_pool.json")
    with open(final_pool_path, "w") as f:
        json.dump(pool.to_json_dict(), f)

    # Save convergence data
    csv_path = conv_logger.save_csv()
    conv_logger.save_json()
    try:
        plot_convergence(csv_path)
    except Exception:
        pass

    # Evaluate final pool
    valid_ic, valid_rank_ic = pool.test_ensemble(calculators[1])
    test_ic, test_rank_ic = pool.test_ensemble(calculators[2])

    result = {
        "window_id": wid,
        **window,
        "valid_ic": float(valid_ic),
        "valid_rank_ic": float(valid_rank_ic),
        "test_ic": float(test_ic),
        "test_rank_ic": float(test_rank_ic),
        "pool_size": int(pool.size),
        "pool_path": final_pool_path,
    }

    summary = conv_logger.summary()
    if summary:
        result["best_pool_ic"] = summary.get("best_pool_ic", 0.0)

    print(f"[Window {wid}] Done: valid_ic={valid_ic:.4f}, test_ic={test_ic:.4f}")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(
    seed: int = 0,
    instruments: Union[str, List[str]] = '["510300.sh"]',
    pool_capacity: int = 20,
    steps_per_window: int = 150_000,
    data_root: str = "~/EquityLevel2/stock",
    use_level2_features: bool = True,
    use_regime_features: bool = False,
    # Bar size
    bar_size_sec: int = 3,
    # Rolling window parameters
    global_start: str = "2022-01-01",
    global_end: str = "2026-12-31",
    train_months: int = 12,
    valid_months: int = 6,
    test_months: int = 6,
    step_months: int = 6,
    # Bar-level lookback/forward
    max_backtrack_bars: int = 4800,
    max_future_bars: int = 4800,
    # IO
    cache_dir: Optional[str] = "./out/l2_cache",
    max_workers: int = 4,
    # Diversity
    ic_mut_threshold: float = 0.99,
    diversity_bonus: float = 0.0,
    # Warm-start
    warm_start: bool = True,
):
    """
    Walk-forward rolling-window RL alpha training on Level 2 data.

    :param seed: Random seed
    :param instruments: Stock/ETF codes as JSON list (e.g. '["510300.sh"]')
    :param pool_capacity: Maximum alpha pool size per window
    :param steps_per_window: RL training steps per window
    :param data_root: Root path to Level 2 HDF5 data
    :param use_level2_features: Use all 20 Level 2 features (vs 6 basic OHLCV)
    :param use_regime_features: Include 4 regime/conditional features (vol/volume/spread/trend)
    :param bar_size_sec: Bar size in seconds (3 = 3-second bars)
    :param global_start: Earliest date for training data
    :param global_end: Latest date for test data
    :param train_months: Training window length in months
    :param valid_months: Validation window length in months
    :param test_months: Test window length in months
    :param step_months: Step size for rolling forward in months
    :param max_backtrack_bars: Max lookback in bars (4800 ≈ 1 day for 3s bars)
    :param max_future_bars: Max forward look in bars (4800 ≈ 1 day for 3s bars)
    :param cache_dir: Cache directory for aggregated bar data
    :param max_workers: Parallel HDF5 IO threads
    :param ic_mut_threshold: Mutual IC threshold for diversity filtering
    :param diversity_bonus: Reward bonus for novel alphas
    :param warm_start: Carry forward alpha pool across windows
    """
    import alphagen_level2.config as l2_config

    reseed_everything(seed)

    # Set DELTA_TIMES based on bar size
    bar_size_min = bar_size_sec / 60.0
    if bar_size_min < 1.0:
        l2_config.DELTA_TIMES = list(DELTA_TIMES_3S)
        print(f"[Rolling] Using 3s-bar DELTA_TIMES: {l2_config.DELTA_TIMES}")
    else:
        l2_config.DELTA_TIMES = list(DELTA_TIMES_3MIN)
        print(f"[Rolling] Using 3min-bar DELTA_TIMES: {l2_config.DELTA_TIMES}")

    # Select features
    if use_regime_features:
        features = list(Level2FeatureType)  # all 24 including regime
    elif use_level2_features:
        features = LEVEL2_FEATURES  # 20 L2 features (no regime)
    else:
        features = BASIC_FEATURES  # 6 OHLCV

    # Parse instrument list
    if isinstance(instruments, str) and instruments.startswith("["):
        instruments = json.loads(instruments)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_root = os.path.join(
        "./out/rolling",
        f"roll_{bar_size_sec}s_pool{pool_capacity}_seed{seed}_{timestamp}",
    )
    os.makedirs(save_root, exist_ok=True)

    # Build rolling schedule
    schedule = build_rolling_schedule(
        global_start=global_start,
        global_end=global_end,
        train_months=train_months,
        valid_months=valid_months,
        test_months=test_months,
        step_months=step_months,
    )

    print(f"[Rolling] {len(schedule)} windows, bar_size={bar_size_sec}s, "
          f"instrument={instruments}, features={len(features)}")
    for w in schedule:
        print(f"  Window {w['window_id']}: "
              f"train {w['train_start']}~{w['train_end']} | "
              f"valid {w['valid_start']}~{w['valid_end']} | "
              f"test {w['test_start']}~{w['test_end']}")

    # Save run config
    run_config = {
        "seed": seed, "instruments": instruments, "pool_capacity": pool_capacity,
        "steps_per_window": steps_per_window, "bar_size_sec": bar_size_sec,
        "use_level2_features": use_level2_features,
        "use_regime_features": use_regime_features,
        "max_backtrack_bars": max_backtrack_bars, "max_future_bars": max_future_bars,
        "schedule": schedule,
    }
    with open(os.path.join(save_root, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Run walk-forward
    all_results = []
    prev_pool_path = None

    for window in schedule:
        result = train_one_window(
            window=window,
            instruments=instruments,
            features=features,
            bar_size_sec=bar_size_sec,
            max_backtrack_bars=max_backtrack_bars,
            max_future_bars=max_future_bars,
            pool_capacity=pool_capacity,
            steps=steps_per_window,
            data_root=data_root,
            cache_dir=cache_dir,
            max_workers=max_workers,
            device=device,
            save_root=save_root,
            seed=seed,
            prev_pool_path=prev_pool_path if warm_start else None,
            ic_mut_threshold=ic_mut_threshold,
            diversity_bonus=diversity_bonus,
        )
        all_results.append(result)
        prev_pool_path = result["pool_path"]

        # Save running results
        with open(os.path.join(save_root, "rolling_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("[Rolling] Walk-forward complete. Summary:")
    print(f"{'='*70}")
    print(f"{'Window':>8} {'Train Period':>25} {'Test Period':>25} {'Test IC':>10} {'Test RkIC':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['window_id']:>8} "
              f"{r['train_start']}~{r['train_end']:>11} "
              f"{r['test_start']}~{r['test_end']:>11} "
              f"{r['test_ic']:>10.4f} {r['test_rank_ic']:>10.4f}")

    avg_test_ic = np.mean([r["test_ic"] for r in all_results])
    avg_test_ric = np.mean([r["test_rank_ic"] for r in all_results])
    print(f"\nAverage test IC:      {avg_test_ic:.4f}")
    print(f"Average test Rank IC: {avg_test_ric:.4f}")
    print(f"Results saved to:     {save_root}")


if __name__ == "__main__":
    fire.Fire(main)

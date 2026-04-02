"""
Rolling-window (walk-forward) RL alpha training on 3-second bar data.

Designed for:
  - 3-second bars (bar_size_sec=3)
  - Single ETF instrument (e.g. 510300.sh)
  - Time range 2022-2026
  - 20 microstructure features
  - Walk-forward windows: 1 year train, 6 months valid, 6 months test

Default schedule:
  Window 0: train 2022-01-01~2022-12-31, valid 2023-01-01~2023-06-30, test 2023-07-01~2023-12-31
  Window 1: train 2022-07-01~2023-06-30, valid 2023-07-01~2023-12-31, test 2024-01-01~2024-06-30
  ...rolling forward in 6-month steps...

Each window:
  1. Loads train/valid/test data as 3s bars with 20 features
  2. Trains an RL agent to discover alpha factors
  3. Carries the best factors (alpha pool) forward as warm-start for next window
  4. Logs per-window and cross-window metrics

Usage:
    python scripts/rl_tick_rolling.py \\
        --data_root=~/EquityLevel2/stock \\
        --instruments='["510300.sh"]'

    # All 20 features are used by default.
    # Use --use_all_features=False for basic subset (7 features).

    python scripts/rl_tick_rolling.py \\
        --data_root=~/EquityLevel2/stock \\
        --instruments='["510300.sh"]' \\
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
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import Feature, Ref, Expression
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_level2.stock_data_tick import TickStockData, TickFeatureType
from alphagen_level2.calculator_tick import TickCalculator
from alphagen_level2.env_wrapper_tick import TickAlphaEnv
from alphagen_level2.config_tick import TICK_FEATURES, TICK_FEATURES_BASIC
from alphagen_level2.convergence_logger import ConvergenceLogger, plot_convergence
from alphagen_level2.diversity_pool import DiversityMseAlphaPool


# ---------------------------------------------------------------------------
# Callback (adapted from rl_level2.py for tick data)
# ---------------------------------------------------------------------------

class TickRollingCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        valid_calculator: TickCalculator,
        test_calculators: List[TickCalculator],
        verbose: int = 0,
        convergence_logger: Optional[ConvergenceLogger] = None,
        plot_interval: int = 10,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.valid_calculator = valid_calculator
        self.test_calculators = test_calculators
        self.conv_logger = convergence_logger
        self._plot_interval = plot_interval
        self._rollout_count = 0
        self._global_eval_cnt = 0
        self._last_pool_eval_cnt = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        pool = self.pool

        current_eval_cnt = int(pool.eval_cnt)
        if current_eval_cnt >= self._last_pool_eval_cnt:
            self._global_eval_cnt += current_eval_cnt - self._last_pool_eval_cnt
        self._last_pool_eval_cnt = current_eval_cnt

        sig_count = int((np.abs(pool.weights[:pool.size]) > 1e-4).sum())
        self.logger.record('pool/size', pool.size)
        self.logger.record('pool/significant', sig_count)
        self.logger.record('pool/best_ic_ret', pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', pool.eval_cnt)

        valid_ic, valid_rank_ic = pool.test_ensemble(self.valid_calculator)
        self.logger.record('valid/ic', valid_ic)
        self.logger.record('valid/rank_ic', valid_rank_ic)

        n_days = sum(calc.data.n_days for calc in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        test_results = []
        for i, test_calc in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = pool.test_ensemble(test_calc)
            test_results.append((ic_test, rank_ic_test))
            if n_days > 0:
                ic_test_mean += ic_test * test_calc.data.n_days / n_days
                rank_ic_test_mean += rank_ic_test * test_calc.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record('test/ic_mean', ic_test_mean)
        self.logger.record('test/rank_ic_mean', rank_ic_test_mean)

        if self.conv_logger is not None:
            self.conv_logger.record_step(
                timestep=self.num_timesteps,
                pool_size=pool.size,
                pool_significant=sig_count,
                pool_best_ic=pool.best_ic_ret,
                pool_eval_cnt=pool.eval_cnt,
                global_eval_cnt=self._global_eval_cnt,
                train_ic=pool.best_ic_ret,
                valid_ic=valid_ic,
                valid_rank_ic=valid_rank_ic,
                test_results=test_results,
            )
            self.conv_logger.save_csv()
            if self._rollout_count % self._plot_interval == 0:
                try:
                    csv_path = os.path.join(self.conv_logger.save_dir, "convergence.csv")
                    plot_convergence(csv_path)
                except Exception:
                    pass

        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_json_dict(), f)

    @property
    def pool(self) -> LinearAlphaPool:
        assert isinstance(self.env_core.pool, LinearAlphaPool)
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        return env


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
    """Generate walk-forward window list."""
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
    features: List[TickFeatureType],
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
    """Train one walk-forward window. Returns dict with results."""
    wid = window["window_id"]
    print(f"\n{'='*70}")
    print(f"[Window {wid}] "
          f"train={window['train_start']}~{window['train_end']}, "
          f"valid={window['valid_start']}~{window['valid_end']}, "
          f"test={window['test_start']}~{window['test_end']}")
    print(f"{'='*70}")

    window_dir = os.path.join(save_root, f"window_{wid:03d}")
    os.makedirs(window_dir, exist_ok=True)

    with open(os.path.join(window_dir, "window_meta.json"), "w") as f:
        json.dump(window, f, indent=2)

    close = Feature(TickFeatureType.CLOSE)
    target = Ref(close, -max_future_bars) / close - 1

    def get_dataset(start: str, end: str) -> TickStockData:
        return TickStockData(
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

    calculators = [TickCalculator(d, target) for d in datasets]

    # Build pool
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

    # Warm-start from previous window's pool
    if prev_pool_path is not None and os.path.exists(prev_pool_path):
        print(f"[Window {wid}] Warm-starting from {prev_pool_path}")
        try:
            with open(prev_pool_path, "r") as f:
                prev_state = json.load(f)
            if "exprs" in prev_state and len(prev_state["exprs"]) > 0:
                from alphagen.data.parser import ExpressionParser
                from alphagen_level2.config_tick import OPERATORS as TICK_OPERATORS
                parser = ExpressionParser(
                    TICK_OPERATORS, ignore_case=True,
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

    env = TickAlphaEnv(
        pool=pool,
        use_all_features=(len(features) > 7),
        device=device,
        print_expr=True,
    )

    conv_logger = ConvergenceLogger(save_dir=window_dir)
    callback = TickRollingCallback(
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

    csv_path = conv_logger.save_csv()
    conv_logger.save_json()
    try:
        plot_convergence(csv_path)
    except Exception:
        pass

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
# Main
# ---------------------------------------------------------------------------

def main(
    seed: int = 0,
    instruments: Union[str, List[str]] = '["510300.sh"]',
    pool_capacity: int = 20,
    steps_per_window: int = 150_000,
    data_root: str = "~/EquityLevel2/stock",
    use_all_features: bool = True,
    # Bar size
    bar_size_sec: int = 3,
    # Rolling window
    global_start: str = "2022-01-01",
    global_end: str = "2026-12-31",
    train_months: int = 12,
    valid_months: int = 6,
    test_months: int = 6,
    step_months: int = 6,
    # Lookback / forward (in bars)
    max_backtrack_bars: int = 4800,
    max_future_bars: int = 1200,
    # IO
    cache_dir: Optional[str] = "./out/tick_cache",
    max_workers: int = 4,
    # Diversity
    ic_mut_threshold: float = 0.99,
    diversity_bonus: float = 0.0,
    # Warm-start
    warm_start: bool = True,
):
    """
    Walk-forward rolling-window RL alpha training on 3s tick-level data.

    20 microstructure features:
      open, high, low, close, ret, volume, turnover, vwap,
      mid, spread, spread_pct, bid_vol1, ask_vol1, total_bid, total_ask,
      imbalance_1, imbalance_total, delta_bid_vol1, delta_ask_vol1, signed_volume

    :param seed: Random seed
    :param instruments: ETF code(s) as JSON list, e.g. '["510300.sh"]'
    :param pool_capacity: Maximum alpha pool size per window
    :param steps_per_window: RL training steps per window
    :param data_root: Root path to Level 2 HDF5 data
    :param use_all_features: Use all 20 features (True) or basic 7 (False)
    :param bar_size_sec: Bar size in seconds (default 3)
    :param global_start: Earliest date for training
    :param global_end: Latest date for test
    :param train_months: Training window in months
    :param valid_months: Validation window in months
    :param test_months: Test window in months
    :param step_months: Rolling step size in months
    :param max_backtrack_bars: Max lookback in bars (4800 ≈ 1 day)
    :param max_future_bars: Max forward look in bars (1200 ≈ 1 hour)
    :param cache_dir: Cache directory
    :param max_workers: Parallel HDF5 IO threads
    :param ic_mut_threshold: Mutual IC rejection threshold
    :param diversity_bonus: Reward bonus for novel alphas
    :param warm_start: Carry forward pool across windows
    """
    reseed_everything(seed)

    if use_all_features:
        features = TICK_FEATURES  # all 20
    else:
        features = TICK_FEATURES_BASIC  # 7 basic

    if isinstance(instruments, str) and instruments.startswith("["):
        instruments = json.loads(instruments)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_root = os.path.join(
        "./out/tick_rolling",
        f"tick_{bar_size_sec}s_pool{pool_capacity}_seed{seed}_{timestamp}",
    )
    os.makedirs(save_root, exist_ok=True)

    schedule = build_rolling_schedule(
        global_start=global_start,
        global_end=global_end,
        train_months=train_months,
        valid_months=valid_months,
        test_months=test_months,
        step_months=step_months,
    )

    print(f"[Tick Rolling] {len(schedule)} windows, bar={bar_size_sec}s, "
          f"instrument={instruments}, features={len(features)}")
    for w in schedule:
        print(f"  Window {w['window_id']}: "
              f"train {w['train_start']}~{w['train_end']} | "
              f"valid {w['valid_start']}~{w['valid_end']} | "
              f"test {w['test_start']}~{w['test_end']}")

    run_config = {
        "seed": seed, "instruments": instruments, "pool_capacity": pool_capacity,
        "steps_per_window": steps_per_window, "bar_size_sec": bar_size_sec,
        "use_all_features": use_all_features, "n_features": len(features),
        "max_backtrack_bars": max_backtrack_bars, "max_future_bars": max_future_bars,
        "schedule": schedule,
    }
    with open(os.path.join(save_root, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

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

        with open(os.path.join(save_root, "rolling_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("[Tick Rolling] Walk-forward complete:")
    print(f"{'='*70}")
    print(f"{'Win':>4} {'Train':>25} {'Test':>25} {'TestIC':>9} {'TestRkIC':>9}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['window_id']:>4} "
              f"{r['train_start']}~{r['train_end']} "
              f"{r['test_start']}~{r['test_end']} "
              f"{r['test_ic']:>9.4f} {r['test_rank_ic']:>9.4f}")

    avg_ic = np.mean([r["test_ic"] for r in all_results])
    avg_ric = np.mean([r["test_rank_ic"] for r in all_results])
    print(f"\nAvg test IC:      {avg_ic:.4f}")
    print(f"Avg test Rank IC: {avg_ric:.4f}")
    print(f"Results: {save_root}")


if __name__ == "__main__":
    fire.Fire(main)

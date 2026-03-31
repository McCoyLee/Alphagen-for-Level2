"""
RL-based alpha generation using local Level 2 HDF5 data.
Show less
Bar-level mode: tick snapshots are resampled to N-minute bars (default 3min).
The time axis for expressions/rolling operators is bars, not days.
For 3min bars: 80 bars ≈ 1 trading day (240min / 3min).
Usage:
    # Basic mode (OHLCV only, 3min bars):
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock
    # Extended Level 2 features (20 features, 3min bars):
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock --use_level2_features
    # 5-minute bars:
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock --bar_size_min=5
    # Custom lookback/forward (in bars):
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock \
        --max_backtrack_bars=160 --max_future_bars=80
    # With custom stock list:
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock \
        --instruments='["000001.sz","000002.sz","000568.sz"]'
"""
import json
import os
from typing import Optional, Tuple, List, Union
from datetime import datetime
import numpy as np
import torch
import fire
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.data.expression import *
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_level2.stock_data import Level2StockData, Level2FeatureType
from alphagen_level2.calculator import Level2Calculator
from alphagen_level2.env_wrapper import Level2AlphaEnv
from alphagen_level2.config import BASIC_FEATURES, LEVEL2_FEATURES
from alphagen_level2.convergence_logger import ConvergenceLogger, plot_convergence
from alphagen_level2.diversity_pool import DiversityMseAlphaPool
class Level2Callback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        valid_calculator: Level2Calculator,
        test_calculators: List[Level2Calculator],
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
        os.makedirs(self.save_path, exist_ok=True)
    def _on_step(self) -> bool:
        return True
    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        pool = self.pool
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
        # Record convergence metrics
        if self.conv_logger is not None:
            self.conv_logger.record_step(
                timestep=self.num_timesteps,
                pool_size=pool.size,
                pool_significant=sig_count,
                pool_best_ic=pool.best_ic_ret,
                pool_eval_cnt=pool.eval_cnt,
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
        # Navigate through any wrapper chain to reach AlphaEnvCore
        while hasattr(env, 'env'):
            env = env.env
        return env
def run_single_experiment(
    seed: int = 0,
    instruments: Union[str, List[str]] = "auto",
    pool_capacity: int = 10,
    steps: int = 200_000,
    data_root: str = "~/EquityLevel2/stock",
    use_level2_features: bool = False,
    train_start: str = "2023-03-01",
    train_end: str = "2023-06-30",
    valid_start: str = "2023-7-01",
    valid_end: str = "2023-7-31",
    test_start: str = "2023-8-01",
    test_end: str = "2023-8-31",
    max_backtrack_bars: int = 80,
    max_future_bars: int = 80,
    cache_dir: Optional[str] = "./out/l2_cache",
    max_workers: int = 4,
    bar_size_min: float = 3.0,
    bar_size_sec: Optional[int] = None,
    # Multi-env parallelism
    n_envs: int = 1,
    # Diversity options
    ic_mut_threshold: float = 0.99,
    diversity_bonus: float = 0.0,
):
    reseed_everything(seed)
    features = LEVEL2_FEATURES if use_level2_features else BASIC_FEATURES
    feature_mode = "level2" if use_level2_features else "basic"
    effective_bar_size_min = (float(bar_size_sec) / 60.0) if bar_size_sec is not None else float(bar_size_min)
    print(f"""[Level2] Starting training
    Seed: {seed}
    Data root: {data_root}
    Feature mode: {feature_mode} ({len(features)} features)
    Bar size: {effective_bar_size_min:.4f} min ({effective_bar_size_min * 60:.1f} sec)
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Steps: {steps}
    Max backtrack: {max_backtrack_bars} bars ({max_backtrack_bars * effective_bar_size_min:.2f} min)
    Max future: {max_future_bars} bars ({max_future_bars * effective_bar_size_min:.2f} min)
    Train: [{train_start}, {train_end}]
    Valid: [{valid_start}, {valid_end}]
    Test:  [{test_start}, {test_end}]""")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name_prefix = f"l2_{feature_mode}_{pool_capacity}_{seed}_{timestamp}"
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Parse instrument list if provided as JSON string
    if isinstance(instruments, str) and instruments.startswith('['):
        instruments = json.loads(instruments)
    if instruments == "auto":
        instruments = "auto"  # Level2StockData will auto-discover stocks
    close = Feature(Level2FeatureType.CLOSE)
    # Forward return target: 80 bars ≈ 1 trading day for 3min bars
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
            bar_size_min=bar_size_min,
            bar_size_sec=bar_size_sec,
        )
    segments = [
        (train_start, train_end),
        (valid_start, valid_end),
        (test_start, test_end),
    ]
    print("[Level2] Loading datasets...")
    datasets = [get_dataset(*s) for s in segments]
    for i, (seg, ds) in enumerate(zip(segments, datasets)):
        print(f"  Segment {i}: {seg[0]} ~ {seg[1]}, "
              f"{ds.n_days} bars, {ds.n_stocks} stocks, {ds.n_features} features")
    calculators = [Level2Calculator(d, target) for d in datasets]
    # Pool: use DiversityMseAlphaPool when diversity features are requested
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
        print(f"  Diversity pool: ic_mut_threshold={ic_mut_threshold}, bonus={diversity_bonus}")
    else:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device,
        )
    # Environment: single or multi-env parallel
    if n_envs > 1:
        from stable_baselines3.common.vec_env import DummyVecEnv
        def make_env(print_expr: bool = False):
            def _init():
                return Level2AlphaEnv(
                    pool=pool,
                    use_level2_features=use_level2_features,
                    device=device,
                    print_expr=print_expr,
                )
            return _init
        # Only first env prints expressions to avoid log spam
        env_fns = [make_env(print_expr=True)] + [make_env(print_expr=False)] * (n_envs - 1)
        env = DummyVecEnv(env_fns)
        print(f"  Multi-env: {n_envs} parallel environments (DummyVecEnv)")
    else:
        env = Level2AlphaEnv(
            pool=pool,
            use_level2_features=use_level2_features,
            device=device,
            print_expr=True,
        )
    conv_logger = ConvergenceLogger(save_dir=save_path)
    callback = Level2Callback(
        save_path=save_path,
        valid_calculator=calculators[1],
        test_calculators=calculators[2:],
        verbose=1,
        convergence_logger=conv_logger,
    )
    # PPO: scale batch_size with n_envs
    batch_size = 128 * n_envs
    n_steps = max(2048 // n_envs, 256)
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
        batch_size=batch_size,
        n_steps=n_steps,
        tensorboard_log="./out/tensorboard",
        device=device,
        verbose=1,
    )
    print("[Level2] Starting RL training...")
    model.learn(
        total_timesteps=steps,
        callback=callback,
        tb_log_name=name_prefix,
    )
    # Save final convergence data
    csv_path = conv_logger.save_csv()
    conv_logger.save_json()
    try:
        plot_path = plot_convergence(csv_path)
        print(f"[Level2] Convergence plot: {plot_path}")
    except Exception as e:
        print(f"[Level2] Plot failed: {e}")
    summary = conv_logger.summary()
    if summary:
        print(f"[Level2] Best pool IC: {summary['best_pool_ic']:.4f} (step {summary['best_pool_ic_step']})")
        print(f"[Level2] Best test IC mean: {summary['best_test_ic_mean']:.4f} (step {summary['best_test_ic_mean_step']})")
    print(f"[Level2] Training complete. Results saved to {save_path}")
def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: Union[str, List[str]] = "auto",
    data_root: str = "~/EquityLevel2/stock",
    use_level2_features: bool = False,
    steps: Optional[int] = None,
    train_start: str = "2023-03-01",
    train_end: str = "2023-06-30",
    valid_start: str = "2023-7-01",
    valid_end: str = "2023-7-31",
    test_start: str = "2023-8-01",
    test_end: str = "2023-8-31",
    max_backtrack_bars: int = 80,
    max_future_bars: int = 80,
    cache_dir: Optional[str] = "./out/l2_cache",
    max_workers: int = 4,
    bar_size_min: float = 3.0,
    bar_size_sec: Optional[int] = None,
    n_envs: int = 1,
    ic_mut_threshold: float = 0.99,
    diversity_bonus: float = 0.0,
):
    """
    Train alpha factors using Level 2 local HDF5 data (bar-level).
    :param random_seeds: Random seed(s)
    :param pool_capacity: Maximum alpha pool size
    :param instruments: Stock list (JSON array string) or "auto" for auto-discovery
    :param data_root: Root path to Level 2 HDF5 data
    :param use_level2_features: Use extended Level 2 features (20) vs basic OHLCV (6)
    :param steps: Total RL iteration steps (None = auto based on pool_capacity)
    :param max_backtrack_bars: Max lookback in bars (80 ≈ 1 day for 3min bars)
    :param max_future_bars: Max forward look in bars (80 ≈ 1 day for 3min bars)
    :param cache_dir: Directory for caching aggregated data (None to disable)
    :param max_workers: Number of threads for parallel HDF5 IO
    :param bar_size_min: Bar size in minutes (default 3)
    :param bar_size_sec: Bar size in seconds; if set, overrides bar_size_min (e.g. 3 means 3-second bars)
    :param n_envs: Number of parallel environments (1=single, 4-8=recommended)
    :param ic_mut_threshold: Mutual IC threshold to reject correlated alphas (0.7=strict, 0.99=permissive)
    :param diversity_bonus: Reward bonus for novel alphas (0=off, 0.05-0.2=typical)
    """
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds,)
    default_steps = {10: 200_000, 20: 250_000, 50: 300_000, 100: 350_000}
    for s in random_seeds:
        actual_steps = default_steps.get(int(pool_capacity), 250_000) if steps is None else int(steps)
        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=pool_capacity,
            steps=actual_steps,
            data_root=data_root,
            use_level2_features=use_level2_features,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            max_backtrack_bars=max_backtrack_bars,
            max_future_bars=max_future_bars,
            cache_dir=cache_dir,
            max_workers=max_workers,
            bar_size_min=bar_size_min,
            bar_size_sec=bar_size_sec,
            n_envs=n_envs,
            ic_mut_threshold=ic_mut_threshold,
            diversity_bonus=diversity_bonus,
        )
if __name__ == '__main__':
    fire.Fire(main)

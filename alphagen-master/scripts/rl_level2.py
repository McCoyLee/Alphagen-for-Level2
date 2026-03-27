"""
RL-based alpha generation using local Level 2 HDF5 data.

Usage:
    # Basic mode (OHLCV only, from Level 2 data):
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock

    # Extended Level 2 features mode:
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock --use_level2_features

    # With custom stock list:
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock \
        --instruments='["000001.sz","000002.sz","000568.sz"]'

    # With custom date ranges:
    python scripts/rl_level2.py --data_root=~/EquityLevel2/stock \
        --train_start=2022-01-01 --train_end=2022-06-30 \
        --valid_start=2022-07-01 --valid_end=2022-09-30 \
        --test_start=2022-10-01 --test_end=2022-12-31
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


class Level2Callback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        test_calculators: List[Level2Calculator],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)

        n_days = sum(calc.data.n_days for calc in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calc in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calc)
            ic_test_mean += ic_test * test_calc.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calc.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record('test/ic_mean', ic_test_mean)
        self.logger.record('test/rank_ic_mean', rank_ic_test_mean)
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
        return self.training_env.envs[0].unwrapped


def run_single_experiment(
    seed: int = 0,
    instruments: Union[str, List[str]] = "auto",
    pool_capacity: int = 10,
    steps: int = 200_000,
    data_root: str = "~/EquityLevel2/stock",
    use_level2_features: bool = False,
    train_start: str = "2022-01-05",
    train_end: str = "2022-09-30",
    valid_start: str = "2022-10-01",
    valid_end: str = "2022-11-30",
    test_start: str = "2022-12-01",
    test_end: str = "2022-12-31",
    max_backtrack_days: int = 100,
    max_future_days: int = 30,
    cache_dir: Optional[str] = "./out/l2_cache",
    max_workers: int = 4,
):
    reseed_everything(seed)

    features = LEVEL2_FEATURES if use_level2_features else BASIC_FEATURES
    feature_mode = "level2" if use_level2_features else "basic"

    print(f"""[Level2] Starting training
    Seed: {seed}
    Data root: {data_root}
    Feature mode: {feature_mode} ({len(features)} features)
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Steps: {steps}
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
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> Level2StockData:
        return Level2StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            max_backtrack_days=max_backtrack_days,
            max_future_days=max_future_days,
            features=features,
            device=device,
            data_root=data_root,
            cache_dir=cache_dir,
            max_workers=max_workers,
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
              f"{ds.n_days} days, {ds.n_stocks} stocks, {ds.n_features} features")

    calculators = [Level2Calculator(d, target) for d in datasets]

    pool = MseAlphaPool(
        capacity=pool_capacity,
        calculator=calculators[0],
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=device,
    )

    env = Level2AlphaEnv(
        pool=pool,
        use_level2_features=use_level2_features,
        device=device,
        print_expr=True,
    )

    callback = Level2Callback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
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
    print(f"[Level2] Training complete. Results saved to {save_path}")


def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: Union[str, List[str]] = "auto",
    data_root: str = "~/EquityLevel2/stock",
    use_level2_features: bool = False,
    steps: Optional[int] = None,
    train_start: str = "2022-01-05",
    train_end: str = "2022-09-30",
    valid_start: str = "2022-10-01",
    valid_end: str = "2022-11-30",
    test_start: str = "2022-12-01",
    test_end: str = "2022-12-31",
    max_backtrack_days: int = 100,
    max_future_days: int = 30,
    cache_dir: Optional[str] = "./out/l2_cache",
    max_workers: int = 4,
):
    """
    Train alpha factors using Level 2 local HDF5 data.

    :param random_seeds: Random seed(s)
    :param pool_capacity: Maximum alpha pool size
    :param instruments: Stock list (JSON array string) or "auto" for auto-discovery
    :param data_root: Root path to Level 2 HDF5 data
    :param use_level2_features: Use extended Level 2 features (20) vs basic OHLCV (6)
    :param steps: Total RL iteration steps (None = auto based on pool_capacity)
    :param cache_dir: Directory for caching aggregated data (None to disable)
    :param max_workers: Number of threads for parallel HDF5 IO
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
            max_backtrack_days=max_backtrack_days,
            max_future_days=max_future_days,
            cache_dir=cache_dir,
            max_workers=max_workers,
        )


if __name__ == '__main__':
    fire.Fire(main)

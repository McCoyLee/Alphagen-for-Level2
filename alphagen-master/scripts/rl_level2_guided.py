"""
RL training with action prior guidance for Level 2 alpha generation.

Uses a pre-trained ActionPriorTransformer to shape rewards during RL,
guiding the agent toward action patterns seen in successful historical alphas.

Workflow:
    1. Train prior: python scripts/train_action_prior.py --result_dirs=...
    2. Guided RL:   python scripts/rl_level2_guided.py --prior_path=./out/action_prior.pt

Usage:
    # Guided RL (prior + reward shaping):
    python scripts/rl_level2_guided.py \
        --data_root=~/EquityLevel2/stock \
        --prior_path=./out/action_prior.pt \
        --prior_beta=0.1

    # With decay (prior influence diminishes over training):
    python scripts/rl_level2_guided.py \
        --data_root=~/EquityLevel2/stock \
        --prior_path=./out/action_prior.pt \
        --prior_beta=0.2 \
        --prior_decay_rate=1e-5

    # With warmup (no prior for first N steps, pure exploration):
    python scripts/rl_level2_guided.py \
        --data_root=~/EquityLevel2/stock \
        --prior_path=./out/action_prior.pt \
        --prior_warmup_steps=10000

    # No prior (fallback to plain RL, same as rl_level2.py):
    python scripts/rl_level2_guided.py --data_root=~/EquityLevel2/stock
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
from alphagen.utils import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore

from alphagen_level2.stock_data import Level2StockData, Level2FeatureType
from alphagen_level2.calculator import Level2Calculator
from alphagen_level2.env_wrapper import Level2AlphaEnv
from alphagen_level2.config import BASIC_FEATURES, LEVEL2_FEATURES
from alphagen_level2.convergence_logger import ConvergenceLogger, plot_convergence
from alphagen_level2.action_prior import (
    ActionPriorTransformer,
    GuidedLevel2EnvWrapper,
)


class GuidedCallback(BaseCallback):
    """Callback with convergence logging for guided RL training."""

    def __init__(
        self,
        save_path: str,
        test_calculators: List[Level2Calculator],
        convergence_logger: ConvergenceLogger,
        verbose: int = 0,
        plot_interval: int = 10,
    ):
        super().__init__(verbose)
        self.save_path = save_path
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

        self.logger.record("pool/size", pool.size)
        self.logger.record("pool/significant", sig_count)
        self.logger.record("pool/best_ic_ret", pool.best_ic_ret)
        self.logger.record("pool/eval_cnt", pool.eval_cnt)

        n_days = sum(calc.data.n_days for calc in self.test_calculators)
        test_results = []
        ic_mean, ric_mean = 0.0, 0.0
        for i, test_calc in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = pool.test_ensemble(test_calc)
            test_results.append((ic_test, rank_ic_test))
            ic_mean += ic_test * test_calc.data.n_days / n_days
            ric_mean += rank_ic_test * test_calc.data.n_days / n_days
            self.logger.record(f"test/ic_{i}", ic_test)
            self.logger.record(f"test/rank_ic_{i}", rank_ic_test)
        self.logger.record("test/ic_mean", ic_mean)
        self.logger.record("test/rank_ic_mean", ric_mean)

        self.conv_logger.record_step(
            timestep=self.num_timesteps,
            pool_size=pool.size,
            pool_significant=sig_count,
            pool_best_ic=pool.best_ic_ret,
            pool_eval_cnt=pool.eval_cnt,
            train_ic=pool.best_ic_ret,
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
        path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
        self.model.save(path)
        with open(f"{path}_pool.json", "w") as f:
            json.dump(self.pool.to_json_dict(), f)

    @property
    def pool(self) -> LinearAlphaPool:
        assert isinstance(self.env_core.pool, LinearAlphaPool)
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        # Navigate through GuidedLevel2EnvWrapper → Level2EnvWrapper → AlphaEnvCore
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
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
    valid_start: str = "2023-07-01",
    valid_end: str = "2023-07-31",
    test_start: str = "2023-08-01",
    test_end: str = "2023-08-31",
    max_backtrack_bars: int = 80,
    max_future_bars: int = 80,
    cache_dir: Optional[str] = "./out/l2_cache",
    max_workers: int = 4,
    bar_size_min: int = 3,
    # Prior guidance options
    prior_path: Optional[str] = None,
    prior_beta: float = 0.1,
    prior_temperature: float = 1.0,
    prior_warmup_steps: int = 0,
    prior_decay_rate: float = 0.0,
    plot_interval: int = 10,
):
    reseed_everything(seed)
    features = LEVEL2_FEATURES if use_level2_features else BASIC_FEATURES
    feature_mode = "level2" if use_level2_features else "basic"
    guided = prior_path is not None

    tag = "guided" if guided else "rl"
    print(f"""[Level2+Prior] Starting training
    Seed: {seed}
    Feature mode: {feature_mode} ({len(features)} features)
    Pool capacity: {pool_capacity}, Steps: {steps}
    Prior guided: {guided}
    Prior path: {prior_path}
    Beta: {prior_beta}, Temperature: {prior_temperature}
    Warmup: {prior_warmup_steps}, Decay: {prior_decay_rate}""")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name_prefix = f"l2_{feature_mode}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(instruments, str) and instruments.startswith("["):
        instruments = json.loads(instruments)
    if instruments == "auto":
        instruments = "auto"

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
            bar_size_min=bar_size_min,
        )

    segments = [
        (train_start, train_end),
        (valid_start, valid_end),
        (test_start, test_end),
    ]
    print("[Level2+Prior] Loading datasets...")
    datasets = [get_dataset(*s) for s in segments]
    calculators = [Level2Calculator(d, target) for d in datasets]

    pool = MseAlphaPool(
        capacity=pool_capacity,
        calculator=calculators[0],
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=device,
    )

    # Create base env
    base_env = Level2AlphaEnv(
        pool=pool,
        use_level2_features=use_level2_features,
        device=device,
        print_expr=True,
    )

    # Load prior and wrap env
    if guided:
        print(f"[Level2+Prior] Loading prior from {prior_path}")
        prior_net = ActionPriorTransformer.load(prior_path, device=device)
        prior_net.eval()
        env = GuidedLevel2EnvWrapper(
            env=base_env,
            prior_net=prior_net,
            beta=prior_beta,
            temperature=prior_temperature,
            warmup_steps=prior_warmup_steps,
            decay_rate=prior_decay_rate,
        )
        print(f"[Level2+Prior] Prior loaded, beta={prior_beta}")
    else:
        env = base_env

    conv_logger = ConvergenceLogger(save_dir=save_path)
    callback = GuidedCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        convergence_logger=conv_logger,
        verbose=1,
        plot_interval=plot_interval,
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
        gamma=1.0,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log="./out/tensorboard",
        device=device,
        verbose=1,
    )

    print("[Level2+Prior] Starting RL training...")
    model.learn(
        total_timesteps=steps,
        callback=callback,
        tb_log_name=name_prefix,
    )

    csv_path = conv_logger.save_csv()
    conv_logger.save_json()
    try:
        plot_convergence(csv_path)
    except Exception:
        pass

    summary = conv_logger.summary()
    if summary:
        print(f"[Level2+Prior] Best pool IC: {summary['best_pool_ic']:.4f} "
              f"(step {summary['best_pool_ic_step']})")
        print(f"[Level2+Prior] Best test IC: {summary['best_test_ic_mean']:.4f} "
              f"(step {summary['best_test_ic_mean_step']})")
    print(f"[Level2+Prior] Results: {save_path}")


def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: Union[str, List[str]] = "auto",
    data_root: str = "~/EquityLevel2/stock",
    use_level2_features: bool = False,
    steps: Optional[int] = None,
    train_start: str = "2023-03-01",
    train_end: str = "2023-06-30",
    valid_start: str = "2023-07-01",
    valid_end: str = "2023-07-31",
    test_start: str = "2023-08-01",
    test_end: str = "2023-08-31",
    max_backtrack_bars: int = 80,
    max_future_bars: int = 80,
    cache_dir: Optional[str] = "./out/l2_cache",
    max_workers: int = 4,
    bar_size_min: int = 3,
    # Prior
    prior_path: Optional[str] = None,
    prior_beta: float = 0.1,
    prior_temperature: float = 1.0,
    prior_warmup_steps: int = 0,
    prior_decay_rate: float = 0.0,
    plot_interval: int = 10,
):
    """
    Guided RL training with optional action prior.

    :param prior_path: path to trained ActionPriorTransformer (.pt file)
    :param prior_beta: reward shaping coefficient (0=no guidance, 0.05-0.2=typical)
    :param prior_temperature: softmax temperature for prior (1.0=default)
    :param prior_warmup_steps: disable prior for first N steps
    :param prior_decay_rate: exponential decay for beta (0=no decay)
    """
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds,)
    default_steps = {10: 200_000, 20: 250_000, 50: 300_000, 100: 350_000}
    for s in random_seeds:
        actual_steps = (
            default_steps.get(int(pool_capacity), 250_000)
            if steps is None
            else int(steps)
        )
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
            prior_path=prior_path,
            prior_beta=prior_beta,
            prior_temperature=prior_temperature,
            prior_warmup_steps=prior_warmup_steps,
            prior_decay_rate=prior_decay_rate,
            plot_interval=plot_interval,
        )


if __name__ == "__main__":
    fire.Fire(main)

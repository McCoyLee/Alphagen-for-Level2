"""
RL training with random-window episodic sampling and GP factor-pool evolution.

Spec
----
* Each episode draws a random 1200-bar slice of the full 3-year dataset.
* The model generates / refines a factor whose reward is the risk-adjusted
  PnL over the next ~100 bars (~5 minutes).
* The factor pool (default capacity = 20) is initialised by LLM, mined by RL
  during training, and periodically evolved by GP (every N epochs).

Usage
-----
    python scripts/rl_random_sampling.py \\
        --data_root=~/EquityLevel2/stock \\
        --instruments='["510300.sh"]' \\
        --start_time=2023-01-01 --end_time=2025-12-31 \\
        --total_steps=300000 --gp_enabled=True --gp_every_n_epochs=20

Disable GP with ``--gp_enabled=False``.  Disable LLM init with
``--llm_warmstart=False`` (pre-built seed expressions are then used).
"""

from __future__ import annotations
import json
import os
import random
from typing import List, Optional, Union

import fire
import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import Feature, Ref, Expression, Greater, Less, Sub
from alphagen.data.parser import ExpressionParser
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen_level2.calculator_tick import TickCalculator
from alphagen_level2.config_random import (
    WINDOW_BARS, FUTURE_BARS, EXECUTION_DELAY, LOOKBACK_BARS,
    EPISODE_HISTORY_BUFFER, POOL_CAPACITY, IC_MUT_THRESHOLD,
    DECORRELATION_BONUS, SORTINO_WEIGHT, SHARPE_WEIGHT, IC_WEIGHT,
    TURNOVER_COST, MAX_DRAWDOWN_THRESHOLD, MAX_DRAWDOWN_PENALTY,
    FAT_TAIL_THRESHOLD, FAT_TAIL_PENALTY, KURTOSIS_THRESHOLD, KURTOSIS_PENALTY,
    COMPLEXITY_PENALTY, GP_ENABLED, GP_EVERY_N_EPOCHS, GP_OFFSPRING,
    GP_CROSSOVER_RATE, GP_MUTATION_RATE, GP_MAX_TRIES, GP_TOURNAMENT_K,
    GP_REPLACE_WORST_N, SCORE_WINDOWS,
)
from alphagen_level2.config_tick import (
    OPERATORS as TICK_OPERATORS, DELTA_TIMES, CONSTANTS,
    TICK_FEATURES, TICK_FEATURES_BASIC,
)
from alphagen_level2.env_wrapper_tick import TickAlphaEnv
from alphagen_level2.gp_evolution import (
    GPGrammar, evolve_pool, expression_size,
)
from alphagen_level2.llm_prompts_tick import get_tick_system_prompt
from alphagen_level2.random_metrics import (
    RandomSamplingMetricsLogger, plot_random_sampling_curves,
)
from alphagen_level2.random_pool import RandomWindowSingleFactorPool
from alphagen_level2.random_window import RandomWindowSampler
from alphagen_level2.risk_reward import RewardWeights
from alphagen_level2.stock_data_tick import TickStockData, TickFeatureType
from alphagen_llm.client import OpenAIClient, ChatConfig
from alphagen_llm.prompts.interaction import DefaultInteraction


# ---------------------------------------------------------------------------
# Parser / LLM helpers (mirrors rl_tick_rolling.py for consistency)
# ---------------------------------------------------------------------------

def build_tick_parser() -> ExpressionParser:
    parser = ExpressionParser(
        TICK_OPERATORS,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater], "Min": [Less], "Delta": [Sub],
        },
    )
    parser._features = {f.name.lower(): f for f in TickFeatureType}
    return parser


def build_chat_client(log_dir: str, base_url: str, api_key: str, model: str):
    from openai import OpenAI
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url=base_url, api_key=api_key),
        config=ChatConfig(system_prompt=get_tick_system_prompt(), logger=logger),
        model=model,
    )


_FALLBACK_SEEDS = [
    "Div(Sub($vwap,$close),Std($close,600))",
    "EMA($imbalance_1,20)",
    "Div(Sum($signed_volume,20),Sum($volume,20))",
    "Corr($delta_bid_vol1,$ret,100)",
    "Sub(EMA($spread_pct,20),EMA($spread_pct,600))",
    "Mul($imbalance_total,Div($volume,Mean($volume,1200)))",
    "Sub(Mean($mid,100),Mean($mid,1200))",
    "Div(Std($ret,100),Std($ret,1200))",
    "Corr($volume,$spread_pct,600)",
    "Sub($high,$low)",
]


def _load_seed_pool(parser: ExpressionParser) -> List[Expression]:
    out = []
    for s in _FALLBACK_SEEDS:
        try:
            out.append(parser.parse(s))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# GP evolution callback
# ---------------------------------------------------------------------------

class GPCallback(BaseCallback):
    """Runs GP evolution on the pool every ``gp_every_n_epochs`` rollouts.

    "Epoch" here means a PPO rollout (``n_steps`` env steps); SB3 calls
    ``_on_rollout_end`` once per rollout.
    """

    def __init__(
        self,
        pool: RandomWindowSingleFactorPool,
        grammar: GPGrammar,
        log_path: str,
        gp_every_n_epochs: int = 20,
        n_offspring: int = 10,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.4,
        tournament_k: int = 3,
        replace_worst_n: int = 5,
        max_tries: int = 50,
        seed: int = 0,
        enabled: bool = True,
        metrics_logger: Optional[RandomSamplingMetricsLogger] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.pool = pool
        self.grammar = grammar
        self.log_path = log_path
        self.gp_every = max(1, int(gp_every_n_epochs))
        self.n_offspring = n_offspring
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.replace_worst_n = replace_worst_n
        self.max_tries = max_tries
        self.rng = random.Random(seed)
        self.enabled = bool(enabled)
        self.metrics_logger = metrics_logger
        self._epoch = 0
        self._history = []
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._epoch += 1
        if not self.enabled:
            return
        if self._epoch % self.gp_every != 0:
            return
        if self.pool.size < 2:
            return
        exprs = list(self.pool.exprs[:self.pool.size])
        scores = list(self.pool._composite_scores[:self.pool.size])
        before = len(exprs)
        new_exprs, new_scores, accepted = evolve_pool(
            pool_exprs=exprs, pool_scores=scores,
            grammar=self.grammar,
            score_fn=self.pool.score_candidate,
            n_offspring=self.n_offspring,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            tournament_k=self.tournament_k,
            replace_worst_n=self.replace_worst_n,
            max_tries_per_offspring=self.max_tries,
            rng=self.rng,
        )
        # Push accepted offspring through pool.try_new_expr so all
        # bookkeeping (mutual ICs, weights, dedup) updates correctly.
        for child, _score in accepted:
            try:
                self.pool.try_new_expr(child)
            except Exception:
                continue
        record = {
            "epoch": self._epoch,
            "pool_size_before": before,
            "pool_size_after": self.pool.size,
            "n_accepted": len(accepted),
            "accepted": [(str(e), float(s)) for e, s in accepted],
        }
        self._history.append(record)
        with open(self.log_path, "w") as f:
            json.dump(self._history, f, indent=2)
        if self.metrics_logger is not None:
            self.metrics_logger.note_gp_accepted(len(accepted))
        if self.verbose:
            print(f"[GP] epoch={self._epoch} accepted={len(accepted)} "
                  f"pool_size={self.pool.size}")


# ---------------------------------------------------------------------------
# Pool snapshot / logging callback
# ---------------------------------------------------------------------------

class PoolSnapshotCallback(BaseCallback):
    """Snapshot pool JSON + record per-rollout metrics row.

    Pulls the latest reward seen by the env (from ``pool.last_reward_info``)
    so the metrics logger can build a recent-reward distribution.  Pool
    JSON is rewritten every ``every_n_rollouts`` rollouts; metrics are
    recorded *every* rollout so the convergence plot is dense.
    """

    def __init__(
        self,
        pool: RandomWindowSingleFactorPool,
        snapshot_path: str,
        metrics_logger: RandomSamplingMetricsLogger,
        every_n_rollouts: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.pool = pool
        self.snapshot_path = snapshot_path
        self.metrics = metrics_logger
        self.every = max(1, int(every_n_rollouts))
        self._epoch = 0
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)

    def _on_step(self) -> bool:
        # Capture the latest reward observed by the env so we can build a
        # rolling reward distribution.  The env appends 0 for non-terminal
        # steps; we only forward the latest non-zero terminal reward.
        info = getattr(self.pool, "last_reward_info", None)
        if info and "reward" in info and info.get("n_windows", 0) > 0:
            self.metrics.note_recent_reward(float(info["reward"]))
        return True

    def _on_rollout_end(self) -> None:
        self._epoch += 1
        # Always record metrics for a dense plot.
        try:
            self.metrics.record(int(self.num_timesteps), self.pool)
        except Exception as exc:
            if self.verbose:
                print(f"[metrics] record failed: {exc}")
        if self._epoch % self.every != 0:
            return
        try:
            with open(self.snapshot_path, "w") as f:
                json.dump(self.pool.to_json_dict(), f, indent=2)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_root: str = "~/EquityLevel2/stock",
    instruments: Union[str, List[str]] = "510300.sh",
    start_time: str = "2023-01-01",
    end_time: str = "2025-12-31",
    bar_size_sec: int = 3,
    use_all_features: bool = True,
    # ``cache_dir`` mirrors rl_tick_rolling.py — TickStockData pickles its
    # parsed tensor here so subsequent runs skip HDF5 -> 3s-bar resampling.
    # Pass ``--cache_dir=None`` to disable.
    cache_dir: Optional[str] = "./out_random/tick_cache",
    save_root: str = "./out_random",
    seed: int = 0,
    device: str = "cuda:0",
    # Window
    window_bars: int = WINDOW_BARS,
    future_bars: int = FUTURE_BARS,
    execution_delay: int = EXECUTION_DELAY,
    lookback_bars: int = LOOKBACK_BARS,
    history_buffer: int = EPISODE_HISTORY_BUFFER,
    # Pool
    pool_capacity: int = POOL_CAPACITY,
    ic_mut_threshold: float = IC_MUT_THRESHOLD,
    score_windows: int = SCORE_WINDOWS,
    # Reward
    sortino_weight: float = SORTINO_WEIGHT,
    sharpe_weight: float = SHARPE_WEIGHT,
    ic_weight: float = IC_WEIGHT,
    turnover_cost: float = TURNOVER_COST,
    max_dd_threshold: float = MAX_DRAWDOWN_THRESHOLD,
    max_dd_penalty: float = MAX_DRAWDOWN_PENALTY,
    fat_tail_threshold: float = FAT_TAIL_THRESHOLD,
    fat_tail_penalty: float = FAT_TAIL_PENALTY,
    kurt_threshold: float = KURTOSIS_THRESHOLD,
    kurt_penalty: float = KURTOSIS_PENALTY,
    decorrelation_bonus: float = DECORRELATION_BONUS,
    complexity_penalty: float = COMPLEXITY_PENALTY,
    # GP
    gp_enabled: bool = GP_ENABLED,
    gp_every_n_epochs: int = GP_EVERY_N_EPOCHS,
    gp_offspring: int = GP_OFFSPRING,
    gp_crossover_rate: float = GP_CROSSOVER_RATE,
    gp_mutation_rate: float = GP_MUTATION_RATE,
    gp_max_tries: int = GP_MAX_TRIES,
    gp_tournament_k: int = GP_TOURNAMENT_K,
    gp_replace_worst_n: int = GP_REPLACE_WORST_N,
    # LLM
    llm_warmstart: bool = True,
    llm_init_updates: int = 4,
    llm_base_url: str = "http://10.2.1.205:8796/v1",
    llm_api_key: str = "sk-GEmM5YHREocL6mOOVEOUQ0Rs0qgWoB_KjJ-fSZUYd30",
    llm_model: str = "MiniMax-M2.5",
    # PPO
    total_steps: int = 300_000,
    n_steps: int = 2048,
    ppo_lr: float = 1e-4,
    batch_size: int = 256,
    gamma: float = 1.0,
    ent_coef: float = 0.01,
):
    reseed_everything(seed)
    os.makedirs(save_root, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    features = TICK_FEATURES if use_all_features else TICK_FEATURES_BASIC
    print(f"[data] Loading {start_time}~{end_time} from {data_root}...")
    full_data = TickStockData(
        instrument=instruments,
        start_time=start_time,
        end_time=end_time,
        max_backtrack_days=max(history_buffer, 4800),
        max_future_days=max(future_bars * 2, 4800),
        features=features,
        device=device_t,
        data_root=data_root,
        cache_dir=cache_dir,
        max_workers=4,
        bar_size_sec=bar_size_sec,
    )
    print(f"[data] total bars={full_data.data.shape[0]} "
          f"usable={full_data.n_days} stocks={full_data.n_stocks}")

    # ---- Sampler / target / weights ----
    sampler = RandomWindowSampler(
        full_data,
        window_bars=window_bars,
        future_bars=future_bars,
        history_buffer=history_buffer,
        execution_delay=execution_delay,
        seed=seed,
    )
    mid_prc = Feature(TickFeatureType.MID)
    target_expr = Ref(mid_prc, -future_bars) / mid_prc - 1

    weights = RewardWeights(
        sortino=sortino_weight, sharpe=sharpe_weight, ic=ic_weight,
        max_dd_threshold=max_dd_threshold, max_dd_penalty=max_dd_penalty,
        fat_tail_threshold=fat_tail_threshold, fat_tail_penalty=fat_tail_penalty,
        kurt_threshold=kurt_threshold, kurt_penalty=kurt_penalty,
        decorrelation_bonus=decorrelation_bonus,
        complexity_penalty=complexity_penalty,
    )

    # ---- Pool ----
    pool = RandomWindowSingleFactorPool(
        capacity=pool_capacity,
        sampler=sampler,
        target_expr=target_expr,
        score_windows=score_windows,
        reward_weights=weights,
        ic_mut_threshold=ic_mut_threshold,
        device=device_t,
        holding_bars=future_bars,
        execution_delay=execution_delay,
        lookback_bars=lookback_bars,
        turnover_cost=turnover_cost,
        complexity_metric=expression_size,
    )

    # ---- LLM init ----
    parser = build_tick_parser()
    if llm_warmstart:
        print("[llm] Initialising factor pool via LLM...")
        try:
            chat_client = build_chat_client(
                save_root, llm_base_url, llm_api_key, llm_model,
            )
            anchor_view = sampler.sample()
            anchor_calc = TickCalculator(
                anchor_view, target=target_expr,
                holding_bars=future_bars,
                execution_delay=execution_delay,
                lookback_bars=lookback_bars,
            )
            inter = DefaultInteraction(
                parser, chat_client,
                pool_factory=lambda exprs: _refresh_pool(pool, exprs),
                calculator_train=anchor_calc,
                calculators_test=[anchor_calc],
                replace_k=3, forgetful=False,
            )
            inter.run(n_updates=llm_init_updates)
        except Exception as exc:
            print(f"[llm] warmstart failed ({type(exc).__name__}: {exc}); "
                  f"falling back to seed expressions.")

    if pool.size < 5:
        seeds = _load_seed_pool(parser)
        print(f"[init] Pool size {pool.size} < 5, loading {len(seeds)} seed exprs.")
        pool.force_load_exprs(seeds)
    print(f"[init] Pool size after init: {pool.size}")

    # ---- Env / PPO ----
    env = TickAlphaEnv(
        pool=pool, use_all_features=use_all_features,
        device=device_t, print_expr=False,
    )

    grammar = GPGrammar(
        operators=TICK_OPERATORS,
        features=features,
        constants=CONSTANTS,
        delta_times=DELTA_TIMES,
        max_depth=4,
    )
    metrics_logger = RandomSamplingMetricsLogger(
        save_path=os.path.join(save_root, "convergence.csv"),
        recent_window=256,
    )
    gp_cb = GPCallback(
        pool=pool, grammar=grammar,
        log_path=os.path.join(save_root, "gp_history.json"),
        gp_every_n_epochs=gp_every_n_epochs,
        n_offspring=gp_offspring,
        crossover_rate=gp_crossover_rate,
        mutation_rate=gp_mutation_rate,
        tournament_k=gp_tournament_k,
        replace_worst_n=gp_replace_worst_n,
        max_tries=gp_max_tries,
        seed=seed,
        enabled=gp_enabled,
        metrics_logger=metrics_logger,
    )
    snap_cb = PoolSnapshotCallback(
        pool=pool,
        snapshot_path=os.path.join(save_root, "pool_latest.json"),
        metrics_logger=metrics_logger,
        every_n_rollouts=5,
    )

    name_prefix = ("MaskablePPO_random_" + str(instruments)
                   .replace("[", "").replace("]", "").replace(" ", ""))
    model = MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2, d_model=128,
                dropout=0.1, device=device_t,
            ),
        ),
        gamma=gamma, ent_coef=ent_coef, batch_size=batch_size,
        n_steps=n_steps, learning_rate=ppo_lr, device=device_t, verbose=1,
        tensorboard_log=os.path.join(save_root, "tb"),
    )
    model.learn(
        total_timesteps=total_steps,
        callback=[gp_cb, snap_cb],
        tb_log_name=name_prefix,
    )

    # ---- Final dump ----
    with open(os.path.join(save_root, "pool_final.json"), "w") as f:
        json.dump(pool.to_json_dict(), f, indent=2)
    pool.dump_reward_stats(os.path.join(save_root, "reward_stats.json"))
    # Render the convergence figure (pool_best vs pool_avg curves + GP /
    # eval / recent-reward panels).
    try:
        png_path = plot_random_sampling_curves(
            csv_path=os.path.join(save_root, "convergence.csv"),
            output_path=os.path.join(save_root, "convergence.png"),
        )
        print(f"[plot] convergence figure -> {png_path}")
    except Exception as exc:
        print(f"[plot] failed: {exc}")
    print(f"[done] Pool size={pool.size}, best_obj={pool.best_obj:.4f}")


def _refresh_pool(
    pool: RandomWindowSingleFactorPool, exprs: List[Expression],
) -> RandomWindowSingleFactorPool:
    """Reset the pool's contents to ``exprs`` (used by ``DefaultInteraction``)."""
    # Clear current pool state
    pool.size = 0
    for i in range(len(pool.exprs)):
        pool.exprs[i] = None
    pool._weights[:] = 0.0
    pool._composite_scores[:] = 0.0
    pool._factor_directions[:] = 1.0
    pool._mutual_ics[:] = np.identity(pool.capacity + 1)
    pool.single_ics[:] = 0.0
    pool.best_obj = -1.0
    pool.best_ic_ret = 0.0
    pool._failure_cache = set()
    if exprs:
        pool.force_load_exprs(exprs)
    return pool


if __name__ == "__main__":
    fire.Fire(main)

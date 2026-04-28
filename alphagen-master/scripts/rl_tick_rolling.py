"""
Rolling-window (walk-forward) RL alpha training on 3-second bar data.

Designed for:
  - 3-second bars (bar_size_sec=3)
  - Single ETF instrument (e.g. 510300.sh)
  - Time range 2023-2025
  - 20 microstructure features
  - Walk-forward windows: 6 months train, 2 months valid, 2 months test

Default schedule:
  Window 0: train 2023-01-01~2023-06-30, valid 2023-07-01~2023-08-31, test 2023-09-01~2023-10-31
  Window 1: train 2023-07-01~2023-12-31, valid 2024-01-01~2024-02-29, test 2024-03-01~2024-04-30
  ...rolling forward in 6-month steps until train 2025-01~2025-06...

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
        --train_months=6 --valid_months=2 --test_months=2 --step_months=6
"""

import json
import os
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Union, Tuple, Dict
from datetime import datetime
from collections import deque, defaultdict
from dateutil.relativedelta import relativedelta

import numpy as np
import torch
import fire
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import Feature, Ref, Expression, Greater, Less, Sub
from alphagen.data.parser import ExpressionParser
from alphagen.models import linear_alpha_pool as _linear_alpha_pool
LinearAlphaPool = _linear_alpha_pool.LinearAlphaPool
MseAlphaPool = _linear_alpha_pool.MseAlphaPool
SingleFactorAlphaPool = getattr(_linear_alpha_pool, "SingleFactorAlphaPool", MseAlphaPool)
HAS_SINGLE_FACTOR_POOL = hasattr(_linear_alpha_pool, "SingleFactorAlphaPool")
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_level2.stock_data_tick import TickStockData, TickFeatureType
from alphagen_level2.calculator_tick import TickCalculator
from alphagen_level2.env_wrapper_tick import TickAlphaEnv
from alphagen_level2.config_tick import TICK_FEATURES, TICK_FEATURES_BASIC, OPERATORS as TICK_OPERATORS
from alphagen_level2.convergence_logger import ConvergenceLogger, plot_convergence
from alphagen_level2.diversity_pool import DiversityMseAlphaPool
from alphagen_level2.llm_prompts_tick import get_tick_system_prompt

from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction
from alphagen_llm.prompts.common import safe_parse_list


# ---------------------------------------------------------------------------
# Parser / LLM client
# ---------------------------------------------------------------------------

def build_tick_parser() -> ExpressionParser:
    parser = ExpressionParser(
        TICK_OPERATORS,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub],
        },
    )
    parser._features = {f.name.lower(): f for f in TickFeatureType}
    return parser


def build_tick_chat_client(
    log_dir: str,
    base_url: str = "http://10.2.1.205:8796/v1",
    api_key: str = "sk-GEmM5YHREocL6mOOVEOUQ0Rs0qgWoB_KjJ-fSZUYd30",
    model: str = "MiniMax-M2.5",
) -> ChatClient:
    from openai import OpenAI

    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url=base_url, api_key=api_key),
        config=ChatConfig(system_prompt=get_tick_system_prompt(), logger=logger),
        model=model,
    )


# ---------------------------------------------------------------------------
# Callback (aligned with rl_level2_llm.py for tick data)
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
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5,
        gentle_inject: bool = True,
        valid_patience: int = 20,
        valid_min_delta: float = 1e-4,
        valid_smooth_window: int = 1,
        valid_restore_cooldown: int = 3,
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
        self.chat_session = chat_session
        self.llm_every_n_steps = llm_every_n_steps
        self._drop_rl_n = drop_rl_n
        self._gentle_inject = gentle_inject
        self.llm_use_count = 0
        self.last_llm_use = 0
        self._valid_min_delta = valid_min_delta
        self._valid_restore_cooldown = valid_restore_cooldown
        self._valid_smooth_window = max(1, int(valid_smooth_window))
        self._valid_hist: deque = deque(maxlen=self._valid_smooth_window)
        self._valid_patience = valid_patience
        self._best_valid_ic: float = -999.0
        self._best_valid_snapshot: Optional[dict] = None
        self._best_valid_model_params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self._best_valid_optimizer_state: Optional[dict] = None
        self._valid_no_improve_count: int = 0
        self._valid_cooldown_count: int = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _pool_factor_mean_ic(self, calculator: TickCalculator) -> Tuple[float, float]:
        """
        Mean single-factor IC/RankIC across current pool expressions.
        Use deployed direction (weight sign) so the metric matches trading orientation.
        """
        pool = self.pool
        if pool.size <= 0:
            return 0.0, 0.0

        ic_vals: List[float] = []
        ric_vals: List[float] = []
        for i in range(pool.size):
            expr = pool.exprs[i]
            if expr is None:
                continue
            try:
                ic = float(calculator.calc_single_IC_ret(expr))
                ric = float(calculator.calc_single_rIC_ret(expr))
                w = float(pool.weights[i]) if i < len(pool.weights) else 1.0
                direction = 1.0 if w >= 0 else -1.0
                ic_vals.append(ic * direction)
                ric_vals.append(ric * direction)
            except Exception:
                continue

        if len(ic_vals) == 0:
            return 0.0, 0.0

        ic_arr = np.array(ic_vals, dtype=float)
        ric_arr = np.array(ric_vals, dtype=float) if len(ric_vals) > 0 else np.array([0.0])
        return float(np.nanmean(ic_arr)), float(np.nanmean(ric_arr))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self.chat_session is not None:
            self._try_use_llm()
        pool = self.pool

        current_eval_cnt = int(pool.eval_cnt)
        if current_eval_cnt >= self._last_pool_eval_cnt:
            self._global_eval_cnt += current_eval_cnt - self._last_pool_eval_cnt
        self._last_pool_eval_cnt = current_eval_cnt

        if HAS_SINGLE_FACTOR_POOL and isinstance(pool, SingleFactorAlphaPool):
            sig_count = int(pool.size)
        else:
            sig_count = int((np.abs(pool.weights[:pool.size]) > 1e-4).sum())
        self.logger.record('pool/size', pool.size)
        self.logger.record('pool/significant', sig_count)
        self.logger.record('pool/best_ic_ret', pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', pool.eval_cnt)
        self.logger.record('pool/global_eval_cnt', self._global_eval_cnt)

        # ---- Reward-component tensorboard metrics (single-factor mode) ----
        if HAS_SINGLE_FACTOR_POOL and isinstance(pool, SingleFactorAlphaPool):
            stats = pool._reward_stats
            tail = 500  # look at the most recent evaluations to track dynamics
            for key in ("abs_ic", "r_bar", "comp_ic", "comp_r", "reward", "pos_abs_mean"):
                arr = stats.get(key, [])
                if not arr:
                    continue
                recent = np.asarray(arr[-tail:], dtype=np.float64)
                self.logger.record(f'reward/{key}_mean',  float(recent.mean()))
                self.logger.record(f'reward/{key}_std',   float(recent.std()))
                self.logger.record(f'reward/{key}_p95',   float(np.quantile(recent, 0.95)))
                self.logger.record(f'reward/{key}_max',   float(recent.max()))

        valid_ic_raw, valid_rank_ic_raw = self._pool_factor_mean_ic(self.valid_calculator)

        n_days = sum(calc.data.n_days for calc in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        test_results = []
        for i, test_calc in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self._pool_factor_mean_ic(test_calc)
            test_results.append((ic_test, rank_ic_test))
            if n_days > 0:
                ic_test_mean += ic_test * test_calc.data.n_days / n_days
                rank_ic_test_mean += rank_ic_test * test_calc.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record('test/ic_mean', ic_test_mean)
        self.logger.record('test/rank_ic_mean', rank_ic_test_mean)

        self._valid_hist.append(valid_ic_raw)
        valid_ic = float(np.mean(self._valid_hist))
        if valid_ic >= self._best_valid_ic + self._valid_min_delta:
            self._best_valid_ic = valid_ic
            self._valid_no_improve_count = 0
            self._best_valid_snapshot = {
                "pool": pool.to_json_dict(),
                "best_obj": float(pool.best_obj),
                "best_ic_ret": float(pool.best_ic_ret),
                "eval_cnt": int(pool.eval_cnt),
            }
            self._best_valid_model_params = copy.deepcopy(self.model.get_parameters())
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                self._best_valid_optimizer_state = copy.deepcopy(self.model.policy.optimizer.state_dict())
        else:
            if self._valid_cooldown_count > 0:
                self._valid_cooldown_count -= 1
            else:
                self._valid_no_improve_count += 1

        if (self._valid_patience > 0
                and self._valid_no_improve_count >= self._valid_patience
                and self._best_valid_snapshot is not None
                and pool.size > 1):
            self._restore_pool_snapshot()
            self._valid_no_improve_count = 0
            self._valid_cooldown_count = self._valid_restore_cooldown
            if self.verbose > 0:
                print(f"[Overfit] Valid IC stale for {self._valid_patience} rollouts, "
                      f"restored best snapshot (valid IC={self._best_valid_ic:.4f})")

        self.logger.record('valid/ic_raw', valid_ic_raw)
        self.logger.record('valid/ic_smooth', valid_ic)
        self.logger.record('valid/rank_ic_raw', valid_rank_ic_raw)
        self.logger.record('valid/best_ic', self._best_valid_ic)
        self.logger.record('valid/no_improve_count', self._valid_no_improve_count)
        self.logger.record('valid/cooldown_count', self._valid_cooldown_count)

        if self.conv_logger is not None:
            self.conv_logger.record_step(
                timestep=self.num_timesteps,
                pool_size=pool.size,
                pool_significant=sig_count,
                pool_best_ic=pool.best_ic_ret,
                pool_eval_cnt=pool.eval_cnt,
                global_eval_cnt=self._global_eval_cnt,
                train_ic=pool.best_ic_ret,
                valid_ic=valid_ic_raw,
                valid_rank_ic=valid_rank_ic_raw,
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

    def _try_use_llm(self) -> None:
        n_steps = self.num_timesteps
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1

        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        logger.debug(
            f"[Step: {n_steps}] Trying LLM (#{self.llm_use_count}): "
            f"IC={self.pool.best_ic_ret:.4f}"
        )
        if self._gentle_inject:
            self._gentle_llm_inject(logger)
        else:
            self._aggressive_llm_inject(logger)

    def _restore_pool_snapshot(self) -> None:
        if self._best_valid_snapshot is None:
            return
        pool = self.pool
        parser = build_tick_parser()
        snapshot = self._best_valid_snapshot
        pool_snapshot = snapshot.get("pool", snapshot)
        exprs = []
        for expr_str in pool_snapshot.get("exprs", []):
            try:
                expr = parser.parse(expr_str)
                exprs.append(expr)
            except Exception:
                continue
        if not exprs:
            return
        weights = pool_snapshot.get("weights", None)
        pool.leave_only([])
        pool.force_load_exprs(exprs, weights=weights)
        if "best_obj" in snapshot:
            pool.best_obj = float(snapshot["best_obj"])
        if "best_ic_ret" in snapshot:
            pool.best_ic_ret = float(snapshot["best_ic_ret"])
        if "eval_cnt" in snapshot:
            pool.eval_cnt = int(snapshot["eval_cnt"])

        # Joint rollback: restore policy parameters + optimizer state together with pool
        if self._best_valid_model_params is not None:
            self.model.set_parameters(self._best_valid_model_params, exact_match=True)
        if (self._best_valid_optimizer_state is not None
                and hasattr(self.model, "policy")
                and hasattr(self.model.policy, "optimizer")):
            self.model.policy.optimizer.load_state_dict(self._best_valid_optimizer_state)

    def _aggressive_llm_inject(self, logger) -> None:
        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed: {type(e).__name__}: {e}")

    def _gentle_llm_inject(self, logger) -> None:
        try:
            report_str, _ = self.chat_session._generate_report(self.pool)
            from alphagen_llm.prompts.common import alpha_phrase
            n_request = self._drop_rl_n
            prompt = (
                "Here are the current alphas and their metrics:\n"
                f"{report_str}\n"
                f"Please generate {alpha_phrase(n_request, 'new')} that are "
                "DIFFERENT from and COMPLEMENTARY to the existing ones. "
                "Focus on capturing different market patterns. "
                "One alpha per line, no numbering, nothing else."
            )
            lines = self.chat_session.client.chat_complete(prompt)
            exprs, invalid = safe_parse_list(
                lines.split("\n"), self.chat_session._parser
            )
            if invalid:
                logger.debug(f"LLM invalid expressions: {invalid}")
            accepted = 0
            for expr in exprs:
                old_obj = self.pool.best_obj
                try:
                    new_obj = self.pool.try_new_expr(expr)
                    if new_obj > old_obj:
                        accepted += 1
                        logger.debug(f"  Accepted: {expr} (obj {old_obj:.4f} -> {new_obj:.4f})")
                    else:
                        logger.debug(f"  Rejected: {expr} (no improvement)")
                except Exception as e:
                    logger.debug(f"  Failed: {expr} ({e})")
            logger.debug(
                f"LLM gentle inject: generated {len(exprs)}, "
                f"accepted {accepted}, pool size {self.pool.size}"
            )
        except Exception as e:
            logger.warning(f"LLM gentle inject failed: {type(e).__name__}: {e}")

    @property
    def pool(self) -> LinearAlphaPool:
        assert isinstance(self.env_core.pool, LinearAlphaPool)
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        vec_env = self.training_env
        if hasattr(vec_env, 'envs'):
            env = vec_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            return env
        # SubprocVecEnv path: pull first worker's wrapper attr
        if hasattr(vec_env, 'get_attr'):
            cores = vec_env.get_attr('env')
            if len(cores) > 0:
                return cores[0]
        raise AttributeError("Unable to locate AlphaEnvCore from current VecEnv")


# ---------------------------------------------------------------------------
# Rolling window schedule
# ---------------------------------------------------------------------------

def build_rolling_schedule(
    global_start: str = "2023-01-01",
    global_end: str = "2025-10-31",
    train_months: int = 6,
    valid_months: int = 2,
    test_months: int = 2,
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


def build_stable_pool_from_results(
    all_results: List[dict],
    min_occurrence: int = 3,
    min_sign_consistency: float = 0.6,
    max_factors: int = 12,
) -> Dict[str, object]:
    """
    Aggregate window final pools and select stable factors by:
      1) occurrence count across windows
      2) sign consistency of weights
      3) median abs(weight)
    """
    stats = defaultdict(lambda: {"count": 0, "weights": [], "windows": []})
    total_windows = len(all_results)

    for result in all_results:
        wid = result.get("window_id")
        pool_path = result.get("pool_path")
        if pool_path is None or (not os.path.exists(pool_path)):
            continue
        try:
            with open(pool_path, "r") as f:
                pool_data = json.load(f)
        except Exception:
            continue
        exprs = pool_data.get("exprs", [])
        weights = pool_data.get("weights", [])
        n = min(len(exprs), len(weights))
        for i in range(n):
            expr = str(exprs[i]).strip()
            w = float(weights[i])
            if expr == "":
                continue
            s = stats[expr]
            s["count"] += 1
            s["weights"].append(w)
            s["windows"].append(wid)

    candidates = []
    for expr, s in stats.items():
        count = int(s["count"])
        ws = np.array(s["weights"], dtype=float)
        if count == 0:
            continue
        pos = int((ws > 0).sum())
        neg = int((ws < 0).sum())
        sign_consistency = max(pos, neg) / count
        med_abs_w = float(np.median(np.abs(ws)))
        if count >= min_occurrence and sign_consistency >= min_sign_consistency:
            candidates.append({
                "expr": expr,
                "count": count,
                "coverage": count / max(total_windows, 1),
                "sign_consistency": sign_consistency,
                "median_abs_weight": med_abs_w,
                "mean_weight": float(ws.mean()),
                "windows": s["windows"],
            })

    candidates.sort(
        key=lambda x: (x["count"], x["sign_consistency"], x["median_abs_weight"]),
        reverse=True,
    )
    selected = candidates[:max_factors]

    return {
        "total_windows": total_windows,
        "min_occurrence": min_occurrence,
        "min_sign_consistency": min_sign_consistency,
        "max_factors": max_factors,
        "n_candidates": len(candidates),
        "selected": selected,
    }


# ---------------------------------------------------------------------------
# Best-factor selection helper
# ---------------------------------------------------------------------------

def _select_best_factor_from_pool(
    pool: LinearAlphaPool,
    valid_calculator: TickCalculator,
    test_calculator: Optional[TickCalculator] = None,
) -> Optional[Dict[str, float]]:
    """
    Pick the single best factor inside ``pool`` by |valid IC| of each
    individual expression (sign-aligned to its deployed weight). Returns
    ``None`` if the pool is empty.
    """
    n = int(pool.size)
    if n <= 0:
        return None

    rows = []
    for i in range(n):
        expr = pool.exprs[i]
        if expr is None:
            continue
        try:
            v_ic  = float(valid_calculator.calc_single_IC_ret(expr))
            v_ric = float(valid_calculator.calc_single_rIC_ret(expr))
        except Exception:
            continue
        w = float(pool.weights[i]) if i < len(pool.weights) else 1.0
        direction = 1.0 if w >= 0 else -1.0
        rows.append({
            "idx": i,
            "expr": str(expr),
            "weight": w,
            "valid_ic": v_ic * direction,
            "valid_rank_ic": v_ric * direction,
            "direction": direction,
        })

    if not rows:
        return None

    rows.sort(key=lambda r: abs(r["valid_ic"]), reverse=True)
    best = rows[0]

    test_ic = float("nan")
    test_rank_ic = float("nan")
    if test_calculator is not None:
        try:
            t_ic  = float(test_calculator.calc_single_IC_ret(pool.exprs[best["idx"]]))
            t_ric = float(test_calculator.calc_single_rIC_ret(pool.exprs[best["idx"]]))
            test_ic = t_ic * best["direction"]
            test_rank_ic = t_ric * best["direction"]
        except Exception:
            pass

    return {
        "expr": best["expr"],
        "weight": best["weight"],
        "valid_ic": best["valid_ic"],
        "valid_rank_ic": best["valid_rank_ic"],
        "test_ic": test_ic,
        "test_rank_ic": test_rank_ic,
    }


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
    single_factor_mode: bool = False,
    # Rolling-zscore reward knobs (SingleFactorAlphaPool, per spec)
    sf_alpha: float = 1.0,
    sf_beta: float = 1.0,
    sf_gamma: float = 1.0,
    sf_tau_ic: float = 0.1,
    sf_tau_r: float = 1e-3,
    sf_tau_c: float = 1e-4,
    sf_lookback_bars: int = 1200,
    sf_turnover_cost: float = 0.0006,
    sf_execution_delay: int = 1,
    sf_trivial_penalty: float = 0.0,
    # LLM options
    llm_warmstart: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3,
    llm_base_url: str = "http://10.2.1.205:8796/v1",
    llm_api_key: str = "sk-GEmM5YHREocL6mOOVEOUQ0Rs0qgWoB_KjJ-fSZUYd30",
    llm_model: str = "MiniMax-M2.5",
    gentle_inject: bool = True,
    llm_init_min_pool_size: int = 5,
    llm_init_updates: int = 4,
    llm_forgetful: bool = False,
    # Validation rollback controls
    valid_patience: int = 20,
    valid_min_delta: float = 1e-4,
    valid_smooth_window: int = 1,
    valid_restore_cooldown: int = 3,
    # Multi-env parallelism
    n_envs: int = 1,
    # Parallel-pool labelling (one worker per parallel pool)
    worker_idx: int = 0,
) -> dict:
    """Train one walk-forward window. Returns dict with results.

    When called from the parallel dispatcher, ``seed`` and ``worker_idx`` are
    distinct per worker; outputs are written under
    ``save_root/window_XXX/seed_<seed>/`` so that concurrent workers cannot
    collide on file paths.
    """
    # Each parallel worker must reseed itself; otherwise spawned subprocesses
    # would inherit identical RNG state and produce duplicate alpha pools.
    reseed_everything(seed)

    wid = window["window_id"]
    tag = f"[Window {wid}|seed {seed}|worker {worker_idx}]"
    print(f"\n{'='*70}")
    print(f"{tag} "
          f"train={window['train_start']}~{window['train_end']}, "
          f"valid={window['valid_start']}~{window['valid_end']}, "
          f"test={window['test_start']}~{window['test_end']}")
    print(f"{'='*70}")

    window_dir = os.path.join(save_root, f"window_{wid:03d}")
    worker_dir = os.path.join(window_dir, f"seed_{seed:03d}")
    os.makedirs(worker_dir, exist_ok=True)

    # window_meta is window-scoped (identical across workers); write once.
    meta_path = os.path.join(window_dir, "window_meta.json")
    if not os.path.exists(meta_path):
        try:
            with open(meta_path, "w") as f:
                json.dump(window, f, indent=2)
        except OSError:
            # Two workers raced; the loser silently skips — content is identical.
            pass

    mid_prc = Feature(TickFeatureType.MID)
    target = Ref(mid_prc, -max_future_bars) / mid_prc - 1

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

    calculators = [
        TickCalculator(
            d,
            target,
            holding_bars=max_future_bars,
            execution_delay=sf_execution_delay,
            lookback_bars=sf_lookback_bars,
        )
        for d in datasets
    ]

    # Build pool
    use_diversity = (diversity_bonus > 0 or ic_mut_threshold < 0.99) and not single_factor_mode

    def build_pool(exprs: Optional[List[Expression]] = None) -> LinearAlphaPool:
        if single_factor_mode:
            print(f"[build_pool] Creating SingleFactorAlphaPool (type={SingleFactorAlphaPool.__name__})")
            p = SingleFactorAlphaPool(
                capacity=pool_capacity,
                calculator=calculators[0],
                ic_lower_bound=None,
                l1_alpha=0.0,
                device=device,
                ic_mut_threshold=ic_mut_threshold,
                holding_bars=max_future_bars,
                execution_delay=sf_execution_delay,
                lookback_bars=sf_lookback_bars,
                turnover_cost=sf_turnover_cost,
                alpha=sf_alpha,
                beta=sf_beta,
                gamma=sf_gamma,
                tau_ic=sf_tau_ic,
                tau_r=sf_tau_r,
                tau_c=sf_tau_c,
                trivial_penalty=sf_trivial_penalty,
            )
        elif single_factor_mode and not HAS_SINGLE_FACTOR_POOL:
            p = MseAlphaPool(
                capacity=pool_capacity,
                calculator=calculators[0],
                ic_lower_bound=None,
                l1_alpha=5e-3,
                device=device,
            )
        elif use_diversity:
            p = DiversityMseAlphaPool(
                capacity=pool_capacity,
                calculator=calculators[0],
                ic_lower_bound=None,
                l1_alpha=5e-3,
                device=device,
                ic_mut_threshold=ic_mut_threshold,
                diversity_bonus=diversity_bonus,
            )
        else:
            p = MseAlphaPool(
                capacity=pool_capacity,
                calculator=calculators[0],
                ic_lower_bound=None,
                l1_alpha=5e-3,
                device=device,
            )
        if exprs:
            p.force_load_exprs(exprs)
        return p
    if single_factor_mode and HAS_SINGLE_FACTOR_POOL:
        print(f"[Window {wid}] Single-factor rolling-zscore reward pool: "
              f"alpha={sf_alpha}, beta={sf_beta}, gamma={sf_gamma}, "
              f"tau_ic={sf_tau_ic}, tau_r={sf_tau_r}, tau_c={sf_tau_c}, "
              f"lookback={sf_lookback_bars}, holding={max_future_bars}, "
              f"delay={sf_execution_delay}, turnover_cost={sf_turnover_cost}, "
              f"ic_mut_threshold={ic_mut_threshold}, "
              f"trivial_penalty={sf_trivial_penalty}")
    elif single_factor_mode and not HAS_SINGLE_FACTOR_POOL:
        print(f"[Window {wid}] [Warn] `SingleFactorAlphaPool` is unavailable in current alphagen package. "
              f"Falling back to MseAlphaPool.")

    pool = build_pool()
    if single_factor_mode:
        assert isinstance(pool, SingleFactorAlphaPool), \
            f"BUG: pool is {type(pool).__name__}, expected SingleFactorAlphaPool"
        print(f"[Window {wid}] Pool type confirmed: {type(pool).__name__}")
        print(f"  optimize() owner: {type(pool).optimize}")
        print(f"  to_json_dict owner: {type(pool).to_json_dict}")

    # Warm-start from previous window's pool
    if prev_pool_path is not None and os.path.exists(prev_pool_path):
        print(f"[Window {wid}] Warm-starting from {prev_pool_path}")
        try:
            with open(prev_pool_path, "r") as f:
                prev_state = json.load(f)
            if "exprs" in prev_state and len(prev_state["exprs"]) > 0:
                parser = build_tick_parser()
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

    chat_session: Optional[InterativeSession] = None
    if llm_warmstart or use_llm:
        print(f"[Window {wid}] Setting up LLM client...")
        chat_client = build_tick_chat_client(
            log_dir=worker_dir,
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
        )
        parser = build_tick_parser()
        inter = DefaultInteraction(
            parser,
            chat_client,
            build_pool,
            calculator_train=calculators[0],
            calculators_test=[calculators[1]],
            replace_k=llm_replace_n,
            forgetful=llm_forgetful,
        )
        print(f"[Window {wid}] LLM generating initial alpha pool...")
        try:
            pool = inter.run(n_updates=llm_init_updates)
        except Exception as exc:  # network / 502 / parser etc.
            print(
                f"[Window {wid}] LLM warmstart failed ({type(exc).__name__}: "
                f"{exc}). Falling back to seed-bootstrap pool."
            )
            # inter.run may have partially mutated its pool; rebuild a clean
            # empty one so the fallback-seed block below is the single source
            # of truth for pool contents.
            pool = build_pool()

        # Fallback: if LLM init pool is too small, bootstrap with robust seed expressions
        if pool.size < llm_init_min_pool_size:
            print(f"[Window {wid}] LLM init pool too small ({pool.size}), applying fallback seeds...")
            parser = build_tick_parser()
            seed_expr_strs = [
                "Div(Sub($vwap,$close),Std($close,600))",
                "EMA($imbalance_1,20)",
                "Div(Sum($signed_volume,20),Sum($volume,20))",
                "Corr($delta_bid_vol1,$ret,100)",
                "Sub(EMA($spread_pct,20),EMA($spread_pct,600))",
                "Mul($imbalance_total,Div($volume,Mean($volume,1200)))",
            ]
            merged_exprs = []
            if hasattr(pool, "exprs"):
                merged_exprs.extend(pool.exprs[:pool.size])
            for s in seed_expr_strs:
                try:
                    merged_exprs.append(parser.parse(s))
                except Exception:
                    continue
            if merged_exprs:
                pool = build_pool(merged_exprs)
        print(f"[Window {wid}] Initial pool: {pool.size} alphas, IC={pool.best_ic_ret:.4f}")
        if use_llm:
            chat_session = inter

    if n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env(print_expr: bool = False):
            def _init():
                return TickAlphaEnv(
                    pool=pool,
                    use_all_features=(len(features) > 7),
                    device=device,
                    print_expr=print_expr,
                )
            return _init

        env_fns = [make_env(print_expr=True)] + [make_env(print_expr=False)] * (n_envs - 1)
        env = SubprocVecEnv(env_fns)
        print(f"[Window {wid}] Multi-env: {n_envs} parallel environments (SubprocVecEnv)")
    else:
        env = TickAlphaEnv(
            pool=pool,
            use_all_features=(len(features) > 7),
            device=device,
            print_expr=True,
        )

    conv_logger = ConvergenceLogger(save_dir=worker_dir)
    callback = TickRollingCallback(
        save_path=worker_dir,
        valid_calculator=calculators[1],
        test_calculators=calculators[2:],
        verbose=1,
        convergence_logger=conv_logger,
        chat_session=chat_session,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n,
        gentle_inject=gentle_inject,
        valid_patience=valid_patience,
        valid_min_delta=valid_min_delta,
        valid_smooth_window=valid_smooth_window,
        valid_restore_cooldown=valid_restore_cooldown,
    )

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
        tensorboard_log=os.path.join(save_root, "tensorboard"),
        device=device,
        verbose=1,
    )

    print(f"{tag} Training for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=callback,
        tb_log_name=f"window_{wid:03d}_seed{seed:03d}",
    )

    # Restore the best validation-IC snapshot so the persisted final_pool /
    # downstream metrics reflect the rollback target rather than whatever
    # state the live pool was left in at the last rollout.
    if callback._best_valid_snapshot is not None:
        callback._restore_pool_snapshot()
        print(f"{tag} Restored best-valid snapshot for final pool "
              f"(valid IC={callback._best_valid_ic:.4f})")
    else:
        print(f"{tag} No valid snapshot recorded; final pool = last live pool")

    # Save final pool
    final_pool_path = os.path.join(worker_dir, "final_pool.json")
    pool_dict = pool.to_json_dict()
    if single_factor_mode:
        bad = [w for w in pool_dict["weights"] if abs(abs(w) - 1.0) > 1e-6]
        if bad:
            print(f"[BUG] Final pool has non-±1 weights: {pool_dict['weights']}")
            print(f"  pool type: {type(pool).__name__}")
            print(f"  _factor_directions: {list(pool._factor_directions[:pool.size])}")
            print(f"  raw _weights: {list(pool._weights[:pool.size])}")
        else:
            print(f"[Window {wid}] Final pool weights OK (all ±1): {pool_dict['weights']}")
    with open(final_pool_path, "w") as f:
        json.dump(pool_dict, f)

    csv_path = conv_logger.save_csv()
    conv_logger.save_json()
    try:
        plot_convergence(csv_path)
    except Exception:
        pass

    # Dump reward-component magnitude & distribution stats (single-factor mode)
    reward_stats_summary: Optional[Dict[str, Dict[str, float]]] = None
    if single_factor_mode and hasattr(pool, "dump_reward_stats"):
        stats_path = os.path.join(worker_dir, "reward_component_stats.json")
        try:
            reward_stats_summary = pool.dump_reward_stats(stats_path, include_raw=True)
            comp_ic = reward_stats_summary.get("comp_ic", {})
            comp_r  = reward_stats_summary.get("comp_r",  {})
            abs_ic  = reward_stats_summary.get("abs_ic",  {})
            r_bar   = reward_stats_summary.get("r_bar",   {})
            print(
                f"{tag} Reward-component stats (n={comp_ic.get('count', 0)}):\n"
                f"  |IC|       mean={abs_ic.get('mean', 0):.4f}  std={abs_ic.get('std', 0):.4f}  "
                f"p50={abs_ic.get('p50', 0):.4f}  max={abs_ic.get('max', 0):.4f}\n"
                f"  r_bar      mean={r_bar.get('mean', 0):.3e}  std={r_bar.get('std', 0):.3e}  "
                f"p50={r_bar.get('p50', 0):.3e}  max={r_bar.get('max', 0):.3e}\n"
                f"  a*tanh(|IC|/t) mean={comp_ic.get('mean', 0):.4f}  std={comp_ic.get('std', 0):.4f}  "
                f"max={comp_ic.get('max', 0):.4f}\n"
                f"  b*tanh(r/t)    mean={comp_r.get('mean', 0):.4f}  std={comp_r.get('std', 0):.4f}  "
                f"max={comp_r.get('max', 0):.4f}"
            )
        except Exception as exc:  # pragma: no cover
            print(f"{tag} Failed to dump reward stats: {exc}")

    valid_ic, valid_rank_ic = pool.test_ensemble(calculators[1])
    test_ic, test_rank_ic = pool.test_ensemble(calculators[2])

    # ---- Per-worker best factor: pick top-1 expr by |valid IC| ----
    best_factor = _select_best_factor_from_pool(
        pool=pool,
        valid_calculator=calculators[1],
        test_calculator=calculators[2],
    )
    best_factor_path = os.path.join(worker_dir, "best_factor.json")
    payload = {
        "window_id": wid,
        "seed": seed,
        "worker_idx": worker_idx,
        "best": best_factor,
    }
    with open(best_factor_path, "w") as f:
        json.dump(payload, f, indent=2)

    if best_factor is not None:
        print(
            f"{tag} BEST FACTOR: {best_factor['expr']}\n"
            f"  weight={best_factor['weight']:+.4f}  "
            f"valid_IC={best_factor['valid_ic']:+.4f}  "
            f"valid_RankIC={best_factor['valid_rank_ic']:+.4f}  "
            f"test_IC={best_factor['test_ic']:+.4f}  "
            f"test_RankIC={best_factor['test_rank_ic']:+.4f}"
        )
    else:
        print(f"{tag} BEST FACTOR: <pool empty>")

    result = {
        "window_id": wid,
        "seed": seed,
        "worker_idx": worker_idx,
        **window,
        "valid_ic": float(valid_ic),
        "valid_rank_ic": float(valid_rank_ic),
        "test_ic": float(test_ic),
        "test_rank_ic": float(test_rank_ic),
        "pool_size": int(pool.size),
        "pool_path": final_pool_path,
        "worker_dir": worker_dir,
        "best_factor": best_factor,
        "best_factor_path": best_factor_path,
    }
    if reward_stats_summary is not None:
        result["reward_stats"] = reward_stats_summary

    summary = conv_logger.summary()
    if summary:
        result["best_pool_ic"] = summary.get("best_pool_ic", 0.0)

    print(f"{tag} Done: valid_ic={valid_ic:.4f}, test_ic={test_ic:.4f}")
    return result


# ---------------------------------------------------------------------------
# Parallel dispatch: N workers per window (each its own pool / seed)
# ---------------------------------------------------------------------------

def _train_window_worker_entry(kwargs: dict) -> dict:
    """Top-level entrypoint executed inside each spawned subprocess.

    Must be importable / picklable, hence kept at module scope.
    """
    # Re-import in the child to make sure CUDA gets initialised lazily.
    return train_one_window(**kwargs)


def _warm_data_cache(
    window: dict,
    instruments: Union[str, List[str]],
    features: List[TickFeatureType],
    bar_size_sec: int,
    max_backtrack_bars: int,
    max_future_bars: int,
    data_root: str,
    cache_dir: Optional[str],
    max_workers: int,
) -> None:
    """
    Touch each train/valid/test segment serially in the parent so the on-disk
    cache is populated before parallel workers race to write it. Datasets are
    discarded immediately to free memory; child workers will reload from the
    warm cache.
    """
    if cache_dir is None:
        return
    print(f"[Window {window['window_id']}] Pre-warming data cache ({cache_dir})...")
    cpu_dev = torch.device("cpu")
    for label, (start, end) in zip(
        ["train", "valid", "test"],
        [
            (window["train_start"], window["train_end"]),
            (window["valid_start"], window["valid_end"]),
            (window["test_start"], window["test_end"]),
        ],
    ):
        try:
            tmp = TickStockData(
                instrument=instruments,
                start_time=start,
                end_time=end,
                max_backtrack_days=max_backtrack_bars,
                max_future_days=max_future_bars,
                features=features,
                device=cpu_dev,
                data_root=data_root,
                cache_dir=cache_dir,
                max_workers=max_workers,
                bar_size_sec=bar_size_sec,
            )
            print(f"  {label}: {start}~{end} cached ({tmp.n_days} bars)")
            del tmp
        except Exception as exc:
            print(f"  {label}: cache warm-up failed ({type(exc).__name__}: {exc}); "
                  f"workers will fall back to on-the-fly load.")


def train_window_parallel(
    window: dict,
    seeds: List[int],
    prev_pool_paths: Dict[int, Optional[str]],
    shared_kwargs: dict,
    n_parallel_pools: int,
) -> List[dict]:
    """
    Run N independent ``train_one_window`` workers concurrently.

    Each worker uses its own seed and its own per-seed warm-start pool. The
    function blocks until every worker finishes, then returns the list of
    per-worker result dicts (sorted by ``worker_idx``).
    """
    wid = window["window_id"]
    n_workers = len(seeds)
    assert n_workers == n_parallel_pools, \
        f"seeds count ({n_workers}) != n_parallel_pools ({n_parallel_pools})"

    # Build one full kwargs dict per worker.
    worker_kwargs_list: List[dict] = []
    for worker_idx, seed in enumerate(seeds):
        kw = dict(shared_kwargs)
        kw["window"] = window
        kw["seed"] = seed
        kw["worker_idx"] = worker_idx
        kw["prev_pool_path"] = prev_pool_paths.get(worker_idx)
        worker_kwargs_list.append(kw)

    if n_workers == 1:
        # Single-pool fast path: skip the subprocess overhead entirely.
        return [_train_window_worker_entry(worker_kwargs_list[0])]

    print(f"\n[Window {wid}] Launching {n_workers} parallel pool workers "
          f"with seeds {seeds} (shared GPU)")

    # 'spawn' is required for CUDA — 'fork' would inherit a poisoned CUDA
    # context from the parent and deadlock.
    ctx = mp.get_context("spawn")
    results: List[Optional[dict]] = [None] * n_workers
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        future_to_idx = {
            executor.submit(_train_window_worker_entry, kw): kw["worker_idx"]
            for kw in worker_kwargs_list
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                print(f"[Window {wid}] worker {idx} (seed {seeds[idx]}) "
                      f"FAILED: {type(exc).__name__}: {exc}")
                # Surface a stub so downstream aggregation does not break.
                results[idx] = {
                    "window_id": wid,
                    "seed": seeds[idx],
                    "worker_idx": idx,
                    **window,
                    "valid_ic": float("nan"),
                    "valid_rank_ic": float("nan"),
                    "test_ic": float("nan"),
                    "test_rank_ic": float("nan"),
                    "pool_size": 0,
                    "pool_path": None,
                    "best_factor": None,
                    "best_factor_path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    return [r for r in results if r is not None]


def _summarize_window_best_factors(
    window: dict,
    worker_results: List[dict],
    save_root: str,
) -> str:
    """
    Print and persist the per-window list of best factors (one per parallel
    pool). Returns the path of the consolidated json file.
    """
    wid = window["window_id"]
    window_dir = os.path.join(save_root, f"window_{wid:03d}")
    os.makedirs(window_dir, exist_ok=True)

    bests = []
    for r in worker_results:
        bests.append({
            "worker_idx": r.get("worker_idx"),
            "seed": r.get("seed"),
            "best": r.get("best_factor"),
            "valid_ic": r.get("valid_ic"),
            "test_ic": r.get("test_ic"),
            "pool_path": r.get("pool_path"),
            "best_factor_path": r.get("best_factor_path"),
            "error": r.get("error"),
        })

    payload = {
        "window_id": wid,
        **window,
        "best_factors": bests,
    }
    out_path = os.path.join(window_dir, "all_best_factors.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Pretty print the per-window roster.
    print(f"\n{'='*70}")
    print(f"[Window {wid}] BEST FACTORS PER PARALLEL POOL")
    print(f"{'='*70}")
    header = f"{'worker':>6} {'seed':>5} {'val_IC':>9} {'val_rIC':>9} {'tst_IC':>9} {'tst_rIC':>9}  expr"
    print(header)
    print("-" * max(len(header), 70))
    for entry in bests:
        bf = entry["best"] or {}
        if entry.get("error"):
            print(f"{entry['worker_idx']:>6} {entry['seed']:>5}  <FAILED: {entry['error']}>")
            continue
        if not bf:
            print(f"{entry['worker_idx']:>6} {entry['seed']:>5}  <empty pool>")
            continue
        print(
            f"{entry['worker_idx']:>6} {entry['seed']:>5} "
            f"{bf.get('valid_ic', float('nan')):>+9.4f} "
            f"{bf.get('valid_rank_ic', float('nan')):>+9.4f} "
            f"{bf.get('test_ic', float('nan')):>+9.4f} "
            f"{bf.get('test_rank_ic', float('nan')):>+9.4f}  "
            f"{bf.get('expr', '')}"
        )
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    seed: int = 0,
    instruments: Union[str, List[str]] = '["510300.sh"]',
    pool_capacity: int = 5,
    steps_per_window: int = 150_000,
    data_root: str = "~/EquityLevel2/stock",
    use_all_features: bool = True,
    # Bar size
    bar_size_sec: int = 3,
    # Rolling window
    global_start: str = "2023-01-01",
    global_end: str = "2025-10-31",
    train_months: int = 6,
    valid_months: int = 2,
    test_months: int = 2,
    step_months: int = 6,
    # Lookback / forward (in bars)
    max_backtrack_bars: int = 1200,
    max_future_bars: int = 100,
    # IO
    cache_dir: Optional[str] = "./out/tick_cache",
    max_workers: int = 8,
    # Diversity
    ic_mut_threshold: float = 0.99,
    diversity_bonus: float = 0.0,
    single_factor_mode: bool = False,
    # Rolling-zscore reward knobs (SingleFactorAlphaPool, per spec)
    sf_alpha: float = 1.0,
    sf_beta: float = 1.0,
    sf_gamma: float = 1.0,
    sf_tau_ic: float = 0.1,
    sf_tau_r: float = 1e-3,
    sf_tau_c: float = 1e-4,
    sf_lookback_bars: int = 1200,
    sf_turnover_cost: float = 0.0006,
    sf_execution_delay: int = 1,
    sf_trivial_penalty: float = 0.0,
    # LLM options
    llm_warmstart: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 2,
    llm_replace_n: int = 1,
    llm_base_url: str = "http://10.2.1.205:8796/v1",
    llm_api_key: str = "sk-GEmM5YHREocL6mOOVEOUQ0Rs0qgWoB_KjJ-fSZUYd30",
    llm_model: str = "MiniMax-M2.5",
    gentle_inject: bool = True,
    llm_init_min_pool_size: int = 2,
    llm_init_updates: int = 3,
    llm_forgetful: bool = False,
    # Validation rollback controls
    valid_patience: int = 20,
    valid_min_delta: float = 1e-4,
    valid_smooth_window: int = 1,
    valid_restore_cooldown: int = 3,
    # Multi-env parallelism
    n_envs: int = 1,
    # Per-window parallel pools (each parallel pool = independent RL agent + own seed)
    n_parallel_pools: int = 2,
    parallel_seeds: Optional[Union[str, List[int]]] = None,
    warm_cache_serial: bool = True,
    # Warm-start
    warm_start: bool = True,
    # Stable pool aggregation (step1)
    stable_pool_min_occurrence: int = 3,
    stable_pool_min_sign_consistency: float = 0.6,
    stable_pool_max_factors: int = 12,
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
    :param max_backtrack_bars: Max lookback in bars (1200 ≈ 1 h)
    :param max_future_bars: Max forward look in bars (100 ≈ 5 minutes)
    :param cache_dir: Cache directory
    :param max_workers: Parallel HDF5 IO threads
    :param ic_mut_threshold: Mutual IC rejection threshold
    :param diversity_bonus: Reward bonus for novel alphas
    :param single_factor_mode: If True, mine standalone factors ranked by the
        rolling-zscore reward (IC + realized PnL tanh-combined) instead of combo IC.
    :param sf_alpha: Weight on ``tanh(|IC|/tau_ic)`` in the reward sum.
    :param sf_beta: Weight on ``tanh(r_bar/tau_r)`` in the reward sum (gross pnl).
    :param sf_gamma: Weight on ``-tanh(tc/tau_c)`` turnover-cost penalty in the reward sum.
    :param sf_tau_ic: Compression scale for the IC component (tau_ic in the spec).
    :param sf_tau_r: Compression scale for the pnl component (tau_r in the spec).
    :param sf_tau_c: Compression scale for the turnover-cost penalty (tau_c in the spec).
    :param sf_lookback_bars: Rolling window for z-score normalization (previous bars
        used to compute mu_t / sigma_t). Default 1200 ≈ 1 hour of 3-second bars.
    :param sf_turnover_cost: Per-bar turnover cost rate lambda_c used in
        ``TC = mean(lambda_c * |p_t - p_{t-1}|)``. r_t itself is gross pnl.
    :param sf_execution_delay: Bars of execution delay between signal observation
        and trade entry. Default 1 means enter at mid[t+1] and exit at mid[t+1+H].
    :param sf_trivial_penalty: If > 0, expressions that are only a single feature combined
        with constants (no rolling operator, ≤1 feature leaf) receive reward = -sf_trivial_penalty
        and are rejected from the pool. 0 disables (default).
    :param llm_warmstart: Use LLM to generate initial alpha pool
    :param use_llm: Enable periodic LLM injection during RL training
    :param llm_every_n_steps: Invoke LLM every N steps
    :param drop_rl_n: Number of RL alphas to replace/inject per LLM round
    :param llm_replace_n: Number of new alphas generated per LLM round
    :param llm_base_url: OpenAI-compatible API base URL
    :param llm_api_key: API key
    :param llm_model: Model name
    :param gentle_inject: Use gentle injection (True) or aggressive replacement
    :param llm_init_min_pool_size: Minimum initial pool size after LLM warmstart
    :param llm_init_updates: Number of LLM warmstart update rounds
    :param llm_forgetful: Whether DefaultInteraction resets dialog each round
    :param valid_patience: Rollout patience before restoring best validation snapshot
    :param valid_min_delta: Minimum validation IC improvement to refresh best snapshot
    :param valid_smooth_window: Smoothing window for validation IC
    :param valid_restore_cooldown: Cooldown rollouts after restoration
    :param n_envs: Number of parallel environments inside a single RL worker
    :param n_parallel_pools: Number of independent RL pools to train concurrently
        per window (each gets its own seed and writes to ``window_XXX/seed_<S>/``).
        Default 2. Set to 1 to disable subprocess parallelism.
    :param parallel_seeds: Optional explicit seed list for the parallel pools.
        Accepts a JSON list (e.g. ``"[0, 1, 7]"``) or python list. If omitted,
        defaults to ``[seed, seed+1, ..., seed+n_parallel_pools-1]``.
    :param warm_cache_serial: When True (default) the parent process touches
        every window's train/valid/test segment before launching parallel
        workers, populating the on-disk cache so workers do not race on writes.
    :param warm_start: Carry forward pool across windows (per-seed chain when
        n_parallel_pools > 1)
    :param stable_pool_min_occurrence: Min cross-window occurrence for stable factor
    :param stable_pool_min_sign_consistency: Min sign consistency for stable factor
    :param stable_pool_max_factors: Max selected stable factors
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

    # Resolve per-pool seeds.
    if parallel_seeds is None:
        seeds_list: List[int] = [seed + i for i in range(n_parallel_pools)]
    else:
        if isinstance(parallel_seeds, str):
            seeds_list = list(json.loads(parallel_seeds))
        else:
            seeds_list = list(parallel_seeds)
        if len(seeds_list) != n_parallel_pools:
            raise ValueError(
                f"parallel_seeds length ({len(seeds_list)}) "
                f"!= n_parallel_pools ({n_parallel_pools})"
            )

    run_config = {
        "seed": seed, "instruments": instruments, "pool_capacity": pool_capacity,
        "steps_per_window": steps_per_window, "bar_size_sec": bar_size_sec,
        "use_all_features": use_all_features, "n_features": len(features),
        "max_backtrack_bars": max_backtrack_bars, "max_future_bars": max_future_bars,
        "single_factor_mode": single_factor_mode,
        "sf_alpha": sf_alpha, "sf_beta": sf_beta, "sf_gamma": sf_gamma,
        "sf_tau_ic": sf_tau_ic, "sf_tau_r": sf_tau_r, "sf_tau_c": sf_tau_c,
        "sf_lookback_bars": sf_lookback_bars,
        "sf_turnover_cost": sf_turnover_cost,
        "sf_execution_delay": sf_execution_delay,
        "sf_trivial_penalty": sf_trivial_penalty,
        "llm_warmstart": llm_warmstart, "use_llm": use_llm,
        "llm_every_n_steps": llm_every_n_steps, "drop_rl_n": drop_rl_n,
        "llm_replace_n": llm_replace_n, "n_envs": n_envs,
        "llm_init_min_pool_size": llm_init_min_pool_size,
        "llm_init_updates": llm_init_updates,
        "llm_forgetful": llm_forgetful,
        "valid_patience": valid_patience, "valid_min_delta": valid_min_delta,
        "valid_smooth_window": valid_smooth_window,
        "valid_restore_cooldown": valid_restore_cooldown,
        "stable_pool_min_occurrence": stable_pool_min_occurrence,
        "stable_pool_min_sign_consistency": stable_pool_min_sign_consistency,
        "stable_pool_max_factors": stable_pool_max_factors,
        "n_parallel_pools": n_parallel_pools,
        "parallel_seeds": seeds_list,
        "warm_cache_serial": warm_cache_serial,
        "schedule": schedule,
    }
    with open(os.path.join(save_root, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Kwargs identical for every parallel worker — only seed / worker_idx /
    # prev_pool_path differ, those are filled in inside ``train_window_parallel``.
    shared_kwargs = dict(
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
        ic_mut_threshold=ic_mut_threshold,
        diversity_bonus=diversity_bonus,
        single_factor_mode=single_factor_mode,
        sf_alpha=sf_alpha,
        sf_beta=sf_beta,
        sf_gamma=sf_gamma,
        sf_tau_ic=sf_tau_ic,
        sf_tau_r=sf_tau_r,
        sf_tau_c=sf_tau_c,
        sf_lookback_bars=sf_lookback_bars,
        sf_turnover_cost=sf_turnover_cost,
        sf_execution_delay=sf_execution_delay,
        sf_trivial_penalty=sf_trivial_penalty,
        llm_warmstart=llm_warmstart,
        use_llm=use_llm,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n,
        llm_replace_n=llm_replace_n,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        gentle_inject=gentle_inject,
        llm_init_min_pool_size=llm_init_min_pool_size,
        llm_init_updates=llm_init_updates,
        llm_forgetful=llm_forgetful,
        valid_patience=valid_patience,
        valid_min_delta=valid_min_delta,
        valid_smooth_window=valid_smooth_window,
        valid_restore_cooldown=valid_restore_cooldown,
        n_envs=n_envs,
    )

    all_results: List[dict] = []                     # flat: one entry per (window, worker)
    prev_pool_paths: Dict[int, Optional[str]] = {i: None for i in range(n_parallel_pools)}
    window_summary: List[dict] = []                  # per-window aggregate

    for window in schedule:
        # Pre-warm the on-disk dataset cache in the parent so concurrent
        # workers do not race on the first write.
        if warm_cache_serial and n_parallel_pools > 1:
            try:
                _warm_data_cache(
                    window=window,
                    instruments=instruments,
                    features=features,
                    bar_size_sec=bar_size_sec,
                    max_backtrack_bars=max_backtrack_bars,
                    max_future_bars=max_future_bars,
                    data_root=data_root,
                    cache_dir=cache_dir,
                    max_workers=max_workers,
                )
            except ValueError as e:
                if "No data in range" in str(e):
                    print(f"[Tick Rolling] Stop at window {window['window_id']}: {e}")
                    print("[Tick Rolling] Remaining windows are skipped due to unavailable date range.")
                    break
                raise

        try:
            worker_results = train_window_parallel(
                window=window,
                seeds=seeds_list,
                prev_pool_paths=prev_pool_paths if warm_start else {i: None for i in range(n_parallel_pools)},
                shared_kwargs=shared_kwargs,
                n_parallel_pools=n_parallel_pools,
            )
        except ValueError as e:
            if "No data in range" in str(e):
                print(f"[Tick Rolling] Stop at window {window['window_id']}: {e}")
                print("[Tick Rolling] Remaining windows are skipped due to unavailable date range.")
                break
            raise

        # Per-window: print + save consolidated best-factor roster.
        _summarize_window_best_factors(
            window=window,
            worker_results=worker_results,
            save_root=save_root,
        )

        # Update per-seed warm-start chain only for workers that succeeded.
        for r in worker_results:
            wi = r.get("worker_idx")
            if wi is None:
                continue
            new_path = r.get("pool_path")
            if new_path is not None:
                prev_pool_paths[wi] = new_path

        all_results.extend(worker_results)
        window_summary.append({
            "window_id": window["window_id"],
            **window,
            "n_workers": len(worker_results),
            "best_factors": [r.get("best_factor") for r in worker_results],
        })

        with open(os.path.join(save_root, "rolling_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        with open(os.path.join(save_root, "window_summary.json"), "w") as f:
            json.dump(window_summary, f, indent=2)

    # Stable cross-window factor pool (step1)
    stable_pool = build_stable_pool_from_results(
        all_results=all_results,
        min_occurrence=stable_pool_min_occurrence,
        min_sign_consistency=stable_pool_min_sign_consistency,
        max_factors=stable_pool_max_factors,
    )
    with open(os.path.join(save_root, "stable_factor_pool.json"), "w") as f:
        json.dump(stable_pool, f, indent=2)
    print(f"[Tick Rolling] Stable factor pool saved: {os.path.join(save_root, 'stable_factor_pool.json')}")
    print(f"[Tick Rolling] Stable selected factors: {len(stable_pool.get('selected', []))}")

    if len(all_results) == 0:
        print("[Tick Rolling] No completed window. Exiting after stable pool export.")
        return

    # Summary
    print(f"\n{'='*82}")
    print("[Tick Rolling] Walk-forward complete:")
    print(f"{'='*82}")
    print(f"{'Win':>4} {'Seed':>5} {'Train':>23} {'Test':>23} {'TestIC':>9} {'TestRkIC':>9}")
    print("-" * 82)
    for r in all_results:
        print(f"{r['window_id']:>4} "
              f"{r.get('seed', 0):>5} "
              f"{r['train_start']}~{r['train_end']} "
              f"{r['test_start']}~{r['test_end']} "
              f"{r['test_ic']:>9.4f} {r['test_rank_ic']:>9.4f}")

    valid_test_ic = [r["test_ic"] for r in all_results if not np.isnan(r["test_ic"])]
    valid_test_ric = [r["test_rank_ic"] for r in all_results if not np.isnan(r["test_rank_ic"])]
    if valid_test_ic:
        print(f"\nAvg test IC      ({len(valid_test_ic)} pools): {np.mean(valid_test_ic):.4f}")
    if valid_test_ric:
        print(f"Avg test Rank IC ({len(valid_test_ric)} pools): {np.mean(valid_test_ric):.4f}")

    # Per-window best-factor recap
    print(f"\n{'='*82}")
    print("[Tick Rolling] BEST FACTOR per window per parallel pool:")
    print(f"{'='*82}")
    for ws in window_summary:
        wid = ws["window_id"]
        print(f"\nWindow {wid}  test {ws['test_start']}~{ws['test_end']}")
        for i, bf in enumerate(ws.get("best_factors", [])):
            if bf is None:
                print(f"  pool {i}: <empty / failed>")
            else:
                print(f"  pool {i} (seed {seeds_list[i]}): "
                      f"valid_IC={bf.get('valid_ic', float('nan')):+.4f}  "
                      f"test_IC={bf.get('test_ic', float('nan')):+.4f}  "
                      f"{bf.get('expr', '')}")

    print(f"\nResults: {save_root}")


if __name__ == "__main__":
    # Required for multiprocessing 'spawn' on platforms where the default
    # start method differs (and harmless when it doesn't).
    mp.freeze_support()
    fire.Fire(main)

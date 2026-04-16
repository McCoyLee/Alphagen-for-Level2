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
        self._valid_no_improve_count: int = 0
        self._valid_cooldown_count: int = 0
        os.makedirs(self.save_path, exist_ok=True)

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

        valid_ic_raw, valid_rank_ic_raw = pool.test_ensemble(self.valid_calculator)

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
    # Composite-reward knobs (SingleFactorAlphaPool)
    sf_ic_weight: float = 0.5,
    sf_profit_weight: float = 0.5,
    sf_use_rank_ic: bool = False,
    sf_window_days: int = 20,
    sf_turnover_penalty: float = 0.001,
    sf_ic_std_penalty: float = 0.0,
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

    calculators = [TickCalculator(d, target) for d in datasets]

    # Build pool
    use_diversity = (diversity_bonus > 0 or ic_mut_threshold < 0.99) and not single_factor_mode

    _bars_per_day = datasets[0].bars_per_day

    def build_pool(exprs: Optional[List[Expression]] = None) -> LinearAlphaPool:
        if single_factor_mode and HAS_SINGLE_FACTOR_POOL:
            p = SingleFactorAlphaPool(
                capacity=pool_capacity,
                calculator=calculators[0],
                ic_lower_bound=None,
                l1_alpha=0.0,
                device=device,
                holding_bars=max_future_bars,
                bars_per_day=_bars_per_day,
                window_days=sf_window_days,
                ic_weight=sf_ic_weight,
                profit_weight=sf_profit_weight,
                use_rank_ic=sf_use_rank_ic,
                turnover_penalty=sf_turnover_penalty,
                ic_std_penalty=sf_ic_std_penalty,
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
        print(f"[Window {wid}] Single-factor composite-reward pool: "
              f"ic_w={sf_ic_weight}, profit_w={sf_profit_weight}, "
              f"rank_ic={sf_use_rank_ic}, window={sf_window_days}d")
    elif single_factor_mode and not HAS_SINGLE_FACTOR_POOL:
        print(f"[Window {wid}] [Warn] `SingleFactorAlphaPool` is unavailable in current alphagen package. "
              f"Falling back to MseAlphaPool.")

    pool = build_pool()

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
            log_dir=window_dir,
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
        pool = inter.run(n_updates=llm_init_updates)

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

    conv_logger = ConvergenceLogger(save_dir=window_dir)
    callback = TickRollingCallback(
        save_path=window_dir,
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
    pool_capacity: int = 10,
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
    # Composite-reward knobs (SingleFactorAlphaPool)
    sf_ic_weight: float = 0.5,
    sf_profit_weight: float = 0.5,
    sf_use_rank_ic: bool = False,
    sf_window_days: int = 20,
    sf_turnover_penalty: float = 0.001,
    sf_ic_std_penalty: float = 0.0,
    # LLM options
    llm_warmstart: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 3,
    llm_replace_n: int = 2,
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
    :param single_factor_mode: If True, mine standalone factors ranked by |single IC| instead of combo IC
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
    :param n_envs: Number of parallel environments
    :param warm_start: Carry forward pool across windows
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

    run_config = {
        "seed": seed, "instruments": instruments, "pool_capacity": pool_capacity,
        "steps_per_window": steps_per_window, "bar_size_sec": bar_size_sec,
        "use_all_features": use_all_features, "n_features": len(features),
        "max_backtrack_bars": max_backtrack_bars, "max_future_bars": max_future_bars,
        "single_factor_mode": single_factor_mode,
        "sf_ic_weight": sf_ic_weight, "sf_profit_weight": sf_profit_weight,
        "sf_use_rank_ic": sf_use_rank_ic, "sf_window_days": sf_window_days,
        "sf_turnover_penalty": sf_turnover_penalty, "sf_ic_std_penalty": sf_ic_std_penalty,
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
        "schedule": schedule,
    }
    with open(os.path.join(save_root, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    all_results = []
    prev_pool_path = None

    for window in schedule:
        try:
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
                single_factor_mode=single_factor_mode,
                sf_ic_weight=sf_ic_weight,
                sf_profit_weight=sf_profit_weight,
                sf_use_rank_ic=sf_use_rank_ic,
                sf_window_days=sf_window_days,
                sf_turnover_penalty=sf_turnover_penalty,
                sf_ic_std_penalty=sf_ic_std_penalty,
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
        except ValueError as e:
            if "No data in range" in str(e):
                print(f"[Tick Rolling] Stop at window {window['window_id']}: {e}")
                print("[Tick Rolling] Remaining windows are skipped due to unavailable date range.")
                break
            raise
        all_results.append(result)
        prev_pool_path = result["pool_path"]

        with open(os.path.join(save_root, "rolling_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

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

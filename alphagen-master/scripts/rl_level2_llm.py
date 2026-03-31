"""
RL + LLM alpha generation using local Level 2 HDF5 data.
 
Supports three modes:
  1. Pure RL (default): same as rl_level2.py but with convergence logging
  2. LLM warm start (--llm_warmstart): LLM generates initial pool, then RL takes over
  3. LLM periodic assist (--use_llm): warm start + LLM replaces worst alphas every N steps
 
Usage:
    # Pure RL with convergence curves:
    python scripts/rl_level2_llm.py --data_root=~/EquityLevel2/stock
 
    # LLM warm start only (LLM fills initial pool, then pure RL):
    python scripts/rl_level2_llm.py --data_root=~/EquityLevel2/stock --llm_warmstart
 
    # Full LLM assist (warm start + periodic replacement):
    python scripts/rl_level2_llm.py --data_root=~/EquityLevel2/stock --use_llm
 
    # With Level 2 extended features:
    python scripts/rl_level2_llm.py --data_root=~/EquityLevel2/stock --use_llm --use_level2_features
 
    # Custom LLM endpoint:
    python scripts/rl_level2_llm.py --data_root=~/EquityLevel2/stock --use_llm \\
        --llm_base_url=http://10.2.1.205:8796/v1 \\
        --llm_api_key=sk-xxx \\
        --llm_model=MiniMax-M2.5
"""
 
import json
import os
from typing import Optional, Tuple, List, Union
from datetime import datetime
from collections import deque
 
import numpy as np
import torch
import fire
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
 
from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
 
from alphagen_level2.stock_data import Level2StockData, Level2FeatureType
from alphagen_level2.calculator import Level2Calculator
from alphagen_level2.env_wrapper import Level2AlphaEnv
from alphagen_level2.config import (
    BASIC_FEATURES, LEVEL2_FEATURES, OPERATORS,
)
from alphagen_level2.convergence_logger import ConvergenceLogger, plot_convergence
from alphagen_level2.llm_prompts import get_level2_system_prompt
from alphagen_level2.diversity_pool import DiversityMseAlphaPool

from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction
from alphagen_llm.prompts.common import safe_parse_list
 
 
# ---------------------------------------------------------------------------
# Expression parser for Level 2 (supports all 20 feature names)
# ---------------------------------------------------------------------------
 
def build_level2_parser() -> ExpressionParser:
    """Build an expression parser that recognizes all Level 2 feature names."""
    parser = ExpressionParser(
        OPERATORS,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub],
        },
    )
    # NOTE:
    # ExpressionParser defaults to alphagen_qlib.FeatureType (OHLCV+VWAP only).
    # Here we override the feature dictionary so LLM outputs that use extended
    # Level 2 names (e.g. $net_order_flow, $txn_vwap) can be parsed correctly.
    parser._features = {f.name.lower(): f for f in Level2FeatureType}
    return parser
 
 
def build_level2_chat_client(
    log_dir: str,
    use_level2_features: bool = True,
    base_url: str = "http://10.2.1.205:8796/v1",
    api_key: str = "sk-GEmM5YHREocL6mOOVEOUQ0Rs0qgWoB_KjJ-fSZUYd30",
    model: str = "MiniMax-M2.5",
) -> ChatClient:
    """Create an OpenAI-compatible chat client with Level 2 system prompt."""
    from openai import OpenAI
 
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    system_prompt = get_level2_system_prompt(use_level2_features)
    return OpenAIClient(
        client=OpenAI(base_url=base_url, api_key=api_key),
        config=ChatConfig(system_prompt=system_prompt, logger=logger),
        model=model,
    )
 
 
# ---------------------------------------------------------------------------
# Callback with convergence logging + optional LLM assist
# ---------------------------------------------------------------------------
 
class Level2LLMCallback(BaseCallback):
    """
    Training callback with:
    - Convergence curve recording (CSV + auto-plot)
    - Optional periodic LLM-assisted alpha replacement
    - Validation-based overfitting detection with pool snapshot rollback
    """
 
    def __init__(
        self,
        save_path: str,
        valid_calculator: Level2Calculator,
        test_calculators: List[Level2Calculator],
        convergence_logger: ConvergenceLogger,
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5,
        plot_interval: int = 10,
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
        os.makedirs(self.save_path, exist_ok=True)
 
        # LLM assist state
        self.chat_session = chat_session
        self.llm_every_n_steps = llm_every_n_steps
        self._drop_rl_n = drop_rl_n
        self.llm_use_count = 0
        self.last_llm_use = 0
        self._gentle_inject = gentle_inject
 
        # Plot every N rollout ends
        self._plot_interval = plot_interval
        self._rollout_count = 0
 
        # Validation-based overfitting control:
        # Track best validation IC and save pool snapshot.
        # If valid IC degrades for `valid_patience` consecutive rollouts,
        # restore the best pool snapshot to prevent overfitting.
        self._valid_patience = valid_patience
        self._valid_min_delta = valid_min_delta
        self._valid_restore_cooldown = valid_restore_cooldown
        self._valid_smooth_window = max(1, int(valid_smooth_window))
        self._valid_hist: deque = deque(maxlen=self._valid_smooth_window)
        self._best_valid_ic: float = -999.0
        self._best_valid_snapshot: Optional[dict] = None
        self._valid_no_improve_count: int = 0
        self._valid_cooldown_count: int = 0
 
    def _on_step(self) -> bool:
        return True
 
    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
 
        # --- LLM assist ---
        if self.chat_session is not None:
            self._try_use_llm()
 
        # --- Record metrics ---
        pool = self.pool
        sig_count = int((np.abs(pool.weights[:pool.size]) > 1e-4).sum())
 
        # Compute train ensemble IC
        train_ic = pool.best_ic_ret
 
        # Compute validation IC (strictly from validation split)
        valid_ic_raw, valid_rank_ic_raw = pool.test_ensemble(self.valid_calculator)
        self.logger.record("valid/rank_ic_raw", valid_rank_ic_raw)

        # Compute test ICs (test split only)
        test_results = []
        n_days = sum(calc.data.n_days for calc in self.test_calculators)
        ic_mean, ric_mean = 0.0, 0.0
        for i, test_calc in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = pool.test_ensemble(test_calc)
            test_results.append((ic_test, rank_ic_test))
            if n_days > 0:
                ic_mean += ic_test * test_calc.data.n_days / n_days
                ric_mean += rank_ic_test * test_calc.data.n_days / n_days
            self.logger.record(f"test/ic_{i}", ic_test)
            self.logger.record(f"test/rank_ic_{i}", rank_ic_test)
 
        # --- Validation-based overfitting control ---
        self._valid_hist.append(valid_ic_raw)
        valid_ic = float(np.mean(self._valid_hist))
        if valid_ic >= self._best_valid_ic + self._valid_min_delta:
            self._best_valid_ic = valid_ic
            self._valid_no_improve_count = 0
            # Snapshot pool + state used by acceptance logic.
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
 
        # If validation IC has not improved for `valid_patience` rollouts,
        # restore the best pool snapshot to prevent further overfitting
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
 
        # SB3 logger
        self.logger.record("pool/size", pool.size)
        self.logger.record("pool/significant", sig_count)
        self.logger.record("pool/best_ic_ret", pool.best_ic_ret)
        self.logger.record("pool/eval_cnt", pool.eval_cnt)
        self.logger.record("test/ic_mean", ic_mean)
        self.logger.record("test/rank_ic_mean", ric_mean)
        self.logger.record("valid/ic_raw", valid_ic_raw)
        self.logger.record("valid/ic_smooth", valid_ic)
        self.logger.record("valid/best_ic", self._best_valid_ic)
        self.logger.record("valid/no_improve_count", self._valid_no_improve_count)
        self.logger.record("valid/cooldown_count", self._valid_cooldown_count)
 
        # Convergence logger
        self.conv_logger.record_step(
            timestep=self.num_timesteps,
            pool_size=pool.size,
            pool_significant=sig_count,
            pool_best_ic=pool.best_ic_ret,
            pool_eval_cnt=pool.eval_cnt,
            train_ic=train_ic,
            valid_ic=valid_ic_raw,
            valid_rank_ic=valid_rank_ic_raw,
            test_results=test_results,
        )
 
        # Save checkpoint + CSV
        self.save_checkpoint()
        self.conv_logger.save_csv()
 
        # Auto-plot periodically
        if self._rollout_count % self._plot_interval == 0:
            try:
                csv_path = os.path.join(self.conv_logger.save_dir, "convergence.csv")
                plot_convergence(csv_path)
                if self.verbose > 0:
                    print(f"[Convergence] Plot saved at step {self.num_timesteps}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[Convergence] Plot failed: {e}")
 
    def save_checkpoint(self):
        path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
        self.model.save(path)
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")
        with open(f"{path}_pool.json", "w") as f:
            json.dump(self.pool.to_json_dict(), f)
    
    def _restore_pool_snapshot(self) -> None:
        """Restore pool from the best validation snapshot."""
        if self._best_valid_snapshot is None:
            return
        pool = self.pool
        parser = build_level2_parser()
        snapshot = self._best_valid_snapshot
        pool_snapshot = snapshot.get("pool", snapshot)
        # Parse saved expressions
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
        # Clear pool and reload
        pool.leave_only([])
        pool.force_load_exprs(exprs, weights=weights)
        if "best_obj" in snapshot:
            pool.best_obj = float(snapshot["best_obj"])
        if "best_ic_ret" in snapshot:
            pool.best_ic_ret = float(snapshot["best_ic_ret"])
        if "eval_cnt" in snapshot:
            pool.eval_cnt = int(snapshot["eval_cnt"])
 
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
 
    def _aggressive_llm_inject(self, logger) -> None:
        """Original behavior: drop worst alphas, let LLM bulk-replace."""
        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed: {type(e).__name__}: {e}")
 
    def _gentle_llm_inject(self, logger) -> None:
        """
        Gentle injection: LLM generates candidates, each goes through
        the normal try_new_expr path. Pool decides whether to accept.
        No forced deletion — reward landscape changes gradually.
        """
        try:
            # Generate a textual report of the current pool for LLM context
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
 
            # Feed each candidate through the normal pool acceptance path
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
        env = self.training_env.envs[0]
        # Navigate through any wrapper chain to reach AlphaEnvCore
        while hasattr(env, 'env'):
            env = env.env
        return env
 
 
# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
 
def run_single_experiment(
    seed: int = 0,
    instruments: Union[str, List[str]] = "auto",
    pool_capacity: int = 10,
    steps: int = 200_000,
    data_root: str = "~/EquityLevel2/stock",
    use_level2_features: bool = False,
    # Data split
    train_start: str = "2023-03-01",
    train_end: str = "2023-06-30",
    valid_start: str = "2023-7-01",
    valid_end: str = "2023-7-31",
    test_start: str = "2023-8-01",
    test_end: str = "2023-8-31",
    # Bar config
    max_backtrack_bars: int = 80,
    max_future_bars: int = 80,
    cache_dir: Optional[str] = "./out/l2_cache",
    max_workers: int = 4,
    bar_size_min: float = 3.0,
    bar_size_sec: Optional[int] = None,
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
    # Multi-env parallelism
    n_envs: int = 1,
    # Diversity options
    ic_mut_threshold: float = 0.99,
    diversity_bonus: float = 0.0,
    # Convergence plot
    plot_interval: int = 10,
    # Validation rollback controls
    valid_patience: int = 20,
    valid_min_delta: float = 1e-4,
    valid_smooth_window: int = 1,
    valid_restore_cooldown: int = 3,
):
    """
    Train alpha factors with optional LLM assistance.
 
    Modes:
      - Pure RL (default): no LLM flags
      - LLM warm start only: --llm_warmstart
        LLM generates initial alpha pool, then RL takes over
      - Full LLM assist: --use_llm
        Warm start + periodic LLM candidate injection every N steps
    """
    reseed_everything(seed)
    features = LEVEL2_FEATURES if use_level2_features else BASIC_FEATURES
    feature_mode = "level2" if use_level2_features else "basic"
 
    # Determine tag
    if use_llm:
        tag = f"llm_d{drop_rl_n}"
    elif llm_warmstart:
        tag = "llm_warmstart"
    else:
        tag = "rl"
 
    effective_bar_size_min = (float(bar_size_sec) / 60.0) if bar_size_sec is not None else float(bar_size_min)
    print(f"""[Level2+LLM] Starting training
    Seed: {seed}
    Data root: {data_root}
    Feature mode: {feature_mode} ({len(features)} features)
    Bar size: {effective_bar_size_min:.4f} min ({effective_bar_size_min * 60:.1f} sec)
    Pool capacity: {pool_capacity}
    Steps: {steps}
    Mode: {tag}
    N envs: {n_envs}
    IC mut threshold: {ic_mut_threshold}
    Diversity bonus: {diversity_bonus}
    LLM warm start: {llm_warmstart or use_llm}
    LLM periodic assist: {use_llm}
    LLM invoke every: {llm_every_n_steps} steps
    LLM drop worst N: {drop_rl_n}
    LLM replace N: {llm_replace_n}
    Train: [{train_start}, {train_end}]
    Valid: [{valid_start}, {valid_end}]
    Test:  [{test_start}, {test_end}]""")
 
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name_prefix = f"l2_{feature_mode}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    # Parse instrument list if provided as JSON string
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
            bar_size_sec=bar_size_sec,
        )
 
    segments = [
        (train_start, train_end),
        (valid_start, valid_end),
        (test_start, test_end),
    ]
    print("[Level2+LLM] Loading datasets...")
    datasets = [get_dataset(*s) for s in segments]
    for i, (seg, ds) in enumerate(zip(segments, datasets)):
        print(
            f"  Segment {i}: {seg[0]} ~ {seg[1]}, "
            f"{ds.n_days} bars, {ds.n_stocks} stocks, {ds.n_features} features"
        )
    calculators = [Level2Calculator(d, target) for d in datasets]
 
    # --- Pool factory (needed for LLM interaction) ---
    use_diversity = diversity_bonus > 0 or ic_mut_threshold < 0.99
    def build_pool(exprs: Optional[List[Expression]] = None) -> MseAlphaPool:
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
        if exprs:
            pool.force_load_exprs(exprs)
        return pool
    if use_diversity:
        print(f"  Diversity pool: ic_mut_threshold={ic_mut_threshold}, bonus={diversity_bonus}")
 
    # --- LLM setup ---
    chat_session: Optional[InterativeSession] = None
    pool = build_pool()
 
    if llm_warmstart or use_llm:
        print("[Level2+LLM] Setting up LLM client...")
        chat_client = build_level2_chat_client(
            log_dir=save_path,
            use_level2_features=use_level2_features,
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
        )
        parser = build_level2_parser()
        inter = DefaultInteraction(
            parser,
            chat_client,
            build_pool,
            calculator_train=calculators[0],
            # IMPORTANT: only pass validation split here to avoid test leakage
            # into LLM prompt-guided pool updates.
            calculators_test=[calculators[1]],
            replace_k=llm_replace_n,
            forgetful=True,
        )
 
        # Warm start: LLM generates initial pool
        print("[Level2+LLM] LLM generating initial alpha pool...")
        pool = inter.run()
        print(f"[Level2+LLM] Initial pool: {pool.size} alphas, IC={pool.best_ic_ret:.4f}")
 
        if use_llm:
            chat_session = inter  # Keep session for periodic assist

    # --- Environment: single or multi-env parallel ---
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
 
    # --- Convergence logger ---
    conv_logger = ConvergenceLogger(save_dir=save_path)
 
    # --- Callback ---
    callback = Level2LLMCallback(
        save_path=save_path,
        valid_calculator=calculators[1],
        test_calculators=calculators[2:],
        convergence_logger=conv_logger,
        verbose=1,
        chat_session=chat_session,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n,
        plot_interval=plot_interval,
        gentle_inject=gentle_inject,
        valid_patience=valid_patience,
        valid_min_delta=valid_min_delta,
        valid_smooth_window=valid_smooth_window,
        valid_restore_cooldown=valid_restore_cooldown,
    )
 
    # --- PPO model (scale batch_size with n_envs) ---
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
        gamma=1.0,
        ent_coef=0.01,
        batch_size=batch_size,
        n_steps=n_steps,
        tensorboard_log="./out/tensorboard",
        device=device,
        verbose=1,
    )
 
    print("[Level2+LLM] Starting RL training...")
    model.learn(
        total_timesteps=steps,
        callback=callback,
        tb_log_name=name_prefix,
    )
 
    # --- Final convergence output ---
    csv_path = conv_logger.save_csv()
    conv_logger.save_json()
    try:
        plot_path = plot_convergence(csv_path)
        print(f"[Level2+LLM] Convergence plot: {plot_path}")
    except Exception as e:
        print(f"[Level2+LLM] Plot failed: {e}")
 
    summary = conv_logger.summary()
    print(f"[Level2+LLM] Training complete.")
    print(f"  Total steps: {summary.get('total_steps', 'N/A')}")
    print(f"  Best pool IC: {summary.get('best_pool_ic', 'N/A'):.4f} "
          f"(at step {summary.get('best_pool_ic_step', 'N/A')})")
    print(f"  Best test IC mean: {summary.get('best_test_ic_mean', 'N/A'):.4f} "
          f"(at step {summary.get('best_test_ic_mean_step', 'N/A')})")
    print(f"  Results: {save_path}")
 
 
def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: Union[str, List[str]] = "auto",
    data_root: str = "~/EquityLevel2/stock",
    use_level2_features: bool = False,
    steps: Optional[int] = None,
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
    # Multi-env parallelism
    n_envs: int = 1,
    # Diversity options
    ic_mut_threshold: float = 0.99,
    diversity_bonus: float = 0.0,
    # Data split
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
    plot_interval: int = 10,
    valid_patience: int = 20,
    valid_min_delta: float = 1e-4,
    valid_smooth_window: int = 1,
    valid_restore_cooldown: int = 3,
):
    """
    Train alpha factors using Level 2 HDF5 data with optional LLM assistance.
 
    :param random_seeds: Random seed(s)
    :param pool_capacity: Maximum alpha pool size
    :param instruments: Stock list (JSON array) or "auto" for auto-discovery
    :param data_root: Root path to Level 2 HDF5 data
    :param use_level2_features: Use extended Level 2 features (20) vs basic OHLCV (6)
    :param steps: Total RL iteration steps (None = auto based on pool_capacity)
    :param llm_warmstart: Use LLM to generate initial alpha pool (warm start only)
    :param use_llm: Enable full LLM assist (warm start + periodic replacement)
    :param llm_every_n_steps: Invoke LLM every N steps (only when use_llm=True)
    :param drop_rl_n: Drop N worst alphas before LLM invocation
    :param llm_replace_n: Number of new alphas LLM generates per invocation
    :param llm_base_url: OpenAI-compatible API base URL
    :param llm_api_key: API key
    :param llm_model: Model name
    :param gentle_inject: Use gentle injection (True) vs aggressive replacement (False)
    :param n_envs: Number of parallel environments (1=single, 4-8=recommended)
    :param ic_mut_threshold: Mutual IC threshold to reject correlated alphas (0.7=strict, 0.99=permissive)
    :param diversity_bonus: Reward bonus for novel alphas (0=off, 0.05-0.2=typical)
    :param bar_size_sec: Bar size in seconds; if set, overrides bar_size_min
    :param plot_interval: Save convergence plot every N rollout ends
    :param valid_patience: Rollouts without validation improvement before rollback
    :param valid_min_delta: Minimum increase in validation IC to count as improvement
    :param valid_smooth_window: Smoothing window size for validation IC in early stopping
    :param valid_restore_cooldown: Cooldown rollouts after rollback before counting stale steps
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
            bar_size_sec=bar_size_sec,
            llm_warmstart=llm_warmstart,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps,
            drop_rl_n=drop_rl_n,
            llm_replace_n=llm_replace_n,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            gentle_inject=gentle_inject,
            n_envs=n_envs,
            ic_mut_threshold=ic_mut_threshold,
            diversity_bonus=diversity_bonus,
            plot_interval=plot_interval,
            valid_patience=valid_patience,
            valid_min_delta=valid_min_delta,
            valid_smooth_window=valid_smooth_window,
            valid_restore_cooldown=valid_restore_cooldown,
        )
 
 
if __name__ == "__main__":
    fire.Fire(main)

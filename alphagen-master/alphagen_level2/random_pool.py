"""
Random-window single-factor alpha pool.

Wraps ``SingleFactorAlphaPool`` so that *every* call to ``try_new_expr``
evaluates the candidate against a freshly-sampled 1200-bar window of the
underlying long-horizon dataset, with the reward being the risk-adjusted
score from :mod:`alphagen_level2.risk_reward`.

The pool keeps the existing dedup/capacity machinery from the parent class;
the only behaviour change is what ``_score_expr`` returns and what data the
calculator points at when ``_calc_ics`` is invoked.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import math
import torch

from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen.models.linear_alpha_pool import SingleFactorAlphaPool
from alphagen_level2.calculator_tick import TickCalculator
from alphagen_level2.random_window import RandomWindowSampler
from alphagen_level2.risk_reward import (
    RewardWeights, compute_risk_components, shape_reward,
)
from alphagen_level2.stock_data_tick import TickStockData


class RandomWindowSingleFactorPool(SingleFactorAlphaPool):
    """Single-factor pool with per-episode random-window scoring.

    Parameters
    ----------
    sampler:
        ``RandomWindowSampler`` over the full dataset.
    target_expr:
        Expression used as the calculator's "target" (e.g. forward 100-bar
        return).  Reused when re-binding the calculator to a fresh window.
    score_windows:
        Number of windows averaged when computing a candidate's reward.
        Higher values -> less noise, more compute.
    reward_weights:
        ``RewardWeights`` controlling Sortino / drawdown / fat-tail / decorr.
    """

    def __init__(
        self,
        capacity: int,
        sampler: RandomWindowSampler,
        target_expr: Optional[Expression] = None,
        score_windows: int = 4,
        reward_weights: Optional[RewardWeights] = None,
        ic_mut_threshold: Optional[float] = 0.95,
        device: torch.device = torch.device("cpu"),
        # Reward knobs (mirror the parent's signature so kwargs pass through).
        holding_bars: int = 100,
        execution_delay: int = 1,
        lookback_bars: int = 1200,
        turnover_cost: float = 0.0006,
        trivial_penalty: float = 0.0,
        complexity_metric=None,
    ):
        # Build an initial calculator at a random window so the parent ctor has
        # something to chew on; it'll be replaced each episode.
        init_view = sampler.sample()
        init_calc = TickCalculator(
            init_view, target=target_expr,
            holding_bars=holding_bars,
            execution_delay=execution_delay,
            lookback_bars=lookback_bars,
        )
        super().__init__(
            capacity=capacity,
            calculator=init_calc,
            ic_lower_bound=None,
            l1_alpha=0.0,
            device=device,
            ic_mut_threshold=ic_mut_threshold,
            holding_bars=holding_bars,
            execution_delay=execution_delay,
            lookback_bars=lookback_bars,
            turnover_cost=turnover_cost,
            alpha=1.0, beta=1.0, gamma=1.0,
            tau_ic=0.1, tau_r=1e-3, tau_c=1e-4,
            trivial_penalty=trivial_penalty,
        )
        self._sampler = sampler
        self._target_expr = target_expr
        self._score_windows = max(1, int(score_windows))
        self._reward_weights = reward_weights or RewardWeights()
        self._complexity_metric = complexity_metric or (lambda e: 0)
        # Latest reward breakdown (populated by ``_score_expr``); useful for
        # logging in the training loop.
        self.last_reward_info: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def resample_window(self) -> None:
        """Bind the calculator to a freshly-sampled window."""
        view = self._sampler.sample()
        self.calculator = TickCalculator(
            view, target=self._target_expr,
            holding_bars=self.holding_bars,
            execution_delay=self.execution_delay,
            lookback_bars=self.lookback_bars,
        )

    # ------------------------------------------------------------------
    # try_new_expr: resample window before parent runs IC filter + scoring
    # ------------------------------------------------------------------

    def try_new_expr(self, expr: Expression) -> float:
        self.resample_window()
        return super().try_new_expr(expr)

    # ------------------------------------------------------------------
    # Reward scoring (override)
    # ------------------------------------------------------------------

    def _avg_corr_with_pool(self, expr: Expression) -> float:
        """Average |IC| of `expr` with current pool members (rough decorr)."""
        if self.size == 0:
            return 0.0
        try:
            cs = []
            for i in range(self.size):
                other = self.exprs[i]
                if other is None:
                    continue
                ic = self.calculator.calc_mutual_IC(expr, other)
                if ic is None or math.isnan(ic):
                    continue
                cs.append(abs(float(ic)))
            return sum(cs) / max(len(cs), 1)
        except Exception:
            return 0.0

    def _score_expr(
        self, expr: Expression,
    ) -> Tuple[float, float, float, float]:
        """Risk-adjusted reward averaged over ``score_windows`` random windows.

        Returns (reward, signed_ic, mean_pnl, direction).
        """
        rewards: List[float] = []
        ics: List[float] = []
        rbars: List[float] = []
        infos: List[Dict[str, float]] = []
        avg_corr = self._avg_corr_with_pool(expr)
        complexity = int(self._complexity_metric(expr))

        for k in range(self._score_windows):
            if k > 0:
                self.resample_window()
            try:
                rc = compute_risk_components(
                    expr, self.calculator,
                    future_bars=self.holding_bars,
                    execution_delay=self.execution_delay,
                    lookback_bars=self.lookback_bars,
                    turnover_cost=self.turnover_cost,
                )
            except OutOfDataRangeError:
                rc = None
            if rc is None:
                continue
            info = shape_reward(
                rc, self._reward_weights,
                avg_corr_with_pool=avg_corr,
                expr_complexity=complexity,
            )
            rewards.append(info["reward"])
            ics.append(rc.ic)
            rbars.append(rc.mean_pnl)
            infos.append(info)

        if not rewards:
            self.last_reward_info = {"reward": 0.0, "n_windows": 0}
            return 0.0, 0.0, 0.0, 1.0

        reward = sum(rewards) / len(rewards)
        ic = sum(ics) / len(ics)
        rbar = sum(rbars) / len(rbars)
        direction = 1.0 if ic >= 0 else -1.0

        # Aggregate per-component means for later inspection.
        agg: Dict[str, float] = {"n_windows": float(len(rewards))}
        keys = set().union(*[set(i.keys()) for i in infos])
        for key in keys:
            vals = [i.get(key, 0.0) for i in infos if key in i]
            if vals:
                agg[key] = sum(vals) / len(vals)
        self.last_reward_info = agg

        # Push into parent's stats buffers so dump_reward_stats still works.
        s = self._reward_stats
        s["abs_ic"].append(abs(float(ic)))
        s["r_bar"].append(float(rbar))
        s["tc"].append(float(agg.get("turnover", 0.0)))
        s["comp_ic"].append(float(agg.get("base", 0.0)))
        s["comp_r"].append(float(agg.get("dd_pen", 0.0)))
        s["comp_tc"].append(float(agg.get("fat_pen", 0.0)))
        s["reward"].append(float(reward))
        s["pos_abs_mean"].append(float(agg.get("decorr", 0.0)))

        return float(reward), float(ic), float(rbar), float(direction)

    # ------------------------------------------------------------------
    # Score helper used by GP evolution
    # ------------------------------------------------------------------

    def score_candidate(self, expr: Expression) -> Optional[float]:
        """Quick score for an externally-generated candidate (no insertion).

        Used by the GP evolution module.
        """
        try:
            reward, _ic, _r, _d = self._score_expr(expr)
        except (OutOfDataRangeError, RuntimeError, ValueError):
            return None
        return float(reward)

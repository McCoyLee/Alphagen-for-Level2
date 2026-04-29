"""
Risk-adjusted reward for the random-window training loop.

Given a factor expression and a ``TickStockData`` view, produce reward
components driven by the *bar-by-bar* PnL of holding the factor's z-score
position for the next ``future_bars`` bars.  Mirrors the cost / position
conventions of ``backtest_tick_pnl.py`` so reward and backtest agree.

PnL convention (per bar)
------------------------
At every bar t inside the observation window we form a position
``p_t = clip(zscore(factor_t, lookback), -3, 3) / 3`` whose direction follows
``sign(IC_train)``.  The realised return between bar ``t+delay`` and
``t+delay+1`` is the 1-bar mid-price return ``r_{t+1}``.  The bar PnL is

    pnl_t = sign(IC) * p_t * r_{t+1}  -  turnover_cost * |p_t - p_{t-1}|

We compute the FUTURE 100-bar PnL series by iterating ``t`` over the last
``future_bars`` indices that fit inside the slice — this is the model's
forward-looking sample for the reward.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import math
import torch
from torch import Tensor

from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen_level2.calculator_tick import TickCalculator
from alphagen_level2.stock_data_tick import TickFeatureType


@dataclass
class RiskComponents:
    sortino: float
    sharpe: float
    mean_pnl: float
    max_drawdown: float
    max_abs_bar_ret: float
    excess_kurt: float
    ic: float
    rank_ic: float
    turnover: float
    n_valid: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _safe_std(x: Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    return float(x.std(unbiased=False).item())


def _drawdown(cum: Tensor) -> float:
    if cum.numel() == 0:
        return 0.0
    running_max = torch.cummax(cum, dim=0).values
    dd = running_max - cum
    return float(dd.max().item())


def _kurtosis(x: Tensor) -> float:
    if x.numel() < 4:
        return 0.0
    mu = x.mean()
    centered = x - mu
    var = (centered ** 2).mean()
    if var.item() < 1e-18:
        return 0.0
    m4 = (centered ** 4).mean()
    return float((m4 / (var * var)).item() - 3.0)


def compute_risk_components(
    expr: Expression,
    calculator: TickCalculator,
    future_bars: int = 100,
    execution_delay: int = 1,
    lookback_bars: int = 1200,
    turnover_cost: float = 0.0006,
) -> Optional[RiskComponents]:
    """Compute risk-adjusted reward primitives for a single factor.

    The position is sized by the rolling z-score (matching the existing
    ``calc_single_reward_components`` convention).  The PnL series used for
    Sortino / drawdown / fat-tail is the *future* ``future_bars`` bar-by-bar
    realised PnL — i.e. the same horizon the spec asks the model to predict.

    Returns ``None`` when there are not enough valid bars.
    """
    data = calculator.data
    try:
        x_raw = expr.evaluate(data)
    except OutOfDataRangeError:
        return None
    if not calculator._single_instrument:
        # Multi-instrument support is out of scope for this reward.
        return None

    x = x_raw.squeeze(-1)
    T = x.shape[0]
    L = int(lookback_bars)
    H = int(future_bars)
    D = int(execution_delay)
    if T < L + H + D + 4:
        return None

    z = calculator._rolling_zscore_1d(x, L)
    p = torch.clamp(z, -3.0, 3.0) / 3.0

    mid_idx = int(TickFeatureType.MID)
    mid_raw = data.data[:, mid_idx, 0]
    bt = data.max_backtrack_days
    max_valid = mid_raw.shape[0] - 1

    # 1-bar forward returns (used for bar-by-bar PnL of the future window).
    t_idx = torch.arange(T, device=x.device)
    enter = bt + t_idx + D
    exit_ = enter + 1
    valid = (enter >= 0) & (exit_ <= max_valid)
    enter_s = enter.clamp(0, max_valid)
    exit_s = exit_.clamp(0, max_valid)
    m_in = mid_raw[enter_s]
    m_out = mid_raw[exit_s]
    bar_ret = torch.where(
        valid & (m_in > 0) & ~torch.isnan(m_in) & ~torch.isnan(m_out),
        m_out / m_in - 1.0,
        torch.full_like(m_in, float("nan")),
    )

    # IC for direction — computed on the *observation* window, i.e. the
    # leading T - H - D - 1 bars where horizon-H forward return is defined.
    horizon_ret = _horizon_return(mid_raw, bt, t_idx, D, H, max_valid)
    obs_mask = (~torch.isnan(z)) & (~torch.isnan(horizon_ret))
    obs_mask[-(H + D + 1):] = False  # leave the trailing window for the future PnL
    if int(obs_mask.sum().item()) < 8:
        return None

    z_obs = z[obs_mask]
    r_obs = horizon_ret[obs_mask]
    ic = calculator._pearson_1d(z_obs, r_obs)
    rank_ic = calculator._pearson_1d(
        calculator._rank_1d(z_obs), calculator._rank_1d(r_obs)
    )
    sign_ic = 1.0 if ic >= 0 else -1.0

    # Future 100-bar PnL series
    future_lo = T - H - D - 1
    future_hi = T - D - 1
    p_fut = p[future_lo:future_hi]
    r_fut = bar_ret[future_lo:future_hi]
    fut_mask = (~torch.isnan(p_fut)) & (~torch.isnan(r_fut))
    if int(fut_mask.sum().item()) < max(8, H // 4):
        return None

    p_fut = torch.where(fut_mask, p_fut, torch.zeros_like(p_fut))
    r_fut = torch.where(fut_mask, r_fut, torch.zeros_like(r_fut))

    gross = sign_ic * p_fut * r_fut
    # Per-bar turnover cost (matches backtest's |Δp| * cost_frac convention).
    if p_fut.numel() >= 2:
        dp = torch.zeros_like(p_fut)
        dp[1:] = (p_fut[1:] - p_fut[:-1]).abs()
        cost = turnover_cost * dp
    else:
        cost = torch.zeros_like(p_fut)
    pnl = gross - cost

    mean_pnl = float(pnl.mean().item())
    sd_all = _safe_std(pnl)
    downside = pnl[pnl < 0]
    sd_dn = _safe_std(downside) if downside.numel() >= 2 else sd_all
    sortino = mean_pnl / sd_dn if sd_dn > 1e-12 else 0.0
    sharpe = mean_pnl / sd_all if sd_all > 1e-12 else 0.0
    cum = torch.cumsum(pnl, dim=0)
    max_dd = _drawdown(cum)
    max_abs = float(pnl.abs().max().item())
    kurt = _kurtosis(pnl)
    turnover = float(cost.sum().item())

    return RiskComponents(
        sortino=float(sortino),
        sharpe=float(sharpe),
        mean_pnl=mean_pnl,
        max_drawdown=max_dd,
        max_abs_bar_ret=max_abs,
        excess_kurt=kurt,
        ic=float(ic),
        rank_ic=float(rank_ic),
        turnover=turnover,
        n_valid=int(fut_mask.sum().item()),
    )


def _horizon_return(
    mid_raw: Tensor, bt: int, t_idx: Tensor, D: int, H: int, max_valid: int,
) -> Tensor:
    enter = bt + t_idx + D
    exit_ = enter + H
    valid = (enter >= 0) & (exit_ <= max_valid)
    e_s = enter.clamp(0, max_valid)
    x_s = exit_.clamp(0, max_valid)
    m_in = mid_raw[e_s]
    m_out = mid_raw[x_s]
    return torch.where(
        valid & (m_in > 0) & ~torch.isnan(m_in) & ~torch.isnan(m_out),
        m_out / m_in - 1.0,
        torch.full_like(m_in, float("nan")),
    )


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    sortino: float = 1.0
    sharpe: float = 0.0
    ic: float = 0.5
    max_dd_threshold: float = 0.02
    max_dd_penalty: float = 2.0
    fat_tail_threshold: float = 0.01
    fat_tail_penalty: float = 2.0
    kurt_threshold: float = 8.0
    kurt_penalty: float = 1.0
    decorrelation_bonus: float = 0.5
    complexity_penalty: float = 0.005


def shape_reward(
    rc: RiskComponents,
    weights: RewardWeights,
    avg_corr_with_pool: float = 0.0,
    expr_complexity: int = 0,
) -> Dict[str, float]:
    """Combine ``RiskComponents`` into a scalar reward + breakdown.

    All risk metrics are squashed by ``tanh`` so the reward is bounded.
    Penalties are *additive* and only fire above their threshold so that
    well-behaved factors are never punished gratuitously.
    """
    w = weights
    base = (
        w.sortino * math.tanh(rc.sortino)
        + w.sharpe * math.tanh(rc.sharpe)
        + w.ic * math.tanh(rc.rank_ic / 0.05)
    )
    dd_breach = max(0.0, rc.max_drawdown - w.max_dd_threshold)
    dd_pen = -w.max_dd_penalty * math.tanh(dd_breach / max(w.max_dd_threshold, 1e-6))

    fat_breach = max(0.0, rc.max_abs_bar_ret - w.fat_tail_threshold)
    fat_pen = -w.fat_tail_penalty * math.tanh(fat_breach / max(w.fat_tail_threshold, 1e-6))

    kurt_breach = max(0.0, rc.excess_kurt - w.kurt_threshold)
    kurt_pen = -w.kurt_penalty * math.tanh(kurt_breach / max(w.kurt_threshold, 1e-6))

    decorr = w.decorrelation_bonus * (1.0 - max(0.0, min(1.0, avg_corr_with_pool)))
    complexity = -w.complexity_penalty * float(expr_complexity)

    reward = base + dd_pen + fat_pen + kurt_pen + decorr + complexity
    return {
        "reward":     float(reward),
        "base":       float(base),
        "dd_pen":     float(dd_pen),
        "fat_pen":    float(fat_pen),
        "kurt_pen":   float(kurt_pen),
        "decorr":     float(decorr),
        "complexity": float(complexity),
        **rc.as_dict(),
    }

"""
Calculator for tick-level (3s bar) data.

Drop-in replacement for Level2Calculator, using TickStockData.
"""

from typing import Optional, Dict
import torch
from torch import Tensor
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_level2.stock_data_tick import TickStockData, TickFeatureType


class TickCalculator(TensorAlphaCalculator):
    """
    Calculator for tick-level 3s bar data.

    Uses TickStockData instead of Level2StockData.
    """

    def __init__(self, data: TickStockData, target: Optional[Expression] = None):
        self.data = data
        self._single_instrument = data.n_stocks == 1
        target_tensor = None
        if target is not None:
            raw_target_tensor = target.evaluate(data)
            target_tensor = (
                self._normalize_single(raw_target_tensor)
                if self._single_instrument else normalize_by_day(raw_target_tensor)
            )
        super().__init__(target_tensor)

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        value = expr.evaluate(self.data)
        if self._single_instrument:
            return self._normalize_single(value)
        return normalize_by_day(value)

    def _normalize_single(self, value: Tensor) -> Tensor:
        # value: [days, 1], normalize along time for single-instrument setup
        x = value.squeeze(-1)
        mask = torch.isnan(x)
        valid = ~mask
        if valid.sum() <= 1:
            return torch.zeros_like(value)
        xv = x[valid]
        mean = xv.mean()
        std = xv.std(unbiased=False)
        if std < 1e-6:
            z = torch.zeros_like(x)
        else:
            z = (x - mean) / std
            z[mask] = 0.
        return z.unsqueeze(-1)

    @staticmethod
    def _rank_1d(x: Tensor) -> Tensor:
        mask = torch.isnan(x)
        valid = ~mask
        r = torch.zeros_like(x)
        if valid.sum() <= 1:
            return r
        xv = x[valid]
        order = torch.argsort(xv)
        ranks = torch.empty_like(xv)
        ranks[order] = torch.arange(len(xv), device=x.device, dtype=x.dtype)
        r[valid] = ranks
        return r

    @staticmethod
    def _pearson_1d(x: Tensor, y: Tensor) -> float:
        mask = torch.isnan(x) | torch.isnan(y)
        x = x[~mask]
        y = y[~mask]
        if x.numel() <= 1:
            return 0.0
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt((x * x).sum() * (y * y).sum())
        if denom < 1e-12:
            return 0.0
        return float((x * y).sum() / denom)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        if not self._single_instrument:
            return super()._calc_IC(value1, value2)
        return self._pearson_1d(value1.squeeze(-1), value2.squeeze(-1))

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        if not self._single_instrument:
            return super()._calc_rIC(value1, value2)
        x = self._rank_1d(value1.squeeze(-1))
        y = self._rank_1d(value2.squeeze(-1))
        return self._pearson_1d(x, y)

    @property
    def n_days(self) -> int:
        return self.data.n_days

    # ------------------------------------------------------------------
    # Rolling z-score + forward-return reward components
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_zscore_1d(
        x: Tensor, window: int, eps: float = 1e-8,
    ) -> Tensor:
        """Rolling z-score using *previous* `window` bars (strictly before t).

        For each t, mu_t = mean(x[t-window : t]), sigma_t likewise.
        Returns a tensor of the same shape as `x`; entries with insufficient
        history (t < window) or NaN current value are set to NaN.
        """
        T = x.shape[0]
        device = x.device
        dtype = x.dtype
        mask_nan = torch.isnan(x)
        x_c = torch.where(mask_nan, torch.zeros_like(x), x)
        one = (~mask_nan).to(dtype)

        zero = torch.zeros(1, device=device, dtype=dtype)
        cum_x  = torch.cat([zero, torch.cumsum(x_c, dim=0)])
        cum_x2 = torch.cat([zero, torch.cumsum(x_c * x_c, dim=0)])
        cum_n  = torch.cat([zero, torch.cumsum(one, dim=0)])

        t_idx = torch.arange(1, T + 1, device=device)
        lo = (t_idx - window).clamp(min=0)
        s  = cum_x[t_idx]  - cum_x[lo]
        s2 = cum_x2[t_idx] - cum_x2[lo]
        n  = cum_n[t_idx]  - cum_n[lo]

        n_safe = n.clamp(min=1.0)
        mu = s / n_safe
        var = (s2 / n_safe) - mu * mu
        var = var.clamp(min=0.0)
        sigma = torch.sqrt(var)

        z = (x - mu) / (sigma + eps)
        insufficient = n < window
        invalid = insufficient | mask_nan
        z = torch.where(invalid, torch.full_like(z, float('nan')), z)
        return z

    def calc_single_reward_components(
        self,
        expr: Expression,
        holding_bars: int = 100,
        execution_delay: int = 1,
        lookback_bars: int = 1200,
        turnover_cost: float = 0.0006,
    ) -> Optional[Dict[str, float]]:
        """Compute per-image reward primitives for a single factor.

        Formulas:
            z_t = (x_t - mu_t) / (sigma_t + eps),  (mu, sigma over previous `lookback_bars`)
            p_t = clip(z_t, -3, 3) / 3
            ret_t = mid[t + delay + H] / mid[t + delay] - 1
            IC = rank-corr(z_t, ret_t)   (over the full training span)
            r_t = sign(IC) * p_t * ret_t              (gross per-bar pnl)
            r_bar = mean(r_t)
            tc    = mean(turnover_cost * |p_t - p_{t-1}|)   (turnover cost as separate penalty)

        Returns:
            dict with keys {ic, abs_ic, r_bar, tc, n_valid, pos_abs_mean}
            or None if fewer than 2 valid bars.
        """
        if not self._single_instrument:
            raise NotImplementedError(
                "calc_single_reward_components is only implemented for single-instrument data."
            )

        data = self.data
        try:
            x_raw = expr.evaluate(data)
        except OutOfDataRangeError:
            return None
        x = x_raw.squeeze(-1)  # (T,)
        T = x.shape[0]
        if T < lookback_bars + holding_bars + execution_delay + 2:
            return None

        # Rolling z-score and position
        z = self._rolling_zscore_1d(x, lookback_bars)
        p = torch.clamp(z, -3.0, 3.0) / 3.0

        # Forward return via direct raw-mid indexing (avoids mutating the
        # data-loading buffers). For each usable t, we read mid at
        # (bt + t + delay) and (bt + t + delay + H).
        mid_idx = int(TickFeatureType.MID)
        mid_raw = data.data[:, mid_idx, 0]  # (n_bars_total,)
        bt = data.max_backtrack_days
        max_valid = mid_raw.shape[0] - 1

        t_idx = torch.arange(T, device=x.device)
        enter = bt + t_idx + execution_delay
        exit_ = bt + t_idx + execution_delay + holding_bars

        valid_idx = (enter >= 0) & (exit_ <= max_valid)
        enter_s = enter.clamp(0, max_valid)
        exit_s  = exit_.clamp(0, max_valid)
        m_enter = mid_raw[enter_s]
        m_exit  = mid_raw[exit_s]

        ret = torch.where(
            valid_idx & (m_enter > 0) & ~torch.isnan(m_enter) & ~torch.isnan(m_exit),
            m_exit / m_enter - 1.0,
            torch.full_like(m_enter, float('nan')),
        )

        mask = (~torch.isnan(z)) & (~torch.isnan(ret))
        if int(mask.sum().item()) < 2:
            return None

        z_v = z[mask]
        p_v = p[mask]
        r_v = ret[mask]

        # Rank IC of z_t vs forward return (Spearman); sign determines direction
        ic = self._pearson_1d(self._rank_1d(z_v), self._rank_1d(r_v))
        sign_ic = 1.0 if ic >= 0 else -1.0

        # Gross per-bar pnl (transaction cost moved to separate TC penalty)
        r_t = sign_ic * p_v * r_v
        r_bar = float(r_t.mean().item())

        # Turnover cost: TC = (1/N) * sum_t lambda_c * |p_t - p_{t-1}|
        if p_v.shape[0] >= 2:
            tc = float(turnover_cost * (p_v[1:] - p_v[:-1]).abs().sum().item()
                       / p_v.shape[0])
        else:
            tc = 0.0

        return {
            "ic": float(ic),
            "abs_ic": abs(float(ic)),
            "r_bar": r_bar,
            "tc": tc,
            "n_valid": int(mask.sum().item()),
            "pos_abs_mean": float(p_v.abs().mean().item()),
        }

"""
Calculator for tick-level (3s bar) data.

Drop-in replacement for Level2Calculator, using TickStockData.
"""

from typing import Optional
import torch
from torch import Tensor
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_level2.stock_data_tick import TickStockData


class TickCalculator(TensorAlphaCalculator):
    """
    Calculator for tick-level 3s bar data.

    Uses TickStockData instead of Level2StockData.
    """

    def __init__(self, data: TickStockData, target: Optional[Expression] = None):
        self.data = data
        self._single_instrument = data.n_stocks == 1
        target_tensor = None
        raw_target_tensor = None
        if target is not None:
            raw_target_tensor = target.evaluate(data)
            target_tensor = (
                self._normalize_single(raw_target_tensor)
                if self._single_instrument else normalize_by_day(raw_target_tensor)
            )
        super().__init__(
            target_tensor,
            raw_target=raw_target_tensor,
        )

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

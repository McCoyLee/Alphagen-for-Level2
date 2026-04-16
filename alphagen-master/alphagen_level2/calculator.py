"""
Calculator for Level 2 data - compatible with the existing AlphaCalculator interface.

This is a thin wrapper that adapts Level2StockData for use with the expression
evaluation and IC/RankIC calculation system.
"""

from typing import Optional
from torch import Tensor
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_level2.stock_data import Level2StockData


class Level2Calculator(TensorAlphaCalculator):
    """
    Drop-in replacement for QLibStockDataCalculator.

    Uses Level2StockData instead of StockData, but the evaluation
    and IC calculation logic is identical.
    """

    def __init__(self, data: Level2StockData, target: Optional[Expression] = None):
        raw = target.evaluate(data) if target is not None else None
        normed = normalize_by_day(raw) if raw is not None else None
        super().__init__(normed, raw_target=raw)
        self.data = data

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    @property
    def n_days(self) -> int:
        return self.data.n_days

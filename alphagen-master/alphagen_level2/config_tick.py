"""
Configuration for 3-second bar (tick-level) alpha generation.

Bar mode: DELTA_TIMES are in units of 3s bars.
  20 bars =  1 min
  100 bars =  5 min
  600 bars = 30 min
  1200 bars =  1 h
  4800 bars ≈  1 day (4h session = 14,400s / 3s)
"""

from typing import Type, List
from alphagen.data.expression import *
from alphagen_level2.stock_data_tick import TickFeatureType


MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 256

OPERATORS: List[Type[Operator]] = [
    # Unary
    Abs, Log,
    # Binary
    Add, Sub, Mul, Div, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var,
    Max, Min,
    Med, Mad,
    Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]

# 3s-bar delta times (units = bars)
# 20=1min, 100=5min, 600=30min, 1200=1h, 4800≈1day
DELTA_TIMES = [10, 20, 100, 600, 1200]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.

# All 20 tick features
TICK_FEATURES: List[TickFeatureType] = list(TickFeatureType)

# Minimal subset (OHLCV + ret + vwap) for quick tests
TICK_FEATURES_BASIC: List[TickFeatureType] = [
    TickFeatureType.OPEN,
    TickFeatureType.HIGH,
    TickFeatureType.LOW,
    TickFeatureType.CLOSE,
    TickFeatureType.RET,
    TickFeatureType.VOLUME,
    TickFeatureType.VWAP,
]

"""
Configuration for Level 2 alpha generation.

Bar mode: DELTA_TIMES are in units of bars (not days).
For 3s bars: 20 bars = 1min, 1200 bars = 1h, ~4800 bars ≈ 1 day (4h session).
For 3min bars: 5 bars = 15min, 20 bars = 1h, 80 bars ≈ 1 day.
"""
from typing import Type, List
from alphagen.data.expression import *
from alphagen_level2.stock_data import Level2FeatureType

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

# ---------------------------------------------------------------------------
# Delta-time presets by bar size
# ---------------------------------------------------------------------------
# For 3s bars: 20=1min, 100=5min, 600=30min, 1200=1h, 4800≈1day
DELTA_TIMES_3S = [20, 100, 600, 1200, 4800]
# For 3min bars: 5=15min, 10=30min, 20=1h, 40=2h, 80≈1day
DELTA_TIMES_3MIN = [5, 10, 20, 40, 80]

# Active delta times (switched at runtime by rolling script)
DELTA_TIMES = DELTA_TIMES_3S

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]
REWARD_PER_STEP = 0.

# Level 2 feature set for the RL action space
LEVEL2_FEATURES: List[Level2FeatureType] = list(Level2FeatureType)

# Basic feature set (OHLCV + VWAP only) - compatible with original pipeline
BASIC_FEATURES: List[Level2FeatureType] = [
    Level2FeatureType.OPEN,
    Level2FeatureType.CLOSE,
    Level2FeatureType.HIGH,
    Level2FeatureType.LOW,
    Level2FeatureType.VOLUME,
    Level2FeatureType.VWAP,
]


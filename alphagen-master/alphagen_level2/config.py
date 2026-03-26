"""
Configuration for Level 2 alpha generation.

Extends the base config with Level 2 features available in the action space.
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

DELTA_TIMES = [1, 5, 10, 20, 40]

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

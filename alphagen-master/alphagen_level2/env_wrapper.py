"""
Level 2 RL environment wrapper.

Extends the action space to include Level 2 features while maintaining
full compatibility with the original AlphaEnvCore.
"""

from typing import Tuple, List, Optional
import gymnasium as gym
import numpy as np

from alphagen_level2.config import (
    OPERATORS, DELTA_TIMES, CONSTANTS, LEVEL2_FEATURES,
    BASIC_FEATURES, MAX_EXPR_LENGTH, REWARD_PER_STEP,
)
from alphagen_level2.stock_data import Level2FeatureType
from alphagen.data.tokens import (
    Token, OperatorToken, FeatureToken, ConstantToken,
    DeltaTimeToken, SequenceIndicatorToken, SequenceIndicatorType,
    ExpressionToken,
)
from alphagen.data.expression import (
    Expression, Operator, UnaryOperator, BinaryOperator,
    RollingOperator, PairRollingOperator,
)
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.core import AlphaEnvCore


class Level2EnvWrapper(gym.Wrapper):
    """
    RL environment wrapper with Level 2 feature action space.

    When use_level2_features=True, the action space includes all Level2FeatureType
    features. When False, only the basic 6 OHLCV+VWAP features are available
    (backward compatible with the original).
    """
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(
        self,
        env: AlphaEnvCore,
        use_level2_features: bool = True,
        subexprs: Optional[List[Expression]] = None,
    ):
        super().__init__(env)
        self.subexprs = subexprs or []
        self._features = LEVEL2_FEATURES if use_level2_features else BASIC_FEATURES

        self.size_op = len(OPERATORS)
        self.size_feature = len(self._features)
        self.size_constant = len(CONSTANTS)
        self.size_delta_time = len(DELTA_TIMES)
        self.size_subexpr = len(self.subexprs)
        self.size_sep = 1

        self.size_action = (
            self.size_op + self.size_feature + self.size_constant +
            self.size_delta_time + self.size_subexpr + self.size_sep
        )
        self.action_space = gym.spaces.Discrete(self.size_action)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.size_action,
            shape=(MAX_EXPR_LENGTH,),
            dtype=np.uint8
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset()
        return self.state, {}

    def step(self, action: int):
        _, reward, done, truncated, info = self.env.step(self.action(action))
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        return self.state, self.reward(reward), done, truncated, info

    def action(self, action: int) -> Token:
        return self.action_to_token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()

        offset = 0
        # Operators
        for i in range(self.size_op):
            if valid['op'][OPERATORS[i].category_type()]:
                res[offset + i] = True
        offset += self.size_op

        # Features
        if valid['select'][1]:
            res[offset:offset + self.size_feature] = True
        offset += self.size_feature

        # Constants
        if valid['select'][2]:
            res[offset:offset + self.size_constant] = True
        offset += self.size_constant

        # Delta time
        if valid['select'][3]:
            res[offset:offset + self.size_delta_time] = True
        offset += self.size_delta_time

        # Sub-expressions
        if valid['select'][1]:
            res[offset:offset + self.size_subexpr] = True
        offset += self.size_subexpr

        # SEP
        if valid['select'][4]:
            res[offset] = True

        return res

    def action_to_token(self, action: int) -> Token:
        if action < 0:
            raise ValueError(f"Invalid action: {action}")

        if action < self.size_op:
            return OperatorToken(OPERATORS[action])
        action -= self.size_op

        if action < self.size_feature:
            return FeatureToken(self._features[action])
        action -= self.size_feature

        if action < self.size_constant:
            return ConstantToken(CONSTANTS[action])
        action -= self.size_constant

        if action < self.size_delta_time:
            return DeltaTimeToken(DELTA_TIMES[action])
        action -= self.size_delta_time

        if action < self.size_subexpr:
            return ExpressionToken(self.subexprs[action])
        action -= self.size_subexpr

        if action == 0:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)

        raise ValueError(f"Action index out of range")


def Level2AlphaEnv(
    pool: AlphaPoolBase,
    use_level2_features: bool = True,
    subexprs: Optional[List[Expression]] = None,
    **kwargs,
):
    """Factory function for creating the Level 2 RL environment."""
    return Level2EnvWrapper(
        AlphaEnvCore(pool=pool, **kwargs),
        use_level2_features=use_level2_features,
        subexprs=subexprs,
    )

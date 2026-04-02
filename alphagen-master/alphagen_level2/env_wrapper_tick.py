"""
RL environment wrapper for tick-level (3s bar) alpha generation.

Uses config_tick.py for operators, delta_times, features.
"""

from typing import Tuple, List, Optional
import gymnasium as gym
import numpy as np

from alphagen_level2.config_tick import (
    OPERATORS, DELTA_TIMES, CONSTANTS, TICK_FEATURES,
    TICK_FEATURES_BASIC, MAX_EXPR_LENGTH, REWARD_PER_STEP,
)
from alphagen_level2.stock_data_tick import TickFeatureType
from alphagen.data.tokens import (
    Token, OperatorToken, FeatureToken, ConstantToken,
    DeltaTimeToken, SequenceIndicatorToken, SequenceIndicatorType,
    ExpressionToken,
)
from alphagen.data.expression import Expression
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.core import AlphaEnvCore


class TickEnvWrapper(gym.Wrapper):
    """
    RL environment wrapper with 20 tick-level feature action space.
    """
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(
        self,
        env: AlphaEnvCore,
        use_all_features: bool = True,
        subexprs: Optional[List[Expression]] = None,
    ):
        super().__init__(env)
        self.subexprs = subexprs or []
        self._features = TICK_FEATURES if use_all_features else TICK_FEATURES_BASIC

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
            dtype=np.uint8,
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
        for i in range(self.size_op):
            if valid['op'][OPERATORS[i].category_type()]:
                res[offset + i] = True
        offset += self.size_op

        if valid['select'][1]:
            res[offset:offset + self.size_feature] = True
        offset += self.size_feature

        if valid['select'][2]:
            res[offset:offset + self.size_constant] = True
        offset += self.size_constant

        if valid['select'][3]:
            res[offset:offset + self.size_delta_time] = True
        offset += self.size_delta_time

        if valid['select'][1]:
            res[offset:offset + self.size_subexpr] = True
        offset += self.size_subexpr

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


def TickAlphaEnv(
    pool: AlphaPoolBase,
    use_all_features: bool = True,
    subexprs: Optional[List[Expression]] = None,
    **kwargs,
):
    """Factory function for creating the tick-level RL environment."""
    return TickEnvWrapper(
        AlphaEnvCore(pool=pool, **kwargs),
        use_all_features=use_all_features,
        subexprs=subexprs,
    )

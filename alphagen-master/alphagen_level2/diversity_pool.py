"""
MseAlphaPool with optional diversity reward bonus and configurable
mutual-IC rejection threshold.

Drop-in replacement for MseAlphaPool. When diversity_bonus=0 and
ic_mut_threshold=0.99, behavior is identical to the original.

Usage:
    pool = DiversityMseAlphaPool(
        capacity=20,
        calculator=calc,
        ic_mut_threshold=0.7,     # reject highly correlated alphas
        diversity_bonus=0.1,      # reward bonus for novel alphas
    )
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen.data.pool_update import AddRemoveAlphas
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.data.calculator import AlphaCalculator


class DiversityMseAlphaPool(MseAlphaPool):
    """
    MseAlphaPool extended with:
      1. Configurable ic_mut_threshold (default 0.99 → recommended 0.7)
      2. Optional diversity bonus in reward
      3. Periodic pool deduplication

    The diversity bonus adds a term to the reward:
        reward = ensemble_obj + diversity_bonus * (1 - avg_abs_mutual_ic)

    This encourages the RL agent to generate alphas that are dissimilar
    to existing pool members, not just ones that improve train IC.
    """

    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device("cpu"),
        ic_mut_threshold: float = 0.99,
        diversity_bonus: float = 0.0,
    ):
        """
        Args:
            capacity, calculator, ic_lower_bound, l1_alpha, device:
                same as MseAlphaPool
            ic_mut_threshold: reject new alpha if mutual IC with any
                existing alpha exceeds this value.
                - 0.99 = nearly no filtering (original behavior)
                - 0.7  = recommended, rejects redundant factors
            diversity_bonus: coefficient for diversity reward term.
                - 0.0 = disabled (original behavior)
                - 0.05~0.2 = typical range
        """
        super().__init__(
            capacity=capacity,
            calculator=calculator,
            ic_lower_bound=ic_lower_bound,
            l1_alpha=l1_alpha,
            device=device,
        )
        self._ic_mut_threshold = ic_mut_threshold
        self._diversity_bonus = diversity_bonus

    def try_new_expr(self, expr: Expression) -> float:
        """
        Override: use configurable ic_mut_threshold and add diversity bonus.
        """
        ic_ret, ic_mut = self._calc_ics(
            expr, ic_mut_threshold=self._ic_mut_threshold
        )
        if ic_ret is None or ic_mut is None:
            return 0.
        if np.isnan(ic_ret) or np.isnan(ic_mut).any():
            return 0.
        if str(expr) in self._failure_cache:
            return self.best_obj

        self.eval_cnt += 1
        old_pool: List[Expression] = self.exprs[:self.size]  # type: ignore

        # Compute diversity before adding (for bonus calculation)
        avg_abs_corr = float(np.mean(np.abs(ic_mut))) if len(ic_mut) > 0 else 0.0

        self._add_factor(expr, ic_ret, ic_mut)
        if self.size > 1:
            new_weights = self.optimize()
            worst_idx = None
            if self.size > self.capacity:
                worst_idx = int(np.argmin(np.abs(new_weights)))
                if worst_idx == self.capacity:
                    self._pop(worst_idx)
                    self._failure_cache.add(str(expr))
                    return self.best_obj
            removed_idx = [worst_idx] if worst_idx is not None else []
            self.weights = new_weights
            self.update_history.append(AddRemoveAlphas(
                added_exprs=[expr],
                removed_idx=removed_idx,
                old_pool=old_pool,
                old_pool_ic=self.best_ic_ret,
                new_pool_ic=ic_ret
            ))
            if worst_idx is not None:
                self._pop(worst_idx)
        else:
            self.update_history.append(AddRemoveAlphas(
                added_exprs=[expr],
                removed_idx=[],
                old_pool=[],
                old_pool_ic=0.,
                new_pool_ic=ic_ret
            ))

        self._failure_cache = set()
        new_ic_ret, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic_ret, new_obj)

        # Add diversity bonus
        if self._diversity_bonus > 0 and len(ic_mut) > 0:
            new_obj = new_obj + self._diversity_bonus * (1.0 - avg_abs_corr)

        return new_obj

    def deduplicate(self, corr_threshold: Optional[float] = None) -> int:
        """
        Remove highly correlated alphas from the pool, keeping the
        higher-weight one in each correlated pair.

        Args:
            corr_threshold: mutual IC above which to deduplicate.
                Defaults to self._ic_mut_threshold.

        Returns:
            Number of alphas removed.
        """
        if self.size < 2:
            return 0

        threshold = corr_threshold or self._ic_mut_threshold
        weights = np.abs(self.weights[:self.size])
        mutual_ics = self._mutual_ics[:self.size, :self.size]

        to_remove = set()
        for i in range(self.size):
            if i in to_remove:
                continue
            for j in range(i + 1, self.size):
                if j in to_remove:
                    continue
                if abs(mutual_ics[i][j]) > threshold:
                    victim = i if weights[i] < weights[j] else j
                    to_remove.add(victim)

        if not to_remove:
            return 0

        remain = [i for i in range(self.size) if i not in to_remove]
        n_removed = len(to_remove)
        self.leave_only(remain)
        if self.size > 0:
            self.weights = self.optimize()
        return n_removed

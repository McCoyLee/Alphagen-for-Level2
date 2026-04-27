import math
from itertools import count
from typing import List, Optional, Tuple, Iterable, Dict, Any, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from .alpha_pool import AlphaPoolBase
from ..data.calculator import AlphaCalculator, TensorAlphaCalculator
from ..data.expression import Expression, OutOfDataRangeError, is_trivial_expr
from ..data.pool_update import PoolUpdate, AddRemoveAlphas
from ..utils.correlation import batch_pearsonr


class LinearAlphaPool(AlphaPoolBase, metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(capacity, calculator, device)
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self._weights: np.ndarray = np.zeros(capacity + 1)
        self._mutual_ics: np.ndarray = np.identity(capacity + 1)
        self._extra_info = [None for _ in range(capacity + 1)]
        self._ic_lower_bound = -1. if ic_lower_bound is None else ic_lower_bound
        self.best_obj = -1.
        self.update_history: List[PoolUpdate] = []
        self._failure_cache: Set[str] = set()

    @property
    def weights(self) -> np.ndarray:
        "Get the weights of the linear model as a numpy array of shape (size,)."
        return self._weights[:self.size]

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        "Set the weights of the linear model with a numpy array of shape (size,)."
        assert value.shape == (self.size,), f"Invalid weights shape: {value.shape}"
        self._weights[:self.size] = value

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "exprs": self.exprs[:self.size],
            "ics_ret": list(self.single_ics[:self.size]),
            "weights": list(self.weights),
            "best_ic_ret": self.best_ic_ret
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights)
        }

    def try_new_expr(self, expr: Expression) -> float:
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
            return 0.
        if str(expr) in self._failure_cache:
            return self.best_obj

        self.eval_cnt += 1
        old_pool: List[Expression] = self.exprs[:self.size]     # type: ignore
        self._add_factor(expr, ic_ret, ic_mut)
        if self.size > 1:
            new_weights = self.optimize()
            worst_idx = None
            if self.size > self.capacity:   # Need to remove one
                worst_idx = int(np.argmin(np.abs(new_weights)))
                # The one added this time is the worst, revert the changes
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
        return new_obj

    def force_load_exprs(self, exprs: List[Expression], weights: Optional[List[float]] = None) -> None:
        self._failure_cache = set()
        old_ic = self.evaluate_ensemble()
        old_pool: List[Expression] = self.exprs[:self.size] # type: ignore
        added = []
        for expr in exprs:
            if self.size >= self.capacity:
                break
            try:
                ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            except (OutOfDataRangeError, TypeError):
                continue
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, ic_ret, ic_mut)
            added.append(expr)
            assert self.size <= self.capacity
        if weights is not None:
            if len(weights) != self.size:
                raise ValueError(f"Invalid weights length: got {len(weights)}, expected {self.size}")
            self.weights = np.array(weights)
        else:
            self.weights = self.optimize()
        new_ic, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic, new_obj)
        self.update_history.append(AddRemoveAlphas(
            added_exprs=added,
            removed_idx=[],
            old_pool=old_pool,
            old_pool_ic=old_ic,
            new_pool_ic=new_ic
        ))

    def calculate_ic_and_objective(self) -> Tuple[float, float]:
        ic = self.evaluate_ensemble()
        obj = self._calc_main_objective()
        if obj is None:
            obj = ic
        return ic, obj

    def _calc_main_objective(self) -> Optional[float]:
        "Get the main optimization objective, return None for the default (ensemble IC)."

    def _maybe_update_best(self, ic: float, obj: float) -> bool:
        if obj <= self.best_obj:
            return False
        self.best_obj = obj
        self.best_ic_ret = ic
        return True

    @abstractmethod
    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        "Optimize the weights of the linear model and return the new weights as a numpy array."

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        return calculator.calc_pool_all_ret(self.exprs[:self.size], self.weights)      # type: ignore

    def evaluate_ensemble(self) -> float:
        if self.size == 0:
            return 0.
        return self.calculator.calc_pool_IC_ret(self.exprs[:self.size], self.weights)  # type: ignore

    @property
    def _under_thres_alpha(self) -> bool:
        if self._ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self._ic_lower_bound

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]]]:
        single_ic = self.calculator.calc_single_IC_ret(expr)
        if not self._under_thres_alpha and single_ic < self._ic_lower_bound:
            return single_ic, None

        mutual_ics = []
        for i in range(self.size):
            mutual_ic = self.calculator.calc_mutual_IC(expr, self.exprs[i])     # type: ignore
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics
    
    def _get_extra_info(self, expr: Expression) -> Any:
        "Override this method to save extra data for a newly added expression."

    def _add_factor(
        self,
        expr: Expression,
        ic_ret: float,
        ic_mut: List[float]
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.single_ics[n] = ic_ret
        for i in range(n):
            self._mutual_ics[i][n] = self._mutual_ics[n][i] = ic_mut[i]
        self._extra_info[n] = self._get_extra_info(expr)
        new_weight = max(ic_ret, 0.01) if n == 0 else self.weights.mean()
        self._weights[n] = new_weight   # Assign an initial weight
        self.size += 1

    def _pop(self, index_hint: Optional[int] = None) -> None:
        if self.size <= self.capacity:
            return
        index = int(np.argmin(np.abs(self.weights))) if index_hint is None else index_hint
        self._swap_idx(index, self.capacity)
        self.size = self.capacity

    def most_significant_indices(self, k: int) -> List[int]:
        if self.size == 0:
            return []
        ranks = (-np.abs(self.weights)).argsort().argsort()
        return [i for i in range(self.size) if ranks[i] < k]

    def leave_only(self, indices: Iterable[int]) -> None:
        "Leaves only the alphas at the given indices intact, and removes all others."
        self._failure_cache = set()
        indices = sorted(indices)
        for i, j in enumerate(indices):
            self._swap_idx(i, j)
        self.size = len(indices)

    def bulk_edit(self, removed_indices: Iterable[int], added_exprs: List[Expression]) -> None:
        self._failure_cache = set()
        old_ic = self.evaluate_ensemble()
        old_pool: List[Expression] = self.exprs[:self.size] # type: ignore
        removed_indices = set(removed_indices)
        remain = [i for i in range(self.size) if i not in removed_indices]
        old_exprs = {id(self.exprs[i]): i for i in range(self.size)}
        old_update_history_count = len(self.update_history)
        self.leave_only(remain)
        for e in added_exprs:
            self.try_new_expr(e)
        self.update_history = self.update_history[:old_update_history_count]
        new_exprs = {id(e): e for e in self.exprs[:self.size]}
        added_exprs = [e for e in self.exprs[:self.size] if id(e) not in old_exprs] # type: ignore
        removed_indices = list(sorted(i for eid, i in old_exprs.items() if eid not in new_exprs))
        new_ic, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic, new_obj)
        self.update_history.append(AddRemoveAlphas(
            added_exprs=added_exprs,
            removed_idx=removed_indices,
            old_pool=old_pool,
            old_pool_ic=old_ic,
            new_pool_ic=new_ic
        ))

    def _swap_idx(self, i: int, j: int) -> None:
        if i == j:
            return
        
        def swap_in_list(lst, i: int, j: int) -> None:
            lst[i], lst[j] = lst[j], lst[i]

        swap_in_list(self.exprs, i, j)
        swap_in_list(self.single_ics, i, j)
        self._mutual_ics[:, [i, j]] = self._mutual_ics[:, [j, i]]
        self._mutual_ics[[i, j], :] = self._mutual_ics[[j, i], :]
        swap_in_list(self._weights, i, j)
        swap_in_list(self._extra_info, i, j)


class MseAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(capacity, calculator, ic_lower_bound, device)
        self._l1_alpha = l1_alpha

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        alpha = self._l1_alpha
        if math.isclose(alpha, 0.):     # No L1 regularization, use the faster least-squares method
            return self._optimize_lstsq()
            
        ics_ret = torch.tensor(self.single_ics[:self.size], device=self.device)
        ics_mut = torch.tensor(self._mutual_ics[:self.size, :self.size], device=self.device)
        weights = torch.tensor(self.weights, device=self.device, requires_grad=True)
        optim = torch.optim.Adam([weights], lr=lr)
    
        loss_ic_min = float("inf")
        best_weights = weights
        tolerance_count = 0
        for step in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()
    
            loss_l1 = torch.norm(weights, p=1)  # type: ignore
            loss = loss_ic + alpha * loss_l1
    
            optim.zero_grad()
            loss.backward()
            optim.step()
    
            if loss_ic_min - loss_ic_curr > 1e-6:
                tolerance_count = 0
            else:
                tolerance_count += 1
    
            if loss_ic_curr < loss_ic_min:
                best_weights = weights
                loss_ic_min = loss_ic_curr
    
            if tolerance_count >= tolerance or step >= max_steps:
                break
    
        return best_weights.cpu().detach().numpy()
    
    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(self._mutual_ics[:self.size, :self.size],self.single_ics[:self.size])[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights


class SingleFactorAlphaPool(MseAlphaPool):
    """Mine standalone factors with a rolling-zscore reward.

    Reward formulation (matches the spec):
        z_t     = (x_t - mu_t) / (sigma_t + eps)    # mu,sigma over prev
                                                      # `lookback_bars` bars
        p_t     = clip(z_t, -3, 3) / 3
        ret_t   = mid[t + delay + H] / mid[t + delay] - 1
        IC      = rank-corr(z_t, ret_t)             # one scalar per factor
        r_t     = sign(IC) * p_t * ret_t - lambda * |p_t|
        r_bar   = mean(r_t)
        R       = alpha * tanh(|IC| / tau_ic) + beta * tanh(r_bar / tau_r)

    Per-factor reward statistics (magnitude and distribution of the two tanh
    components) are accumulated on every `try_new_expr` call and can be
    dumped to a JSON file via ``dump_reward_stats``.

    Key differences from the combination pool:
    - Factors are ranked independently (no linear-combination objective).
    - Weights are ±1 signs only (direction, not magnitude).
    - Pool / test score is based on the best single factor in the pool.
    """

    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 0.0,
        device: torch.device = torch.device("cpu"),
        ic_mut_threshold: Optional[float] = 0.99,
        # ---- reward knobs (image spec) ----
        holding_bars: int = 100,
        execution_delay: int = 1,
        lookback_bars: int = 1200,
        turnover_cost: float = 0.0006,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        tau_ic: float = 0.1,
        tau_r: float = 1e-3,
        tau_c: float = 1e-4,
        trivial_penalty: float = 0.0,
    ):
        super().__init__(capacity, calculator, ic_lower_bound, l1_alpha, device)
        self.holding_bars = int(holding_bars)
        self.execution_delay = int(execution_delay)
        self.lookback_bars = int(lookback_bars)
        self.turnover_cost = float(turnover_cost)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tau_ic = max(float(tau_ic), 1e-12)
        self.tau_r = max(float(tau_r), 1e-12)
        self.tau_c = max(float(tau_c), 1e-12)
        self.trivial_penalty = float(trivial_penalty)
        self._ic_mut_threshold = ic_mut_threshold
        # _composite_scores mirrors single_ics but stores the composite score
        self._composite_scores: np.ndarray = np.zeros(capacity + 1)
        # _factor_directions: +1 / -1 per factor (IC sign)
        self._factor_directions: np.ndarray = np.ones(capacity + 1)
        # Per-call reward statistics (in order of insertion)
        self._reward_stats: Dict[str, List[float]] = {
            "abs_ic":    [],   # |IC|
            "r_bar":     [],   # mean per-bar gross pnl
            "tc":        [],   # mean per-bar turnover cost lambda_c * |dp|
            "comp_ic":   [],   # alpha * tanh(|IC|  / tau_ic)
            "comp_r":    [],   # beta  * tanh(r_bar / tau_r)
            "comp_tc":   [],   # -gamma * tanh(tc   / tau_c)
            "reward":    [],   # R = comp_ic + comp_r + comp_tc
            "pos_abs_mean": [],
        }

    # ---- scoring helpers ------------------------------------------------

    def _score_expr(self, expr: Expression) -> Tuple[float, float, float, float]:
        """Return (reward, ic_signed, r_bar, direction).

        When reward components cannot be computed the call returns a zero
        reward, zero IC/r_bar and direction = +1 (treated as a skip).
        """
        try:
            parts = self.calculator.calc_single_reward_components(
                expr,
                holding_bars=self.holding_bars,
                execution_delay=self.execution_delay,
                lookback_bars=self.lookback_bars,
                turnover_cost=self.turnover_cost,
            )
        except AttributeError as exc:
            raise RuntimeError(
                "Calculator is missing `calc_single_reward_components`; the "
                "single-factor reward requires a TickCalculator-compatible "
                "instance."
            ) from exc
        if parts is None:
            return 0.0, 0.0, 0.0, 1.0

        ic = parts["ic"]
        abs_ic = parts["abs_ic"]
        r_bar = parts["r_bar"]
        tc = parts.get("tc", 0.0)
        pos_abs = parts.get("pos_abs_mean", 0.0)

        comp_ic = self.alpha * math.tanh(abs_ic / self.tau_ic)
        comp_r  = self.beta  * math.tanh(r_bar  / self.tau_r)
        comp_tc = -self.gamma * math.tanh(tc    / self.tau_c)
        reward  = comp_ic + comp_r + comp_tc

        # Direction follows sign(IC)
        direction = 1.0 if ic >= 0 else -1.0

        stats = self._reward_stats
        stats["abs_ic"].append(float(abs_ic))
        stats["r_bar"].append(float(r_bar))
        stats["tc"].append(float(tc))
        stats["comp_ic"].append(float(comp_ic))
        stats["comp_r"].append(float(comp_r))
        stats["comp_tc"].append(float(comp_tc))
        stats["reward"].append(float(reward))
        stats["pos_abs_mean"].append(float(pos_abs))

        return float(reward), float(ic), float(r_bar), float(direction)

    # ---- reward statistics ---------------------------------------------

    def compute_reward_stats(self) -> Dict[str, Dict[str, float]]:
        """Aggregate magnitude/distribution statistics for each component.

        Returns a mapping ``name -> {count, mean, std, abs_mean, min, max,
        p05, p25, p50, p75, p95}`` covering both raw primitives (|IC|, r_bar)
        and the tanh-squashed reward components (comp_ic, comp_r, reward).
        """
        def _summarize(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"count": 0}
            a = np.asarray(arr, dtype=np.float64)
            return {
                "count":     int(a.size),
                "mean":      float(a.mean()),
                "std":       float(a.std()),
                "abs_mean":  float(np.abs(a).mean()),
                "min":       float(a.min()),
                "max":       float(a.max()),
                "p05":       float(np.quantile(a, 0.05)),
                "p25":       float(np.quantile(a, 0.25)),
                "p50":       float(np.quantile(a, 0.50)),
                "p75":       float(np.quantile(a, 0.75)),
                "p95":       float(np.quantile(a, 0.95)),
            }
        return {k: _summarize(v) for k, v in self._reward_stats.items()}

    def dump_reward_stats(
        self, path: str, include_raw: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Write accumulated reward-component statistics to a JSON file.

        When ``include_raw`` is True the per-call arrays are stored alongside
        the summary, which is useful for plotting distributions later.
        """
        import json
        import os
        summary = self.compute_reward_stats()
        payload: Dict[str, Any] = {
            "config": {
                "alpha":            self.alpha,
                "beta":             self.beta,
                "gamma":            self.gamma,
                "tau_ic":           self.tau_ic,
                "tau_r":            self.tau_r,
                "tau_c":            self.tau_c,
                "holding_bars":     self.holding_bars,
                "execution_delay":  self.execution_delay,
                "lookback_bars":    self.lookback_bars,
                "turnover_cost":    self.turnover_cost,
            },
            "summary": summary,
        }
        if include_raw:
            payload["raw"] = {k: list(v) for k, v in self._reward_stats.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return summary

    def reset_reward_stats(self) -> None:
        for k in self._reward_stats:
            self._reward_stats[k] = []

    # ---- overrides ------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        w = self._weights[:self.size]
        if self.size == 0:
            return w
        return np.where(w >= 0, 1.0, -1.0)

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        assert value.shape == (self.size,), f"Invalid weights shape: {value.shape}"
        self._weights[:self.size] = np.where(value >= 0, 1.0, -1.0)

    def optimize(self, lr=5e-4, max_steps=10000, tolerance=500) -> np.ndarray:
        if self.size == 0:
            return np.array([])
        return self._factor_directions[:self.size].copy()

    def _add_factor(self, expr: Expression, ic_ret: float, ic_mut: List[float]) -> None:
        n = self.size
        self.exprs[n] = expr
        self.single_ics[n] = ic_ret
        for i in range(n):
            self._mutual_ics[i][n] = self._mutual_ics[n][i] = ic_mut[i]
        self._extra_info[n] = self._get_extra_info(expr)
        self._weights[n] = self._factor_directions[n]
        self.size += 1

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None,
    ) -> Tuple[float, Optional[List[float]]]:
        """Use global IC + optional mutual-IC filter for de-correlation."""
        single_ic = self.calculator.calc_single_IC_ret(expr)
        if np.isnan(single_ic):
            return single_ic, None
        if self._ic_lower_bound is not None and abs(single_ic) < self._ic_lower_bound:
            return single_ic, None

        mutual_ics: List[float] = []
        for i in range(self.size):
            mutual_ic = self.calculator.calc_mutual_IC(expr, self.exprs[i])  # type: ignore[arg-type]
            if np.isnan(mutual_ic):
                return single_ic, None
            if ic_mut_threshold is not None and abs(mutual_ic) > ic_mut_threshold:
                return single_ic, None
            mutual_ics.append(mutual_ic)
        return single_ic, mutual_ics

    def _best_single_index(self) -> Optional[int]:
        if self.size == 0:
            return None
        return int(np.argmax(self._composite_scores[:self.size]))

    def _calc_main_objective(self) -> Optional[float]:
        if self.size == 0:
            return 0.0
        return float(np.max(self._composite_scores[:self.size]))

    def evaluate_ensemble(self) -> float:
        idx = self._best_single_index()
        if idx is None:
            return 0.0
        return float(self._composite_scores[idx])

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        idx = self._best_single_index()
        if idx is None:
            return 0.0, 0.0
        expr = self.exprs[idx]
        assert expr is not None
        ic, ric = calculator.calc_single_all_ret(expr)
        d = self._factor_directions[idx]
        return ic * d, ric * d

    def _get_extra_info(self, expr: Expression):
        """Cache nothing; direction is stored in _factor_directions."""
        return None

    def to_json_dict(self) -> Dict[str, Any]:
        dirs = self._factor_directions[:self.size]
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": [1.0 if d >= 0 else -1.0 for d in dirs],
        }

    @property
    def state(self) -> Dict[str, Any]:
        dirs = self._factor_directions[:self.size]
        return {
            "exprs": self.exprs[:self.size],
            "ics_ret": list(self.single_ics[:self.size]),
            "weights": [1.0 if d >= 0 else -1.0 for d in dirs],
            "best_ic_ret": self.best_ic_ret
        }

    def _swap_idx(self, i: int, j: int) -> None:
        super()._swap_idx(i, j)
        self._composite_scores[i], self._composite_scores[j] = (
            self._composite_scores[j], self._composite_scores[i],
        )
        self._factor_directions[i], self._factor_directions[j] = (
            self._factor_directions[j], self._factor_directions[i],
        )

    # ---- main entry point -----------------------------------------------

    def force_load_exprs(
        self,
        exprs: List[Expression],
        weights: Optional[List[float]] = None,
    ) -> None:
        """Warm-load factors with composite scoring (weights are always ±1).

        Any externally supplied ``weights`` are reduced to their signs so the
        ±1 invariant of the single-factor pool holds.
        """
        self._failure_cache = set()
        old_ic = self.evaluate_ensemble()
        old_pool: List[Expression] = self.exprs[:self.size]  # type: ignore
        added: List[Expression] = []
        for expr in exprs:
            if self.size >= self.capacity:
                break
            if self.trivial_penalty > 0.0 and is_trivial_expr(expr):
                continue
            try:
                ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=self._ic_mut_threshold)
            except (OutOfDataRangeError, TypeError):
                continue
            if ic_ret is None or ic_mut is None or np.isnan(ic_ret):
                continue
            try:
                composite, ts_ic, profit, direction = self._score_expr(expr)
            except (OutOfDataRangeError, TypeError, ValueError):
                continue
            if np.isnan(composite) or np.isnan(direction):
                continue
            n = self.size
            self._composite_scores[n] = composite
            self._factor_directions[n] = direction
            self._add_factor(expr, ic_ret, ic_mut)
            added.append(expr)

        # Always use ±1 directions; ignore provided weights' magnitudes
        self.weights = self.optimize()

        new_ic, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic, new_obj)
        self.update_history.append(AddRemoveAlphas(
            added_exprs=added,
            removed_idx=[],
            old_pool=old_pool,
            old_pool_ic=old_ic,
            new_pool_ic=new_ic,
        ))

    def try_new_expr(self, expr: Expression) -> float:
        # Reject "single feature + constant" trivial expressions: no rolling,
        # at most one feature leaf. Returns a negative reward and skips pool insertion.
        if self.trivial_penalty > 0.0 and is_trivial_expr(expr):
            return -self.trivial_penalty

        # Quick IC filter (cheap)
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=self._ic_mut_threshold)
        if ic_ret is None or ic_mut is None or np.isnan(ic_ret):
            return 0.0
        if str(expr) in self._failure_cache:
            return self.best_obj

        self.eval_cnt += 1

        # Compute composite score (expensive)
        composite, ts_ic, profit, direction = self._score_expr(expr)

        old_pool: List[Expression] = self.exprs[:self.size]  # type: ignore
        old_obj = self.evaluate_ensemble()

        # Store composite score & direction, then add to pool
        n = self.size
        self._composite_scores[n] = composite
        self._factor_directions[n] = direction
        self._add_factor(expr, ic_ret, ic_mut)      # increments self.size
        self.weights = self.optimize()

        # Prune weakest if over capacity (keep top-K by composite score)
        removed_idx: List[int] = []
        if self.size > self.capacity:
            keep = np.argsort(-self._composite_scores[:self.size])[:self.capacity]
            keep_set = set(keep.tolist())
            removed_idx = sorted(i for i in range(self.size) if i not in keep_set)
            self.leave_only(keep.tolist())
            self.weights = self.optimize()

        self.update_history.append(AddRemoveAlphas(
            added_exprs=[expr],
            removed_idx=removed_idx,
            old_pool=old_pool,
            old_pool_ic=old_obj,
            new_pool_ic=self.evaluate_ensemble(),
        ))

        self._failure_cache = set()
        new_ic_ret, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic_ret, new_obj)
        improvement = max(new_obj - old_obj, 0.0)
        return composite + improvement


# Note: Currently the weights are only updated when the new IC is higher.
# It might be better to update the weights according to the actual objective,
# in this case the ICIR or the LCB of the IC.

class MeanStdAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: TensorAlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        lcb_beta: Optional[float] = None,
        device: torch.device = torch.device("cpu")
    ):
        """
        l1_alpha: the L1 regularization coefficient.
        lcb_beta: for optimizing the lower-confidence-bound: LCB = mean - beta * std, \
                  when this is None, optimize ICIR (mean / std) instead.
        """
        super().__init__(capacity, calculator, ic_lower_bound, device)
        self.calculator: TensorAlphaCalculator
        self._l1_alpha = l1_alpha
        self._lcb_beta = lcb_beta

    def _get_extra_info(self, expr: Expression) -> Any:
        return self.calculator.evaluate_alpha(expr)
    
    def _calc_main_objective(self) -> float:
        alpha_values = torch.stack(self._extra_info[:self.size])    # type: ignore | shape: n * days * stocks
        weights = torch.tensor(self.weights, device=self.device)
        return self._calc_obj_impl(alpha_values, weights).item()
    
    def _calc_obj_impl(self, alpha_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        target_value = self.calculator.target
        weighted = (weights[:, None, None] * alpha_values).sum(dim=0)
        ics = batch_pearsonr(weighted, target_value)
        mean, std = ics.mean(), ics.std()
        if self._lcb_beta is not None:
            return mean - self._lcb_beta * std
        else:
            return mean / std

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        alpha_values = torch.stack(self._extra_info[:self.size])    # type: ignore | shape: n * days * stocks
        weights = torch.tensor(self.weights, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([weights], lr=lr)
    
        min_loss = float("inf")
        best_weights = weights
        tol_count = 0
        for step in count():
            obj = self._calc_obj_impl(alpha_values, weights)
            loss_l1 = torch.norm(weights, p=1)
            loss = self._l1_alpha * loss_l1 - obj   # Maximize the objective
            curr_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if min_loss - curr_loss > 1e-6:
                tol_count = 0
            else:
                tol_count += 1
    
            if curr_loss < min_loss:
                best_weights = weights
                min_loss = curr_loss
    
            if tol_count >= tolerance or step >= max_steps:
                break
    
        return best_weights.cpu().detach().numpy()

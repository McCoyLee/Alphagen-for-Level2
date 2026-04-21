from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Sequence
from torch import Tensor
import torch

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr


class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression) -> float:
        'Calculate Rank IC between a single alpha and a predefined target.'

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        return self.calc_single_IC_ret(expr), self.calc_single_rIC_ret(expr)

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_all_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float]:
        'First combine the alphas linearly,'
        'then Calculate both IC and Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_single_ts_IC_ret(
        self, expr: Expression,
        bars_per_day: int = 1,
        window_days: Optional[int] = 20,
        use_rank: bool = False,
    ) -> Tuple[float, float]:
        """Windowed time-series IC.

        Divide usable bars into calendar days (each *bars_per_day* bars),
        compute per-day IC (or Rank IC) between factor and target, group
        daily ICs into non-overlapping windows of *window_days* days,
        average within each window, then average across windows.

        Returns:
            (mean_ic, std_ic) across windows.
        """

    @abstractmethod
    def calc_single_profit(
        self, expr: Expression,
        holding_bars: int = 100,
        turnover_penalty: float = 0.001,
    ) -> float:
        """Simplified normalised profit metric.

        R = mean(pos_t · r_{t+1}) − λ · mean(|pos_t − pos_{t−1}|)
        normalised by std(r) so the scale is comparable to IC.

        *pos* = factor value clamped to [-1, 1].
        *r*   = raw (un-normalised) forward return stored in *raw_target*.
        """


class TensorAlphaCalculator(AlphaCalculator):
    def __init__(
        self,
        target: Optional[Tensor],
        raw_target: Optional[Tensor] = None,
    ) -> None:
        self._target = target
        # raw_target: un-normalised forward returns, needed by calc_single_profit
        self._raw_target = raw_target

    @property
    @abstractmethod
    def n_days(self) -> int: ...

    @property
    def target(self) -> Tensor:
        if self._target is None:
            raise ValueError("A target must be set before calculating non-mutual IC.")
        return self._target

    @property
    def raw_target(self) -> Optional[Tensor]:
        return self._raw_target

    @abstractmethod
    def evaluate_alpha(self, expr: Expression) -> Tensor:
        'Evaluate an alpha into a `Tensor` of shape (days, stocks).'

    def make_ensemble_alpha(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tensor:
        n = len(exprs)
        factors = [self.evaluate_alpha(exprs[i]) * weights[i] for i in range(n)]
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).nanmean().item()
    
    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).nanmean().item()
    
    def _IR_from_batch(self, batch: Tensor) -> float:
        mean, std = batch.mean(), batch.std()
        return (mean / std).item()
    
    def _calc_ICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_pearsonr(value1, value2))
    
    def _calc_rICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_spearmanr(value1, value2))

    def calc_single_IC_ret(self, expr: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr), self.target)
    
    def calc_single_IC_ret_daily(self, expr: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr), self.target)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return self._calc_rIC(self.evaluate_alpha(expr), self.target)
    
    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self.evaluate_alpha(expr)
        target = self.target
        return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_mutual_IC_daily(self, expr1: Expression, expr2: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_pool_IC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(value, self.target)

    def calc_pool_rIC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(value, self.target)

    def calc_pool_all_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            return self._calc_IC(value, target), self._calc_rIC(value, target)
        
    def calc_pool_all_ret_with_ir(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float, float, float]:
        "Returns IC, ICIR, Rank IC, Rank ICIR"
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            ics = batch_pearsonr(value, target)
            rics = batch_spearmanr(value, target)
            ic_mean, ic_std = ics.mean().item(), ics.std().item()
            ric_mean, ric_std = rics.mean().item(), rics.std().item()
            return ic_mean, ic_mean / ic_std, ric_mean, ric_mean / ric_std

    # ------------------------------------------------------------------
    # Time-series windowed IC
    # ------------------------------------------------------------------

    def calc_single_ts_IC_ret(
        self, expr: Expression,
        bars_per_day: int = 1,
        window_days: Optional[int] = 20,
        use_rank: bool = False,
    ) -> Tuple[float, float]:
        with torch.no_grad():
            factor = self.evaluate_alpha(expr)
            target = self.target
            daily_ics = self._daily_ics(factor, target, bars_per_day, use_rank)
            return self._window_mean_std(daily_ics, window_days)

    def _daily_ics(
        self, factor: Tensor, target: Tensor,
        bars_per_day: int, use_rank: bool,
    ) -> Tensor:
        """Compute per-day IC tensor.

        For single-instrument bar-level data (bars_per_day > 1, n_stocks == 1)
        the time axis is reshaped into (n_calendar_days, bars_per_day) so that
        each row represents one trading day and the correlation is computed
        across bars within the day.

        For multi-stock data the standard cross-sectional IC per time-step is
        used (``batch_pearsonr`` already treats dim-0 as days, dim-1 as stocks).
        """
        corr_fn = batch_spearmanr if use_rank else batch_pearsonr

        if bars_per_day > 1 and factor.shape[-1] == 1:
            f = factor.squeeze(-1)
            t = target.squeeze(-1)
            n = f.shape[0]
            n_full = n // bars_per_day
            if n_full < 1:
                return torch.tensor([])
            f = f[: n_full * bars_per_day].view(n_full, bars_per_day)
            t = t[: n_full * bars_per_day].view(n_full, bars_per_day)
            return corr_fn(f, t)

        return corr_fn(factor, target)

    @staticmethod
    def _window_mean_std(
        daily_ics: Tensor, window_days: Optional[int]
    ) -> Tuple[float, float]:
        valid = daily_ics[~daily_ics.isnan()]
        n = len(valid)
        if n == 0:
            return 0.0, 0.0
        if window_days is None or window_days <= 0:
            return valid.mean().item(), (valid.std().item() if n > 1 else 0.0)
        n_win = n // window_days
        if n_win < 1:
            return valid.mean().item(), (valid.std().item() if n > 1 else 0.0)
        usable = n_win * window_days
        win_means = valid[:usable].view(n_win, window_days).mean(dim=1)
        return win_means.mean().item(), (win_means.std().item() if n_win > 1 else 0.0)

    # ------------------------------------------------------------------
    # Simplified normalised profit
    # ------------------------------------------------------------------

    def calc_single_profit(
        self, expr: Expression,
        holding_bars: int = 100,
        turnover_penalty: float = 0.001,
    ) -> float:
        raw_ret = self._raw_target
        if raw_ret is None:
            return 0.0

        with torch.no_grad():
            factor = self.evaluate_alpha(expr)

            # Flatten to 1-D for single instrument, else mean across stocks
            if factor.shape[-1] == 1:
                pos = factor.squeeze(-1).clamp(-1.0, 1.0)
                r = raw_ret.squeeze(-1)
            else:
                pos = factor.mean(dim=-1).clamp(-1.0, 1.0)
                r = raw_ret.mean(dim=-1)

            mask = ~(pos.isnan() | r.isnan())
            pos_v = pos[mask]
            r_v = r[mask]
            if len(pos_v) < 2:
                return 0.0

            profit = (pos_v * r_v).mean()
            turnover = (pos_v[1:] - pos_v[:-1]).abs().mean()
            R = profit - turnover_penalty * turnover

            ret_std = r_v.std()
            if ret_std < 1e-8:
                return 0.0
            return float(R / ret_std)

"""
Random-window sampler for tick-level alpha generation.

Given a single full-period ``TickStockData`` (e.g. 3 years of 3-second bars),
this module produces cheap *views* covering an arbitrarily-placed
[t0, t0 + window_bars] segment plus the backtrack history and forward-lookup
that the reward calculator needs.

A "view" is a ``TickStockData`` instance sharing the parent tensor (no I/O,
no GPU copy) — instantiation is O(1) plus a NaN-prune over stocks.
"""

from __future__ import annotations
from typing import Optional
import random
import torch

from alphagen_level2.stock_data_tick import TickStockData


class RandomWindowSampler:
    """Yield ``TickStockData`` views of a random ``window_bars`` segment.

    Parameters
    ----------
    full_data:
        The pre-loaded long-horizon dataset.  Its ``max_backtrack_days`` and
        ``max_future_days`` margins are reused as-is.
    window_bars:
        Number of *observation* bars per episode (1200 = 1h @ 3s).
    future_bars:
        Number of forward bars retained for the reward (100 = 5min @ 3s).
    history_buffer:
        Extra bars kept *before* the observation window so rolling operators
        (Ref, Mean, ...) and the rolling-zscore lookback still have history
        available inside the slice.  Defaults to ``window_bars``.
    execution_delay:
        Bars between signal and entry (must be aligned with the calculator).
    seed:
        RNG seed.  ``None`` reuses the global RNG.
    """

    def __init__(
        self,
        full_data: TickStockData,
        window_bars: int = 1200,
        future_bars: int = 100,
        history_buffer: Optional[int] = None,
        execution_delay: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.full = full_data
        self.window_bars = int(window_bars)
        self.future_bars = int(future_bars)
        self.history_buffer = int(history_buffer if history_buffer is not None
                                  else window_bars)
        self.execution_delay = int(execution_delay)
        self._rng = random.Random(seed)

        # Tensor index space:
        #   total bars  = full.data.shape[0]
        #   usable t0   ∈ [backtrack + history_buffer,
        #                  total - future_bars - execution_delay - window_bars)
        T = full.data.shape[0]
        bt = full.max_backtrack_days
        ft = full.max_future_days
        # Required margin past the observation window for forward-return lookup.
        forward_margin = self.future_bars + self.execution_delay + 1
        lo = bt + self.history_buffer
        hi = T - ft - forward_margin - self.window_bars
        if hi <= lo:
            raise ValueError(
                f"Full dataset too short for sampling: T={T}, "
                f"window={window_bars}, history_buffer={self.history_buffer}, "
                f"future={future_bars}, available [{lo}, {hi})"
            )
        self._lo = lo
        self._hi = hi

    # ------------------------------------------------------------------
    @property
    def n_possible(self) -> int:
        return self._hi - self._lo

    def sample(self) -> TickStockData:
        """Return a fresh ``TickStockData`` view at a random offset."""
        t0 = self._rng.randrange(self._lo, self._hi)
        return self.view_at(t0)

    def view_at(self, t0: int) -> TickStockData:
        """Build a ``TickStockData`` view centered on bar index ``t0``.

        ``t0`` is the *first observation bar* in the absolute index space
        of the parent tensor (i.e. counting the parent's backtrack margin).
        """
        full = self.full
        # We pack [history_buffer | window_bars | forward_margin] into the
        # child tensor and label them as backtrack / usable / future so that
        # ``n_days == window_bars``.
        forward_margin = self.future_bars + self.execution_delay + 1
        slice_lo = t0 - self.history_buffer
        slice_hi = t0 + self.window_bars + forward_margin
        slice_lo = max(0, slice_lo)
        slice_hi = min(full.data.shape[0], slice_hi)

        sub_tensor = full.data[slice_lo:slice_hi]
        # Drop stocks that are all-NaN in the slice (mirrors __getitem__).
        keep = (
            sub_tensor.isnan()
            .reshape(-1, sub_tensor.shape[-1])
            .all(dim=0)
            .logical_not()
            .nonzero()
            .flatten()
        )
        if keep.numel() == 0:
            # Degenerate slice — let the caller resample.
            raise RuntimeError("Sampled window has no valid stocks")
        sub_tensor = sub_tensor[:, :, keep]
        sub_dates = full._dates[slice_lo:slice_hi]
        sub_ids = full._stock_ids[keep.tolist()]

        view = TickStockData.__new__(TickStockData)
        # Mirror __init__'s attribute layout without re-loading data.
        view._instrument = full._instrument
        view._start_time = full._start_time
        view._end_time = full._end_time
        view._features = full._features
        view.device = full.device
        view._data_root = full._data_root
        view._cache_dir = full._cache_dir
        view._max_workers = full._max_workers
        view._bar_size_sec = full._bar_size_sec
        view.max_backtrack_days = self.history_buffer
        view.max_future_days = forward_margin
        view.data = sub_tensor
        view._dates = sub_dates
        view._stock_ids = sub_ids
        return view

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

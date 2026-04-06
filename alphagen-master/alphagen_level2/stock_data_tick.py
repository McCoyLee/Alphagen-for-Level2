"""
Tick-level StockData for 3-second bars.

Resamples Level 2 tick/transaction snapshots into 3-second bars and computes
20 microstructure features. Drop-in replacement for Level2StockData.

Feature list (20 features, index = enum value):
  0  OPEN            open_t
  1  HIGH            high_t
  2  LOW             low_t
  3  CLOSE           close_t
  4  RET             log(close_t / close_{t-1})
  5  VOLUME          volume_t
  6  TURNOVER        turnover_t
  7  VWAP            turnover_t / max(volume_t, EPS)
  8  MID             (ask1 + bid1) / 2
  9  SPREAD          ask1 - bid1
  10 SPREAD_PCT      spread / max(mid, EPS)
  11 BID_VOL1        bid_volume1_t
  12 ASK_VOL1        ask_volume1_t
  13 TOTAL_BID       total_bid_volume_t
  14 TOTAL_ASK       total_ask_volume_t
  15 IMBALANCE_1     (bid_vol1 - ask_vol1) / max(bid_vol1 + ask_vol1, EPS)
  16 IMBALANCE_TOTAL (total_bid - total_ask) / max(total_bid + total_ask, EPS)
  17 DELTA_BID_VOL1  bid_volume1_t - bid_volume1_{t-1}
  18 DELTA_ASK_VOL1  ask_volume1_t - ask_volume1_{t-1}
  19 SIGNED_VOLUME   buy_volume_t - sell_volume_t

Tensor shape: (n_bars_total, n_features, n_stocks)
"""

import os
import pickle
import hashlib
from typing import List, Optional, Tuple, Union
from enum import IntEnum
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from alphagen_level2.hdf5_reader import Level2HDF5Reader, match_char


EPS = 1e-9


class TickFeatureType(IntEnum):
    """20 microstructure features for 3-second bars."""
    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    RET = 4               # log return
    VOLUME = 5
    TURNOVER = 6
    VWAP = 7
    MID = 8               # (ask1 + bid1) / 2
    SPREAD = 9            # ask1 - bid1
    SPREAD_PCT = 10       # spread / mid
    BID_VOL1 = 11
    ASK_VOL1 = 12
    TOTAL_BID = 13
    TOTAL_ASK = 14
    IMBALANCE_1 = 15      # L1 volume imbalance
    IMBALANCE_TOTAL = 16  # total volume imbalance
    DELTA_BID_VOL1 = 17
    DELTA_ASK_VOL1 = 18
    SIGNED_VOLUME = 19    # buy - sell volume


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def _parse_tick_time_seconds(time_arr: np.ndarray) -> np.ndarray:
    """
    Parse tick Time field to seconds-since-midnight (float64).
    Common formats: HHMMSSMMM (int, e.g. 93000000), HHMMSS (e.g. 93000).
    """
    t = time_arr.astype(np.float64)
    if len(t) == 0:
        return t
    sample = t[t > 0]
    if len(sample) == 0:
        return np.zeros_like(t)
    max_val = sample.max()

    if max_val > 1e8:
        # HHMMSSMMM  e.g. 93000000 = 09:30:00.000
        hours = (t // 10000000).astype(int)
        mins = ((t % 10000000) // 100000).astype(int)
        secs = ((t % 100000) // 1000).astype(int)
        ms = (t % 1000).astype(int)
        return hours * 3600.0 + mins * 60.0 + secs + ms / 1000.0
    elif max_val > 1e6:
        hours = (t // 10000000).astype(int)
        mins = ((t % 10000000) // 100000).astype(int)
        secs = ((t % 100000) // 1000).astype(int)
        return hours * 3600.0 + mins * 60.0 + secs
    elif max_val > 1e4:
        # HHMMSS
        hours = (t // 10000).astype(int)
        mins = ((t % 10000) // 100).astype(int)
        secs = (t % 100).astype(int)
        return hours * 3600.0 + mins * 60.0 + secs
    elif max_val > 100:
        # HHMM (minutes-only)
        hours = (t // 100).astype(int)
        mins = (t % 100).astype(int)
        return hours * 3600.0 + mins * 60.0
    else:
        # Already seconds or minutes
        if max_val < 24:
            return t * 3600.0  # hours
        return t


def _compute_bar_edges_seconds(bar_size_sec: int = 3) -> np.ndarray:
    """
    Compute bar edge timestamps (seconds since midnight) for A-share trading hours.
    Morning: 9:30-11:30, Afternoon: 13:00-15:00 → 14400 sec total.
    Returns array of bar start times (seconds).
    """
    edges = []
    # Morning: 9:30:00 → 11:30:00  (34200 → 41400)
    t = 9 * 3600 + 30 * 60  # 34200
    end_am = 11 * 3600 + 30 * 60  # 41400
    while t < end_am:
        edges.append(float(t))
        t += bar_size_sec
    # Afternoon: 13:00:00 → 15:00:00 (46800 → 54000)
    t = 13 * 3600  # 46800
    end_pm = 15 * 3600  # 54000
    while t < end_pm:
        edges.append(float(t))
        t += bar_size_sec
    return np.array(edges, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tick → 3s bar resampling (20 features)
# ---------------------------------------------------------------------------

def _resample_tick_to_bars_v2(
    tick_data: dict,
    bar_edges: np.ndarray,
    bar_size_sec: int,
) -> np.ndarray:
    """
    Resample raw tick snapshots into N-second bars, producing 20 features.

    Returns: (n_bars, 20) float32 array, NaN for empty bars.
    """
    n_bars = len(bar_edges)
    n_features = len(TickFeatureType)
    result = np.full((n_bars, n_features), np.nan, dtype=np.float32)

    time_arr = tick_data.get('Time')
    price = tick_data.get('Price')
    if time_arr is None or price is None or len(price) == 0:
        return result

    seconds = _parse_tick_time_seconds(time_arr)
    price = price.astype(np.float64)

    # Assign ticks to bars
    bar_idx = np.digitize(seconds, bar_edges) - 1

    # Pre-fetch arrays
    volume = tick_data.get('Volume')
    volume = volume.astype(np.float64) if volume is not None and len(volume) > 0 else None
    turnover = tick_data.get('Turnover')
    turnover = turnover.astype(np.float64) if turnover is not None and len(turnover) > 0 else None

    # L1 book
    bp10 = tick_data.get('BidPrice10')
    ap10 = tick_data.get('AskPrice10')
    bv10 = tick_data.get('BidVolume10')
    av10 = tick_data.get('AskVolume10')

    # Scalars for bid1/ask1
    bid1 = bp10[:, 0].astype(np.float64) if bp10 is not None and bp10.ndim == 2 else None
    ask1 = ap10[:, 0].astype(np.float64) if ap10 is not None and ap10.ndim == 2 else None
    bid_vol1 = bv10[:, 0].astype(np.float64) if bv10 is not None and bv10.ndim == 2 else None
    ask_vol1 = av10[:, 0].astype(np.float64) if av10 is not None and av10.ndim == 2 else None

    # Total bid/ask volume
    total_bid = tick_data.get('TotalBidVolume')
    total_bid = total_bid.astype(np.float64) if total_bid is not None and len(total_bid) > 0 else None
    total_ask = tick_data.get('TotalAskVolume')
    total_ask = total_ask.astype(np.float64) if total_ask is not None and len(total_ask) > 0 else None

    # BSFlag for signed volume (from tick snapshots)
    bs_flag = tick_data.get('BSFlag')

    fi = TickFeatureType
    prev_close = np.nan
    prev_bid_vol1 = np.nan
    prev_ask_vol1 = np.nan

    for bi in range(n_bars):
        mask = (bar_idx == bi) & (price > 0)
        if not mask.any():
            continue

        p = price[mask]

        # --- OHLC ---
        result[bi, fi.OPEN] = p[0]
        result[bi, fi.HIGH] = p.max()
        result[bi, fi.LOW] = p.min()
        result[bi, fi.CLOSE] = p[-1]

        # --- RET: log(close / prev_close) ---
        if not np.isnan(prev_close) and prev_close > 0:
            result[bi, fi.RET] = np.log(p[-1] / prev_close)
        else:
            result[bi, fi.RET] = 0.0
        prev_close = p[-1]

        # --- VOLUME ---
        if volume is not None:
            v = volume[mask]
            bar_vol = v.sum()
            result[bi, fi.VOLUME] = bar_vol
        else:
            bar_vol = float(mask.sum())
            result[bi, fi.VOLUME] = bar_vol

        # --- TURNOVER ---
        if turnover is not None:
            to = turnover[mask]
            bar_to = to.sum()
            result[bi, fi.TURNOVER] = bar_to
        else:
            bar_to = (p * (volume[mask] if volume is not None else np.ones(mask.sum()))).sum()
            result[bi, fi.TURNOVER] = bar_to

        # --- VWAP ---
        result[bi, fi.VWAP] = bar_to / max(bar_vol, EPS)

        # --- MID, SPREAD, SPREAD_PCT ---
        if bid1 is not None and ask1 is not None:
            b1 = bid1[mask]
            a1 = ask1[mask]
            valid_ba = (b1 > 0) & (a1 > 0)
            if valid_ba.any():
                b1v = b1[valid_ba]
                a1v = a1[valid_ba]
                mid_val = ((a1v + b1v) * 0.5).mean()
                spread_val = (a1v - b1v).mean()
                result[bi, fi.MID] = mid_val
                result[bi, fi.SPREAD] = spread_val
                result[bi, fi.SPREAD_PCT] = spread_val / max(mid_val, EPS)

        # --- BID_VOL1, ASK_VOL1 ---
        if bid_vol1 is not None:
            bv1 = bid_vol1[mask]
            result[bi, fi.BID_VOL1] = bv1[-1]  # snapshot at bar close

            # DELTA_BID_VOL1
            if not np.isnan(prev_bid_vol1):
                result[bi, fi.DELTA_BID_VOL1] = bv1[-1] - prev_bid_vol1
            else:
                result[bi, fi.DELTA_BID_VOL1] = 0.0
            prev_bid_vol1 = bv1[-1]

        if ask_vol1 is not None:
            av1 = ask_vol1[mask]
            result[bi, fi.ASK_VOL1] = av1[-1]

            # DELTA_ASK_VOL1
            if not np.isnan(prev_ask_vol1):
                result[bi, fi.DELTA_ASK_VOL1] = av1[-1] - prev_ask_vol1
            else:
                result[bi, fi.DELTA_ASK_VOL1] = 0.0
            prev_ask_vol1 = av1[-1]

        # --- TOTAL_BID, TOTAL_ASK ---
        if total_bid is not None:
            tb = total_bid[mask]
            result[bi, fi.TOTAL_BID] = tb[-1]
        if total_ask is not None:
            ta = total_ask[mask]
            result[bi, fi.TOTAL_ASK] = ta[-1]

        # --- IMBALANCE_1: (bid_vol1 - ask_vol1) / max(bid_vol1 + ask_vol1, EPS) ---
        if bid_vol1 is not None and ask_vol1 is not None:
            bv1_last = bid_vol1[mask][-1]
            av1_last = ask_vol1[mask][-1]
            denom = bv1_last + av1_last
            result[bi, fi.IMBALANCE_1] = (bv1_last - av1_last) / max(denom, EPS)

        # --- IMBALANCE_TOTAL ---
        if total_bid is not None and total_ask is not None:
            tb_last = total_bid[mask][-1]
            ta_last = total_ask[mask][-1]
            denom = tb_last + ta_last
            result[bi, fi.IMBALANCE_TOTAL] = (tb_last - ta_last) / max(denom, EPS)

        # --- SIGNED_VOLUME: buy_vol - sell_vol ---
        if bs_flag is not None and len(bs_flag) > 0 and volume is not None:
            bs = bs_flag[mask]
            v = volume[mask]
            buy_mask = match_char(bs, 'B')
            sell_mask = match_char(bs, 'S')
            buy_vol = v[buy_mask].sum() if buy_mask.any() else 0.0
            sell_vol = v[sell_mask].sum() if sell_mask.any() else 0.0
            result[bi, fi.SIGNED_VOLUME] = buy_vol - sell_vol
        else:
            result[bi, fi.SIGNED_VOLUME] = 0.0

    return result


# ---------------------------------------------------------------------------
# TickStockData
# ---------------------------------------------------------------------------

class TickStockData:
    """
    StockData for 3-second bars with 20 microstructure features.

    Drop-in replacement for Level2StockData. Interface-compatible with the
    expression system (data.data[start:stop, feature, :]).

    Tensor shape: (n_bars_total, 20, n_stocks)
    """

    def __init__(
        self,
        instrument: Union[str, List[str]],
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 4800,
        max_future_days: int = 4800,
        features: Optional[List[TickFeatureType]] = None,
        device: torch.device = torch.device("cuda:0"),
        data_root: str = "~/EquityLevel2/stock",
        cache_dir: Optional[str] = None,
        max_workers: int = 4,
        bar_size_sec: int = 3,
        preloaded_data: Optional[Tuple[torch.Tensor, pd.Index, pd.Index]] = None,
    ) -> None:
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(TickFeatureType)
        self.device = device
        self._data_root = data_root
        self._cache_dir = cache_dir
        self._max_workers = max_workers
        self._bar_size_sec = bar_size_sec

        if preloaded_data is not None:
            self.data, self._dates, self._stock_ids = preloaded_data
        else:
            self.data, self._dates, self._stock_ids = self._load_data()

    def _cache_key(self) -> str:
        key_str = (f"tick|{self._instrument}|{self._start_time}|{self._end_time}|"
                   f"{self.max_backtrack_days}|{self.max_future_days}|"
                   f"{[int(f) for f in self._features]}|bar{self._bar_size_sec}s")
        return hashlib.md5(key_str.encode()).hexdigest()

    def _try_load_cache(self) -> Optional[Tuple[torch.Tensor, pd.Index, pd.Index]]:
        if self._cache_dir is None:
            return None
        path = os.path.join(self._cache_dir, f"tick_{self._cache_key()}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            return (d['tensor'].to(self.device), d['dates'], d['stock_ids'])
        return None

    def _save_cache(self, tensor: torch.Tensor, dates: pd.Index, stock_ids: pd.Index):
        if self._cache_dir is None:
            return
        os.makedirs(self._cache_dir, exist_ok=True)
        path = os.path.join(self._cache_dir, f"tick_{self._cache_key()}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(
                {'tensor': tensor.cpu(), 'dates': dates, 'stock_ids': stock_ids},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _load_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        cached = self._try_load_cache()
        if cached is not None:
            return cached
        reader = Level2HDF5Reader(self._data_root, max_workers=self._max_workers)
        try:
            result = self._build_tensor(reader)
            self._save_cache(*result)
            return result
        finally:
            reader.close()

    @property
    def bars_per_day(self) -> int:
        return len(_compute_bar_edges_seconds(self._bar_size_sec))

    def _build_tensor(
        self, reader: Level2HDF5Reader
    ) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """Build tensor with 3s-bar time axis."""
        all_dates = reader.common_dates()
        if not all_dates:
            all_dates = reader.available_dates('tick')
        if not all_dates:
            raise FileNotFoundError(f"No HDF5 data found in {self._data_root}")

        start_ts = pd.Timestamp(self._start_time)
        end_ts = pd.Timestamp(self._end_time)
        date_timestamps = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        start_idx = date_timestamps.searchsorted(start_ts)
        end_idx = date_timestamps.searchsorted(end_ts, side='right') - 1
        if end_idx < start_idx:
            raise ValueError(f"No data in range [{self._start_time}, {self._end_time}]")

        bpd = self.bars_per_day
        backtrack_dates = (self.max_backtrack_days + bpd - 1) // bpd
        future_dates = (self.max_future_days + bpd - 1) // bpd
        real_start_idx = max(0, start_idx - backtrack_dates)
        real_end_idx = min(len(all_dates) - 1, end_idx + future_dates)
        selected_dates = all_dates[real_start_idx:real_end_idx + 1]

        stock_codes = self._resolve_stocks(reader, selected_dates)
        if not stock_codes:
            raise ValueError("No stocks found in data")
        stock_ids = pd.Index(stock_codes)

        bar_edges = _compute_bar_edges_seconds(self._bar_size_sec)
        n_bars_per_day = len(bar_edges)
        n_dates = len(selected_dates)
        n_bars_total = n_dates * n_bars_per_day
        n_features = len(self._features)
        n_stocks = len(stock_codes)
        feat_idx = {f.name: i for i, f in enumerate(self._features)}

        values = np.full((n_bars_total, n_features, n_stocks), np.nan, dtype=np.float32)

        # Build bar DatetimeIndex
        bar_timestamps = []
        for date_str in selected_dates:
            base = pd.Timestamp(date_str)
            for edge in bar_edges:
                h = int(edge // 3600)
                m = int((edge % 3600) // 60)
                s = int(edge % 60)
                bar_timestamps.append(base.replace(hour=h, minute=m, second=s))
        bar_index = pd.DatetimeIndex(bar_timestamps)

        def process_date(args):
            d_idx, date_str = args
            day_data = reader.read_date_all_categories(date_str, stock_codes)
            day_values = np.full(
                (n_bars_per_day, n_features, n_stocks), np.nan, dtype=np.float32,
            )

            for s_idx, code in enumerate(stock_codes):
                tick_d = day_data.get('tick', {}).get(code, {})
                if tick_d:
                    tick_bars = _resample_tick_to_bars_v2(
                        tick_d, bar_edges, self._bar_size_sec,
                    )
                    # tick_bars: (n_bars_per_day, 20)
                    for feat in TickFeatureType:
                        if feat.name in feat_idx and feat.value < tick_bars.shape[1]:
                            day_values[:, feat_idx[feat.name], s_idx] = tick_bars[:, feat.value]

            return d_idx, d_idx * n_bars_per_day, day_values

        if self._max_workers > 1 and n_dates > 5:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = [
                    pool.submit(process_date, (i, d))
                    for i, d in enumerate(selected_dates)
                ]
                for future in futures:
                    _, bar_off, day_vals = future.result()
                    values[bar_off:bar_off + n_bars_per_day] = day_vals
        else:
            for i, d in enumerate(selected_dates):
                _, bar_off, day_vals = process_date((i, d))
                values[bar_off:bar_off + n_bars_per_day] = day_vals

        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        return tensor, bar_index, stock_ids

    def _resolve_stocks(
        self, reader: Level2HDF5Reader, dates: List[str]
    ) -> List[str]:
        if isinstance(self._instrument, list):
            return [code.lower() for code in self._instrument]
        if isinstance(self._instrument, str) and self._instrument.lower() != "auto":
            return [self._instrument.lower()]
        sample_dates = dates[::max(1, len(dates) // 10)][:10]
        stock_sets = []
        for d in sample_dates:
            stocks = set(reader.available_stocks('tick', d))
            if stocks:
                stock_sets.append(stocks)
        if not stock_sets:
            return []
        common = stock_sets[0]
        for s in stock_sets[1:]:
            common &= s
        return sorted(common)

    # ----- Interface (duck-type compatible with StockData) -----

    def __getitem__(self, slc: slice) -> "TickStockData":
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        if isinstance(slc.start, str):
            return self[self.find_date_slice(slc.start, slc.stop)]
        start, stop = slc.start, slc.stop
        start = start if start is not None else 0
        stop = (
            (stop if stop is not None else self.n_days)
            + self.max_future_days
            + self.max_backtrack_days
        )
        start = max(0, start)
        stop = min(self.data.shape[0], stop)
        idx_range = slice(start, stop)
        data = self.data[idx_range]
        remaining = (
            data.isnan()
            .reshape(-1, data.shape[-1])
            .all(dim=0)
            .logical_not()
            .nonzero()
            .flatten()
        )
        data = data[:, :, remaining]
        return TickStockData(
            instrument=self._instrument,
            start_time=self._start_time,
            end_time=self._end_time,
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            data_root=self._data_root,
            bar_size_sec=self._bar_size_sec,
            preloaded_data=(data, self._dates[idx_range], self._stock_ids[remaining.tolist()]),
        )

    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        ts = pd.Timestamp(date)
        idx: int = self._dates.searchsorted(ts)
        if exclusive and idx < len(self._dates) and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(
                f"Date {date} out of range: [{self._start_time}, {self._end_time}]"
            )
        return idx

    def find_date_slice(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> slice:
        start = None if start_time is None else self.find_date_index(start_time)
        stop = None if end_time is None else self.find_date_index(end_time, exclusive=False)
        return slice(start, stop)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        """Number of usable time steps (bars). Named 'n_days' for interface compat."""
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def stock_ids(self) -> pd.Index:
        return self._stock_ids

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"n_bars mismatch: tensor {n_days} vs data {self.n_days}")
        if self.n_stocks != n_stocks:
            raise ValueError(f"n_stocks mismatch: tensor {n_stocks} vs data {self.n_stocks}")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)

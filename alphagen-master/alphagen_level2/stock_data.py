"""
Level 2 StockData - drop-in replacement for alphagen_qlib.StockData.
Show less
Supports two modes:
  1. Bar mode (default): Resample tick snapshots to N-minute bars.
     Time axis = bars across all dates. For 3min bars × 250 days ≈ 20,000 bars.
     Expression rolling operators (Mean, Std, etc.) operate on bars, not days.
  2. Daily mode: Aggregate intraday data into daily features (legacy).
Tensor shape: (n_bars_total, n_features, n_stocks)
  where n_bars_total = n_bars + max_backtrack_bars + max_future_bars
The expression system only sees `data.data[start:stop, feature, :]` — it doesn't
care whether the time axis is days or bars, so it works unchanged.
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
class Level2FeatureType(IntEnum):
    """
    Extended feature types including Level 2 order book features.
    The first 6 (OPEN..VWAP) match the original FeatureType for compatibility.
    """
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    BID_ASK_SPREAD = 6       # Bar mean spread: ask1 - bid1
    BID_ASK_SPREAD_PCT = 7   # Spread / mid_price
    MID_PRICE = 8            # Bar mean (bid1 + ask1) / 2
    WEIGHTED_BID = 9         # Volume-weighted bid across 10 levels
    WEIGHTED_ASK = 10        # Volume-weighted ask across 10 levels
    ORDER_IMBALANCE = 11     # (TBV - TAV) / (TBV + TAV) bar mean
    DEPTH_IMBALANCE = 12     # 10-level bid/ask volume imbalance
    TXN_VOLUME = 13          # Bar transaction volume (actual trades)
    TXN_VWAP = 14            # Bar transaction VWAP
    BUY_RATIO = 15           # Active buy volume / total (BSFlag)
    LARGE_ORDER_RATIO = 16   # Top 10% volume / total
    TXN_COUNT = 17           # Number of trades in bar
    ORDER_CANCEL_RATIO = 18  # OrderKind=='D' count / total
    NET_ORDER_FLOW = 19      # (buy_vol - sell_vol) / total
# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------
def _parse_tick_time(time_arr: np.ndarray) -> np.ndarray:
    """
    Parse tick Time field to minutes-since-midnight.
    Common formats: HHMMSS000 (int), HHMM (int), or float seconds.
    Returns array of minutes (float64).
    """
    t = time_arr.astype(np.float64)
    if len(t) == 0:
        return t
    sample = t[t > 0]
    if len(sample) == 0:
        return np.zeros_like(t)
    max_val = sample.max()
    if max_val > 1e8:
        # Format: HHMMSSMMM (e.g., 93000000 = 09:30:00.000)
        hours = (t // 10000000).astype(int)
        mins = ((t % 10000000) // 100000).astype(int)
        secs = ((t % 100000) // 1000).astype(int)
        return hours * 60.0 + mins + secs / 60.0
    elif max_val > 1e6:
        # Format: HHMMSS000 (e.g., 93000000 but as 9300000)
        hours = (t // 10000000).astype(int)
        mins = ((t % 10000000) // 100000).astype(int)
        return hours * 60.0 + mins
    elif max_val > 1e4:
        # Format: HHMMSS (e.g., 93000)
        hours = (t // 10000).astype(int)
        mins = ((t % 10000) // 100).astype(int)
        secs = (t % 100).astype(int)
        return hours * 60.0 + mins + secs / 60.0
    elif max_val > 100:
        # Format: HHMM (e.g., 930)
        hours = (t // 100).astype(int)
        mins = (t % 100).astype(int)
        return hours * 60.0 + mins
    else:
        # Already in minutes or seconds
        return t
def _compute_bar_edges(bar_size_min: float) -> np.ndarray:
    """
    Compute bar edge timestamps (in minutes since midnight) for A-share trading hours.
    Morning: 9:30-11:30, Afternoon: 13:00-15:00 → 240 min total.
    Returns array of bar start times.
    """
    edges = []
    # Morning session: 9:30 (570 min) to 11:30 (690 min)
    t = 570.0
    while t < 690.0:
        edges.append(t)
        t += bar_size_min
    # Afternoon session: 13:00 (780 min) to 15:00 (900 min)
    t = 780.0
    while t < 900.0:
        edges.append(t)
        t += bar_size_min
    return np.array(edges, dtype=np.float64)
# ---------------------------------------------------------------------------
# Bar-level resampling (tick snapshots → N-min bars)
# ---------------------------------------------------------------------------
def _resample_tick_to_bars(
    tick_data: dict,
    bar_edges: np.ndarray,
    bar_size_min: float,
) -> np.ndarray:
    """
    Resample tick snapshot data into N-minute bars.
    Args:
        tick_data: dict of arrays from one stock, one date
        bar_edges: array of bar start times (minutes since midnight)
        bar_size_min: bar duration in minutes
    Returns:
        features array shape (n_bars, n_features=20), NaN for empty bars
    """
    n_bars = len(bar_edges)
    n_features = len(Level2FeatureType)
    result = np.full((n_bars, n_features), np.nan, dtype=np.float32)
    time_arr = tick_data.get('Time')
    price = tick_data.get('Price')
    if time_arr is None or price is None or len(price) == 0:
        return result
    minutes = _parse_tick_time(time_arr)
    price = price.astype(np.float64)
    # Assign each tick to a bar via digitize
    bar_ends = bar_edges + bar_size_min
    # tick belongs to bar i if bar_edges[i] <= tick_minute < bar_ends[i]
    bar_idx = np.digitize(minutes, bar_edges) - 1  # 0-based bar index
    # Pre-fetch arrays
    volume = tick_data.get('Volume')
    volume = volume.astype(np.float64) if volume is not None and len(volume) > 0 else None
    turnover = tick_data.get('Turnover')
    turnover = turnover.astype(np.float64) if turnover is not None and len(turnover) > 0 else None
    bp10 = tick_data.get('BidPrice10')
    ap10 = tick_data.get('AskPrice10')
    bv10 = tick_data.get('BidVolume10')
    av10 = tick_data.get('AskVolume10')
    has_book = (bp10 is not None and ap10 is not None and len(bp10) > 0 and bp10.ndim == 2)
    tbv_arr = tick_data.get('TotalBidVolume')
    tav_arr = tick_data.get('TotalAskVolume')
    for bi in range(n_bars):
        mask = (bar_idx == bi) & (price > 0)
        if not mask.any():
            continue
        p = price[mask]
        fi = Level2FeatureType
        # OHLCV
        result[bi, fi.OPEN] = p[0]
        result[bi, fi.CLOSE] = p[-1]
        result[bi, fi.HIGH] = p.max()
        result[bi, fi.LOW] = p.min()
        if volume is not None:
            v = volume[mask]
            bar_vol = v.sum()
            result[bi, fi.VOLUME] = bar_vol
            if bar_vol > 0:
                result[bi, fi.VWAP] = (p * v).sum() / bar_vol
            else:
                result[bi, fi.VWAP] = p.mean()
        elif turnover is not None:
            to = turnover[mask]
            result[bi, fi.VOLUME] = to.sum()  # use turnover as proxy
            result[bi, fi.VWAP] = p.mean()
        else:
            result[bi, fi.VOLUME] = float(mask.sum())
            result[bi, fi.VWAP] = p.mean()
        # Bid-ask features from 10-level book
        if has_book:
            bp = bp10[mask].astype(np.float64)  # (ticks_in_bar, 10)
            ap = ap10[mask].astype(np.float64)
            bid1 = bp[:, 0]
            ask1 = ap[:, 0]
            vba = (bid1 > 0) & (ask1 > 0)
            if vba.any():
                b1 = bid1[vba]
                a1 = ask1[vba]
                spread = a1 - b1
                mid = (a1 + b1) * 0.5
                result[bi, fi.BID_ASK_SPREAD] = spread.mean()
                result[bi, fi.BID_ASK_SPREAD_PCT] = (spread / mid).mean()
                result[bi, fi.MID_PRICE] = mid.mean()
            if bv10 is not None and av10 is not None:
                bv = bv10[mask].astype(np.float64)
                av = av10[mask].astype(np.float64)
                # Weighted bid
                tbv_l = bv.sum(axis=1)
                safe = np.where(tbv_l > 0, tbv_l, 1.0)
                wb = (bp * bv).sum(axis=1) / safe
                good = tbv_l > 0
                result[bi, fi.WEIGHTED_BID] = wb[good].mean() if good.any() else 0.0
                # Weighted ask
                tav_l = av.sum(axis=1)
                safe = np.where(tav_l > 0, tav_l, 1.0)
                wa = (ap * av).sum(axis=1) / safe
                good = tav_l > 0
                result[bi, fi.WEIGHTED_ASK] = wa[good].mean() if good.any() else 0.0
                # Depth imbalance (10 levels)
                tb = bv.sum(axis=1)
                ta = av.sum(axis=1)
                d = tb + ta
                safe_d = np.where(d > 0, d, 1.0)
                di = (tb - ta) / safe_d
                result[bi, fi.DEPTH_IMBALANCE] = di[d > 0].mean() if (d > 0).any() else 0.0
        # Order imbalance from TotalBidVolume / TotalAskVolume
        if tbv_arr is not None and tav_arr is not None:
            tb = tbv_arr[mask].astype(np.float64)
            ta = tav_arr[mask].astype(np.float64)
            d = tb + ta
            safe_d = np.where(d > 0, d, 1.0)
            oi = (tb - ta) / safe_d
            result[bi, fi.ORDER_IMBALANCE] = oi[d > 0].mean() if (d > 0).any() else 0.0
    return result
def _resample_txn_to_bars(
    txn_data: dict,
    bar_edges: np.ndarray,
    bar_size_min: float,
) -> np.ndarray:
    """Resample transaction data into bars. Returns (n_bars, 5) for txn features."""
    n_bars = len(bar_edges)
    # Features: TXN_VOLUME, TXN_VWAP, BUY_RATIO, LARGE_ORDER_RATIO, TXN_COUNT
    result = np.full((n_bars, 5), np.nan, dtype=np.float32)
    time_arr = txn_data.get('Time')
    price = txn_data.get('Price')
    volume = txn_data.get('Volume')
    if time_arr is None or price is None or len(price) == 0:
        return result
    minutes = _parse_tick_time(time_arr)
    price = price.astype(np.float64)
    volume = volume.astype(np.float64) if volume is not None and len(volume) > 0 else np.ones_like(price)
    # Filter cancels
    func_code = txn_data.get('FunctionCode')
    if func_code is not None and len(func_code) > 0:
        trade_mask = match_char(func_code, '0')
        if not trade_mask.any() and func_code.dtype.kind in ('i', 'u', 'f'):
            trade_mask = func_code == 0
        if trade_mask.any():
            minutes = minutes[trade_mask]
            price = price[trade_mask]
            volume = volume[trade_mask]
            bs_flag = txn_data.get('BSFlag')
            if bs_flag is not None and len(bs_flag) > 0:
                txn_data = {**txn_data, 'BSFlag': bs_flag[trade_mask]}
    valid = price > 0
    minutes = minutes[valid]
    price = price[valid]
    volume = volume[valid]
    bs_flag = txn_data.get('BSFlag')
    if bs_flag is not None and len(bs_flag) > 0:
        bs_flag = bs_flag[valid] if len(bs_flag) == len(valid) else bs_flag
    bar_idx = np.digitize(minutes, bar_edges) - 1
    for bi in range(n_bars):
        mask = bar_idx == bi
        if not mask.any():
            continue
        p = price[mask]
        v = volume[mask]
        total_v = v.sum()
        result[bi, 0] = total_v  # TXN_VOLUME
        result[bi, 1] = (p * v).sum() / total_v if total_v > 0 else 0.0  # TXN_VWAP
        result[bi, 4] = float(mask.sum())  # TXN_COUNT
        # BUY_RATIO from BSFlag
        if bs_flag is not None and len(bs_flag) > 0:
            bs = bs_flag[mask] if len(bs_flag) >= len(price) else bs_flag
            if len(bs) == mask.sum():
                buy_vol = v[match_char(bs, 'B')].sum()
                result[bi, 2] = buy_vol / total_v if total_v > 0 else 0.5
            else:
                result[bi, 2] = 0.5
        else:
            result[bi, 2] = 0.5
        # LARGE_ORDER_RATIO
        if total_v > 0 and len(v) > 10:
            thr = np.percentile(v, 90)
            result[bi, 3] = v[v >= thr].sum() / total_v
        else:
            result[bi, 3] = 0.0
    return result
def _resample_order_to_bars(
    order_data: dict,
    bar_edges: np.ndarray,
    bar_size_min: float,
) -> np.ndarray:
    """Resample order data into bars. Returns (n_bars, 2) for order features."""
    n_bars = len(bar_edges)
    result = np.full((n_bars, 2), np.nan, dtype=np.float32)  # CANCEL_RATIO, NET_FLOW
    time_arr = order_data.get('Time')
    volume = order_data.get('Volume')
    if time_arr is None or volume is None or len(volume) == 0:
        return result
    minutes = _parse_tick_time(time_arr)
    volume = volume.astype(np.float64)
    order_kind = order_data.get('OrderKind')
    func_code = order_data.get('FunctionCode')
    bar_idx = np.digitize(minutes, bar_edges) - 1
    for bi in range(n_bars):
        mask = bar_idx == bi
        if not mask.any():
            continue
        # Cancel ratio
        if order_kind is not None and len(order_kind) > 0:
            ok = order_kind[mask]
            cancel = match_char(ok, 'D')
            total = len(ok)
            result[bi, 0] = cancel.sum() / total if total > 0 else 0.0
        # Net order flow
        if func_code is not None and len(func_code) > 0:
            fc = func_code[mask]
            v = volume[mask]
            buy_v = v[match_char(fc, 'B')].sum()
            sell_v = v[match_char(fc, 'S')].sum()
            total_v = buy_v + sell_v
            result[bi, 1] = (buy_v - sell_v) / total_v if total_v > 0 else 0.0
    return result
# ---------------------------------------------------------------------------
# Level2StockData
# ---------------------------------------------------------------------------
class Level2StockData:
    """
    Drop-in replacement for alphagen_qlib.StockData using local Level 2 HDF5 data.
    In bar mode (default), the time axis is N-minute bars, not days.
    - self.data: Tensor (n_total_bars, n_features, n_stocks)
    - self.n_days: actually n_bars (the "days" naming is for interface compat)
    - max_backtrack_days: actually max_backtrack_bars
    - max_future_days: actually max_future_bars
    The expression system, calculator, and RL env work unchanged because they
    only index data.data[start:stop, feature, :] along dim0.
    """
    def __init__(
        self,
        instrument: Union[str, List[str]],
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[Level2FeatureType]] = None,
        device: torch.device = torch.device("cuda:0"),
        data_root: str = "~/EquityLevel2/stock",
        cache_dir: Optional[str] = None,
        max_workers: int = 4,
        bar_size_min: float = 3.0,
        bar_size_sec: Optional[int] = None,
        preloaded_data: Optional[Tuple[torch.Tensor, pd.Index, pd.Index]] = None,
    ) -> None:
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(Level2FeatureType)
        self.device = device
        self._data_root = data_root
        self._cache_dir = cache_dir
        self._max_workers = max_workers
        self._bar_size_min = (float(bar_size_sec) / 60.0) if bar_size_sec is not None else float(bar_size_min)
        if self._bar_size_min <= 0:
            raise ValueError(f"bar_size_min must be > 0, got {self._bar_size_min}")
        if preloaded_data is not None:
            self.data, self._dates, self._stock_ids = preloaded_data
        else:
            self.data, self._dates, self._stock_ids = self._load_data()
    def _cache_key(self) -> str:
        key_str = (f"{self._instrument}|{self._start_time}|{self._end_time}|"
                   f"{self.max_backtrack_days}|{self.max_future_days}|"
                   f"{[int(f) for f in self._features]}|bar{self._bar_size_min:.8f}")
        return hashlib.md5(key_str.encode()).hexdigest()
    def _try_load_cache(self) -> Optional[Tuple[torch.Tensor, pd.Index, pd.Index]]:
        if self._cache_dir is None:
            return None
        path = os.path.join(self._cache_dir, f"l2bar_{self._cache_key()}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            return (d['tensor'].to(self.device), d['dates'], d['stock_ids'])
        return None
    def _save_cache(self, tensor: torch.Tensor, dates: pd.Index, stock_ids: pd.Index):
        if self._cache_dir is None:
            return
        os.makedirs(self._cache_dir, exist_ok=True)
        path = os.path.join(self._cache_dir, f"l2bar_{self._cache_key()}.pkl")
        with open(path, 'wb') as f:
            pickle.dump({'tensor': tensor.cpu(), 'dates': dates, 'stock_ids': stock_ids}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    def _load_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        cached = self._try_load_cache()
        if cached is not None:
            return cached
        reader = Level2HDF5Reader(self._data_root, max_workers=self._max_workers)
        try:
            result = self._build_tensor_bars(reader)
            self._save_cache(*result)
            return result
        finally:
            reader.close()
    @property
    def bars_per_day(self) -> int:
        return len(_compute_bar_edges(self._bar_size_min))
    def _build_tensor_bars(self, reader: Level2HDF5Reader) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """Build tensor with bar-level time axis."""
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
        # Convert bar-based backtrack/future to date-based margins
        bpd = self.bars_per_day
        backtrack_dates = (self.max_backtrack_days + bpd - 1) // bpd  # ceil
        future_dates = (self.max_future_days + bpd - 1) // bpd
        real_start_idx = max(0, start_idx - backtrack_dates)
        real_end_idx = min(len(all_dates) - 1, end_idx + future_dates)
        selected_dates = all_dates[real_start_idx:real_end_idx + 1]
        stock_codes = self._resolve_stocks(reader, selected_dates)
        if not stock_codes:
            raise ValueError("No stocks found in data")
        stock_ids = pd.Index(stock_codes)
        bar_edges = _compute_bar_edges(self._bar_size_min)
        n_bars_per_day = len(bar_edges)
        n_dates = len(selected_dates)
        n_bars_total = n_dates * n_bars_per_day
        n_features = len(self._features)
        n_stocks = len(stock_codes)
        feat_idx = {f.name: i for i, f in enumerate(self._features)}
        # Pre-allocate
        values = np.full((n_bars_total, n_features, n_stocks), np.nan, dtype=np.float32)
        # Build bar-level DatetimeIndex
        bar_timestamps = []
        for date_str in selected_dates:
            base = pd.Timestamp(date_str)
            for edge in bar_edges:
                h = int(edge // 60)
                m = int(edge % 60)
                bar_timestamps.append(base.replace(hour=h, minute=m))
        bar_index = pd.DatetimeIndex(bar_timestamps)
        def process_date(args):
            d_idx, date_str = args
            day_data = reader.read_date_all_categories(date_str, stock_codes)
            bar_offset = d_idx * n_bars_per_day
            day_values = np.full((n_bars_per_day, n_features, n_stocks), np.nan, dtype=np.float32)
            for s_idx, code in enumerate(stock_codes):
                # Tick → bars (OHLCV + book features)
                tick_d = day_data.get('tick', {}).get(code, {})
                if tick_d:
                    tick_bars = _resample_tick_to_bars(tick_d, bar_edges, self._bar_size_min)
                    # tick_bars shape: (n_bars_per_day, 20)
                    for feat in Level2FeatureType:
                        if feat.name in feat_idx and feat.value < tick_bars.shape[1]:
                            day_values[:, feat_idx[feat.name], s_idx] = tick_bars[:, feat.value]
                # Transaction → bars
                txn_d = day_data.get('transaction', {}).get(code, {})
                if txn_d:
                    txn_bars = _resample_txn_to_bars(txn_d, bar_edges, self._bar_size_min)
                    # txn_bars: (n_bars, 5) → TXN_VOLUME, TXN_VWAP, BUY_RATIO, LARGE_ORDER_RATIO, TXN_COUNT
                    txn_feat_map = [
                        ('TXN_VOLUME', 0), ('TXN_VWAP', 1), ('BUY_RATIO', 2),
                        ('LARGE_ORDER_RATIO', 3), ('TXN_COUNT', 4),
                    ]
                    for fname, col in txn_feat_map:
                        if fname in feat_idx:
                            day_values[:, feat_idx[fname], s_idx] = txn_bars[:, col]
                # Order → bars
                ord_d = day_data.get('order', {}).get(code, {})
                if ord_d:
                    ord_bars = _resample_order_to_bars(ord_d, bar_edges, self._bar_size_min)
                    for fname, col in [('ORDER_CANCEL_RATIO', 0), ('NET_ORDER_FLOW', 1)]:
                        if fname in feat_idx:
                            day_values[:, feat_idx[fname], s_idx] = ord_bars[:, col]
            return d_idx, bar_offset, day_values
        # Parallel date loading
        if self._max_workers > 1 and n_dates > 5:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = [pool.submit(process_date, (i, d)) for i, d in enumerate(selected_dates)]
                for future in futures:
                    _, bar_off, day_vals = future.result()
                    values[bar_off:bar_off + n_bars_per_day] = day_vals
        else:
            for i, d in enumerate(selected_dates):
                _, bar_off, day_vals = process_date((i, d))
                values[bar_off:bar_off + n_bars_per_day] = day_vals
        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        return tensor, bar_index, stock_ids
    def _resolve_stocks(self, reader: Level2HDF5Reader, dates: List[str]) -> List[str]:
        if isinstance(self._instrument, list):
            return [code.lower() for code in self._instrument]
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
    def __getitem__(self, slc: slice) -> "Level2StockData":
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        if isinstance(slc.start, str):
            return self[self.find_date_slice(slc.start, slc.stop)]
        start, stop = slc.start, slc.stop
        start = start if start is not None else 0
        stop = (stop if stop is not None else self.n_days) + self.max_future_days + self.max_backtrack_days
        start = max(0, start)
        stop = min(self.data.shape[0], stop)
        idx_range = slice(start, stop)
        data = self.data[idx_range]
        remaining = data.isnan().reshape(-1, data.shape[-1]).all(dim=0).logical_not().nonzero().flatten()
        data = data[:, :, remaining]
        return Level2StockData(
            instrument=self._instrument,
            start_time=self._start_time,
            end_time=self._end_time,
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            data_root=self._data_root,
            bar_size_min=self._bar_size_min,
            preloaded_data=(data, self._dates[idx_range], self._stock_ids[remaining.tolist()])
        )
    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        ts = pd.Timestamp(date)
        idx: int = self._dates.searchsorted(ts)
        if exclusive and idx < len(self._dates) and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(f"Date {date} out of range: [{self._start_time}, {self._end_time}]")
        return idx
    def find_date_slice(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> slice:
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
        columns: Optional[List[str]] = None
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


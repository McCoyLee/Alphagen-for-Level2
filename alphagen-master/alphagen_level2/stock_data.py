"""
Level 2 StockData - drop-in replacement for alphagen_qlib.StockData.

Reads Level 2 HDF5 data (tick/order/transaction), aggregates intraday data
into daily features, and presents the same tensor interface that the expression
system expects: data shape (time_steps, n_features, n_stocks).

Actual data fields:
  tick:  Price, Volume, Turnover, AccVolume, AccTurnover, MatchItem, BSFlag,
         BidAvgPrice, AskAvgPrice, BidPrice10(N,10), BidVolume10(N,10),
         AskPrice10(N,10), AskVolume10(N,10), TotalBidVolume, TotalAskVolume,
         Open, High, Low, PreClose, Time
  order: BizIndex, OrderOriNo, OrderNumber, FunctionCode(B/S),
         OrderKind(0/1/A/D/U/S), Price, Volume, Time
  transaction: BidOrder, AskOrder, BSFlag, Channel, FunctionCode(0/1),
               BizIndex, Index, OrderKind, Price, Volume, Time

Performance:
  - Daily aggregation at load time (one-shot, then pure tensor ops)
  - Optional pickle cache to skip re-aggregation on subsequent loads
  - Parallel date loading via ThreadPoolExecutor
  - Vectorized numpy aggregation (no Python loops over ticks)
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
    # === Original daily features (indices 0-5 match FeatureType) ===
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5

    # === Level 2 tick-derived features ===
    BID_ASK_SPREAD = 6       # Daily mean spread: ask1 - bid1
    BID_ASK_SPREAD_PCT = 7   # Spread / mid_price percentage
    MID_PRICE = 8            # Daily mean (bid1 + ask1) / 2
    WEIGHTED_BID = 9         # Volume-weighted average bid across 10 levels
    WEIGHTED_ASK = 10        # Volume-weighted average ask across 10 levels
    ORDER_IMBALANCE = 11     # (TotalBidVolume - TotalAskVolume) / total, daily mean
    DEPTH_IMBALANCE = 12     # 10-level bid/ask volume imbalance

    # === Transaction-derived features ===
    TXN_VOLUME = 13          # Total actual trade volume (excl. cancels)
    TXN_VWAP = 14            # Transaction VWAP (actual trades only)
    BUY_RATIO = 15           # Active buy volume / total (from BSFlag)
    LARGE_ORDER_RATIO = 16   # Top 10% volume / total volume
    TXN_COUNT = 17           # Number of actual trades

    # === Order-derived features ===
    ORDER_CANCEL_RATIO = 18  # OrderKind=='D' count / total orders
    NET_ORDER_FLOW = 19      # (buy_vol - sell_vol) / total, from FunctionCode B/S


# ---------------------------------------------------------------------------
# Aggregation functions (fully vectorized numpy, no per-tick Python loops)
# ---------------------------------------------------------------------------

def _aggregate_tick_daily(td: dict) -> dict:
    """
    Aggregate tick snapshot data → daily features.

    Uses pre-computed Open/High/Low from tick snapshots and AccVolume/AccTurnover
    for accurate daily volume/VWAP. Handles 2D bid/ask arrays (N×10 levels).
    """
    r = {}
    price = td.get('Price')
    if price is None or len(price) == 0:
        return r
    price = price.astype(np.float64)
    valid = price > 0
    if not valid.any():
        return r

    # --- OHLCV from tick snapshot fields (no recomputation needed) ---
    open_arr = td.get('Open')
    high_arr = td.get('High')
    low_arr = td.get('Low')

    if open_arr is not None and len(open_arr) > 0:
        ov = open_arr.astype(np.float64)
        r['OPEN'] = float(ov[ov > 0][0]) if (ov > 0).any() else float(price[valid][0])
    else:
        r['OPEN'] = float(price[valid][0])

    r['CLOSE'] = float(price[valid][-1])

    if high_arr is not None and len(high_arr) > 0:
        hv = high_arr.astype(np.float64)
        # High is running max, last snapshot is daily high
        r['HIGH'] = float(hv[-1]) if hv[-1] > 0 else float(price[valid].max())
    else:
        r['HIGH'] = float(price[valid].max())

    if low_arr is not None and len(low_arr) > 0:
        lv = low_arr.astype(np.float64)
        vl = lv[lv > 0]
        r['LOW'] = float(vl[-1]) if len(vl) > 0 else float(price[valid].min())
    else:
        r['LOW'] = float(price[valid].min())

    # Volume / VWAP from cumulative fields
    acc_vol = td.get('AccVolume')
    acc_turn = td.get('AccTurnover')
    if acc_vol is not None and len(acc_vol) > 0 and float(acc_vol[-1]) > 0:
        daily_vol = float(acc_vol[-1])
        r['VOLUME'] = daily_vol
        if acc_turn is not None and len(acc_turn) > 0:
            r['VWAP'] = float(acc_turn[-1]) / daily_vol
        else:
            r['VWAP'] = float(price[valid].mean())
    else:
        vol = td.get('Volume')
        if vol is not None and len(vol) > 0:
            r['VOLUME'] = float(vol.astype(np.float64).sum())
        else:
            r['VOLUME'] = 0.0
        r['VWAP'] = float(price[valid].mean())

    # --- Bid-ask features from 2D arrays (N×10 levels) ---
    bp10 = td.get('BidPrice10')   # shape (N, 10)
    ap10 = td.get('AskPrice10')
    bv10 = td.get('BidVolume10')
    av10 = td.get('AskVolume10')

    has_book = (bp10 is not None and ap10 is not None
                and len(bp10) > 0 and bp10.ndim == 2)

    if has_book:
        bp = bp10.astype(np.float64)
        ap = ap10.astype(np.float64)
        bid1 = bp[:, 0]
        ask1 = ap[:, 0]

        # Filter valid snapshots (both bid1 and ask1 positive)
        vba = (bid1 > 0) & (ask1 > 0)
        if vba.any():
            b1 = bid1[vba]
            a1 = ask1[vba]
            spread = a1 - b1
            mid = (a1 + b1) * 0.5
            r['BID_ASK_SPREAD'] = float(spread.mean())
            r['BID_ASK_SPREAD_PCT'] = float((spread / mid).mean())
            r['MID_PRICE'] = float(mid.mean())
        else:
            r['BID_ASK_SPREAD'] = 0.0
            r['BID_ASK_SPREAD_PCT'] = 0.0
            r['MID_PRICE'] = float(price[valid].mean())

        # Weighted bid/ask across all 10 levels (vectorized, no loop)
        if bv10 is not None and av10 is not None and len(bv10) > 0:
            bv = bv10.astype(np.float64)  # (N, 10)
            av = av10.astype(np.float64)

            # Weighted bid price per snapshot, then daily mean
            tbv = bv.sum(axis=1)            # (N,)
            safe_tbv = np.where(tbv > 0, tbv, 1.0)
            w_bid = (bp * bv).sum(axis=1) / safe_tbv
            r['WEIGHTED_BID'] = float(w_bid[tbv > 0].mean()) if (tbv > 0).any() else 0.0

            tav = av.sum(axis=1)
            safe_tav = np.where(tav > 0, tav, 1.0)
            w_ask = (ap * av).sum(axis=1) / safe_tav
            r['WEIGHTED_ASK'] = float(w_ask[tav > 0].mean()) if (tav > 0).any() else 0.0

            # Depth imbalance: per-snapshot across 10 levels
            total_bid = bv.sum(axis=1)
            total_ask = av.sum(axis=1)
            denom = total_bid + total_ask
            safe_d = np.where(denom > 0, denom, 1.0)
            depth_imb = (total_bid - total_ask) / safe_d
            r['DEPTH_IMBALANCE'] = float(depth_imb[denom > 0].mean()) if (denom > 0).any() else 0.0
        else:
            r['WEIGHTED_BID'] = 0.0
            r['WEIGHTED_ASK'] = 0.0
            r['DEPTH_IMBALANCE'] = 0.0
    else:
        r['BID_ASK_SPREAD'] = 0.0
        r['BID_ASK_SPREAD_PCT'] = 0.0
        r['MID_PRICE'] = float(price[valid].mean())
        r['WEIGHTED_BID'] = 0.0
        r['WEIGHTED_ASK'] = 0.0
        r['DEPTH_IMBALANCE'] = 0.0

    # Order imbalance: use TotalBidVolume / TotalAskVolume (all outstanding)
    tbv_arr = td.get('TotalBidVolume')
    tav_arr = td.get('TotalAskVolume')
    if tbv_arr is not None and tav_arr is not None and len(tbv_arr) > 0:
        tb = tbv_arr.astype(np.float64)
        ta = tav_arr.astype(np.float64)
        denom = tb + ta
        safe_d = np.where(denom > 0, denom, 1.0)
        oi = (tb - ta) / safe_d
        r['ORDER_IMBALANCE'] = float(oi[denom > 0].mean()) if (denom > 0).any() else 0.0
    else:
        r['ORDER_IMBALANCE'] = 0.0

    return r


def _aggregate_transaction_daily(td: dict) -> dict:
    """
    Aggregate transaction data → daily features.

    Uses FunctionCode to filter actual trades (0) vs cancels (1).
    Uses BSFlag (B/S) for precise active buy/sell classification.
    """
    r = {}
    price = td.get('Price')
    volume = td.get('Volume')

    if price is None or len(price) == 0:
        return {'TXN_VOLUME': 0.0, 'TXN_VWAP': 0.0, 'BUY_RATIO': 0.5,
                'LARGE_ORDER_RATIO': 0.0, 'TXN_COUNT': 0.0}

    price = price.astype(np.float64)
    volume = volume.astype(np.float64) if volume is not None and len(volume) > 0 else np.ones_like(price)

    # Filter to actual trades (FunctionCode == 0), exclude cancels
    func_code = td.get('FunctionCode')
    if func_code is not None and len(func_code) > 0:
        trade_mask = match_char(func_code, '0')
        # Also handle integer 0
        if not trade_mask.any():
            if func_code.dtype.kind in ('i', 'u', 'f'):
                trade_mask = func_code == 0
        if trade_mask.any():
            price = price[trade_mask]
            volume = volume[trade_mask]
            bs_flag = td.get('BSFlag')
            if bs_flag is not None and len(bs_flag) > 0:
                bs_flag = bs_flag[trade_mask]
                td = {**td, 'BSFlag': bs_flag}  # update for buy_ratio below

    valid = price > 0
    price = price[valid]
    volume = volume[valid]

    r['TXN_COUNT'] = float(len(price))
    r['TXN_VOLUME'] = float(volume.sum())
    r['TXN_VWAP'] = float((price * volume).sum() / volume.sum()) if volume.sum() > 0 else 0.0

    # Buy ratio: from BSFlag (B=active buy, S=active sell)
    bs_flag = td.get('BSFlag')
    if bs_flag is not None and len(bs_flag) > 0:
        bs = bs_flag[valid] if len(bs_flag) == len(valid) else bs_flag
        if len(bs) == len(volume):
            buy_mask = match_char(bs, 'B')
            buy_vol = volume[buy_mask].sum()
            total_vol = volume.sum()
            r['BUY_RATIO'] = float(buy_vol / total_vol) if total_vol > 0 else 0.5
        else:
            r['BUY_RATIO'] = 0.5
    else:
        # Fallback: tick rule (less accurate)
        if len(price) > 1:
            pdiff = np.diff(price)
            buy_m = np.concatenate([[False], pdiff > 0])
            bvol = volume[buy_m].sum()
            svol = volume[np.concatenate([[False], pdiff < 0])].sum()
            t = bvol + svol
            r['BUY_RATIO'] = float(bvol / t) if t > 0 else 0.5
        else:
            r['BUY_RATIO'] = 0.5

    # Large order ratio: top 10% by volume
    if len(volume) > 0 and volume.sum() > 0:
        threshold = np.percentile(volume, 90)
        r['LARGE_ORDER_RATIO'] = float(volume[volume >= threshold].sum() / volume.sum())
    else:
        r['LARGE_ORDER_RATIO'] = 0.0

    return r


def _aggregate_order_daily(od: dict) -> dict:
    """
    Aggregate order data → daily features.

    OrderKind == 'D' → cancel orders.
    FunctionCode == 'B'/'S' → buy/sell direction.
    """
    r = {'ORDER_CANCEL_RATIO': 0.0, 'NET_ORDER_FLOW': 0.0}
    volume = od.get('Volume')
    if volume is None or len(volume) == 0:
        return r
    volume = volume.astype(np.float64)

    # Cancel ratio: OrderKind == 'D'
    order_kind = od.get('OrderKind')
    if order_kind is not None and len(order_kind) > 0:
        cancel_mask = match_char(order_kind, 'D')
        total = len(order_kind)
        r['ORDER_CANCEL_RATIO'] = float(cancel_mask.sum() / total) if total > 0 else 0.0

    # Net order flow: FunctionCode B(buy) vs S(sell)
    func_code = od.get('FunctionCode')
    if func_code is not None and len(func_code) > 0 and len(volume) == len(func_code):
        buy_mask = match_char(func_code, 'B')
        sell_mask = match_char(func_code, 'S')
        buy_vol = volume[buy_mask].sum()
        sell_vol = volume[sell_mask].sum()
        total_vol = buy_vol + sell_vol
        r['NET_ORDER_FLOW'] = float((buy_vol - sell_vol) / total_vol) if total_vol > 0 else 0.0

    return r


# ---------------------------------------------------------------------------
# Level2StockData
# ---------------------------------------------------------------------------

class Level2StockData:
    """
    Drop-in replacement for alphagen_qlib.StockData using local Level 2 HDF5 data.

    Interface contract (duck-type compatible with StockData):
      - self.data: Tensor (total_days, n_features, n_stocks)
      - self.n_days, self.n_features, self.n_stocks
      - self.stock_ids: pd.Index
      - self.max_backtrack_days, self.max_future_days, self.device
      - __getitem__(slice), make_dataframe(...)
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

        if preloaded_data is not None:
            self.data, self._dates, self._stock_ids = preloaded_data
        else:
            self.data, self._dates, self._stock_ids = self._load_data()

    def _cache_key(self) -> str:
        """Generate a unique cache key for the current configuration."""
        key_str = f"{self._instrument}|{self._start_time}|{self._end_time}|" \
                  f"{self.max_backtrack_days}|{self.max_future_days}|" \
                  f"{[int(f) for f in self._features]}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _try_load_cache(self) -> Optional[Tuple[torch.Tensor, pd.Index, pd.Index]]:
        if self._cache_dir is None:
            return None
        cache_path = os.path.join(self._cache_dir, f"l2cache_{self._cache_key()}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return (data['tensor'].to(self.device), data['dates'], data['stock_ids'])
        return None

    def _save_cache(self, tensor: torch.Tensor, dates: pd.Index, stock_ids: pd.Index):
        if self._cache_dir is None:
            return
        os.makedirs(self._cache_dir, exist_ok=True)
        cache_path = os.path.join(self._cache_dir, f"l2cache_{self._cache_key()}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump({'tensor': tensor.cpu(), 'dates': dates, 'stock_ids': stock_ids}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def _load_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        # Try cache first
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

    def _build_tensor(self, reader: Level2HDF5Reader) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        all_dates = reader.common_dates()
        if not all_dates:
            all_dates = reader.available_dates('tick')
        if not all_dates:
            raise FileNotFoundError(
                f"No HDF5 data found in {self._data_root}. "
                f"Expected: {{data_root}}/tick/YYYYMMDD.h5"
            )

        start_ts = pd.Timestamp(self._start_time)
        end_ts = pd.Timestamp(self._end_time)
        date_timestamps = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])

        start_idx = date_timestamps.searchsorted(start_ts)
        end_idx = date_timestamps.searchsorted(end_ts, side='right') - 1
        if end_idx < start_idx:
            raise ValueError(f"No data in range [{self._start_time}, {self._end_time}]")

        real_start_idx = max(0, start_idx - self.max_backtrack_days)
        real_end_idx = min(len(all_dates) - 1, end_idx + self.max_future_days)

        selected_dates = all_dates[real_start_idx:real_end_idx + 1]
        date_index = pd.DatetimeIndex([pd.Timestamp(d) for d in selected_dates])

        stock_codes = self._resolve_stocks(reader, selected_dates)
        if not stock_codes:
            raise ValueError("No stocks found in the data for the given date range")

        stock_ids = pd.Index(stock_codes)
        n_dates = len(selected_dates)
        n_features = len(self._features)
        n_stocks = len(stock_codes)

        # Feature name -> column index
        feat_idx = {f.name: i for i, f in enumerate(self._features)}

        # Pre-allocate with NaN (float32 for GPU efficiency)
        values = np.full((n_dates, n_features, n_stocks), np.nan, dtype=np.float32)

        # --- Parallel date loading ---
        def process_date(d_idx_date):
            d_idx, date_str = d_idx_date
            day_data = reader.read_date_all_categories(date_str, stock_codes)
            day_values = np.full((n_features, n_stocks), np.nan, dtype=np.float32)

            for s_idx, code in enumerate(stock_codes):
                # Tick aggregation
                tick_d = day_data.get('tick', {}).get(code, {})
                if tick_d:
                    for fname, val in _aggregate_tick_daily(tick_d).items():
                        if fname in feat_idx:
                            day_values[feat_idx[fname], s_idx] = val

                # Transaction aggregation
                txn_d = day_data.get('transaction', {}).get(code, {})
                if txn_d:
                    for fname, val in _aggregate_transaction_daily(txn_d).items():
                        if fname in feat_idx:
                            day_values[feat_idx[fname], s_idx] = val

                # Order aggregation
                ord_d = day_data.get('order', {}).get(code, {})
                if ord_d:
                    for fname, val in _aggregate_order_daily(ord_d).items():
                        if fname in feat_idx:
                            day_values[feat_idx[fname], s_idx] = val

            return d_idx, day_values

        # Use ThreadPool for IO-bound HDF5 reads (GIL released during h5py reads)
        if self._max_workers > 1 and n_dates > 10:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = [pool.submit(process_date, (i, d)) for i, d in enumerate(selected_dates)]
                for future in futures:
                    d_idx, day_vals = future.result()
                    values[d_idx] = day_vals
        else:
            for i, d in enumerate(selected_dates):
                d_idx, day_vals = process_date((i, d))
                values[d_idx] = day_vals

        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        return tensor, date_index, stock_ids

    def _resolve_stocks(self, reader: Level2HDF5Reader, dates: List[str]) -> List[str]:
        if isinstance(self._instrument, list):
            return [code.lower() for code in self._instrument]

        # Auto-discover: intersection of stocks across sampled dates
        sample_dates = dates[::max(1, len(dates) // 10)][:10]
        stock_sets = []
        for date_str in sample_dates:
            stocks = set(reader.available_stocks('tick', date_str))
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
            start_time=self._dates[start + self.max_backtrack_days].strftime("%Y-%m-%d"),
            end_time=self._dates[stop - 1 - self.max_future_days].strftime("%Y-%m-%d"),
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            data_root=self._data_root,
            preloaded_data=(data, self._dates[idx_range], self._stock_ids[remaining.tolist()])
        )

    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        ts = pd.Timestamp(date)
        idx: int = self._dates.searchsorted(ts)
        if exclusive and idx < len(self._dates) and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(f"Date {date} is out of range: [{self._start_time}, {self._end_time}]")
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
            raise ValueError(f"n_days mismatch: tensor {n_days} vs data {self.n_days}")
        if self.n_stocks != n_stocks:
            raise ValueError(f"n_stocks mismatch: tensor {n_stocks} vs data {self.n_stocks}")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)

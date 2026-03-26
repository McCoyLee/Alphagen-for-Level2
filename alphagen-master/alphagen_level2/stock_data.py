"""
Level 2 StockData - drop-in replacement for alphagen_qlib.StockData.

Reads Level 2 HDF5 data (tick/order/transaction), aggregates intraday data
into daily features, and presents the same tensor interface that the expression
system expects: data shape (time_steps, n_features, n_stocks).

Design choices:
  - Daily aggregation at load time: Level 2 data is aggregated into per-day
    feature vectors once during __init__. This avoids repeated IO during
    expression evaluation and keeps the expression system unchanged.
  - Extended feature set: Beyond OHLCV, we add Level 2 features like
    bid-ask spread, order imbalance, volume-weighted metrics, etc.
  - Memory layout: features are contiguous along dim=1 for cache-friendly
    access during per-feature expression evaluation.
"""

import os
from typing import List, Optional, Tuple, Union
from enum import IntEnum

import numpy as np
import pandas as pd
import torch

from alphagen_level2.hdf5_reader import Level2HDF5Reader


class Level2FeatureType(IntEnum):
    """
    Extended feature types including Level 2 order book features.
    The first 6 (OPEN..VWAP) match the original FeatureType for compatibility.
    """
    # === Original daily features (compatible with FeatureType) ===
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5

    # === Level 2 tick-derived features ===
    BID_ASK_SPREAD = 6       # Average daily bid-ask spread (ask1 - bid1)
    BID_ASK_SPREAD_PCT = 7   # Spread as percentage of mid price
    MID_PRICE = 8            # Average mid price = (bid1 + ask1) / 2
    WEIGHTED_BID = 9         # Volume-weighted average bid price (top 5 levels)
    WEIGHTED_ASK = 10        # Volume-weighted average ask price (top 5 levels)
    ORDER_IMBALANCE = 11     # (bid_vol - ask_vol) / (bid_vol + ask_vol) at top level
    DEPTH_IMBALANCE = 12     # Order book depth imbalance across 5 levels

    # === Transaction-derived features ===
    TXN_VOLUME = 13          # Total transaction volume
    TXN_VWAP = 14            # Transaction VWAP
    BUY_RATIO = 15           # Buy-initiated volume / total volume
    LARGE_ORDER_RATIO = 16   # Large order volume ratio (top 10% by size)
    TXN_COUNT = 17           # Number of transactions

    # === Order-derived features ===
    ORDER_CANCEL_RATIO = 18  # Cancel order ratio (if available)
    NET_ORDER_FLOW = 19      # Net order flow = buy_orders - sell_orders


# Mapping from original FeatureType values to Level2FeatureType
_COMPAT_FEATURE_MAP = {
    0: Level2FeatureType.OPEN,
    1: Level2FeatureType.CLOSE,
    2: Level2FeatureType.HIGH,
    3: Level2FeatureType.LOW,
    4: Level2FeatureType.VOLUME,
    5: Level2FeatureType.VWAP,
}


def _aggregate_tick_daily(tick_data: dict) -> dict:
    """
    Aggregate intraday tick data into daily OHLCV + Level 2 features.

    Args:
        tick_data: dict with keys like 'Price', 'Volume', 'BidPrice1', 'AskPrice1', etc.

    Returns:
        dict of daily aggregated feature values (scalars).
    """
    result = {}
    price = tick_data.get('Price', np.array([]))
    volume = tick_data.get('Volume', np.array([]))

    if len(price) == 0:
        return result

    price = price.astype(np.float64)
    # Filter out zero/negative prices
    valid = price > 0
    if not valid.any():
        return result
    price_valid = price[valid]

    result['OPEN'] = float(price_valid[0])
    result['CLOSE'] = float(price_valid[-1])
    result['HIGH'] = float(price_valid.max())
    result['LOW'] = float(price_valid.min())

    if len(volume) > 0:
        volume = volume.astype(np.float64)
        vol_valid = volume[valid] if len(volume) == len(price) else volume
        result['VOLUME'] = float(vol_valid.sum())
        # VWAP
        if vol_valid.sum() > 0:
            pv = price_valid[:len(vol_valid)] * vol_valid[:len(price_valid)]
            result['VWAP'] = float(pv.sum() / vol_valid.sum())
        else:
            result['VWAP'] = float(price_valid.mean())
    else:
        result['VOLUME'] = 0.0
        result['VWAP'] = float(price_valid.mean())

    # Bid-ask features
    bid1 = tick_data.get('BidPrice1', np.array([]))
    ask1 = tick_data.get('AskPrice1', np.array([]))
    if len(bid1) > 0 and len(ask1) > 0:
        bid1 = bid1.astype(np.float64)
        ask1 = ask1.astype(np.float64)
        valid_ba = (bid1 > 0) & (ask1 > 0)
        if valid_ba.any():
            b = bid1[valid_ba]
            a = ask1[valid_ba]
            spread = a - b
            mid = (a + b) / 2.0
            result['BID_ASK_SPREAD'] = float(spread.mean())
            result['BID_ASK_SPREAD_PCT'] = float((spread / mid).mean())
            result['MID_PRICE'] = float(mid.mean())
        else:
            result['BID_ASK_SPREAD'] = 0.0
            result['BID_ASK_SPREAD_PCT'] = 0.0
            result['MID_PRICE'] = float(price_valid.mean())
    else:
        result['BID_ASK_SPREAD'] = 0.0
        result['BID_ASK_SPREAD_PCT'] = 0.0
        result['MID_PRICE'] = float(price_valid.mean())

    # Weighted bid/ask across 5 levels
    for side, prefix in [('WEIGHTED_BID', 'Bid'), ('WEIGHTED_ASK', 'Ask')]:
        prices_levels = []
        volumes_levels = []
        for i in range(1, 6):
            p = tick_data.get(f'{prefix}Price{i}', np.array([]))
            v = tick_data.get(f'{prefix}Volume{i}', np.array([]))
            if len(p) > 0 and len(v) > 0:
                prices_levels.append(p.astype(np.float64))
                volumes_levels.append(v.astype(np.float64))
        if prices_levels:
            all_p = np.stack(prices_levels, axis=1)  # (ticks, levels)
            all_v = np.stack(volumes_levels, axis=1)
            # Per-tick weighted price, then average across ticks
            total_v = all_v.sum(axis=1)
            safe_total = np.where(total_v > 0, total_v, 1.0)
            weighted = (all_p * all_v).sum(axis=1) / safe_total
            result[side] = float(weighted.mean())
        else:
            result[side] = 0.0

    # Order imbalance at top level
    bid_vol1 = tick_data.get('BidVolume1', np.array([]))
    ask_vol1 = tick_data.get('AskVolume1', np.array([]))
    if len(bid_vol1) > 0 and len(ask_vol1) > 0:
        bv = bid_vol1.astype(np.float64)
        av = ask_vol1.astype(np.float64)
        total = bv + av
        safe_total = np.where(total > 0, total, 1.0)
        imb = (bv - av) / safe_total
        result['ORDER_IMBALANCE'] = float(imb.mean())
    else:
        result['ORDER_IMBALANCE'] = 0.0

    # Depth imbalance across 5 levels
    total_bid_vol = 0.0
    total_ask_vol = 0.0
    for i in range(1, 6):
        bv = tick_data.get(f'BidVolume{i}', np.array([]))
        av = tick_data.get(f'AskVolume{i}', np.array([]))
        if len(bv) > 0:
            total_bid_vol += float(bv.astype(np.float64).sum())
        if len(av) > 0:
            total_ask_vol += float(av.astype(np.float64).sum())
    denom = total_bid_vol + total_ask_vol
    if denom > 0:
        result['DEPTH_IMBALANCE'] = (total_bid_vol - total_ask_vol) / denom
    else:
        result['DEPTH_IMBALANCE'] = 0.0

    return result


def _aggregate_transaction_daily(txn_data: dict) -> dict:
    """Aggregate intraday transaction data into daily features."""
    result = {}
    price = txn_data.get('Price', np.array([]))
    volume = txn_data.get('Volume', np.array([]))

    if len(price) == 0:
        result['TXN_VOLUME'] = 0.0
        result['TXN_VWAP'] = 0.0
        result['BUY_RATIO'] = 0.5
        result['LARGE_ORDER_RATIO'] = 0.0
        result['TXN_COUNT'] = 0.0
        return result

    price = price.astype(np.float64)
    volume = volume.astype(np.float64) if len(volume) > 0 else np.ones_like(price)

    result['TXN_COUNT'] = float(len(price))
    result['TXN_VOLUME'] = float(volume.sum())

    if volume.sum() > 0:
        result['TXN_VWAP'] = float((price * volume).sum() / volume.sum())
    else:
        result['TXN_VWAP'] = float(price.mean())

    # Buy ratio: estimate from price movement (tick rule)
    # If transaction price > previous price -> buy-initiated
    if len(price) > 1:
        price_diff = np.diff(price)
        buy_mask = np.concatenate([[False], price_diff > 0])
        sell_mask = np.concatenate([[False], price_diff < 0])
        buy_vol = volume[buy_mask].sum()
        sell_vol = volume[sell_mask].sum()
        total = buy_vol + sell_vol
        result['BUY_RATIO'] = float(buy_vol / total) if total > 0 else 0.5
    else:
        result['BUY_RATIO'] = 0.5

    # Large order ratio: top 10% by volume
    if len(volume) > 0 and volume.sum() > 0:
        threshold = np.percentile(volume, 90)
        large_vol = volume[volume >= threshold].sum()
        result['LARGE_ORDER_RATIO'] = float(large_vol / volume.sum())
    else:
        result['LARGE_ORDER_RATIO'] = 0.0

    return result


def _aggregate_order_daily(order_data: dict) -> dict:
    """Aggregate intraday order data into daily features."""
    result = {}
    price = order_data.get('Price', np.array([]))
    volume = order_data.get('Volume', np.array([]))

    result['ORDER_CANCEL_RATIO'] = 0.0  # Requires OrderType field
    result['NET_ORDER_FLOW'] = 0.0

    if len(price) == 0:
        return result

    # If we have direction/type info, we can compute cancel ratio and net flow
    # For now, use a simple heuristic based on available data
    order_type = order_data.get('OrderType', np.array([]))
    direction = order_data.get('Direction', np.array([]))

    if len(order_type) > 0:
        # Common encoding: 0=limit, 1=market, 2=cancel
        cancel_mask = (order_type == 2) | (order_type == ord('D'))
        total = len(order_type)
        result['ORDER_CANCEL_RATIO'] = float(cancel_mask.sum() / total) if total > 0 else 0.0

    if len(direction) > 0 and len(volume) > 0:
        volume = volume.astype(np.float64)
        direction = direction.astype(np.float64)
        # Common encoding: 1=buy, -1=sell, or 1=buy, 2=sell
        if direction.max() <= 1:
            # -1/1 encoding
            net = (volume * direction).sum()
        else:
            # 1/2 encoding: 1=buy, 2=sell
            buy_vol = volume[direction == 1].sum()
            sell_vol = volume[direction == 2].sum()
            net = buy_vol - sell_vol
        total_vol = volume.sum()
        result['NET_ORDER_FLOW'] = float(net / total_vol) if total_vol > 0 else 0.0

    return result


class Level2StockData:
    """
    Drop-in replacement for alphagen_qlib.StockData using local Level 2 HDF5 data.

    Interface contract (same as StockData):
      - self.data: Tensor of shape (total_days, n_features, n_stocks)
        where total_days = n_days + max_backtrack_days + max_future_days
      - self.n_days, self.n_features, self.n_stocks: int properties
      - self.stock_ids: pd.Index
      - self.max_backtrack_days, self.max_future_days: int
      - self.device: torch.device
      - __getitem__(slice) -> Level2StockData  (date slicing)
      - make_dataframe(...) -> pd.DataFrame
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

        if preloaded_data is not None:
            self.data, self._dates, self._stock_ids = preloaded_data
        else:
            self.data, self._dates, self._stock_ids = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """Load and aggregate Level 2 data into the standard tensor format."""
        reader = Level2HDF5Reader(self._data_root)
        try:
            return self._build_tensor(reader)
        finally:
            reader.close()

    def _build_tensor(self, reader: Level2HDF5Reader) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        # Determine the date range with backtrack/future margins
        all_dates = reader.common_dates()
        if not all_dates:
            # Fallback: try tick dates only
            all_dates = reader.available_dates('tick')
        if not all_dates:
            raise FileNotFoundError(
                f"No HDF5 data files found in {self._data_root}. "
                f"Expected structure: {{data_root}}/tick/YYYYMMDD.h5"
            )

        start_ts = pd.Timestamp(self._start_time)
        end_ts = pd.Timestamp(self._end_time)
        date_timestamps = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])

        # Find indices for the requested range
        start_idx = date_timestamps.searchsorted(start_ts)
        end_idx = date_timestamps.searchsorted(end_ts, side='right') - 1
        if end_idx < start_idx:
            raise ValueError(f"No data found in range [{self._start_time}, {self._end_time}]")

        # Expand range for backtrack and future margins
        real_start_idx = max(0, start_idx - self.max_backtrack_days)
        real_end_idx = min(len(all_dates) - 1, end_idx + self.max_future_days)

        selected_dates = all_dates[real_start_idx:real_end_idx + 1]
        date_index = pd.DatetimeIndex([pd.Timestamp(d) for d in selected_dates])

        # Determine stock universe
        stock_codes = self._resolve_stocks(reader, selected_dates)
        if not stock_codes:
            raise ValueError("No stocks found in the data for the given date range")

        stock_ids = pd.Index(stock_codes)
        n_dates = len(selected_dates)
        n_features = len(self._features)
        n_stocks = len(stock_codes)

        # Pre-allocate with NaN
        values = np.full((n_dates, n_features, n_stocks), np.nan, dtype=np.float32)

        # Feature name to index mapping
        feat_idx = {f.name: i for i, f in enumerate(self._features)}

        # Load data date by date
        for d_idx, date_str in enumerate(selected_dates):
            # Read tick data for all stocks at once
            has_tick = os.path.exists(reader._get_h5_path('tick', date_str))
            has_txn = os.path.exists(reader._get_h5_path('transaction', date_str))
            has_order = os.path.exists(reader._get_h5_path('order', date_str))

            tick_batch = reader.read_stocks_batch('tick', date_str, stock_codes) if has_tick else {}
            txn_batch = reader.read_stocks_batch('transaction', date_str, stock_codes) if has_txn else {}
            order_batch = reader.read_stocks_batch('order', date_str, stock_codes) if has_order else {}

            for s_idx, code in enumerate(stock_codes):
                # Aggregate tick data
                tick_d = tick_batch.get(code, {})
                if tick_d:
                    agg = _aggregate_tick_daily(tick_d)
                    for feat_name, val in agg.items():
                        if feat_name in feat_idx:
                            values[d_idx, feat_idx[feat_name], s_idx] = val

                # Aggregate transaction data
                txn_d = txn_batch.get(code, {})
                if txn_d:
                    agg = _aggregate_transaction_daily(txn_d)
                    for feat_name, val in agg.items():
                        if feat_name in feat_idx:
                            values[d_idx, feat_idx[feat_name], s_idx] = val

                # Aggregate order data
                order_d = order_batch.get(code, {})
                if order_d:
                    agg = _aggregate_order_daily(order_d)
                    for feat_name, val in agg.items():
                        if feat_name in feat_idx:
                            values[d_idx, feat_idx[feat_name], s_idx] = val

        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        return tensor, date_index, stock_ids

    def _resolve_stocks(self, reader: Level2HDF5Reader, dates: List[str]) -> List[str]:
        """Determine the stock universe from the data or instrument config."""
        if isinstance(self._instrument, list):
            # Explicit stock list provided
            return [code.lower() for code in self._instrument]

        # Auto-discover stocks from data: use the intersection of stocks
        # across a sample of dates (ensures all stocks have data for most dates)
        sample_dates = dates[::max(1, len(dates) // 10)][:10]
        stock_sets = []
        for date_str in sample_dates:
            stocks = set(reader.available_stocks('tick', date_str))
            if stocks:
                stock_sets.append(stocks)

        if not stock_sets:
            return []

        # Use intersection so we only get stocks with consistent data
        common_stocks = stock_sets[0]
        for s in stock_sets[1:]:
            common_stocks &= s

        return sorted(common_stocks)

    def __getitem__(self, slc: slice) -> "Level2StockData":
        """Get a subview of the data given a date slice or an index slice."""
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
        # Remove stocks that are all NaN in this slice
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
            raise ValueError(f"Date {date} is out of range: available [{self._start_time}, {self._end_time}]")
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
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)

"""
HDF5 reader for Level 2 equity data.

Expected directory layout:
    {data_root}/
    ├── order/
    │   ├── 20220105.h5
    │   │   └── /{stock_code}/ -> BizIndex, OrderOriNo, OrderNumber,
    │   │                         FunctionCode(B/S), OrderKind(0/1/A/D/U/S),
    │   │                         Price, Volume, Time
    │   └── ...
    ├── tick/
    │   ├── 20220105.h5
    │   │   └── /{stock_code}/ -> Price, Volume, Turnover, AccVolume, AccTurnover,
    │   │                         MatchItem, BSFlag(B/S/""),
    │   │                         BidAvgPrice, AskAvgPrice,
    │   │                         BidPrice10(N,10), BidVolume10(N,10),
    │   │                         AskPrice10(N,10), AskVolume10(N,10),
    │   │                         TotalBidVolume, TotalAskVolume,
    │   │                         Open, High, Low, PreClose, Time
    │   └── ...
    └── transaction/
        ├── 20220105.h5
        │   └── /{stock_code}/ -> BidOrder, AskOrder, BSFlag, Channel,
        │                         FunctionCode(0=trade/1=cancel), BizIndex,
        │                         Index, OrderKind, Price, Volume, Time
        └── ...
"""

import os
import re
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np


# Actual field names per data category
DEFAULT_TICK_FIELDS = [
    'Price', 'Volume', 'Turnover', 'AccVolume', 'AccTurnover', 'MatchItem',
    'BSFlag',
    'BidAvgPrice', 'AskAvgPrice',
    'BidPrice10', 'BidVolume10',       # shape (N, 10)
    'AskPrice10', 'AskVolume10',       # shape (N, 10)
    'TotalBidVolume', 'TotalAskVolume',
    'Open', 'High', 'Low', 'PreClose',
    'Time',
]

DEFAULT_ORDER_FIELDS = [
    'BizIndex', 'OrderOriNo', 'OrderNumber',
    'FunctionCode', 'OrderKind',
    'Price', 'Volume', 'Time',
]

DEFAULT_TRANSACTION_FIELDS = [
    'BidOrder', 'AskOrder', 'BSFlag', 'Channel',
    'FunctionCode', 'BizIndex', 'Index', 'OrderKind',
    'Price', 'Volume', 'Time',
]


def _normalize_stock_code(code: str) -> str:
    """Normalize stock code to lowercase format like '000001.sz'."""
    return code.strip().lower()


def _list_h5_dates(directory: str) -> List[str]:
    """List all available dates (YYYYMMDD) from .h5 files in a directory."""
    if not os.path.isdir(directory):
        return []
    dates = []
    for fname in os.listdir(directory):
        if fname.endswith('.h5'):
            date_str = fname.replace('.h5', '')
            if re.match(r'^\d{8}$', date_str):
                dates.append(date_str)
    dates.sort()
    return dates


def match_char(arr: np.ndarray, char: str) -> np.ndarray:
    """
    Match a character in an HDF5-loaded array that may be bytes, str, or int.
    Handles: b'B', ord('B'), 'B', numpy bytes, etc.
    """
    if len(arr) == 0:
        return np.array([], dtype=bool)
    sample = arr.flat[0]
    if isinstance(sample, (bytes, np.bytes_)):
        return arr == char.encode()
    elif isinstance(sample, (str, np.str_)):
        return arr == char
    elif isinstance(sample, (int, np.integer)):
        return arr == ord(char)
    else:
        return arr == char.encode()


class Level2HDF5Reader:
    """
    Reads Level 2 HDF5 data from local filesystem.

    Features:
    - Lazy file open with caching (each .h5 opened at most once)
    - Batch stock reading (one file open per date, read all stocks)
    - Parallel date loading via ThreadPoolExecutor
    - Handles 2D datasets (BidPrice10/AskPrice10 shape N×10)
    - Handles byte-string fields (BSFlag, FunctionCode, OrderKind)
    """

    def __init__(
        self,
        data_root: str,
        tick_fields: Optional[List[str]] = None,
        order_fields: Optional[List[str]] = None,
        transaction_fields: Optional[List[str]] = None,
        max_workers: int = 4,
    ):
        self.data_root = os.path.expanduser(data_root)
        self.tick_dir = os.path.join(self.data_root, 'tick')
        self.order_dir = os.path.join(self.data_root, 'order')
        self.transaction_dir = os.path.join(self.data_root, 'transaction')

        self.tick_fields = tick_fields or DEFAULT_TICK_FIELDS
        self.order_fields = order_fields or DEFAULT_ORDER_FIELDS
        self.transaction_fields = transaction_fields or DEFAULT_TRANSACTION_FIELDS
        self.max_workers = max_workers

        self._file_cache: Dict[str, h5py.File] = {}
        self._dir_map = {
            'tick': self.tick_dir,
            'order': self.order_dir,
            'transaction': self.transaction_dir,
        }
        self._field_map = {
            'tick': self.tick_fields,
            'order': self.order_fields,
            'transaction': self.transaction_fields,
        }

    def available_dates(self, category: str = 'tick') -> List[str]:
        return _list_h5_dates(self._dir_map[category])

    def available_stocks(self, category: str, date: str) -> List[str]:
        fpath = self._get_h5_path(category, date)
        if not os.path.exists(fpath):
            return []
        f = self._open_h5(fpath)
        return [_normalize_stock_code(k) for k in f.keys()]

    def common_dates(self) -> List[str]:
        tick_dates = set(self.available_dates('tick'))
        order_dates = set(self.available_dates('order'))
        txn_dates = set(self.available_dates('transaction'))
        common = tick_dates & order_dates & txn_dates
        if not common:
            common = tick_dates
        return sorted(common)

    def _get_h5_path(self, category: str, date: str) -> str:
        return os.path.join(self._dir_map[category], f'{date}.h5')

    def _open_h5(self, fpath: str) -> h5py.File:
        if fpath not in self._file_cache:
            self._file_cache[fpath] = h5py.File(fpath, 'r')
        return self._file_cache[fpath]

    def read_stocks_batch(
        self,
        category: str,
        date: str,
        stock_codes: List[str],
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Read data for multiple stocks on a single date.
        Opens the HDF5 file only once for all stocks.
        Handles 2D datasets (e.g. BidPrice10 shape N×10) correctly.
        """
        fpath = self._get_h5_path(category, date)
        if not os.path.exists(fpath):
            return {code: {} for code in stock_codes}

        f = self._open_h5(fpath)
        if fields is None:
            fields = self._field_map[category]

        # Build normalized code -> actual HDF5 key mapping (once per file)
        file_keys = {_normalize_stock_code(k): k for k in f.keys()}

        result = {}
        for code in stock_codes:
            norm_code = _normalize_stock_code(code)
            if norm_code not in file_keys:
                result[code] = {}
                continue

            group = f[file_keys[norm_code]]
            group_keys_lower = {k.lower(): k for k in group.keys()}
            stock_data = {}
            for field in fields:
                actual_key = group_keys_lower.get(field.lower())
                if actual_key is not None:
                    stock_data[field] = group[actual_key][:]
                # Missing fields are simply not included (aggregation handles absence)
            result[code] = stock_data

        return result

    def read_date_all_categories(
        self,
        date: str,
        stock_codes: List[str],
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Read tick + transaction + order for a single date, all stocks.
        Returns: {category -> {stock_code -> {field -> array}}}
        """
        result = {}
        for cat in ('tick', 'transaction', 'order'):
            fpath = self._get_h5_path(cat, date)
            if os.path.exists(fpath):
                result[cat] = self.read_stocks_batch(cat, date, stock_codes)
            else:
                result[cat] = {}
        return result

    def close(self):
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_cache.clear()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

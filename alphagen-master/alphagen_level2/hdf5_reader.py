"""
HDF5 reader for Level 2 equity data.

Expected directory layout:
    {data_root}/
    ├── order/
    │   ├── 20220105.h5      # groups: /{stock_code}/Price, /{stock_code}/Time, ...
    │   └── ...
    ├── tick/
    │   ├── 20220105.h5      # groups: /{stock_code}/Price, /{stock_code}/Volume, ...
    │   └── ...
    └── transaction/
        ├── 20220105.h5      # groups: /{stock_code}/Price, /{stock_code}/Volume, ...
        └── ...

Each .h5 file is keyed by date (YYYYMMDD).
Within each file, groups are named by stock code (e.g., '000001.sz').
Each group contains datasets like Price, Time, Volume, etc.
"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np


# Which dataset keys to read from each data category
DEFAULT_TICK_FIELDS = ['Price', 'Volume', 'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1',
                       'BidPrice2', 'AskPrice2', 'BidVolume2', 'AskVolume2',
                       'BidPrice3', 'AskPrice3', 'BidVolume3', 'AskVolume3',
                       'BidPrice4', 'AskPrice4', 'BidVolume4', 'AskVolume4',
                       'BidPrice5', 'AskPrice5', 'BidVolume5', 'AskVolume5',
                       'Time']
DEFAULT_ORDER_FIELDS = ['Price', 'Volume', 'Time']
DEFAULT_TRANSACTION_FIELDS = ['Price', 'Volume', 'Time']


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


class Level2HDF5Reader:
    """
    Reads Level 2 HDF5 data from local filesystem.

    Provides lazy file access - files are opened on demand and cached.
    Supports reading tick, order, and transaction data.
    """

    def __init__(
        self,
        data_root: str,
        tick_fields: Optional[List[str]] = None,
        order_fields: Optional[List[str]] = None,
        transaction_fields: Optional[List[str]] = None,
    ):
        self.data_root = os.path.expanduser(data_root)
        self.tick_dir = os.path.join(self.data_root, 'tick')
        self.order_dir = os.path.join(self.data_root, 'order')
        self.transaction_dir = os.path.join(self.data_root, 'transaction')

        self.tick_fields = tick_fields or DEFAULT_TICK_FIELDS
        self.order_fields = order_fields or DEFAULT_ORDER_FIELDS
        self.transaction_fields = transaction_fields or DEFAULT_TRANSACTION_FIELDS

        # Cache for opened HDF5 file handles
        self._file_cache: Dict[str, h5py.File] = {}

    def available_dates(self, category: str = 'tick') -> List[str]:
        """Get sorted list of available dates for a given category."""
        dir_map = {'tick': self.tick_dir, 'order': self.order_dir, 'transaction': self.transaction_dir}
        return _list_h5_dates(dir_map[category])

    def available_stocks(self, category: str, date: str) -> List[str]:
        """Get list of available stock codes for a given category and date."""
        fpath = self._get_h5_path(category, date)
        if not os.path.exists(fpath):
            return []
        f = self._open_h5(fpath)
        return [_normalize_stock_code(k) for k in f.keys()]

    def common_dates(self) -> List[str]:
        """Get dates available across all three categories."""
        tick_dates = set(self.available_dates('tick'))
        order_dates = set(self.available_dates('order'))
        txn_dates = set(self.available_dates('transaction'))
        common = tick_dates & order_dates & txn_dates
        if not common:
            # Fall back: use tick dates if order/transaction are missing
            common = tick_dates
        return sorted(common)

    def _get_h5_path(self, category: str, date: str) -> str:
        dir_map = {'tick': self.tick_dir, 'order': self.order_dir, 'transaction': self.transaction_dir}
        return os.path.join(dir_map[category], f'{date}.h5')

    def _open_h5(self, fpath: str) -> h5py.File:
        if fpath not in self._file_cache:
            self._file_cache[fpath] = h5py.File(fpath, 'r')
        return self._file_cache[fpath]

    def read_stock_data(
        self,
        category: str,
        date: str,
        stock_code: str,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Read data for a single stock on a single date.

        Returns dict mapping field name -> 1D numpy array.
        Missing fields return empty arrays.
        """
        fpath = self._get_h5_path(category, date)
        if not os.path.exists(fpath):
            return {}

        f = self._open_h5(fpath)
        code = _normalize_stock_code(stock_code)

        # Try to find the stock group (case-insensitive match)
        group = None
        for key in f.keys():
            if _normalize_stock_code(key) == code:
                group = f[key]
                break

        if group is None:
            return {}

        if fields is None:
            field_map = {
                'tick': self.tick_fields,
                'order': self.order_fields,
                'transaction': self.transaction_fields,
            }
            fields = field_map[category]

        result = {}
        for field in fields:
            # Case-insensitive field lookup
            matched_key = None
            for k in group.keys():
                if k.lower() == field.lower():
                    matched_key = k
                    break
            if matched_key is not None:
                result[field] = group[matched_key][:]
            else:
                result[field] = np.array([], dtype=np.float64)

        return result

    def read_stocks_batch(
        self,
        category: str,
        date: str,
        stock_codes: List[str],
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Read data for multiple stocks on a single date.
        Returns dict: stock_code -> {field -> array}.

        Opens the HDF5 file only once for all stocks (efficient IO).
        """
        fpath = self._get_h5_path(category, date)
        if not os.path.exists(fpath):
            return {code: {} for code in stock_codes}

        f = self._open_h5(fpath)

        if fields is None:
            field_map = {
                'tick': self.tick_fields,
                'order': self.order_fields,
                'transaction': self.transaction_fields,
            }
            fields = field_map[category]

        # Build a mapping from normalized code -> actual key in the file
        file_keys = {}
        for key in f.keys():
            file_keys[_normalize_stock_code(key)] = key

        result = {}
        for code in stock_codes:
            norm_code = _normalize_stock_code(code)
            if norm_code not in file_keys:
                result[code] = {}
                continue

            group = f[file_keys[norm_code]]
            stock_data = {}
            # Build case-insensitive field mapping once per group
            group_keys = {k.lower(): k for k in group.keys()}
            for field in fields:
                actual_key = group_keys.get(field.lower())
                if actual_key is not None:
                    stock_data[field] = group[actual_key][:]
                else:
                    stock_data[field] = np.array([], dtype=np.float64)
            result[code] = stock_data

        return result

    def close(self):
        """Close all cached HDF5 file handles."""
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

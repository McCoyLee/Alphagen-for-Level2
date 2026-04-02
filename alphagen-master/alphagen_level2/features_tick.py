"""
Feature definitions for 3-second bar (tick-level) data.

20 microstructure features designed for high-frequency (3s) bars.
"""

from alphagen.data.expression import Feature, Ref
from alphagen_level2.stock_data_tick import TickFeatureType

# === Price ===
open_ = Open = OPEN = Feature(TickFeatureType.OPEN)
high = High = HIGH = Feature(TickFeatureType.HIGH)
low = Low = LOW = Feature(TickFeatureType.LOW)
close = Close = CLOSE = Feature(TickFeatureType.CLOSE)
ret = Ret = RET = Feature(TickFeatureType.RET)

# === Volume / Turnover ===
volume = Volume = VOLUME = Feature(TickFeatureType.VOLUME)
turnover = Turnover = TURNOVER = Feature(TickFeatureType.TURNOVER)
vwap = Vwap = VWAP = Feature(TickFeatureType.VWAP)

# === Bid-Ask ===
mid = Mid = MID = Feature(TickFeatureType.MID)
spread = Spread = SPREAD = Feature(TickFeatureType.SPREAD)
spread_pct = SpreadPct = SPREAD_PCT = Feature(TickFeatureType.SPREAD_PCT)

# === Book Volume ===
bid_vol1 = BidVol1 = BID_VOL1 = Feature(TickFeatureType.BID_VOL1)
ask_vol1 = AskVol1 = ASK_VOL1 = Feature(TickFeatureType.ASK_VOL1)
total_bid = TotalBid = TOTAL_BID = Feature(TickFeatureType.TOTAL_BID)
total_ask = TotalAsk = TOTAL_ASK = Feature(TickFeatureType.TOTAL_ASK)

# === Imbalance ===
imbalance_1 = Imbalance1 = IMBALANCE_1 = Feature(TickFeatureType.IMBALANCE_1)
imbalance_total = ImbalanceTotal = IMBALANCE_TOTAL = Feature(TickFeatureType.IMBALANCE_TOTAL)

# === Delta / Flow ===
delta_bid_vol1 = DeltaBidVol1 = DELTA_BID_VOL1 = Feature(TickFeatureType.DELTA_BID_VOL1)
delta_ask_vol1 = DeltaAskVol1 = DELTA_ASK_VOL1 = Feature(TickFeatureType.DELTA_ASK_VOL1)
signed_volume = SignedVolume = SIGNED_VOLUME = Feature(TickFeatureType.SIGNED_VOLUME)

# Default target: forward return over 1200 bars ≈ 1 hour for 3s bars
target = Ref(close, -1200) / close - 1

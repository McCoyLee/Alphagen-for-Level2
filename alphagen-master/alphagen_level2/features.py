"""
Feature definitions for Level 2 data.

Extends the original features.py with Level 2 order book features.
"""

from alphagen.data.expression import Feature, Ref
from alphagen_level2.stock_data import Level2FeatureType

# === Original features (backward compatible) ===
high = High = HIGH = Feature(Level2FeatureType.HIGH)
low = Low = LOW = Feature(Level2FeatureType.LOW)
volume = Volume = VOLUME = Feature(Level2FeatureType.VOLUME)
open_ = Open = OPEN = Feature(Level2FeatureType.OPEN)
close = Close = CLOSE = Feature(Level2FeatureType.CLOSE)
vwap = Vwap = VWAP = Feature(Level2FeatureType.VWAP)

# === Level 2 tick-derived features ===
bid_ask_spread = BidAskSpread = Feature(Level2FeatureType.BID_ASK_SPREAD)
bid_ask_spread_pct = BidAskSpreadPct = Feature(Level2FeatureType.BID_ASK_SPREAD_PCT)
mid_price = MidPrice = Feature(Level2FeatureType.MID_PRICE)
weighted_bid = WeightedBid = Feature(Level2FeatureType.WEIGHTED_BID)
weighted_ask = WeightedAsk = Feature(Level2FeatureType.WEIGHTED_ASK)
order_imbalance = OrderImbalance = Feature(Level2FeatureType.ORDER_IMBALANCE)
depth_imbalance = DepthImbalance = Feature(Level2FeatureType.DEPTH_IMBALANCE)

# === Transaction-derived features ===
txn_volume = TxnVolume = Feature(Level2FeatureType.TXN_VOLUME)
txn_vwap = TxnVwap = Feature(Level2FeatureType.TXN_VWAP)
buy_ratio = BuyRatio = Feature(Level2FeatureType.BUY_RATIO)
large_order_ratio = LargeOrderRatio = Feature(Level2FeatureType.LARGE_ORDER_RATIO)
txn_count = TxnCount = Feature(Level2FeatureType.TXN_COUNT)

# === Order-derived features ===
order_cancel_ratio = OrderCancelRatio = Feature(Level2FeatureType.ORDER_CANCEL_RATIO)
net_order_flow = NetOrderFlow = Feature(Level2FeatureType.NET_ORDER_FLOW)

# Default target: 20-day forward return
target = Ref(close, -20) / close - 1

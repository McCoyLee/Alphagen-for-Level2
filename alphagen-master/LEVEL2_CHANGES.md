# Level 2 本地数据集成 — 变更说明

## 一、改了什么

新增 `alphagen_level2/` 包（7 个文件）和训练脚本 `scripts/rl_level2.py`，共 8 个新文件：

| 文件 | 职责 |
|------|------|
| `alphagen_level2/__init__.py` | 包入口，导出核心类 |
| `alphagen_level2/hdf5_reader.py` | HDF5 文件读取器（tick/order/transaction） |
| `alphagen_level2/stock_data.py` | `Level2StockData` 类 — 替代原 Qlib 的 `StockData` |
| `alphagen_level2/calculator.py` | `Level2Calculator` — 替代原 `QLibStockDataCalculator` |
| `alphagen_level2/config.py` | Level 2 配置（算子、特征集、常量） |
| `alphagen_level2/env_wrapper.py` | `Level2EnvWrapper` — 支持 20 维特征的 RL 动作空间 |
| `alphagen_level2/features.py` | 20 个特征变量定义（OHLCV + 14 个 Level 2 特征） |
| `scripts/rl_level2.py` | 完整训练脚本，直接读取本地 HDF5 数据 |

**原有代码零修改** — 所有改动都是新增文件，不影响已有的 Qlib 流程。

---

## 二、为什么要改

### 2.1 原有架构的问题

原系统通过 `Qlib + BaoStock` 加载日频 OHLCV 数据：

```
StockData.__init__()
  → initialize_qlib()        # 需要安装/初始化 Qlib
  → QlibDataLoader.load()    # 只支持 Qlib 二进制格式（.bin）
  → 6 个特征: OPEN, CLOSE, HIGH, LOW, VOLUME, VWAP
```

**核心痛点：**

1. **无法使用本地 Level 2 数据**：Qlib 不支持 HDF5 格式的逐笔数据
2. **特征维度有限**：仅 6 维日频特征，丢失了 Level 2 盘口深度信息
3. **依赖外部服务**：需要 BaoStock 下载 + Qlib 转换，流程繁琐
4. **信息损失严重**：Order book 的微观结构信息（买卖盘不平衡、大单占比等）完全丢失

### 2.2 改造目标

- 直接读取 `~/EquityLevel2/stock/{tick,order,transaction}/YYYYMMDD.h5` 数据
- 保持与现有表达式系统、RL 系统的完全兼容（duck-typing 接口不变）
- 将特征从 6 维扩展到 20 维，充分利用 Level 2 信息

---

## 三、怎么改的

### 3.1 架构设计

```
原架构：
  Qlib (.bin) → StockData → Tensor(days, 6, stocks) → Expression → Calculator → RL

新架构：
  HDF5 (.h5) → Level2HDF5Reader → Level2StockData → Tensor(days, 20, stocks) → Expression → Level2Calculator → RL
                    ↑                    ↑                                          ↑
               [新增] 文件IO层      [新增] 数据聚合层                         [不变] 完全复用
```

关键设计决策：**在数据加载时完成日内聚合，输出与原系统相同形状的 Tensor**。这样表达式系统、算子、RL 环境全部无需修改。

### 3.2 数据读取层 (`hdf5_reader.py`)

```python
class Level2HDF5Reader:
    # 懒加载 + 文件句柄缓存，避免重复 open/close
    # 支持批量读取（一次打开 .h5，读取所有股票），减少 IO 次数
    # 大小写不敏感的 stock code / field name 匹配
```

**性能考量：**
- 每个日期的 .h5 文件只打开一次，缓存在 `_file_cache` 中
- `read_stocks_batch()` 一次读取所有股票，避免 N 次文件打开
- 使用 `h5py` 直接读取为 numpy 数组，零拷贝

### 3.3 数据聚合层 (`stock_data.py`)

**20 维特征定义：**

| 索引 | 特征名 | 来源 | 说明 |
|------|--------|------|------|
| 0-5 | OPEN/CLOSE/HIGH/LOW/VOLUME/VWAP | tick | 与原系统完全兼容 |
| 6 | BID_ASK_SPREAD | tick | 买一卖一价差均值 |
| 7 | BID_ASK_SPREAD_PCT | tick | 价差占中间价百分比 |
| 8 | MID_PRICE | tick | 中间价均值 |
| 9 | WEIGHTED_BID | tick | 5 档加权买价 |
| 10 | WEIGHTED_ASK | tick | 5 档加权卖价 |
| 11 | ORDER_IMBALANCE | tick | 买卖量不平衡 (bid-ask)/(bid+ask) |
| 12 | DEPTH_IMBALANCE | tick | 5 档深度不平衡 |
| 13 | TXN_VOLUME | transaction | 成交量 |
| 14 | TXN_VWAP | transaction | 成交 VWAP |
| 15 | BUY_RATIO | transaction | 主买占比（tick rule 推断） |
| 16 | LARGE_ORDER_RATIO | transaction | 大单占比（>90 分位） |
| 17 | TXN_COUNT | transaction | 成交笔数 |
| 18 | ORDER_CANCEL_RATIO | order | 撤单率 |
| 19 | NET_ORDER_FLOW | order | 净委托流 |

**聚合逻辑：**
- `_aggregate_tick_daily()`: 逐笔快照 → 日频 OHLCV + 盘口特征
- `_aggregate_transaction_daily()`: 逐笔成交 → 成交统计特征
- `_aggregate_order_daily()`: 逐笔委托 → 委托行为特征

### 3.4 接口兼容性

`Level2StockData` 实现了与 `StockData` 完全相同的 duck-typing 接口：

```python
# 原系统使用方式（不变）：
data.data[start:stop, int(feature), :]   # Expression.evaluate() 中的索引
data.max_backtrack_days                   # 回溯天数
data.max_future_days                      # 前瞻天数
data.n_days / data.n_stocks / data.n_features  # 维度属性
data[slice]                               # 日期切片
data.make_dataframe(...)                  # 转 DataFrame
```

`Level2FeatureType(IntEnum)` 的前 6 个值（0-5）与原 `FeatureType` 一一对应，所以用原特征（OHLCV）创建的 `Feature` 对象可以直接在 `Level2StockData` 上 evaluate。

### 3.5 RL 环境扩展 (`env_wrapper.py`)

```python
# 原系统：6 个特征 → action space = ops + 6 features + constants + dt + sep
# 新系统：20 个特征 → action space = ops + 20 features + constants + dt + sep

# 支持两种模式：
Level2AlphaEnv(pool, use_level2_features=True)   # 20 维（Level 2）
Level2AlphaEnv(pool, use_level2_features=False)  # 6 维（兼容模式）
```

---

## 四、如何使用

### 4.1 基本用法（仅 OHLCV，从 Level 2 数据提取）

```bash
python scripts/rl_level2.py \
    --data_root=~/EquityLevel2/stock \
    --train_start=2022-01-05 --train_end=2022-09-30 \
    --valid_start=2022-10-01 --valid_end=2022-11-30 \
    --test_start=2022-12-01 --test_end=2022-12-31
```

### 4.2 Level 2 全特征模式

```bash
python scripts/rl_level2.py \
    --data_root=~/EquityLevel2/stock \
    --use_level2_features \
    --pool_capacity=20 \
    --steps=250000
```

### 4.3 指定股票列表

```bash
python scripts/rl_level2.py \
    --data_root=~/EquityLevel2/stock \
    --instruments='["000001.sz","000002.sz","000568.sz"]'
```

### 4.4 在代码中使用

```python
from alphagen_level2 import Level2StockData, Level2Calculator, Level2FeatureType
from alphagen.data.expression import Feature, Mean, Ref

# 加载数据
data = Level2StockData(
    instrument=["000001.sz", "000002.sz"],
    start_time="2022-01-05",
    end_time="2022-06-30",
    data_root="~/EquityLevel2/stock",
)

# 使用 Level 2 特征构建 alpha
order_imb = Feature(Level2FeatureType.ORDER_IMBALANCE)
alpha = Mean(order_imb, 5)  # 5 日均值买卖不平衡

# 计算 IC
close = Feature(Level2FeatureType.CLOSE)
target = Ref(close, -20) / close - 1
calc = Level2Calculator(data, target)
ic = calc.calc_single_IC_ret(alpha)
```

---

## 五、已知潜在问题与后续优化方向

| 问题 | 当前状态 | 优化方向 |
|------|---------|---------|
| 日内数据聚合在 `__init__` 时完成，大数据量首次加载慢 | 可用 | 可加缓存（pickle/parquet），第二次加载直接读聚合结果 |
| HDF5 文件逐日打开，未并行化 | 可用 | 可用 `concurrent.futures` 多线程读取 |
| 股票宇宙自动发现用交集，可能过于保守 | 可用 | 可改为出现率 >80% 的股票 |
| `BUY_RATIO` 使用 tick rule 推断，精度有限 | 可用 | 如有 BSFlag 字段可直接使用 |
| `ORDER_CANCEL_RATIO` / `NET_ORDER_FLOW` 依赖 OrderType/Direction 字段 | 降级为 0 | 需确认 HDF5 中是否有这些字段 |
| 单线程聚合，未向量化 | 可用 | numpy 已向量化，瓶颈在 IO 而非计算 |

---

## 六、依赖变更

新增依赖：`h5py`（用于读取 HDF5 文件）

```bash
pip install h5py
```

原有依赖（torch, pandas, numpy, gymnasium 等）不变。**不再需要 qlib 和 baostock**（仅 Level 2 训练时）。

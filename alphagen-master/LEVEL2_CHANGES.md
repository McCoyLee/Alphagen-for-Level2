# Level2 改造总览（更新版）

> 本文是对当前仓库 **现状** 的整理（含历史改动 + 最近补丁），用于回答“到底改了什么、怎么用、有哪些坑”。

---

## 1. 架构与数据时间轴

当前 Level2 方案是 **bar-level 时序**，不是日频聚合：

- `dim0` 时间轴是连续 bars。
- 表达式 `Mean(x, 5)` 的单位是 5 个 bars。
- `max_backtrack_days / max_future_days` 在 Level2 实际上是 bar 数（命名为 days 仅为兼容旧接口）。

数据流：

```text
tick/order/transaction HDF5 -> 按 bar 聚合 -> Tensor[n_bars_total, n_features, n_stocks] -> Calculator/RL
```

---

## 2. Level2 核心模块（现有）

### 2.1 数据层
- `alphagen_level2/hdf5_reader.py`
  - HDF5 文件读取、文件句柄缓存、按日期批量读取。
- `alphagen_level2/stock_data.py`
  - `Level2StockData`：把 tick / transaction / order 聚合成 bar 特征。
  - 支持 cache（pickle）与并行加载（`max_workers`）。

### 2.2 计算与环境
- `alphagen_level2/calculator.py`
  - `Level2Calculator` 负责 IC / RankIC 相关计算。
- `alphagen_level2/env_wrapper.py`
  - RL 环境包装，适配 Level2 特征与动作空间。
- `alphagen_level2/config.py`
  - Level2 特征集合、算子集合、delta 时间窗口配置。

### 2.3 训练入口
- `scripts/rl_level2.py`：纯 RL 训练入口。
- `scripts/rl_level2_llm.py`：RL + LLM（warmstart / 周期注入 / 回滚控制）。

### 2.4 多样性池
- `alphagen_level2/diversity_pool.py`
  - `DiversityMseAlphaPool`：可配置 `ic_mut_threshold` 与 `diversity_bonus`。

---

## 3. 最近重要行为变化（请重点关注）

## 3.1 Bar 粒度参数：支持“秒级配置”

`Level2StockData` 新增并统一支持：

- `bar_size_min: float`（支持小数分钟）
- `bar_size_sec: Optional[int]`（秒级输入，优先级高于 `bar_size_min`）

实际逻辑：

```python
self._bar_size_min = bar_size_sec / 60.0 if bar_size_sec is not None else bar_size_min
```

这意味着：

- 15 秒 bar：`--bar_size_sec=15`（推荐）
- 或者写成分钟：`--bar_size_min=0.25`
- **不是** `bar_size_min=5`（那是 5 分钟）

同时，bar 相关函数签名已改为浮点分钟：
- `_compute_bar_edges(bar_size_min: float)`
- `_resample_tick_to_bars(..., bar_size_min: float)`
- `_resample_txn_to_bars(..., bar_size_min: float)`
- `_resample_order_to_bars(..., bar_size_min: float)`

缓存 key 也带上高精度 bar 大小，避免不同粒度共用缓存：

```text
...|bar{self._bar_size_min:.8f}
```

---

## 3.2 训练脚本参数透传修正

`rl_level2.py` 与 `rl_level2_llm.py` 均已支持：
- `bar_size_min`
- `bar_size_sec`

并在启动时打印 **effective bar size**（分钟 + 秒），避免参数误判。

另外，`rl_level2.py` 的 `main -> run_single_experiment` 参数透传已补齐（bar 参数会真正生效）。

---

## 3.3 LLM 回滚逻辑：明确“按验证集回滚，不按测试集”

`Level2LLMCallback` 现已改为：

- 显式接收 `valid_calculator`（单独验证集）
- `test_calculators` 仅用于测试指标记录

即：
- 回滚判断使用 validation IC
- test IC 不参与回滚触发

额外增强：
- `valid_smooth_window`：验证 IC 平滑
- `valid_min_delta`：最小提升阈值
- `valid_restore_cooldown`：回滚后冷却
- 快照恢复时包含 `best_obj / best_ic_ret / eval_cnt`，避免恢复后状态不一致

---

## 4. 当前 20 维特征（简版）

- 基础价量：`OPEN/CLOSE/HIGH/LOW/VOLUME/VWAP`
- 盘口类：`BID_ASK_SPREAD`、`MID_PRICE`、`ORDER_IMBALANCE`、`DEPTH_IMBALANCE` 等
- 成交类：`TXN_VOLUME`、`TXN_VWAP`、`BUY_RATIO`、`LARGE_ORDER_RATIO`、`TXN_COUNT`
- 委托类：`ORDER_CANCEL_RATIO`、`NET_ORDER_FLOW`

---

## 5. 性能现状与建议

已具备：
- pickle cache（推荐固定 `cache_dir`）
- `ThreadPoolExecutor(max_workers=...)` 日期并行
- 多环境并行训练 `n_envs`

建议调参顺序：
1. 先把 `cache_dir` 固定下来，避免重复聚合。
2. 调 `max_workers` 到机器可承受范围（例如 8/16/24，按 IO 与 CPU 观测）。
3. 再调 `n_envs`（如 8/12/16）。
4. 先用 `basic` 特征做快速迭代，稳定后再上 `use_level2_features`。

> 注意：bar 越细（例如 15s），单日 bars 越多，表达式评估成本会明显上升。

---

## 6. 常用命令（更新）

### 6.1 纯 RL（15 秒 bar）

```bash
python scripts/rl_level2.py \
  --data_root=~/EquityLevel2/stock \
  --bar_size_sec=15 \
  --max_workers=16 \
  --n_envs=8
```

### 6.2 RL + LLM（15 秒 bar）

```bash
python scripts/rl_level2_llm.py \
  --data_root=~/EquityLevel2/stock \
  --bar_size_sec=15 \
  --max_workers=16 \
  --n_envs=8 \
  --valid_patience=20 \
  --valid_smooth_window=5 \
  --valid_min_delta=0.001
```

### 6.3 如果你坚持用分钟参数

```bash
# 15 秒 = 0.25 分钟
python scripts/rl_level2.py --bar_size_min=0.25
```

---

## 7. 结论

这套改造当前已经覆盖：
- 本地 Level2 数据读取与 bar 聚合
- 纯 RL 与 RL+LLM 双训练入口
- 秒级 bar 参数化
- 多样性控制
- 基于验证集的回滚防过拟合

若后续要进一步提高吞吐，建议下一步做：
- 离线预聚合（按 bar 落盘）
- 更细粒度并行（按股票分片）
- 表达式求值缓存 / JIT

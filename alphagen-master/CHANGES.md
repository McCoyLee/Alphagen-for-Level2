# 改动说明：单因子复合奖励函数（时间序列 IC + 归一化收益）

## 背景

原来的 `SingleFactorAlphaPool` 使用全局 `|IC|` 作为 RL 奖励。对于 tick 级单只 ETF 场景（n\_stocks=1），截面 IC 无意义——只有一个标的，无法在截面上做相关性计算。因此需要：

1. 将 IC 改为**时间序列 IC**（按日计算、按窗口聚合）
2. 引入**归一化收益**作为第二个奖励信号
3. 用加权复合得到最终奖励

---

## 做了什么

### 1. 基类 `AlphaCalculator` / `TensorAlphaCalculator` (`alphagen/data/calculator.py`)

**新增两个抽象方法并在 `TensorAlphaCalculator` 中实现：**

#### `calc_single_ts_IC_ret(expr, bars_per_day, window_days, use_rank)`

时间序列窗口化 IC 计算流程：

```
输入数据: factor[n_bars, 1], target[n_bars, 1]  (单只 ETF)

Step 1: 按 bars_per_day 重塑为 (n_calendar_days, bars_per_day)
Step 2: 每日计算 Pearson / Spearman 相关 → daily_ics[n_days]
Step 3: 去除 NaN 后分为 window_days 天一组的不重叠窗口
Step 4: 每窗口取均值 → window_means[n_windows]
Step 5: 返回 (mean(window_means), std(window_means))
```

- `use_rank=True` 时使用 Rank IC (Spearman)
- `use_rank=False` 时使用 Pearson IC
- 对多标的数据（n\_stocks > 1）自动退化为标准的截面 IC

#### `calc_single_profit(expr, holding_bars, turnover_penalty)`

简化收益指标：

```
pos = clamp(factor_normalized, -1, 1)   # 归一化因子值作为仓位
r   = raw_target                          # 未归一化的 forward return

R = mean(pos · r) − λ · mean(|pos_t − pos_{t-1}|)
R_norm = R / std(r)                       # 归一化使量纲与 IC 可比
```

- `raw_target`：构造函数新增的参数，保存归一化前的原始目标值
- 归一化方法：除以 `std(r)` 使收益项和 IC 在同一量级（约 [-1, 1]）

**`TensorAlphaCalculator.__init__` 新增 `raw_target` 参数**，所有子类 calculator（TickCalculator、Level2Calculator、QLibCalculator）已同步更新传入 raw\_target。

---

### 2. `TickCalculator` (`alphagen_level2/calculator_tick.py`)

- 构造函数保存 `raw_target_tensor`（归一化前的原始目标张量）并传给基类
- 无需额外覆写 `calc_single_ts_IC_ret` 和 `calc_single_profit`，基类实现已自动处理单标的场景

---

### 3. `SingleFactorAlphaPool` (`alphagen/models/linear_alpha_pool.py`)

**完全重写**，原有的纯 `|IC|` 奖励逻辑已删除。

#### 新构造函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `holding_bars` | 100 | 持仓周期（bars），与训练 target 对齐 |
| `bars_per_day` | 4800 | 每个交易日的 bar 数（3s bar = 4800） |
| `window_days` | 20 | IC 计算的窗口大小（交易日数） |
| `ic_weight` | 0.5 | 复合奖励中 IC 项的权重 α |
| `profit_weight` | 0.5 | 复合奖励中收益项的权重 β |
| `use_rank_ic` | False | True=Rank IC, False=Pearson IC |
| `turnover_penalty` | 0.001 | 换手惩罚系数 λ |
| `ic_std_penalty` | 0.0 | 窗口间 IC 标准差的惩罚系数 γ |

#### 奖励公式

```
ts_ic_mean, ts_ic_std = calc_single_ts_IC_ret(expr, bars_per_day, window_days, use_rank)
profit_norm = calc_single_profit(expr, holding_bars, turnover_penalty)

direction = sign(ts_ic_mean)
ic_score  = |ts_ic_mean| − γ · ts_ic_std
profit_directed = profit_norm · direction

composite = α · max(ic_score, 0) + β · max(profit_directed, 0)

RL_reward = composite + max(pool_improvement, 0)
```

#### 池管理

- 因子按 `composite_score` 排名（而非 `|IC|`）
- 超出容量时淘汰 composite 最低的因子
- 权重仍为 ±1（方向由 IC 符号决定）
- `test_ensemble` 仍返回 `(IC, RankIC)` 以兼容 callback 日志

#### Bugfix：`force_load_exprs` 路径现在也走复合评分

此前 LLM warmstart（`DefaultInteraction._parse_and_add → pool.force_load_exprs`）
以及 `_restore_pool_snapshot / build_pool(merged_exprs)` 路径走的是父类
`LinearAlphaPool.force_load_exprs`，该方法调用 `_add_factor` 时会把权重设置
为 `max(ic_ret, 0.01)` 或 `self.weights.mean()`（浮点数），而且从未触达
`_score_expr`，导致 `_composite_scores` 与 `_factor_directions` 保持默认值，
最终 `optimize()` 返回的方向数组对 warmstart 进来的因子无意义。

`SingleFactorAlphaPool` 现在**重写** `force_load_exprs`：
1. 对每个 expr 调用 `_score_expr` 计算 composite + direction；
2. 在 `_add_factor` 之前把方向与得分写入 `_factor_directions` / `_composite_scores`；
3. 最终 `self.weights = self.optimize()` 返回 ±1；
4. 如果调用方显式传入 `weights`，也会被忽略（只保留方向），保持单因子池 ±1 不变量。

---

### 4. 训练脚本 (`scripts/rl_tick_rolling.py`)

新增 CLI 参数：
- `--sf_ic_weight` (default 0.5)
- `--sf_profit_weight` (default 0.5)
- `--sf_use_rank_ic` (default False)
- `--sf_window_days` (default 20)
- `--sf_turnover_penalty` (default 0.001)
- `--sf_ic_std_penalty` (default 0.0)

这些参数在 `--single_factor_mode=True` 时生效，传入 `SingleFactorAlphaPool` 构造函数。

`bars_per_day` 自动从 `TickStockData.bars_per_day` 获取。

移除了 `HAS_SINGLE_FACTOR_POOL` 兼容检查（不再需要 fallback）。

---

### 5. `scripts/rl_level2.py`

同步更新 `SingleFactorAlphaPool` 的构造调用，传入 `holding_bars` 和 `bars_per_day`，使 Level2 bar 级训练也可使用新的复合奖励。

---

### 6. 回测脚本 (`backtest_tick_pnl.py`)

- `simulate_pnl` 返回值新增 `avg_profit_per_trade`（平均每笔 profit，单位为原始收益率）
- 逐因子打印行和图表标题均增加 AvgPPT 列（以 bps 显示）
- 表头增加 `AvgPPT` 列

---

## 使用示例

```bash
# 使用复合奖励训练单因子（默认 α=β=0.5, Pearson IC, 20天窗口）
python scripts/rl_tick_rolling.py \
    --data_root=~/EquityLevel2/stock \
    --instruments='["510300.sh"]' \
    --single_factor_mode=True

# 使用 Rank IC + 更高收益权重 + IC 标准差惩罚
python scripts/rl_tick_rolling.py \
    --data_root=~/EquityLevel2/stock \
    --instruments='["510300.sh"]' \
    --single_factor_mode=True \
    --sf_use_rank_ic=True \
    --sf_ic_weight=0.3 \
    --sf_profit_weight=0.7 \
    --sf_ic_std_penalty=0.5

# 回测并查看平均每笔 profit
python backtest_tick_pnl.py \
    --data_root=~/EquityLevel2/stock \
    --instrument=510300.sh \
    --start=2024-01-01 --end=2024-06-30
```

---

## 文件变更清单

| 文件 | 变更类型 |
|------|---------|
| `alphagen/data/calculator.py` | 基类新增 `calc_single_ts_IC_ret`、`calc_single_profit`；`TensorAlphaCalculator` 增加 `raw_target` |
| `alphagen_level2/calculator_tick.py` | 保存 raw\_target 传给基类 |
| `alphagen_level2/calculator.py` | 保存 raw\_target 传给基类 |
| `alphagen_qlib/calculator.py` | 保存 raw\_target 传给基类 |
| `alphagen/models/linear_alpha_pool.py` | 重写 `SingleFactorAlphaPool`（复合奖励） |
| `scripts/rl_tick_rolling.py` | 新增 sf\_\* CLI 参数，更新 pool 构建 |
| `scripts/rl_level2.py` | 同步 SingleFactorAlphaPool 构造参数 |
| `backtest_tick_pnl.py` | 新增 avg\_profit\_per\_trade 指标 |
| `CHANGES.md` | 本文件 |

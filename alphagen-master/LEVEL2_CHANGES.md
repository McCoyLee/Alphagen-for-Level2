# Level 2 本地数据集成 — 变更说明
## 零、核心架构：Bar-level 计算
Show less
**本系统采用 bar-level（N 分钟 K 线）计算，而非日频聚合。**
```
Tick 快照（~4832/日）→ 3min K 线（~80 bars/日）→ 表达式/RL 以 bar 为时间单位
```
关键设计：
- 时间轴是连续的 bars，不是 days。`data.data` 的 dim0 是 bar 索引
- 表达式系统的 `Rolling(x, 5)` 意味着"过去 5 个 bar"（= 15 分钟），而不是"过去 5 天"
- `max_backtrack_days` 和 `max_future_days` 的单位实际上是 **bars**（接口名保持不变以兼容）
- 对于 3min bars：80 bars ≈ 1 交易日（240min / 3min = 80）
- `DELTA_TIMES = [5, 10, 20, 40, 80]` 对应 15min, 30min, 1h, 2h, ~1day
---
## 一、数据格式说明
### 目录结构
```
~/EquityLevel2/stock/
├── order/
│   └── YYYYMMDD.h5
│       └── /{stock_code}/  (e.g. /0000561.sz/)
│           ├── BizIndex      {N}      成交编号
│           ├── OrderOriNo    {N}      原始委托号（用于撤单）
│           ├── OrderNumber   {N}      委托编号
│           ├── FunctionCode  {N}      委托类型: B=买, S=卖
│           ├── OrderKind     {N}      委托类别: 0=限价, 1=市价, A=新增, D=撤单, U=修改, S=特殊
│           ├── Price         {N}      委托价格
│           ├── Volume        {N}      委托数量
│           └── Time          {N}      时间戳
├── tick/
│   └── YYYYMMDD.h5
│       └── /{stock_code}/  (e.g. /000001.sz/)
│           ├── Price          {N}        最新价
│           ├── Volume         {N}        当前 tick 成交量
│           ├── Turnover       {N}        当前 tick 成交金额
│           ├── AccVolume      {N}        当日累计成交量
│           ├── AccTurnover    {N}        当日累计成交额
│           ├── MatchItem      {N}        当前 tick 成交笔数
│           ├── BSFlag         {N}        主动方向: B=主买, S=主卖, ""=无成交
│           ├── BidAvgPrice    {N}        买方平均价 (BAP)
│           ├── AskAvgPrice    {N}        卖方平均价 (AAP)
│           ├── BidPrice10     {N, 10}    10 档买价
│           ├── BidVolume10    {N, 10}    10 档买量
│           ├── AskPrice10     {N, 10}    10 档卖价
│           ├── AskVolume10    {N, 10}    10 档卖量
│           ├── TotalBidVolume {N}        所有买单挂单量 (TBV)
│           ├── TotalAskVolume {N}        所有卖单挂单量 (TAV)
│           ├── Open           {N}        开盘价（快照维护）
│           ├── High           {N}        最高价（快照维护）
│           ├── Low            {N}        最低价（快照维护）
│           ├── PreClose       {N}        昨日收盘价
│           └── Time           {N}        时间戳
└── transaction/
    └── YYYYMMDD.h5
        └── /{stock_code}/
            ├── BidOrder       {N}      买方 ID
            ├── AskOrder       {N}      卖方 ID
            ├── BSFlag         {N}      主动买卖方向
            ├── Channel        {N}      撮合通道
            ├── FunctionCode   {N}      业务类型: 0=成交, 1=撤单
            ├── BizIndex       {N}      交易业务流水序号
            ├── Index          {N}      成交序号
            ├── OrderKind      {N}      订单类别
            ├── Price          {N}      成交价格
            ├── Volume         {N}      成交数量
            └── Time           {N}      时间戳
```
### 数据量级（单日单股参考）
| 类别 | 典型记录数 | 说明 |
|------|-----------|------|
| tick | ~4,832 | 3 秒快照，含 10 档盘口 2D 数组 |
| transaction | ~135,706 | 逐笔成交 |
| order | ~19,277 | 逐笔委托 |
---
## 二、改了什么
### 新增文件
| 文件 | 职责 |
|------|------|
| `alphagen_level2/__init__.py` | 包入口 |
| `alphagen_level2/hdf5_reader.py` | HDF5 IO 层：懒加载、文件缓存、批量读取、字符串字段处理 |
| `alphagen_level2/stock_data.py` | `Level2StockData`：日内数据聚合、pickle 缓存、多线程并行加载 |
| `alphagen_level2/calculator.py` | `Level2Calculator`：IC/RankIC 计算（复用原有算法） |
| `alphagen_level2/config.py` | 配置：算子集、特征集、常量 |
| `alphagen_level2/env_wrapper.py` | `Level2EnvWrapper`：支持 20 维特征的 RL 动作空间 |
| `alphagen_level2/features.py` | 20 个特征变量定义 |
| `scripts/rl_level2.py` | 训练入口脚本 |
**原有代码零修改。**
---
## 三、特征定义（20 维）
### 3.1 基础特征（索引 0-5，与原系统兼容）
每个 bar（3min 窗口）内从 tick 快照聚合：
| 索引 | 名称 | 来源 | 聚合逻辑（per bar） |
|------|------|------|---------|
| 0 | OPEN | tick.Price | bar 内首个 tick 的 Price |
| 1 | CLOSE | tick.Price | bar 内末个 tick 的 Price |
| 2 | HIGH | tick.Price | bar 内 max(Price) |
| 3 | LOW | tick.Price | bar 内 min(Price) |
| 4 | VOLUME | tick.Volume | bar 内 sum(Volume) |
| 5 | VWAP | tick.Price × Volume | `Σ(P×V) / ΣV` |
### 3.2 盘口特征（索引 6-12，从 tick 10 档数据聚合）
| 索引 | 名称 | 聚合逻辑（per bar） |
|------|------|---------|
| 6 | BID_ASK_SPREAD | bar 内 `mean(AskPrice10[:,0] - BidPrice10[:,0])` |
| 7 | BID_ASK_SPREAD_PCT | bar 内 `mean(spread / mid)` |
| 8 | MID_PRICE | bar 内 `mean((Ask1 + Bid1) / 2)` |
| 9 | WEIGHTED_BID | bar 内每 tick 10 档买量加权均价的均值 |
| 10 | WEIGHTED_ASK | bar 内每 tick 10 档卖量加权均价的均值 |
| 11 | ORDER_IMBALANCE | bar 内 `mean((TBV - TAV) / (TBV + TAV))` |
| 12 | DEPTH_IMBALANCE | bar 内 10 档总量不平衡均值 |
### 3.3 成交特征（索引 13-17，从 transaction 聚合）
| 索引 | 名称 | 聚合逻辑（per bar） |
|------|------|---------|
| 13 | TXN_VOLUME | bar 内实际成交量（`FunctionCode==0` 过滤撤单） |
| 14 | TXN_VWAP | bar 内实际成交 VWAP |
| 15 | BUY_RATIO | bar 内 `sum(vol[BSFlag=='B']) / total_vol` |
| 16 | LARGE_ORDER_RATIO | bar 内 `sum(vol[vol >= P90]) / total_vol` |
| 17 | TXN_COUNT | bar 内实际成交笔数 |
### 3.4 委托特征（索引 18-19，从 order 聚合）
| 索引 | 名称 | 聚合逻辑（per bar） |
|------|------|---------|
| 18 | ORDER_CANCEL_RATIO | bar 内 `count(OrderKind=='D') / total_orders` |
| 19 | NET_ORDER_FLOW | bar 内 `(vol[FunctionCode=='B'] - vol[FunctionCode=='S']) / total_vol` |
---
## 四、为什么这样改
### 4.1 vs 上一版的修正
| 问题 | 上一版（错误） | 本版（正确） |
|------|--------------|------------|
| **时间粒度** | 日频聚合（每日 1 行） | Bar-level 聚合（每日 ~80 行 @3min） |
| 盘口档位 | `BidPrice1..5`（5 个 1D 字段） | `BidPrice10` shape `(N,10)` 一个 2D 数组 |
| OHLC | 从日内 Open/High/Low 快照字段取值 | 从 bar 内 tick Price 计算 OHLC |
| 成交量 | `AccVolume[-1]` 日累计 | bar 内 `sum(Volume)` |
| VWAP | `AccTurnover[-1] / AccVolume[-1]` | bar 内 `Σ(P×V) / ΣV` |
| 主买判断 | tick rule（价格变动方向推断） | 直接用 `BSFlag` 字段（B/S 明确标记） |
| 撤单检测 | 查找不存在的 `OrderType==2` | 用 `OrderKind=='D'` |
| 买卖方向 | 查找不存在的 `Direction` 字段 | 用 `FunctionCode=='B'/'S'` |
| 字符串字段 | 未处理 HDF5 bytes 编码 | `match_char()` 统一处理 `b'B'`/`ord('B')`/`'B'` |
### 4.2 为什么不修改原有代码
`Level2StockData` 实现了与 `StockData` 完全相同的 duck-typing 接口：
```python
data.data[start:stop, int(feature), :]   # Feature.evaluate() 中的张量索引
data.max_backtrack_days / max_future_days # 回溯/前瞻（单位是 bars，但属性名保持兼容）
data.n_days / n_stocks / n_features       # n_days 实际返回 n_bars
```
表达式系统（`Expression`）、算子（`Operator`）、RL 环境（`AlphaEnvCore`）全部无需修改。
关键洞察：**表达式系统只关心 dim0 的索引，不关心它代表"天"还是"bar"。**
---
## 五、性能优化
### 5.1 已实现的优化
| 优化 | 位置 | 效果 |
|------|------|------|
| **Pickle 缓存** | `Level2StockData(cache_dir=...)` | 首次聚合后缓存，后续加载 <1s（vs 分钟级） |
| **多线程 IO** | `ThreadPoolExecutor(max_workers=4)` | h5py 读取时释放 GIL，4 线程 ~3x 加速 |
| **批量读取** | `read_stocks_batch()` | 每日 .h5 只打开一次，读所有股票 |
| **2D 向量化** | `(bp * bv).sum(axis=1)` | 10 档加权均价一行 numpy，无 Python 循环 |
| **文件句柄缓存** | `_file_cache` | 避免重复 open/close 同一 .h5 文件 |
| **智能过滤** | `FunctionCode==0` | transaction 先过滤撤单再聚合，减少无效计算 |
### 5.2 使用缓存
```bash
# 首次运行：读取 HDF5 并聚合（较慢，取决于数据量）
python scripts/rl_level2.py --cache_dir=./out/l2_cache ...
# 后续运行：直接读 pickle 缓存（<1秒）
python scripts/rl_level2.py --cache_dir=./out/l2_cache ...
# 禁用缓存（数据更新后需要重新聚合）
python scripts/rl_level2.py --cache_dir=None ...
```
### 5.3 进一步优化建议（需要重构）
以下是当前架构下仍可改进的方向，按优先级排序：
#### P0：瓶颈分析
当前系统的性能瓶颈分布：
```
┌──────────────────────────────────────────────────────┐
│ 阶段                   │ 首次    │ 缓存后  │ 瓶颈    │
│────────────────────────│─────────│─────────│─────────│
│ HDF5 读取 (IO)         │ 60-80%  │   0%    │ IO      │
│ Bar 级聚合 (CPU)       │ 15-30%  │   0%    │ CPU     │
│ Expression eval (GPU)  │  5-10%  │  70%    │ GPU     │
│ RL step (GPU)          │  <5%    │  30%    │ GPU     │
└──────────────────────────────────────────────────────┘
```
**注意**：Bar 模式下 tensor dim0 更大（~80x vs 日频），GPU 端 Expression eval 耗时会增加。
使用 pickle 缓存后，IO 和聚合成本归零；剩余瓶颈在 GPU 端的表达式求值。
#### P1：如果需要更大规模数据
| 方向 | 描述 | 预计收益 |
|------|------|---------|
| **预聚合工具** | 写一个离线脚本，将 HDF5 聚合结果存为 `daily_features.parquet`，后续直接读 | 10-100x IO 加速 |
| **内存映射** | 对超大数据集，用 `np.memmap` 替代全量加载 | 减少内存占用 |
| **分片读取** | 按股票分片并行处理，而非按日期 | 对 >1000 股有效 |
#### P2：计算密集场景
| 方向 | 描述 | 预计收益 |
|------|------|---------|
| **GPU 聚合** | 将日内聚合搬到 GPU（`cupy` 或 `torch`） | 对大 tick 数据 2-5x |
| **JIT 编译** | 用 `numba.jit` 加速聚合函数 | 对 CPU 聚合 3-10x |
| **Expression 缓存** | 缓存常用子表达式的求值结果 | 减少重复计算 |
#### P3：架构级优化（大改动）
| 方向 | 描述 | 何时需要 |
|------|------|---------|
| **流式聚合** | 不预聚合为日频，直接在 tick 级别做滚动计算 | 需要 tick 级别的 alpha |
| **Arrow/Parquet 替换 HDF5** | 列式存储，更适合批量读取少量字段 | 数据重建时 |
| **分布式计算** | Dask/Ray 多机并行 | >10TB 数据 |
---
## 六、使用方式
### 基本用法（3min bars）
```bash
# OHLCV 训练（6 维特征，3min bars，默认）
python scripts/rl_level2.py \
    --data_root=~/EquityLevel2/stock \
    --train_start=2022-01-05 --train_end=2022-09-30 \
    --valid_start=2022-10-01 --valid_end=2022-11-30 \
    --test_start=2022-12-01 --test_end=2022-12-31
# Level 2 全特征训练（20 维特征，3min bars）
python scripts/rl_level2.py \
    --data_root=~/EquityLevel2/stock \
    --use_level2_features \
    --pool_capacity=20
# 5 分钟 bars
python scripts/rl_level2.py \
    --data_root=~/EquityLevel2/stock \
    --bar_size_min=5
# 自定义 lookback/forward（单位：bars）
# 160 bars @3min ≈ 2 天回溯
python scripts/rl_level2.py \
    --data_root=~/EquityLevel2/stock \
    --max_backtrack_bars=160 --max_future_bars=80
# 指定股票 + 并行加载
python scripts/rl_level2.py \
    --data_root=~/EquityLevel2/stock \
    --instruments='["000001.sz","000002.sz"]' \
    --max_workers=8
```
### Bar 参数对照表
| bar_size_min | bars/day | 80 bars = | DELTA_TIMES 含义 |
|-------------|----------|-----------|-----------------|
| 1 | 240 | 20min | 5/10/20/40/80 min |
| 3 (默认) | 80 | 1 day | 15min/30min/1h/2h/1day |
| 5 | 48 | ~1.7h | 25min/50min/100min/... |
### 代码中使用
```python
from alphagen_level2 import Level2StockData, Level2Calculator, Level2FeatureType
from alphagen.data.expression import Feature, Mean, Ref
# 加载数据（3min bars，自动缓存）
data = Level2StockData(
    instrument=["000001.sz", "000002.sz"],
    start_time="2022-01-05",
    end_time="2022-06-30",
    data_root="~/EquityLevel2/stock",
    cache_dir="./out/l2_cache",
    max_workers=4,
    bar_size_min=3,        # 3 分钟 K 线
    max_backtrack_days=80,  # 80 bars 回溯（实际是 bars，属性名保持兼容）
    max_future_days=80,     # 80 bars 前瞻
)
# 用 Level 2 特征构建 alpha
oi = Feature(Level2FeatureType.ORDER_IMBALANCE)
alpha = Mean(oi, 5)  # 5 bars 滚动订单不平衡（= 15 分钟）
# 计算 IC
close = Feature(Level2FeatureType.CLOSE)
target = Ref(close, -80) / close - 1  # 80 bars ≈ 1 天的未来收益
calc = Level2Calculator(data, target)
print(f"IC = {calc.calc_single_IC_ret(alpha):.4f}")
```



## 八、RL+LLM 辅助训练 (`rl_level2_llm.py`)
 
### 8.1 训练模式
 
| 模式 | 启动参数 | 说明 |
|------|---------|------|
| 纯 RL | （默认） | 与 `rl_level2.py` 相同，附带收敛曲线记录 |
| LLM 暖启动 | `--llm_warmstart` | LLM 生成初始 pool，然后纯 RL 训练 |
| LLM 定期辅助 | `--use_llm` | 暖启动 + 每 N 步 LLM 注入候选 alpha |
 
### 8.2 LLM 注入方式
 
| 方式 | 参数 | 行为 | 适用场景 |
|------|------|------|---------|
| 温和注入（默认） | `--gentle_inject` | LLM 生成候选，走 `try_new_expr` 正常路径，pool 自主决定是否接受 | 推荐，稳定 |
| 激进注入 | `--gentle_inject=False` | 先删除最差 N 个 alpha，LLM 做 20 轮 `bulk_edit` 替换 | 不推荐，见下方 Bug 分析 |
 
### 8.3 多环境并行
 
```bash
python scripts/rl_level2_llm.py --n_envs=4 --data_root=...
```
 
- 使用 SB3 `DummyVecEnv`（同进程，单线程安全）
- 所有 env 共享同一个 pool 对象
- PPO 参数自动缩放：`batch_size = 128 * n_envs`, `n_steps = max(2048 // n_envs, 256)`
 
### 8.4 多样性 Alpha 池
 
```bash
python scripts/rl_level2_llm.py --ic_mut_threshold=0.85 --diversity_bonus=0.1
```
 
- 使用 `DiversityMseAlphaPool` 替代 `MseAlphaPool`
- `ic_mut_threshold`: 新 alpha 与现有 alpha 的 **最大** mutual IC 超过阈值则拒绝
- `diversity_bonus`: 奖励中附加多样性项 `bonus * (1 - avg_abs_mutual_ic)`
 
### 8.5 验证集过拟合控制
 
- Callback 跟踪验证集 IC（第一个 test calculator）
- 记录最佳验证 IC 时的 pool 快照
- 连续 `valid_patience`（默认 20）个 rollout 验证 IC 未改善时，回滚到最佳快照
- `valid_patience=0` 关闭此功能
 

## 十、收敛曲线日志 (`convergence_logger.py`)
 
### 输出文件
| 文件 | 内容 |
|------|------|
| `convergence.csv` | 每个 rollout 的指标：timestep, pool_size, train_ic, test_ic, eval_cnt |
| `convergence.json` | 完整运行摘要，含最佳 IC 及对应步数 |
| `convergence.png` | 2×2 图：train IC, test IC, pool size, eval_cnt |
 
### 可视化
```python
from alphagen_level2.convergence_logger import plot_convergence, compare_runs
# 单次运行
plot_convergence("out/results/xxx/convergence.csv")
# 多次运行对比
compare_runs(["run1/convergence.csv", "run2/convergence.csv"])
```
 
---
## 十一、可训练 Action Prior (`action_prior.py`)
 
### 架构
Transformer encoder + 分类头，输入已生成的 token 前缀，输出下一个 action 的概率分布。
 
### 训练数据
从历史成功 alpha 的表达式树中提取 (prefix → next_action) 监督学习对，以 IC 加权采样。
 
### 使用
```bash
# 训练 prior 模型
python scripts/train_action_prior.py --data_dirs="out/results/run1,out/results/run2"
 
# 用 prior 引导 RL
python scripts/rl_level2_guided.py --prior_path=out/prior/model.pt --beta=0.1
```
 
`beta` 控制 prior 引导强度：`shaped_reward = reward + beta * log(prior_prob[action])`

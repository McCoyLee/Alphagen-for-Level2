# Tick Rolling RL + LLM (Main Branch Guide)

本说明面向当前 `main` 分支的 tick 级别（3 秒 bar）滚动训练脚本：

- 入口脚本：`scripts/rl_tick_rolling.py`
- 数据与特征：`alphagen_level2/stock_data_tick.py` + `alphagen_level2/config_tick.py`
- 评估器：`alphagen_level2/calculator_tick.py`
- 环境封装：`alphagen_level2/env_wrapper_tick.py`

---

## 1. 目标与能力

`rl_tick_rolling.py` 实现 walk-forward（滚动窗口）训练流程，核心能力包括：

1. **滚动窗口训练**（train/valid/test）
2. **RL Alpha 搜索**（Maskable PPO）
3. **可选 LLM warm-start 与周期注入**
4. **验证集平滑与回滚（防过拟合）**
5. **并行环境支持**（`SubprocVecEnv`）

---

## 2. 目录结构与职责

### 2.1 训练入口

- `scripts/rl_tick_rolling.py`
  - 构建 rolling schedule
  - 分窗加载数据
  - 训练/验证/测试评估
  - LLM 交互（可选）
  - 保存每个窗口的模型、pool、收敛日志

### 2.2 数据层

- `alphagen_level2/stock_data_tick.py`
  - 读取 Level2 HDF5
  - 将 tick 快照重采样为 3 秒 bar
  - 产出 20 个微观结构特征
  - 支持缓存与多线程读取

### 2.3 评估层

- `alphagen_level2/calculator_tick.py`
  - 计算单 alpha / pool 的 IC、RankIC
  - 自动识别单标的模式（`n_stocks == 1`）并走时间序列归一化与时序相关计算

### 2.4 环境层

- `alphagen_level2/env_wrapper_tick.py`
  - 将 `AlphaEnvCore` 封装为 Gym 环境
  - 定义动作空间（算子、特征、常数、时间窗口等）

### 2.5 配置层

- `alphagen_level2/config_tick.py`
  - 算子集合（含 `Cov/Corr`）
  - `DELTA_TIMES`（默认含 `[20, 100, 600, 1200, 4800]`）
  - 常量集合、特征集合

---

## 3. 训练流程（按窗口）

每个窗口执行：

1. 加载 train/valid/test 三段数据
2. 构建 target：`Ref(close, -max_future_bars) / close - 1`
3. 构建 alpha pool（支持 diversity pool）
4. 可选：
   - 从上一个窗口 pool warm-start
   - LLM 生成初始 pool（`llm_warmstart`）
5. 构建环境（`n_envs > 1` 时用 `SubprocVecEnv`）
6. PPO 训练
7. 回调中记录指标、可选 LLM 注入、验证回滚
8. 保存窗口结果与全局汇总

---

## 4. 关键参数说明

以下参数都在 `scripts/rl_tick_rolling.py::main` 中：

### 4.1 基础参数

- `seed`：随机种子
- `instruments`：标的（JSON list 或字符串）
- `data_root`：HDF5 根目录
- `use_all_features`：是否使用 20 特征（否则 7 基础特征）
- `bar_size_sec`：bar 粒度（默认 3 秒）

### 4.2 滚动窗口参数

- `global_start`, `global_end`
- `train_months`, `valid_months`, `test_months`, `step_months`

### 4.3 序列长度参数

- `max_backtrack_bars`
- `max_future_bars`

### 4.4 训练并行与IO

- `n_envs`：环境并行数（>1 使用 `SubprocVecEnv`）
- `max_workers`：数据读取线程数

### 4.5 LLM 参数

- `llm_warmstart`
- `use_llm`
- `llm_every_n_steps`
- `drop_rl_n`
- `llm_replace_n`
- `llm_base_url`, `llm_api_key`, `llm_model`
- `gentle_inject`
- `llm_init_min_pool_size`（LLM 初始化池过小时自动回退填充）

### 4.6 验证回滚参数

- `valid_patience`
- `valid_min_delta`
- `valid_smooth_window`
- `valid_restore_cooldown`

### 4.7 跨窗口稳定因子聚合（Step1）

- `stable_pool_min_occurrence`
- `stable_pool_min_sign_consistency`
- `stable_pool_max_factors`

---

## 5. 快速开始

> 建议从纯 RL、单环境开始，先跑通再开 LLM 与并行。

```bash
cd /path/to/alphagen-master

PYTHONPATH=. python scripts/rl_tick_rolling.py \
  --data_root=/root/data/subset_data \
  --instruments='["159845.sz"]' \
  --llm_warmstart=False \
  --use_llm=False \
  --n_envs=1 \
  --steps_per_window=20000
```

查看完整参数：

```bash
PYTHONPATH=. python scripts/rl_tick_rolling.py -- --help
```

---

## 6. 输出文件

训练结果默认在：

- `./out/tick_rolling/tick_{bar_size}s_pool{pool_capacity}_seed{seed}_{timestamp}/`

每个窗口会产生：

- `window_xxx/window_meta.json`
- `window_xxx/final_pool.json`
- `window_xxx/convergence.csv`
- checkpoint 模型与 `_pool.json`

全局会产生：

- `run_config.json`
- `rolling_results.json`
- `stable_factor_pool.json`（跨窗口稳定因子筛选结果）

---

## 7. 常见问题

### Q1: 明明 `n_envs=1`，还是 GPU OOM？

`n_envs` 不是唯一因素。表达式评估（尤其 `Cov/Corr/Std/Var/Mad` 等 rolling/pair rolling）会产生大中间张量。

建议：

1. 先缩小窗口（`train_months/valid_months/test_months`）
2. 降 `max_backtrack_bars` / `max_future_bars`
3. 必要时收窄 `DELTA_TIMES`（去掉 4800）
4. 先关 LLM，再逐步打开

### Q2: Subproc 训练出现 `EOFError`？

常见是子进程 OOM 后主进程收不到返回。先把 `n_envs` 降到 1 排查，再逐步增加。

### Q3: 单标的 IC 全 0？

当前分支已在 `TickCalculator` 里加入单标的路径（时序归一化 + 时序 IC/RankIC）。

---

## 8. 推荐调参顺序

1. `n_envs=1` + `use_llm=False` 跑通
2. 缩小窗口验证稳定性
3. 打开 LLM warmstart
4. 再尝试 `n_envs=2/4` 并监控显存
5. 最后调 `valid_*` 回滚参数

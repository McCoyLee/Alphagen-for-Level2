#!/bin/bash

LOG_DIR="./out/paper_logs"
mkdir -p "$LOG_DIR"

run_timed() {
    local name="$1"
    local log="$LOG_DIR/${name}.log"
    shift
    if [ -s "$log" ]; then
        echo "跳过: $name (已完成)"
        return 0
    fi
    echo ""
    echo "========================================"
    echo "开始: $name | $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    local start=$(date +%s)
    "$@" 2>&1 | tee "$log" ; true
    local end=$(date +%s)
    local elapsed=$((end - start))
    echo "[完成] $name 耗时: $((elapsed/3600))h $(($((elapsed%3600))/60))m $((elapsed%60))s" \
        | tee -a "$LOG_DIR/timing_summary.txt"
}

echo "========================================"  >> "$LOG_DIR/timing_summary.txt"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"   >> "$LOG_DIR/timing_summary.txt"

# 1. RL - CSI300 (5 seeds)
run_timed "RL_CSI300" \
    python scripts/rl.py \
        --random_seeds 0,1,2,3,4 \
        --instruments csi300 \
        --pool_capacity 10

# 2. RL - CSI500 (5 seeds)
run_timed "RL_CSI500" \
    python scripts/rl.py \
        --random_seeds 0,1,2,3,4 \
        --instruments csi500 \
        --pool_capacity 10

# 3. LLM-only - CSI300 (每个seed单独跑)
for seed in 0 1 2 3 4; do
    run_timed "LLM_CSI300_seed${seed}" \
        python scripts/llm_only.py \
            --pool_size 10 \
            --n_updates 20
done

# 4. HARLA (RL+LLM) - CSI300
run_timed "HARLA_CSI300" \
    python scripts/rl.py \
        --random_seeds 0,1,2,3,4 \
        --instruments csi300 \
        --pool_capacity 10 \
        --use_llm True \
        --llm_every_n_steps 25000 \
        --drop_rl_n 5

# 5. HARLA (RL+LLM) - CSI500
run_timed "HARLA_CSI500" \
    python scripts/rl.py \
        --random_seeds 0,1,2,3,4 \
        --instruments csi500 \
        --pool_capacity 10 \
        --use_llm True \
        --llm_every_n_steps 25000 \
        --drop_rl_n 5

# 6. GP baseline
run_timed "GP" \
    python gp.py

# 7. 回测
run_timed "Backtest" \
    python backtest.py

# 8. 汇总结果
echo ""
echo "========================================"
echo "结果汇总"
echo "========================================"
python -c "
import json
import numpy as np
from pathlib import Path

print()
print('=' * 70)
print('论文复现结果汇总 (测试期: 2020-01-01 ~ 2021-12-31)')
print('=' * 70)
print(f\"{'方法':<20} {'超额年化收益':>12} {'IR':>8} {'Sharpe':>8} {'最大回撤':>10}\")
print('-' * 70)

methods = {}
for f in sorted(Path('out/backtests').rglob('*-result.json')):
    data = json.load(open(f))
    method = f.parts[-3] if len(f.parts) >= 3 else f.parent.name
    if method not in methods:
        methods[method] = []
    methods[method].append(data)

if not methods:
    print('暂无回测结果，请先运行回测')
else:
    for method, results in sorted(methods.items()):
        exc = [r['annual_excess_return'] for r in results]
        ir  = [r['information_ratio'] for r in results]
        sh  = [r['sharpe'] for r in results]
        mdd = [r['max_drawdown'] for r in results]
        print(f\"{method:<20} {np.mean(exc):>11.4f} {np.mean(ir):>8.4f} {np.mean(sh):>8.4f} {np.mean(mdd):>10.4f}\")
        if len(results) > 1:
            print(f\"{'  ±std':<20} {np.std(exc):>11.4f} {np.std(ir):>8.4f} {np.std(sh):>8.4f} {np.std(mdd):>10.4f}\")
    print('=' * 70)

print()
print('各阶段耗时：')
try:
    print(open('out/paper_logs/timing_summary.txt').read())
except:
    pass
"

echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_DIR/timing_summary.txt"
echo "全部完成！"

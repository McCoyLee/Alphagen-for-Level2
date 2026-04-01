#!/bin/bash

set -e  # 任意一个失败就退出

echo "=== Run 3: Pure RL baseline ==="
python scripts/rl_level2.py \
  --data_root=~/EquityLevel2/stock \
  --random_seeds=0 \
  --use_level2_features=True \
  --pool_capacity=20 \
  --steps=250000 \
  --train_start=2023-03-01 --train_end=2023-06-30 \
  --valid_start=2023-07-01 --valid_end=2023-07-31 \
  --test_start=2023-08-01 --test_end=2023-08-31 \
  --bar_size_min=3 \
  --max_backtrack_bars=80 --max_future_bars=80 \
  --cache_dir=./out/l2_cache \
  --max_workers=16 \
  --n_envs=8

echo "=== Run 1: RL + LLM ==="
python scripts/rl_level2_llm.py \
  --data_root=~/EquityLevel2/stock \
  --random_seeds=0 \
  --use_level2_features=True \
  --pool_capacity=20 \
  --steps=250000 \
  --llm_warmstart=True \
  --use_llm=True \
  --llm_every_n_steps=25000 \
  --drop_rl_n=5 \
  --llm_replace_n=3 \
  --gentle_inject=True \
  --train_start=2023-03-01 --train_end=2023-06-30 \
  --valid_start=2023-07-01 --valid_end=2023-07-31 \
  --test_start=2023-08-01 --test_end=2023-08-31 \
  --bar_size_min=3 \
  --max_backtrack_bars=80 --max_future_bars=80 \
  --cache_dir=./out/l2_cache \
  --max_workers=16 \
  --n_envs=8 \
  --ic_mut_threshold=0.9 \
  --diversity_bonus=0.04 \
  --plot_interval=10 \
  --valid_patience=20 \
  --valid_min_delta=0.001 \
  --valid_smooth_window=5 \
  --valid_restore_cooldown=5

echo "=== Run 2: RL (no LLM) ==="
python scripts/rl_level2_llm.py \
  --data_root=~/EquityLevel2/stock \
  --random_seeds=0 \
  --use_level2_features=True \
  --pool_capacity=20 \
  --steps=250000 \
  --llm_warmstart=True \
  --use_llm=False \
  --llm_replace_n=3 \
  --train_start=2023-03-01 --train_end=2023-06-30 \
  --valid_start=2023-07-01 --valid_end=2023-07-31 \
  --test_start=2023-08-01 --test_end=2023-08-31 \
  --bar_size_min=3 \
  --max_backtrack_bars=80 --max_future_bars=80 \
  --cache_dir=./out/l2_cache \
  --max_workers=16 \
  --plot_interval=10 



echo "=== All Done ==="
#!/usr/bin/env bash
# Run the full walk-forward rolling pipeline (5 windows by default) with N
# parallel factor pools per window. Each pool is an independent RL agent with
# its own seed; results are written to ./out/tick_rolling/<run-tag>/.
#
# Usage:
#   bash scripts/run_full_rolling.sh                       # 1 round, default args
#   bash scripts/run_full_rolling.sh --rounds 3            # 3 sequential rounds
#   bash scripts/run_full_rolling.sh --n_parallel_pools 4  # 4 pools per window
#   bash scripts/run_full_rolling.sh --steps 200000        # override RL steps
#
# Any unknown flag is forwarded verbatim to rl_tick_rolling.py, e.g.
#   bash scripts/run_full_rolling.sh --single_factor_mode True --sf_alpha 1.0
set -euo pipefail

# ---- Defaults (override via CLI flags below) ----
ROUNDS=1
BASE_SEED=0
N_PARALLEL_POOLS=2
STEPS_PER_WINDOW=150000
INSTRUMENTS='["510300.sh"]'
DATA_ROOT="${DATA_ROOT:-$HOME/EquityLevel2/stock}"
GLOBAL_START="2023-01-01"
GLOBAL_END="2025-10-31"
TRAIN_MONTHS=6
VALID_MONTHS=2
TEST_MONTHS=2
STEP_MONTHS=6

# ---- Parse known flags; collect the rest into EXTRA_ARGS ----
# Handles both "--flag value" and "--flag=value" formats.
_val() {
    # Extract value: if $1 contains '=', return the part after it;
    # otherwise return $2 (next positional) and signal caller to shift twice.
    if [[ "$1" == *=* ]]; then
        echo "${1#*=}"
        return 1   # shift 1
    else
        echo "$2"
        return 0   # shift 2
    fi
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    key="${1%%=*}"   # flag name without any =value suffix
    case "$key" in
        --rounds)
            ROUNDS="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --base_seed)
            BASE_SEED="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --n_parallel_pools)
            N_PARALLEL_POOLS="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --steps|--steps_per_window)
            STEPS_PER_WINDOW="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --instruments)
            INSTRUMENTS="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --data_root)
            DATA_ROOT="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --global_start)
            GLOBAL_START="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --global_end)
            GLOBAL_END="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --train_months)
            TRAIN_MONTHS="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --valid_months)
            VALID_MONTHS="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --test_months)
            TEST_MONTHS="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        --step_months)
            STEP_MONTHS="$(_val "$1" "${2:-}")" && shift 2 || shift ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================================================"
echo "[run_full_rolling] $(date)"
echo "  rounds            : $ROUNDS"
echo "  base_seed         : $BASE_SEED"
echo "  n_parallel_pools  : $N_PARALLEL_POOLS"
echo "  steps_per_window  : $STEPS_PER_WINDOW"
echo "  instruments       : $INSTRUMENTS"
echo "  data_root         : $DATA_ROOT"
echo "  schedule          : ${GLOBAL_START}~${GLOBAL_END} "
echo "                      (${TRAIN_MONTHS}m train / ${VALID_MONTHS}m valid"
echo "                       / ${TEST_MONTHS}m test, step=${STEP_MONTHS}m)"
echo "  extra args        : ${EXTRA_ARGS[*]:-<none>}"
echo "================================================================"

for ((round=0; round < ROUNDS; round++)); do
    seed=$((BASE_SEED + round * N_PARALLEL_POOLS))
    echo
    echo "================================================================"
    echo "[run_full_rolling] Round $((round+1))/$ROUNDS  base_seed=$seed"
    echo "================================================================"

    python scripts/rl_tick_rolling.py \
        --seed "$seed" \
        --instruments "$INSTRUMENTS" \
        --data_root "$DATA_ROOT" \
        --steps_per_window "$STEPS_PER_WINDOW" \
        --global_start "$GLOBAL_START" \
        --global_end "$GLOBAL_END" \
        --train_months "$TRAIN_MONTHS" \
        --valid_months "$VALID_MONTHS" \
        --test_months "$TEST_MONTHS" \
        --step_months "$STEP_MONTHS" \
        --n_parallel_pools "$N_PARALLEL_POOLS" \
        "${EXTRA_ARGS[@]}"
done

echo
echo "[run_full_rolling] All $ROUNDS round(s) finished at $(date)"

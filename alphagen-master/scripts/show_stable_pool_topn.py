#!/usr/bin/env python3
"""Print top-N stable factors with occurrence and sign consistency.

Usage examples:
  python scripts/show_stable_pool_topn.py --run-dir out/tick_rolling/tick_3s_pool20_seed0_20260403
  python scripts/show_stable_pool_topn.py --stable-json out/tick_rolling/.../stable_factor_pool.json --top-n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _resolve_stable_json(run_dir: str | None, stable_json: str | None) -> Path:
    if stable_json:
        path = Path(stable_json)
    elif run_dir:
        path = Path(run_dir) / "stable_factor_pool.json"
    else:
        raise ValueError("Please provide either --run-dir or --stable-json")

    if not path.exists():
        raise FileNotFoundError(f"stable pool file not found: {path}")
    return path


def _format_row(idx: int, item: Dict[str, Any]) -> str:
    expr = str(item.get("expr", "")).strip()
    count = int(item.get("count", 0))
    sc = float(item.get("sign_consistency", 0.0))
    cov = float(item.get("coverage", 0.0))
    med_abs = float(item.get("median_abs_weight", 0.0))
    mean_w = float(item.get("mean_weight", 0.0))
    return (
        f"{idx:>3d} | count={count:>2d} | sign_consistency={sc:>5.2f} | "
        f"coverage={cov:>5.2f} | median_abs_w={med_abs:>8.5f} | "
        f"mean_w={mean_w:>+9.5f} | {expr}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show top-N stable factors from stable_factor_pool.json")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to one tick rolling run directory")
    parser.add_argument("--stable-json", type=str, default=None, help="Direct path to stable_factor_pool.json")
    parser.add_argument("--top-n", type=int, default=12, help="How many factors to print")
    args = parser.parse_args()

    stable_path = _resolve_stable_json(args.run_dir, args.stable_json)
    with stable_path.open("r") as f:
        payload: Dict[str, Any] = json.load(f)

    selected: List[Dict[str, Any]] = payload.get("selected", [])
    top_n = max(1, int(args.top_n))
    top_items = selected[:top_n]

    print(f"stable_json: {stable_path}")
    print(
        "summary: "
        f"total_windows={payload.get('total_windows', 0)}, "
        f"n_candidates={payload.get('n_candidates', 0)}, "
        f"selected={len(selected)}, "
        f"min_occurrence={payload.get('min_occurrence', '-')}, "
        f"min_sign_consistency={payload.get('min_sign_consistency', '-')}, "
        f"max_factors={payload.get('max_factors', '-')}")

    if not top_items:
        print("No selected stable factors found.")
        return

    print("\nTop factors:")
    for i, item in enumerate(top_items, start=1):
        print(_format_row(i, item))


if __name__ == "__main__":
    main()

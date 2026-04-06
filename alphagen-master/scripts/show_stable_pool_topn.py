#!/usr/bin/env python3
"""Print top-N stable factors with occurrence and sign consistency.

If `stable_factor_pool.json` is missing, this script can build it from each
window's `final_pool.json` under the provided run directory.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DEFAULT_STABLE_NAME = "stable_factor_pool.json"


def _build_stable_from_run_dir(
    run_dir: Path,
    min_occurrence: int,
    min_sign_consistency: float,
    max_factors: int,
) -> Dict[str, Any]:
    window_dirs = sorted([p for p in run_dir.glob("window_*") if p.is_dir()])
    total_windows = len(window_dirs)

    stats = defaultdict(lambda: {"count": 0, "weights": [], "windows": []})
    for wdir in window_dirs:
        pool_path = wdir / "final_pool.json"
        if not pool_path.exists():
            continue
        try:
            with pool_path.open("r") as f:
                pool_data = json.load(f)
        except Exception:
            continue

        exprs = pool_data.get("exprs", [])
        weights = pool_data.get("weights", [])
        n = min(len(exprs), len(weights))
        for i in range(n):
            expr = str(exprs[i]).strip()
            if not expr:
                continue
            w = float(weights[i])
            s = stats[expr]
            s["count"] += 1
            s["weights"].append(w)
            s["windows"].append(wdir.name)

    candidates: List[Dict[str, Any]] = []
    for expr, s in stats.items():
        count = int(s["count"])
        if count == 0:
            continue
        ws = np.array(s["weights"], dtype=float)
        pos = int((ws > 0).sum())
        neg = int((ws < 0).sum())
        sign_consistency = max(pos, neg) / count
        med_abs_w = float(np.median(np.abs(ws)))
        if count >= min_occurrence and sign_consistency >= min_sign_consistency:
            candidates.append({
                "expr": expr,
                "count": count,
                "coverage": count / max(total_windows, 1),
                "sign_consistency": sign_consistency,
                "median_abs_weight": med_abs_w,
                "mean_weight": float(ws.mean()),
                "windows": s["windows"],
            })

    candidates.sort(
        key=lambda x: (x["count"], x["sign_consistency"], x["median_abs_weight"]),
        reverse=True,
    )
    selected = candidates[:max_factors]

    return {
        "total_windows": total_windows,
        "min_occurrence": min_occurrence,
        "min_sign_consistency": min_sign_consistency,
        "max_factors": max_factors,
        "n_candidates": len(candidates),
        "selected": selected,
    }


def _resolve_or_build_stable_json(
    run_dir: str | None,
    stable_json: str | None,
    min_occurrence: int,
    min_sign_consistency: float,
    max_factors: int,
    save_when_built: bool,
) -> tuple[Path, Dict[str, Any]]:
    if stable_json:
        path = Path(stable_json)
        if not path.exists():
            raise FileNotFoundError(f"stable pool file not found: {path}")
        with path.open("r") as f:
            return path, json.load(f)

    if not run_dir:
        raise ValueError("Please provide either --run-dir or --stable-json")

    run_path = Path(run_dir)
    stable_path = run_path / DEFAULT_STABLE_NAME
    if stable_path.exists():
        with stable_path.open("r") as f:
            return stable_path, json.load(f)

    # Fallback: build from existing window outputs.
    payload = _build_stable_from_run_dir(
        run_dir=run_path,
        min_occurrence=min_occurrence,
        min_sign_consistency=min_sign_consistency,
        max_factors=max_factors,
    )
    if save_when_built:
        with stable_path.open("w") as f:
            json.dump(payload, f, indent=2)
    return stable_path, payload


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
    parser.add_argument("--min-occurrence", type=int, default=3, help="Fallback build: min windows an expr appears in")
    parser.add_argument("--min-sign-consistency", type=float, default=0.6, help="Fallback build: min sign consistency")
    parser.add_argument("--max-factors", type=int, default=12, help="Fallback build: max stable factors to keep")
    parser.add_argument("--no-save", action="store_true", help="Do not save generated stable_factor_pool.json")
    args = parser.parse_args()

    stable_path, payload = _resolve_or_build_stable_json(
        run_dir=args.run_dir,
        stable_json=args.stable_json,
        min_occurrence=args.min_occurrence,
        min_sign_consistency=args.min_sign_consistency,
        max_factors=args.max_factors,
        save_when_built=not args.no_save,
    )

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

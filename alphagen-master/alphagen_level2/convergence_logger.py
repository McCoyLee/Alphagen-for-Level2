"""
RL training convergence logger & plotter.
 
Records training metrics at each rollout end and provides:
- CSV export for post-hoc analysis
- Matplotlib convergence curve plotting
- Integration as a callback mixin for SB3
 
Usage standalone:
    from alphagen_level2.convergence_logger import ConvergenceLogger, plot_convergence
    logger = ConvergenceLogger(save_dir="./out/results/my_run")
    # ... during training, call logger.record_step(...)
    logger.save_csv()
    plot_convergence("./out/results/my_run/convergence.csv")
 
Usage as callback (see ConvergenceCallback):
    callback = ConvergenceCallback(save_path=..., test_calculators=..., plot_interval=10)
"""
 
import os
import csv
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
 
import numpy as np
 
 
@dataclass
class StepRecord:
    """Single training step metrics snapshot."""
    timestep: int
    pool_size: int
    pool_significant: int
    pool_best_ic: float
    pool_eval_cnt: int
    train_ic: float = 0.0
    train_rank_ic: float = 0.0
    valid_ic: float = 0.0
    valid_rank_ic: float = 0.0
    test_ic: float = 0.0
    test_rank_ic: float = 0.0
    test_ic_mean: float = 0.0
    test_rank_ic_mean: float = 0.0
    # Per-test-set breakdown (variable length, stored as JSON string)
    test_details: str = ""
 
 
class ConvergenceLogger:
    """
    Accumulates training metrics and provides CSV/JSON export + plotting.
 
    Attributes:
        records: list of StepRecord objects
        save_dir: directory for output files
    """
 
    COLUMNS = [
        "timestep", "pool_size", "pool_significant", "pool_best_ic",
        "pool_eval_cnt", "train_ic", "train_rank_ic",
        "valid_ic", "valid_rank_ic", "test_ic", "test_rank_ic",
        "test_ic_mean", "test_rank_ic_mean", "test_details",
    ]
 
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.records: List[StepRecord] = []
        os.makedirs(save_dir, exist_ok=True)
 
    def record_step(
        self,
        timestep: int,
        pool_size: int,
        pool_significant: int,
        pool_best_ic: float,
        pool_eval_cnt: int,
        train_ic: float = 0.0,
        train_rank_ic: float = 0.0,
        valid_ic: Optional[float] = None,
        valid_rank_ic: Optional[float] = None,
        test_results: Optional[List[Tuple[float, float]]] = None,
    ) -> StepRecord:
        """
        Record metrics for a single rollout end.
 
        Args:
            timestep: current training timestep
            pool_size: number of alphas in pool
            pool_significant: number of significant alphas (|weight| > 1e-4)
            pool_best_ic: best ensemble IC on training set
            pool_eval_cnt: total number of expressions evaluated
            train_ic: ensemble IC on training set
            train_rank_ic: ensemble Rank IC on training set
            valid_ic: optional explicit validation IC
            valid_rank_ic: optional explicit validation Rank IC
            test_results: list of (ic, rank_ic) tuples per test calculator
                          Convention: [valid, test] or [test1, test2, ...]
 
        Returns:
            The recorded StepRecord
        """
        test_results = test_results or []
 
        # Weighted mean across test sets
        n_total = len(test_results)
        if n_total > 0:
            ic_mean = np.mean([t[0] for t in test_results])
            ric_mean = np.mean([t[1] for t in test_results])
        else:
            ic_mean, ric_mean = 0.0, 0.0
 
        # Valid/Test separation
        # 1) Prefer explicit validation inputs when provided.
        # 2) Backward compatible fallback: treat first test_results item as valid.
        if valid_ic is not None:
            valid_ic_val = float(valid_ic)
            valid_ric_val = float(valid_rank_ic) if valid_rank_ic is not None else 0.0
            test_ic = test_results[0][0] if len(test_results) > 0 else 0.0
            test_ric = test_results[0][1] if len(test_results) > 0 else 0.0
        else:
            valid_ic_val = test_results[0][0] if len(test_results) > 0 else 0.0
            valid_ric_val = test_results[0][1] if len(test_results) > 0 else 0.0
            test_ic = test_results[1][0] if len(test_results) > 1 else 0.0
            test_ric = test_results[1][1] if len(test_results) > 1 else 0.0
 
        rec = StepRecord(
            timestep=timestep,
            pool_size=pool_size,
            pool_significant=pool_significant,
            pool_best_ic=pool_best_ic,
            pool_eval_cnt=pool_eval_cnt,
            train_ic=train_ic,
            train_rank_ic=train_rank_ic,
            valid_ic=valid_ic_val,
            valid_rank_ic=valid_ric_val,
            test_ic=test_ic,
            test_rank_ic=test_ric,
            test_ic_mean=float(ic_mean),
            test_rank_ic_mean=float(ric_mean),
            test_details=json.dumps(test_results),
        )
        self.records.append(rec)
        return rec
 
    def save_csv(self, filename: str = "convergence.csv") -> str:
        """Save all records to CSV. Returns the file path."""
        path = os.path.join(self.save_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(asdict(rec))
        return path
 
    def save_json(self, filename: str = "convergence.json") -> str:
        """Save all records to JSON. Returns the file path."""
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        return path
 
    def get_series(self, key: str) -> Tuple[List[int], List[float]]:
        """Extract a metric time series. Returns (timesteps, values)."""
        steps = [r.timestep for r in self.records]
        values = [getattr(r, key) for r in self.records]
        return steps, values
 
    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of the training run."""
        if not self.records:
            return {}
        last = self.records[-1]
        best_ic_idx = int(np.argmax([r.pool_best_ic for r in self.records]))
        best_test_idx = int(np.argmax([r.test_ic_mean for r in self.records]))
        return {
            "total_steps": last.timestep,
            "total_records": len(self.records),
            "final_pool_size": last.pool_size,
            "final_pool_best_ic": last.pool_best_ic,
            "final_test_ic_mean": last.test_ic_mean,
            "best_pool_ic": self.records[best_ic_idx].pool_best_ic,
            "best_pool_ic_step": self.records[best_ic_idx].timestep,
            "best_test_ic_mean": self.records[best_test_idx].test_ic_mean,
            "best_test_ic_mean_step": self.records[best_test_idx].timestep,
        }
 
 
def plot_convergence(
    csv_path: str,
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 10),
) -> str:
    """
    Plot convergence curves from a CSV file.
 
    Generates a 2x2 figure:
      - Top-left: Pool Best IC (train) over steps
      - Top-right: Valid IC & Test IC over steps
      - Bottom-left: Pool Size & Significant Alphas over steps
      - Bottom-right: Eval Count over steps
 
    Args:
        csv_path: path to convergence.csv
        output_path: where to save the plot (default: same dir as csv, .png)
        show: whether to call plt.show()
        figsize: figure size
 
    Returns:
        Path to the saved plot image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    # Read CSV
    steps, pool_best_ic, valid_ic, test_ic = [], [], [], []
    pool_size, pool_sig, eval_cnt = [], [], []
    test_ic_mean, test_ric_mean = [], []
    valid_ric, test_ric = [], []
 
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["timestep"]))
            pool_best_ic.append(float(row["pool_best_ic"]))
            valid_ic.append(float(row["valid_ic"]))
            test_ic.append(float(row["test_ic"]))
            pool_size.append(int(row["pool_size"]))
            pool_sig.append(int(row["pool_significant"]))
            eval_cnt.append(int(row["pool_eval_cnt"]))
            test_ic_mean.append(float(row["test_ic_mean"]))
            test_ric_mean.append(float(row["test_rank_ic_mean"]))
            valid_ric.append(float(row["valid_rank_ic"]))
            test_ric.append(float(row["test_rank_ic"]))
 
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("RL Training Convergence", fontsize=14, fontweight="bold")
 
    # Top-left: Pool Best IC
    ax = axes[0, 0]
    ax.plot(steps, pool_best_ic, "b-", linewidth=1.5, label="Pool Best IC (Train)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("IC")
    ax.set_title("Training IC Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    # Top-right: Valid & Test IC
    ax = axes[0, 1]
    ax.plot(steps, valid_ic, "g-", linewidth=1.2, label="Valid IC")
    ax.plot(steps, test_ic, "r-", linewidth=1.2, label="Test IC")
    ax.plot(steps, test_ic_mean, "m--", linewidth=1.0, label="Test IC Mean")
    ax2 = ax.twinx()
    ax2.plot(steps, valid_ric, "g:", linewidth=0.8, alpha=0.5, label="Valid RankIC")
    ax2.plot(steps, test_ric, "r:", linewidth=0.8, alpha=0.5, label="Test RankIC")
    ax2.set_ylabel("Rank IC", color="gray")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("IC")
    ax.set_title("Validation & Test IC")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
 
    # Bottom-left: Pool Size
    ax = axes[1, 0]
    ax.plot(steps, pool_size, "b-", linewidth=1.2, label="Pool Size")
    ax.plot(steps, pool_sig, "orange", linewidth=1.2, label="Significant Alphas")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Count")
    ax.set_title("Alpha Pool Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    # Bottom-right: Eval Count
    ax = axes[1, 1]
    ax.plot(steps, eval_cnt, "purple", linewidth=1.2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Eval Count")
    ax.set_title("Expressions Evaluated")
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
 
    if output_path is None:
        output_path = csv_path.replace(".csv", ".png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path
 
 
def compare_runs(
    csv_paths: List[str],
    labels: Optional[List[str]] = None,
    output_path: str = "comparison.png",
    metric: str = "pool_best_ic",
    show: bool = False,
) -> str:
    """
    Compare convergence curves from multiple runs on the same plot.
 
    Args:
        csv_paths: list of convergence.csv paths
        labels: names for each run (defaults to filenames)
        output_path: where to save
        metric: which column to compare
        show: whether to plt.show()
 
    Returns:
        Path to the saved comparison plot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    if labels is None:
        labels = [os.path.basename(os.path.dirname(p)) for p in csv_paths]
 
    fig, ax = plt.subplots(figsize=(12, 6))
    for path, label in zip(csv_paths, labels):
        steps_list, values = [], []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps_list.append(int(row["timestep"]))
                values.append(float(row[metric]))
        ax.plot(steps_list, values, linewidth=1.2, label=label)
 
    ax.set_xlabel("Timestep")
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path
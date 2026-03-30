"""
Train an ActionPriorTransformer from collected alpha expression data.

This script reads pool checkpoint JSONs from previous RL runs, converts
successful alpha expressions into supervised (prefix → next_action) training
pairs, and trains a small Transformer to predict "good next actions".

The trained model can then be plugged into the RL loop via
GuidedLevel2EnvWrapper for prior-guided exploration.

Usage:
    # Train from a single run:
    python scripts/train_action_prior.py \
        --result_dirs='["./out/results/run1"]' \
        --output_path=./out/action_prior.pt

    # Train from multiple runs:
    python scripts/train_action_prior.py \
        --result_dirs='["./out/results/run1","./out/results/run2"]' \
        --output_path=./out/action_prior.pt

    # With Level 2 extended features:
    python scripts/train_action_prior.py \
        --result_dirs='["./out/results/run1"]' \
        --use_level2_features \
        --output_path=./out/action_prior.pt

    # Customize model:
    python scripts/train_action_prior.py \
        --result_dirs='["./out/results/run1"]' \
        --d_model=64 --n_heads=4 --n_layers=2 \
        --epochs=100 --lr=1e-3
"""

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import fire

from alphagen_level2.action_prior import (
    ActionVocab,
    ActionPriorTransformer,
    PriorTrainingSample,
    build_training_data,
    collect_alphas_from_runs,
)
from alphagen_level2.config import MAX_EXPR_LENGTH


# ============================================================================
# Dataset
# ============================================================================

class PriorDataset(Dataset):
    """PyTorch dataset wrapping PriorTrainingSample list."""

    def __init__(self, samples: List[PriorTrainingSample]):
        self.prefixes = np.stack([s.prefix for s in samples])
        self.targets = np.array([s.target_action for s in samples], dtype=np.int64)
        self.weights = np.array([s.weight for s in samples], dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.prefixes[idx].copy()),
            torch.tensor(self.targets[idx]),
            torch.tensor(self.weights[idx]),
        )


# ============================================================================
# Training Loop
# ============================================================================

def train_prior(
    samples: List[PriorTrainingSample],
    n_actions: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ffn: int = 128,
    dropout: float = 0.1,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.15,
    patience: int = 15,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> Tuple[ActionPriorTransformer, dict]:
    """
    Train an ActionPriorTransformer on supervised data.

    Loss function:
        L = sum_i w_i * CrossEntropy(prior(prefix_i), target_i)

    where w_i is the IC-based sample weight (higher IC → higher weight).

    Args:
        samples: training data from build_training_data()
        n_actions: action space size
        d_model, n_heads, n_layers, d_ffn, dropout: model hyperparameters
        epochs: max training epochs
        batch_size: mini-batch size
        lr: learning rate
        weight_decay: L2 regularization
        val_split: fraction for validation
        patience: early stopping patience (epochs without val loss improvement)
        device: torch device
        verbose: print training progress

    Returns:
        (trained_model, training_history_dict)
    """
    # Build dataset
    dataset = PriorDataset(samples)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    if verbose:
        print(f"[ActionPrior] Training data: {n_train} train, {n_val} val")
        print(f"[ActionPrior] Model: d={d_model}, heads={n_heads}, layers={n_layers}")

    # Build model
    model = ActionPriorTransformer(
        n_actions=n_actions,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss, total_weight = 0.0, 0.0
        for prefixes, targets, weights in train_loader:
            prefixes = prefixes.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            logits = model(prefixes)  # (bs, n_actions)
            # Weighted cross-entropy
            loss_per_sample = F.cross_entropy(logits, targets, reduction="none")
            loss = (loss_per_sample * weights).sum() / weights.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * weights.sum().item()
            total_weight += weights.sum().item()

        train_loss = total_loss / max(total_weight, 1e-8)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss_sum, val_weight_sum = 0.0, 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for prefixes, targets, weights in val_loader:
                prefixes = prefixes.to(device)
                targets = targets.to(device)
                weights = weights.to(device)

                logits = model(prefixes)
                loss_per_sample = F.cross_entropy(logits, targets, reduction="none")
                val_loss_sum += (loss_per_sample * weights).sum().item()
                val_weight_sum += weights.sum().item()

                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += len(targets)

        val_loss = val_loss_sum / max(val_weight_sum, 1e-8)
        val_acc = correct / max(total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.3f}, "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"[ActionPrior] Best val_loss={best_val_loss:.4f}")

    return model, history


# ============================================================================
# CLI Entry Point
# ============================================================================

def main(
    result_dirs: Union[str, List[str]] = '["./out/results"]',
    output_path: str = "./out/action_prior.pt",
    use_level2_features: bool = False,
    # Model hyperparameters
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ffn: int = 128,
    dropout: float = 0.1,
    # Training hyperparameters
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.15,
    patience: int = 15,
    ic_temperature: float = 5.0,
    include_subexprs: bool = True,
):
    """
    Train ActionPriorTransformer from pool checkpoint data.

    :param result_dirs: JSON list of result directories to scan for pool JSONs
    :param output_path: where to save the trained model
    :param use_level2_features: use 20 Level 2 features (vs 6 basic)
    :param d_model: Transformer hidden dimension
    :param n_heads: number of attention heads
    :param n_layers: number of Transformer encoder layers
    :param d_ffn: feed-forward hidden dimension
    :param dropout: dropout rate
    :param epochs: max training epochs
    :param batch_size: mini-batch size
    :param lr: learning rate
    :param weight_decay: L2 regularization
    :param val_split: validation split fraction
    :param patience: early stopping patience
    :param ic_temperature: softmax temperature for IC → sample weight
    :param include_subexprs: extract sub-expressions for data augmentation
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse result_dirs
    if isinstance(result_dirs, str):
        result_dirs = json.loads(result_dirs)

    print(f"[ActionPrior] Scanning {len(result_dirs)} result directories...")
    vocab = ActionVocab(use_level2_features=use_level2_features)

    # Collect alphas
    exprs, weights = collect_alphas_from_runs(result_dirs)
    print(f"[ActionPrior] Collected {len(exprs)} unique alphas")

    if len(exprs) == 0:
        print("[ActionPrior] No alphas found! Exiting.")
        return

    # Build training data
    samples = build_training_data(
        exprs=exprs,
        ics=weights,  # Use weights as proxy for IC quality
        vocab=vocab,
        include_subexprs=include_subexprs,
        ic_temperature=ic_temperature,
    )
    print(f"[ActionPrior] Built {len(samples)} training samples")

    if len(samples) < 10:
        print("[ActionPrior] Too few samples! Need more alphas.")
        return

    # Train
    model, history = train_prior(
        samples=samples,
        n_actions=vocab.n_actions,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        val_split=val_split,
        patience=patience,
        device=device,
    )

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)
    print(f"[ActionPrior] Model saved to {output_path}")

    # Save training history
    hist_path = output_path.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[ActionPrior] History saved to {hist_path}")


if __name__ == "__main__":
    fire.Fire(main)

"""
Trainable Action Prior Network for RL-guided alpha generation.

Provides a Transformer-based network that learns to predict "good next actions"
from historical successful alpha expressions. Integrates with the RL loop via
reward shaping (no SB3 modification needed).

Architecture:
    obs (token seq) → Embedding + PosEnc → TransformerEncoder → Linear → action logits

Workflow:
    1. Collect successful alphas + their ICs from previous RL runs
    2. Convert alphas to action sequences via expr_to_actions()
    3. Train ActionPriorTransformer with train_prior()
    4. Plug into RL via GuidedLevel2EnvWrapper

See scripts/train_action_prior.py for the training pipeline.
"""

import math
import json
from typing import List, Optional, Tuple, Type, Dict, Any, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphagen.data.expression import (
    Expression, Feature, Constant, DeltaTime,
    Operator, UnaryOperator, BinaryOperator,
    RollingOperator, PairRollingOperator,
)
from alphagen_level2.config import (
    OPERATORS, DELTA_TIMES, CONSTANTS,
    LEVEL2_FEATURES, BASIC_FEATURES, MAX_EXPR_LENGTH,
)
from alphagen_level2.stock_data import Level2FeatureType


# ============================================================================
# Expression ↔ Action Sequence Conversion
# ============================================================================

class ActionVocab:
    """
    Maps between tokens and action indices, matching Level2EnvWrapper's layout:
        [operators | features | constants | delta_times | SEP]

    Must be constructed with the same feature set used by the env.
    """

    def __init__(self, use_level2_features: bool = True):
        self.features = LEVEL2_FEATURES if use_level2_features else BASIC_FEATURES
        self.operators = OPERATORS
        self.constants = CONSTANTS
        self.delta_times = DELTA_TIMES

        self.size_op = len(self.operators)
        self.size_feature = len(self.features)
        self.size_constant = len(self.constants)
        self.size_dt = len(self.delta_times)
        self.size_sep = 1

        self.n_actions = (
            self.size_op + self.size_feature + self.size_constant +
            self.size_dt + self.size_sep
        )

        # Build reverse lookup maps
        self._op_to_idx: Dict[Type[Operator], int] = {
            op: i for i, op in enumerate(self.operators)
        }
        self._feat_to_idx: Dict[Level2FeatureType, int] = {
            f: self.size_op + i for i, f in enumerate(self.features)
        }
        self._const_to_idx: Dict[float, int] = {
            c: self.size_op + self.size_feature + i
            for i, c in enumerate(self.constants)
        }
        self._dt_to_idx: Dict[int, int] = {
            dt: self.size_op + self.size_feature + self.size_constant + i
            for i, dt in enumerate(self.delta_times)
        }
        self.sep_idx = self.n_actions - 1

    def operator_idx(self, op: Type[Operator]) -> int:
        return self._op_to_idx[op]

    def feature_idx(self, feat: Level2FeatureType) -> int:
        return self._feat_to_idx[feat]

    def constant_idx(self, val: float) -> int:
        # Find nearest constant
        if val in self._const_to_idx:
            return self._const_to_idx[val]
        closest = min(self.constants, key=lambda c: abs(c - val))
        return self._const_to_idx[closest]

    def delta_time_idx(self, dt: int) -> int:
        return self._dt_to_idx[dt]


def expr_to_actions(expr: Expression, vocab: ActionVocab) -> List[int]:
    """
    Convert an expression tree to a postfix action sequence (matching RL env order).

    The RL environment builds expressions in postfix: operands are pushed first,
    then the operator pops them. This function recursively traverses the tree
    and produces the same postfix token order.

    Example:
        Mean($close, 20) → [feature_close_idx, dt_20_idx, op_mean_idx]
        Add($close, $open) → [feature_close_idx, feature_open_idx, op_add_idx]

    Args:
        expr: expression tree root
        vocab: ActionVocab for index mapping

    Returns:
        List of action indices (without trailing SEP)
    """
    actions: List[int] = []
    _expr_to_postfix(expr, vocab, actions)
    return actions


def _expr_to_postfix(expr: Expression, vocab: ActionVocab, out: List[int]):
    """Recursively convert expression to postfix action sequence."""
    if isinstance(expr, Feature):
        out.append(vocab.feature_idx(expr.feature_type))
    elif isinstance(expr, Constant):
        out.append(vocab.constant_idx(expr.value))
    elif isinstance(expr, DeltaTime):
        out.append(vocab.delta_time_idx(expr.delta_time))
    elif isinstance(expr, UnaryOperator):
        _expr_to_postfix(expr.operand, vocab, out)
        out.append(vocab.operator_idx(type(expr)))
    elif isinstance(expr, BinaryOperator):
        _expr_to_postfix(expr.lhs, vocab, out)
        _expr_to_postfix(expr.rhs, vocab, out)
        out.append(vocab.operator_idx(type(expr)))
    elif isinstance(expr, RollingOperator):
        _expr_to_postfix(expr.operand, vocab, out)
        _expr_to_postfix(expr.delta_time, vocab, out)
        out.append(vocab.operator_idx(type(expr)))
    elif isinstance(expr, PairRollingOperator):
        _expr_to_postfix(expr.lhs, vocab, out)
        _expr_to_postfix(expr.rhs, vocab, out)
        _expr_to_postfix(expr.delta_time, vocab, out)
        out.append(vocab.operator_idx(type(expr)))
    else:
        raise ValueError(f"Unknown expression type: {type(expr)}")


def actions_to_obs(actions: List[int], max_len: int = MAX_EXPR_LENGTH) -> np.ndarray:
    """Convert action list to observation array (zero-padded)."""
    obs = np.zeros(max_len, dtype=np.uint8)
    for i, a in enumerate(actions[:max_len]):
        obs[i] = a
    return obs


# ============================================================================
# Training Data Construction
# ============================================================================

@dataclass
class PriorTrainingSample:
    """A single (prefix, target_action, weight) training example."""
    prefix: np.ndarray       # shape (MAX_EXPR_LENGTH,), zero-padded
    prefix_len: int          # actual prefix length (0 = empty prefix)
    target_action: int       # the next action to predict
    weight: float            # IC-based sample weight


def build_training_data(
    exprs: List[Expression],
    ics: List[float],
    vocab: ActionVocab,
    include_subexprs: bool = True,
    ic_temperature: float = 5.0,
) -> List[PriorTrainingSample]:
    """
    Build supervised training data from collected alpha expressions.

    For each expression, generates (prefix → next_action) pairs at every step.
    Optionally extracts sub-expressions for data augmentation.

    Args:
        exprs: list of successful alpha expressions
        ics: corresponding IC values (used for sample weighting)
        vocab: action vocabulary
        include_subexprs: if True, also extract sub-expressions as additional data
        ic_temperature: temperature for IC → weight softmax (higher = more uniform)

    Returns:
        List of PriorTrainingSample
    """
    # Compute IC-based weights via softmax
    ic_arr = np.array(ics, dtype=np.float32)
    ic_arr = np.clip(ic_arr, 0, None)  # Only positive IC
    weights = np.exp(ic_arr * ic_temperature)
    weights = weights / weights.sum() * len(weights)  # Normalize to mean=1

    samples: List[PriorTrainingSample] = []

    for expr, w in zip(exprs, weights):
        try:
            actions = expr_to_actions(expr, vocab)
        except (KeyError, ValueError):
            continue

        if len(actions) == 0:
            continue

        # Generate (prefix → next_action) pairs
        # prefix=[] → predict actions[0]
        # prefix=[actions[0]] → predict actions[1]
        # ...
        # prefix=[actions[0:n-1]] → predict actions[n-1]
        # prefix=[actions[0:n]] → predict SEP
        full_seq = actions + [vocab.sep_idx]
        for t in range(len(full_seq)):
            prefix = actions[:t]
            target = full_seq[t]
            obs = actions_to_obs(prefix)
            samples.append(PriorTrainingSample(
                prefix=obs,
                prefix_len=t,
                target_action=target,
                weight=float(w),
            ))

        # Sub-expression augmentation: extract inner sub-trees
        if include_subexprs:
            for sub in _extract_subexprs(expr):
                try:
                    sub_actions = expr_to_actions(sub, vocab)
                except (KeyError, ValueError):
                    continue
                if len(sub_actions) < 2:
                    continue
                sub_full = sub_actions + [vocab.sep_idx]
                for t in range(len(sub_full)):
                    prefix = sub_actions[:t]
                    target = sub_full[t]
                    obs = actions_to_obs(prefix)
                    samples.append(PriorTrainingSample(
                        prefix=obs,
                        prefix_len=t,
                        target_action=target,
                        weight=float(w) * 0.5,  # Lower weight for sub-exprs
                    ))

    return samples


def _extract_subexprs(expr: Expression) -> List[Expression]:
    """Recursively extract all non-leaf sub-expressions from an expression tree."""
    subs: List[Expression] = []
    if isinstance(expr, (UnaryOperator,)):
        subs.append(expr.operand)
        subs.extend(_extract_subexprs(expr.operand))
    elif isinstance(expr, (BinaryOperator,)):
        subs.append(expr.lhs)
        subs.append(expr.rhs)
        subs.extend(_extract_subexprs(expr.lhs))
        subs.extend(_extract_subexprs(expr.rhs))
    elif isinstance(expr, (RollingOperator,)):
        subs.append(expr.operand)
        subs.extend(_extract_subexprs(expr.operand))
    elif isinstance(expr, (PairRollingOperator,)):
        subs.append(expr.lhs)
        subs.append(expr.rhs)
        subs.extend(_extract_subexprs(expr.lhs))
        subs.extend(_extract_subexprs(expr.rhs))
    return [s for s in subs if s.is_featured]


# ============================================================================
# Transformer Action Prior Network
# ============================================================================

class ActionPriorTransformer(nn.Module):
    """
    Transformer-based action prior network.

    Takes a partial token sequence (observation) and outputs logits over
    the action space, representing a "learned prior" of what good next
    actions look like based on historical successful alphas.

    Architecture:
        token_indices → Embedding(n_actions+1, d_model) → PositionalEncoding
        → TransformerEncoder(n_layers, n_heads) → mean pool → Linear → logits

    The architecture mirrors the policy network (LSTMSharedNet / TransformerSharedNet)
    but is trained separately via supervised learning.

    Args:
        n_actions: size of action space
        d_model: embedding / hidden dimension
        n_heads: number of attention heads
        n_layers: number of transformer encoder layers
        d_ffn: feed-forward hidden dimension
        dropout: dropout rate
        max_len: maximum sequence length
    """

    def __init__(
        self,
        n_actions: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ffn: int = 128,
        dropout: float = 0.1,
        max_len: int = MAX_EXPR_LENGTH,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.d_model = d_model

        # Token embedding: n_actions for real tokens, +1 for [BEG] token
        self.token_emb = nn.Embedding(n_actions + 1, d_model, padding_idx=0)
        self.beg_token_id = n_actions  # [BEG] uses the last embedding slot

        # Positional encoding
        pe = torch.zeros(max_len + 1, d_model)
        position = torch.arange(0, max_len + 1).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output head: d_model → n_actions
        self.output_head = nn.Linear(d_model, n_actions)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: (batch, seq_len) int tensor of action indices, 0-padded
            return_probs: if True, return softmax probs instead of raw logits

        Returns:
            (batch, n_actions) logits or probs
        """
        bs, seq_len = obs.shape

        # Prepend [BEG] token
        beg = torch.full(
            (bs, 1), self.beg_token_id, dtype=torch.long, device=obs.device
        )
        tokens = torch.cat([beg, obs.long()], dim=1)  # (bs, seq_len+1)

        # Padding mask: True where token == 0 (padding)
        pad_mask = tokens == 0

        # Embed + positional encoding
        x = self.token_emb(tokens)  # (bs, seq_len+1, d_model)
        x = x + self.pe[: tokens.size(1)]

        # Transformer
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (bs, seq_len+1, d_model)

        # Pool: take the last non-padding position
        # For simplicity, use mean of non-padding positions
        mask_float = (~pad_mask).float().unsqueeze(-1)  # (bs, seq_len+1, 1)
        x = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

        logits = self.output_head(x)  # (bs, n_actions)

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits

    def get_action_prior(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Get action prior distribution for a single observation.

        This is the main inference interface, called by GuidedLevel2EnvWrapper
        during RL training.

        Args:
            obs: (seq_len,) observation array
            action_mask: (n_actions,) bool mask of valid actions
            temperature: softmax temperature (lower = more peaked)

        Returns:
            (n_actions,) prior probability distribution
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(
                next(self.parameters()).device
            )
            logits = self.forward(obs_t).squeeze(0)  # (n_actions,)

            # Apply temperature
            logits = logits / max(temperature, 1e-8)

            # Apply action mask (set invalid to -inf)
            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask).bool().to(logits.device)
                logits[~mask_t] = float("-inf")

            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def save(self, path: str):
        """Save model weights and config."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "n_actions": self.n_actions,
                    "d_model": self.d_model,
                    "n_heads": self.transformer.layers[0].self_attn.num_heads,
                    "n_layers": len(self.transformer.layers),
                    "d_ffn": self.transformer.layers[0].linear1.out_features,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "ActionPriorTransformer":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        model = cls(
            n_actions=cfg["n_actions"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ffn=cfg["d_ffn"],
        )
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device)


# ============================================================================
# RL Integration: Guided Environment Wrapper
# ============================================================================

class GuidedLevel2EnvWrapper:
    """
    Wraps a Level2EnvWrapper to add action prior reward shaping.

    At each step, the prior network produces a distribution over actions.
    The agent receives a bonus reward proportional to the log-probability
    of the chosen action under the prior:

        shaped_reward = original_reward + beta * log(prior_prob(action))

    This "softly guides" the RL agent toward action patterns seen in
    successful historical alphas, without hard-constraining exploration.

    The wrapper is a drop-in replacement for Level2EnvWrapper in the
    training script. It implements the gym.Env interface expected by SB3.

    Args:
        env: the underlying Level2EnvWrapper (or Level2AlphaEnv)
        prior_net: trained ActionPriorTransformer (or None to disable)
        beta: reward shaping coefficient (0 = no guidance, higher = stronger)
        temperature: prior softmax temperature
        warmup_steps: disable prior guidance for the first N timesteps
                      (lets RL explore freely at the start)
        decay_rate: exponential decay for beta over timesteps
                    beta_effective = beta * exp(-decay_rate * step)
                    (0 = no decay, gradually reduces prior influence)
    """

    def __init__(
        self,
        env,
        prior_net: Optional[ActionPriorTransformer] = None,
        beta: float = 0.1,
        temperature: float = 1.0,
        warmup_steps: int = 0,
        decay_rate: float = 0.0,
    ):
        self.env = env
        self.prior_net = prior_net
        self.beta = beta
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate

        self._global_step = 0

        # Expose gym interface
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action: int):
        # Get prior bonus BEFORE stepping (need current obs)
        prior_bonus = 0.0
        if self.prior_net is not None and self._global_step >= self.warmup_steps:
            prior_bonus = self._compute_prior_bonus(action)

        obs, reward, done, truncated, info = self.env.step(action)
        self._global_step += 1

        # Add shaped reward
        shaped_reward = reward + prior_bonus
        return obs, shaped_reward, done, truncated, info

    def _compute_prior_bonus(self, action: int) -> float:
        """Compute log-prob reward bonus from the prior network."""
        # Current observation (before this action)
        obs = self.env.state if hasattr(self.env, "state") else np.zeros(MAX_EXPR_LENGTH)
        mask = self.action_masks()

        probs = self.prior_net.get_action_prior(
            obs, action_mask=mask, temperature=self.temperature
        )

        # Log-probability of the chosen action
        prob = max(probs[action], 1e-10)
        log_prob = np.log(prob)

        # Apply decay
        effective_beta = self.beta
        if self.decay_rate > 0:
            elapsed = self._global_step - self.warmup_steps
            effective_beta = self.beta * math.exp(-self.decay_rate * elapsed)

        return effective_beta * log_prob

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        """Proxy all other attributes to the wrapped env."""
        return getattr(self.env, name)


# ============================================================================
# Pool State Collector: extract alphas from pool checkpoints
# ============================================================================

def load_alphas_from_pool_json(
    json_path: str,
    parser=None,
) -> Tuple[List[Expression], List[float]]:
    """
    Load alpha expressions and weights from a pool checkpoint JSON.

    The pool JSON format is:
        {"exprs": ["Mean($close,20)", ...], "weights": [0.32, ...]}

    Args:
        json_path: path to *_pool.json file
        parser: ExpressionParser instance (if None, a default is created)

    Returns:
        (expressions, weights)
    """
    if parser is None:
        from alphagen.data.parser import ExpressionParser
        parser = ExpressionParser(
            OPERATORS,
            ignore_case=True,
            non_positive_time_deltas_allowed=False,
        )

    with open(json_path, "r") as f:
        data = json.load(f)

    exprs = []
    weights = []
    for expr_str, w in zip(data["exprs"], data["weights"]):
        try:
            expr = parser.parse(expr_str)
            exprs.append(expr)
            weights.append(abs(w))
        except Exception:
            continue

    return exprs, weights


def collect_alphas_from_runs(
    result_dirs: List[str],
    parser=None,
) -> Tuple[List[Expression], List[float]]:
    """
    Collect all unique alphas from multiple training run directories.

    Scans for *_pool.json files in each directory and deduplicates by
    expression string.

    Returns:
        (expressions, ic_or_weight_values)
    """
    import glob as glob_mod

    seen = set()
    all_exprs = []
    all_weights = []

    for d in result_dirs:
        for json_path in sorted(glob_mod.glob(f"{d}/*_pool.json")):
            exprs, weights = load_alphas_from_pool_json(json_path, parser)
            for expr, w in zip(exprs, weights):
                key = str(expr)
                if key not in seen:
                    seen.add(key)
                    all_exprs.append(expr)
                    all_weights.append(w)

    return all_exprs, all_weights

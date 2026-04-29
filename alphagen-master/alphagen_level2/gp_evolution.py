"""
Genetic-programming evolution over the Expression tree.

Operates directly on ``alphagen.data.expression.Expression`` ASTs:
  * **crossover** swaps a randomly chosen sub-tree between two parents
  * **mutation**  replaces a random sub-tree by a new random one (or by a
    feature / constant / delta-time node, depending on the slot)

Designed to be called periodically (every ``GP_EVERY_N_EPOCHS`` epochs) on a
``LinearAlphaPool``-style factor pool: parents are sampled by tournament from
the current pool, offspring are scored by the same evaluator used during RL
training, and the worst pool members may be replaced.

Notes
-----
The Expression types expose ``.operands`` (a tuple of children) but no direct
mutator.  We rebuild a node by calling its constructor with the new operands,
which all concrete operator classes support (Unary/Binary/Rolling/PairRolling).
``DeltaTime``, ``Constant`` and ``Feature`` are leaf types.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Type
import random
import copy

from alphagen.data.expression import (
    Expression, Operator, UnaryOperator, BinaryOperator,
    RollingOperator, PairRollingOperator,
    Feature, Constant, DeltaTime, OutOfDataRangeError,
)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def expression_size(expr: Expression) -> int:
    if isinstance(expr, (Feature, Constant, DeltaTime)):
        return 1
    if isinstance(expr, Operator):
        return 1 + sum(expression_size(o) for o in expr.operands)
    return 1


def collect_subtrees(
    expr: Expression,
    only_featured: bool = False,
) -> List[Tuple[Expression, "ReplaceFn"]]:
    """Walk the tree and return ``(node, replacer)`` pairs.

    ``replacer(new_node)`` returns a new full expression with ``node``
    swapped for ``new_node``.  This avoids mutating shared sub-trees in
    place (Expression objects are intentionally treated as immutable).
    """
    out: List[Tuple[Expression, ReplaceFn]] = []

    def walk(node: Expression, rebuild: "ReplaceFn") -> None:
        if (not only_featured) or _is_featured(node):
            out.append((node, rebuild))
        if isinstance(node, Operator):
            ops = list(node.operands)
            for i, child in enumerate(ops):
                cls = type(node)
                # Closure over current i
                def make_rebuild(parent_rebuild, parent_cls, ops_snapshot, idx):
                    def rb(new_child: Expression) -> Expression:
                        new_ops = list(ops_snapshot)
                        new_ops[idx] = new_child
                        return parent_rebuild(parent_cls(*new_ops))
                    return rb
                child_rebuild = make_rebuild(rebuild, cls, ops, i)
                walk(child, child_rebuild)

    walk(expr, lambda x: x)
    return out


ReplaceFn = Callable[[Expression], Expression]


def _is_featured(node: Expression) -> bool:
    try:
        return bool(node.is_featured)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Random AST generator (used for mutation "grow" sub-trees)
# ---------------------------------------------------------------------------

@dataclass
class GPGrammar:
    operators: Sequence[Type[Operator]]
    features: Sequence  # FeatureType enum members
    constants: Sequence[float]
    delta_times: Sequence[int]
    max_depth: int = 4

    def random_feature(self, rng: random.Random) -> Feature:
        return Feature(rng.choice(list(self.features)))

    def random_constant(self, rng: random.Random) -> Constant:
        return Constant(rng.choice(list(self.constants)))

    def random_delta_time(self, rng: random.Random) -> DeltaTime:
        return DeltaTime(rng.choice(list(self.delta_times)))

    def random_terminal(self, rng: random.Random,
                        prefer_featured: bool = True) -> Expression:
        if prefer_featured or rng.random() < 0.7:
            return self.random_feature(rng)
        return self.random_constant(rng)

    def random_expression(self, rng: random.Random,
                          depth: int = 0,
                          must_be_featured: bool = True) -> Expression:
        if depth >= self.max_depth or (depth > 0 and rng.random() < 0.35):
            return self.random_terminal(rng, prefer_featured=must_be_featured)
        op_cls = rng.choice(list(self.operators))
        try:
            return self._build_op(op_cls, rng, depth)
        except Exception:
            return self.random_terminal(rng, prefer_featured=must_be_featured)

    def _build_op(self, op_cls: Type[Operator], rng: random.Random,
                  depth: int) -> Expression:
        if issubclass(op_cls, UnaryOperator):
            return op_cls(self.random_expression(rng, depth + 1, True))
        if issubclass(op_cls, BinaryOperator):
            lhs = self.random_expression(rng, depth + 1, True)
            rhs = self.random_expression(
                rng, depth + 1, must_be_featured=(rng.random() < 0.5),
            )
            return op_cls(lhs, rhs)
        if issubclass(op_cls, RollingOperator):
            return op_cls(
                self.random_expression(rng, depth + 1, True),
                self.random_delta_time(rng),
            )
        if issubclass(op_cls, PairRollingOperator):
            return op_cls(
                self.random_expression(rng, depth + 1, True),
                self.random_expression(rng, depth + 1, True),
                self.random_delta_time(rng),
            )
        raise TypeError(f"Unknown operator category for {op_cls.__name__}")


# ---------------------------------------------------------------------------
# Crossover & mutation
# ---------------------------------------------------------------------------

def crossover(parent_a: Expression, parent_b: Expression,
              rng: random.Random) -> Optional[Expression]:
    """Subtree-swap crossover.  Returns child or ``None`` if no compatible
    swap was found within a few tries."""
    a_subs = collect_subtrees(parent_a, only_featured=True)
    b_subs = collect_subtrees(parent_b, only_featured=True)
    if not a_subs or not b_subs:
        return None
    for _ in range(8):
        node_a, rebuild_a = rng.choice(a_subs)
        node_b, _ = rng.choice(b_subs)
        try:
            child = rebuild_a(_clone(node_b))
            if _is_featured(child):
                return child
        except Exception:
            continue
    return None


def mutate(expr: Expression, grammar: GPGrammar,
           rng: random.Random) -> Optional[Expression]:
    """Replace a random sub-tree with a freshly generated one."""
    subs = collect_subtrees(expr, only_featured=False)
    if not subs:
        return None
    for _ in range(8):
        node, rebuild = rng.choice(subs)
        new_sub: Expression
        if isinstance(node, DeltaTime):
            new_sub = grammar.random_delta_time(rng)
        elif isinstance(node, Constant):
            new_sub = grammar.random_constant(rng) if rng.random() < 0.5 \
                else grammar.random_feature(rng)
        elif isinstance(node, Feature):
            new_sub = grammar.random_feature(rng)
        else:
            new_sub = grammar.random_expression(rng, depth=1, must_be_featured=True)
        try:
            cand = rebuild(new_sub)
            if _is_featured(cand):
                return cand
        except Exception:
            continue
    return None


def _clone(expr: Expression) -> Expression:
    # Expressions are not __deepcopy__-friendly because of FeatureType enum,
    # but copy.deepcopy works for the subset we use here.
    try:
        return copy.deepcopy(expr)
    except Exception:
        return expr


# ---------------------------------------------------------------------------
# Pool-level evolution
# ---------------------------------------------------------------------------

def tournament_select(
    pool_exprs: Sequence[Expression],
    pool_scores: Sequence[float],
    k: int,
    rng: random.Random,
) -> Expression:
    n = len(pool_exprs)
    idx = [rng.randrange(n) for _ in range(min(k, n))]
    best = max(idx, key=lambda i: pool_scores[i])
    return pool_exprs[best]


def evolve_pool(
    pool_exprs: List[Expression],
    pool_scores: List[float],
    grammar: GPGrammar,
    score_fn: Callable[[Expression], Optional[float]],
    n_offspring: int = 10,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.4,
    tournament_k: int = 3,
    replace_worst_n: int = 5,
    max_tries_per_offspring: int = 50,
    rng: Optional[random.Random] = None,
    dedup_strs: Optional[set] = None,
) -> Tuple[List[Expression], List[float], List[Tuple[Expression, float]]]:
    """Run one GP generation against the current pool.

    Returns updated ``(pool_exprs, pool_scores, accepted)`` where ``accepted``
    lists the new (expr, score) pairs that entered the pool.

    Pool members below the worst ``replace_worst_n`` may be evicted in favour
    of a stronger offspring.  No eviction happens when no offspring beats the
    current worst score.
    """
    rng = rng or random.Random()
    dedup = dedup_strs if dedup_strs is not None else {str(e) for e in pool_exprs}
    accepted: List[Tuple[Expression, float]] = []

    if not pool_exprs:
        return pool_exprs, pool_scores, accepted

    for _ in range(n_offspring):
        child: Optional[Expression] = None
        for _t in range(max_tries_per_offspring):
            roll = rng.random()
            total = crossover_rate + mutation_rate
            do_cx = roll < (crossover_rate / max(total, 1e-9))
            if do_cx and len(pool_exprs) >= 2:
                a = tournament_select(pool_exprs, pool_scores, tournament_k, rng)
                b = tournament_select(pool_exprs, pool_scores, tournament_k, rng)
                child = crossover(a, b, rng)
            else:
                a = tournament_select(pool_exprs, pool_scores, tournament_k, rng)
                child = mutate(a, grammar, rng)
            if child is None:
                continue
            cs = str(child)
            if cs in dedup:
                child = None
                continue
            try:
                score = score_fn(child)
            except (OutOfDataRangeError, RuntimeError, ValueError):
                child = None
                continue
            if score is None:
                child = None
                continue
            # Accept if better than current worst eligible slot.
            worst_idx = _argmin(pool_scores)
            if (
                len(pool_exprs) < len(pool_scores) + 1  # always true; reserved
                and score > pool_scores[worst_idx]
                and replace_worst_n > 0
            ):
                pool_exprs[worst_idx] = child
                pool_scores[worst_idx] = float(score)
                dedup.add(cs)
                accepted.append((child, float(score)))
                replace_worst_n -= 1
            break
    return pool_exprs, pool_scores, accepted


def _argmin(xs: Sequence[float]) -> int:
    best = 0
    for i in range(1, len(xs)):
        if xs[i] < xs[best]:
            best = i
    return best

"""Binary classification metrics with explicit threshold and tie rules."""

import math

import numpy as np


def _inputs(y_true, scores):
    labels = np.asarray(y_true)
    try:
        values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be numeric") from exc
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if labels.size == 0 or labels.size != values.size:
        raise ValueError("inputs must be nonempty and have equal lengths")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    if np.isnan(values).any():
        raise ValueError("scores must not contain NaN")
    return labels.astype(np.int64), values


def confusion_at(y_true, scores, threshold: float) -> dict:
    labels, values = _inputs(y_true, scores)
    flagged = values >= threshold
    positive = labels == 1
    return {
        "tp": int(np.count_nonzero(flagged & positive)),
        "fp": int(np.count_nonzero(flagged & ~positive)),
        "tn": int(np.count_nonzero(~flagged & ~positive)),
        "fn": int(np.count_nonzero(~flagged & positive)),
    }


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    counts = confusion_at(y_true, scores, threshold)
    flagged = counts["tp"] + counts["fp"]
    positives = counts["tp"] + counts["fn"]
    return (
        counts["tp"] / flagged if flagged else 0.0,
        counts["tp"] / positives if positives else 0.0,
    )


def precision_at_k(y_true, scores, k: int) -> float:
    labels, values = _inputs(y_true, scores)
    if k <= 0:
        raise ValueError("k must be positive")
    limit = min(k, labels.size)
    order = np.argsort(-values, kind="stable")
    return float(labels[order[:limit]].mean())


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    counts = confusion_at(y_true, scores, threshold)
    return float(cost_fp * counts["fp"] + cost_fn * counts["fn"])


def _threshold_counts(labels, values):
    """Evaluate every distinct score in one stable descending sweep."""
    order = np.argsort(-values, kind="stable")
    ranked_scores = values[order]
    ranked_labels = labels[order]
    ends = np.r_[np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), labels.size - 1]
    tp = np.cumsum(ranked_labels)[ends]
    fp = ends + 1 - tp
    fn = labels.sum() - tp
    return ranked_scores[ends], tp, fp, fn


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    labels, values = _inputs(y_true, scores)
    thresholds, tp, fp, fn = _threshold_counts(labels, values)
    costs = cost_fp * fp + cost_fn * fn
    # Treat +inf with the same >= comparison as confusion_at, including infinite scores.
    at_inf = values >= math.inf
    inf_fp = np.count_nonzero(at_inf & (labels == 0))
    inf_fn = np.count_nonzero(~at_inf & (labels == 1))
    inf_cost = cost_fp * inf_fp + cost_fn * inf_fn
    best = math.inf
    lowest_cost = inf_cost
    for threshold, cost in zip(thresholds, costs):
        if cost < lowest_cost:
            lowest_cost = cost
            best = float(threshold)
    return best


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    labels, values = _inputs(y_true, scores)
    thresholds, tp, fp, _ = _threshold_counts(labels, values)
    precision = tp / (tp + fp)
    valid = np.flatnonzero(precision >= min_precision)
    return float(thresholds[valid[-1]]) if valid.size else None

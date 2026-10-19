import math

import numpy as np
import pytest

from rawaj.metrics import (
    best_threshold, confusion_at, expected_cost, precision_at_k,
    precision_recall_at, threshold_for_precision,
)


def test_inclusive_threshold_and_zero_divisions():
    assert confusion_at([1, 0, 0, 1], [.5, .5, .2, .8], .5) == dict(tp=2, fp=1, tn=1, fn=0)
    assert precision_recall_at([0, 0], [.4, .5], 1) == (0.0, 0.0)
    assert precision_recall_at([1, 1], [.4, .5], .4) == (1.0, 1.0)


def test_top_k_stable_ties_and_bounds():
    assert precision_at_k([0, 1, 1], [.9, .9, .5], 1) == 0
    assert precision_at_k([0, 1, 1], [.9, .9, .5], 9) == 2 / 3
    with pytest.raises(ValueError):
        precision_at_k([1], [.1], 0)


def test_cost_and_tie_preference():
    assert expected_cost([1, 0], [.1, .9], .5, cost_fp=2, cost_fn=8) == 10
    assert best_threshold([1, 0], [.5, .5], cost_fp=1, cost_fn=1) == math.inf


def test_precision_is_not_monotonic():
    # Precision at .9/.8/.7/.6: 1, .5, 2/3, .5.
    assert threshold_for_precision([1, 0, 1, 0], [.9, .8, .7, .6], .6) == .7
    assert threshold_for_precision([0, 0], [.9, .8], .1) is None
    assert threshold_for_precision([0, 0], [.9, .8], 0) == .8


@pytest.mark.parametrize('labels,scores', [([], []), ([0], []), ([2], [.4]), ([1], [np.nan]), ([[1]], [[.3]])])
def test_all_functions_validate(labels, scores):
    calls = [
        lambda: confusion_at(labels, scores, .5),
        lambda: precision_recall_at(labels, scores, .5),
        lambda: precision_at_k(labels, scores, 1),
        lambda: expected_cost(labels, scores, .5, cost_fp=1, cost_fn=2),
        lambda: best_threshold(labels, scores, cost_fp=1, cost_fn=2),
        lambda: threshold_for_precision(labels, scores, .5),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_randomized_against_independent_brute_force():
    rng = np.random.default_rng(20261019)
    for _ in range(500):
        n = int(rng.integers(1, 100))
        y = rng.integers(0, 2, n).tolist()
        scores = (rng.integers(-5, 6, n) / 5).tolist()
        cost_fp, cost_fn = rng.integers(0, 20, 2).tolist()
        candidates = sorted(set(scores) | {math.inf}, reverse=True)
        def cost(t):
            fp = sum(label == 0 and score >= t for label, score in zip(y, scores))
            fn = sum(label == 1 and score < t for label, score in zip(y, scores))
            return (cost_fp * fp + cost_fn * fn) / n
        wanted = min(candidates, key=cost)
        assert best_threshold(y, scores, cost_fp=cost_fp, cost_fn=cost_fn) == wanted
        required = float(rng.choice([0, .25, .5, .8, 1]))
        valid = []
        for t in set(scores):
            selected = [label for label, score in zip(y, scores) if score >= t]
            if sum(selected) / len(selected) >= required:
                valid.append(t)
        assert threshold_for_precision(y, scores, required) == (min(valid) if valid else None)


def test_infinite_scores_follow_inclusive_comparison():
    assert confusion_at([1, 0], [math.inf, -math.inf], math.inf) == dict(tp=1, fp=0, tn=1, fn=0)
    assert best_threshold([1, 0], [math.inf, -math.inf], cost_fp=1, cost_fn=1) == math.inf

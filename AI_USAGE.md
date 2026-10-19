# AI usage

ChatGPT (Codex) assisted with interpreting the live DSC task, implementing `rawaj/audit.py` and all six functions in `rawaj/metrics.py`, drafting these notes, and writing tests. Canva assisted with the separate Sprint 0 deck. This repository is AI-assisted and has not yet received a human code review.

## Verification and a correction

The first AI-generated `expected_cost` divided the weighted false-positive/false-negative cost by the number of transfers. Slide 11 of the official W2 lesson returns total cost instead. A regression case with one false positive costing 2 and one false negative costing 8 failed: expected 10, got 5.0. Removing the division fixed the mismatch. The focused regression then passed as part of the complete suite. Source: https://docs.google.com/presentation/d/1lmbnjmqHIwd8PYKJMQOGurbanQTmOREA/edit

For this implementation, the tests check exact audit outputs, non-mutation, nullable dtypes, missingness leaks, inclusive thresholds, stable ties and invalid input. A seeded randomized test compares threshold selection against a separate brute-force calculation on 500 datasets. At publication, the local suite passes 18 tests. No claim is made that these are DSC's public or hidden tests.

## Work without AI

No function in this version was written by Mohamed without AI assistance. It would be inaccurate to name one. The exercise requests an independently written function; that personal part remains for Mohamed to complete and explain before the Week 8 defence.

## Reproducibility limits

The official starter URL, `uv.lock`, `tests/test_public.py`, and `data/sample_transfers.csv` were not available from the live task links or repository searches. The code follows the published function contracts, with original local tests. Dependency pins in `pyproject.toml` transcribe the task page; they are not presented as the original starter file. Local checks used Python 3.11.15, NumPy 2.4.6, pandas 3.0.5 and pytest 9.1.1, not the grader's complete pinned environment.

# Rawaj Pay: audit the extract, then choose the threshold

DSC AI & Deep Tech, Week 2. Implements the contracts shown on the DSC task page on 19 October 2026.

`rawaj.audit.audit` checks row counts, blank-aware missingness, duplicate IDs and rows, constant columns, labels, timestamps, and value/missingness leakage. It returns exactly the ten required report keys without changing its input.

`rawaj.metrics` provides `confusion_at`, `precision_recall_at`, `precision_at_k`, `expected_cost`, `best_threshold` and `threshold_for_precision`. A score equal to a threshold is flagged. Top-k ties preserve input order. Cost ties choose the highest threshold; a precision constraint chooses the lowest qualifying distinct score. Threshold selection sorts once and scans grouped scores.

## Run the checks

With the task's Python 3.12 environment and published dependency pins available:

```sh
uv sync
uv run pytest -q
```

With a compatible environment already installed:

```sh
python -m pytest -q
```

Local validation: 18 passed, including 500 seeded random datasets checked against independent brute-force threshold selection. Local versions: Python 3.11.15, NumPy 2.4.6, pandas 3.0.5, pytest 9.1.1. This does not establish compatibility with every grader pin or passing DSC hidden tests.

## Audit an approved extract

```python
import pandas as pd
from rawaj.audit import audit

df = pd.read_csv("data/sample_transfers.csv")
print(audit(df, target="is_mule", id_col="txn_id", time_col="sent_at"))
```

The sample file is not included: the official starter could not be located from either repository name on the live page. No replacement is labelled as official data. The starter lockfile and public tests also remain unavailable. The dependency pins here transcribe the page instead of claiming to reproduce its original `pyproject.toml`.

`expected_cost` returns total error cost, matching slide 11 of the W2 lesson. Infinite scores follow the literal `score >= threshold` rule, so a score of positive infinity is flagged at an infinite threshold. The task's description of infinity as "flag nothing" assumes finite scores.

See `FRAMING.md` for the review decision and metric rationale, and `AI_USAGE.md` for assistance, corrections and the remaining personal function-writing requirement.

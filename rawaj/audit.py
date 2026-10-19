"""Profile an in-memory extract without changing it or reading external data."""

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _missing(series):
    return series.isna() | series.map(lambda value: isinstance(value, str) and not value.strip())


def _auc(labels, scores):
    """Mann-Whitney ROC AUC, using average ranks for tied scores."""
    positives = labels.eq(1)
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = scores.rank(method="average")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _extreme(value):
    return value is not None and (value >= 0.98 or value <= 0.02)


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    for column in (target, id_col, time_col):
        if column not in df.columns:
            raise KeyError(column)
    n_rows = len(df)
    missing = {column: _missing(df[column]) for column in df.columns}
    clean = {column: df[column].mask(missing[column]) for column in df.columns}
    labelled = ~missing[target]
    labels = df.loc[labelled, target]
    ids = df.loc[~missing[id_col], id_col]
    repeated_ids = ids[ids.duplicated(keep=False)].unique().tolist()
    parsed = pd.to_datetime(clean[time_col], errors="coerce", format="mixed", utc=True)
    valid_times = parsed.dropna()
    suspects = []
    if len(labels) >= 20 and labels.eq(0).any() and labels.eq(1).any():
        for column in df.columns:
            if column in (target, id_col, time_col):
                continue
            values = clean[column].loc[labelled]
            # Unlike constant_columns, missing counts as a distinct value for this rule.
            if values.nunique(dropna=False) <= 1:
                continue
            missing_score = missing[column].loc[labelled].astype(int)
            suspect = _extreme(_auc(labels, missing_score))
            if is_numeric_dtype(df[column].dtype) or is_bool_dtype(df[column].dtype):
                present = ~missing[column].loc[labelled]
                if int(present.sum()) >= 20:
                    value_auc = _auc(labels.loc[present], df.loc[labelled, column].loc[present])
                    suspect = suspect or _extreme(value_auc)
            if suspect:
                suspects.append(column)
    return {
        "n_rows": n_rows,
        "missing": {column: float(mask.mean()) if n_rows else 0.0 for column, mask in missing.items()},
        "duplicate_ids": sorted(repeated_ids),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": sorted(column for column in df.columns if column != target and clean[column].nunique() <= 1),
        "unlabelled_rows": int(missing[target].sum()),
        "positive_rate": float(labels.eq(1).mean()) if len(labels) else 0.0,
        "bad_timestamps": int((~missing[time_col] & parsed.isna()).sum()),
        "time_range": [valid_times.min().isoformat(), valid_times.max().isoformat()] if len(valid_times) else [None, None],
        "leakage_suspects": sorted(suspects),
    }

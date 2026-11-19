"""Portfolio: the accounts the decisions are made on."""
import numpy as np
import pandas as pd


class Portfolio:
    """One row per account, with an optional stratified sample to develop on.

    Parameters
    ----------
    frame            : DataFrame, one row per account.
    key_column       : unique, non-null account identifier.
    random_sampling  : fraction of accounts kept, in (0, 1]. Develop on a sample, run the full portfolio at the end.
    stratify_columns : sample *within* each combination of these columns (every non-empty stratum keeps at least one
                       account). List every column that will appear in a constraint ``WHERE`` clause, otherwise a
                       segment constraint can end up with too few - or zero - accounts.

    ``full`` is the untouched input, ``data`` is the active (possibly sampled) portfolio, ``scale`` is the factor
    that turns a sample total into an estimate of the full-portfolio total.
    """

    def __init__(self, frame, key_column, random_sampling=1.0, stratify_columns=None, random_state=42):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Portfolio needs a pandas DataFrame")
        if key_column not in frame.columns:
            raise ValueError(f"key column '{key_column}' is not in the DataFrame")
        if frame[key_column].isna().any():
            raise ValueError(f"key column '{key_column}' contains missing values")
        if not frame[key_column].is_unique:
            raise ValueError(f"key column '{key_column}' is not unique (one row per account is required)")
        if not (0 < random_sampling <= 1):
            raise ValueError("random_sampling must be in (0, 1]")
        stratify_columns = list(stratify_columns or [])
        missing = [c for c in stratify_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"stratify columns not in the DataFrame: {missing}")

        self.key_column = key_column
        self.stratify_columns = stratify_columns
        self.random_sampling = random_sampling
        self.random_state = random_state
        self.full = frame.reset_index(drop=True)
        self.data = self._sample()

    def _sample(self):
        n = len(self.full)
        if self.random_sampling >= 1.0:
            return self.full
        rng = np.random.default_rng(self.random_state)
        if self.stratify_columns:
            groups = self.full.groupby(self.stratify_columns, dropna=False, sort=False, observed=True).indices.values()
        else:
            groups = [np.arange(n)]
        picked = []
        for idx in groups:
            k = max(1, int(round(self.random_sampling * len(idx))))
            picked.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        keep = np.sort(np.concatenate(picked))
        return self.full.iloc[keep].reset_index(drop=True)

    @property
    def columns(self):
        return list(self.full.columns)

    @property
    def n_full(self):
        return len(self.full)

    @property
    def n_active(self):
        return len(self.data)

    @property
    def sampling_fraction(self):
        return self.n_active / self.n_full

    @property
    def scale(self):
        """Multiply a sample total by this to estimate the full-portfolio total."""
        return self.n_full / self.n_active

    def __len__(self):
        return self.n_active

    def __repr__(self):
        return (f"Portfolio(key='{self.key_column}', accounts={self.n_active:,} of {self.n_full:,} "
                f"({self.sampling_fraction:.1%}), columns={len(self.columns)})")

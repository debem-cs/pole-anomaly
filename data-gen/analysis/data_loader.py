"""
Canonical loader for the 2015 gamma dose recordings.

All downstream analysis pulls from this loader so that the assumptions
about cleaning (NaN handling, boundary trimming, anomaly masking) are
written in one place. The original file
`data-gen/data/2015_months_DebitDoseA.txt` stores four months in four
columns of varying length; this loader exposes them individually and
concatenated, with NaN-interpolation applied, and with simple
metadata.

Anomaly masking
---------------
Several analyses below need a *clean* estimate of the background — they
must not be biased by the obvious peaks visible in the raw signal. We
offer an optional `mask_anomalies=True` flag that runs a robust
outlier detection (MAD-based, default k=5) and returns either:
  - the values with the outliers interpolated linearly (drop=False), or
  - the boolean mask alone (return_mask=True).

This module purposefully has *no plotting* and does not depend on
matplotlib — it should be importable from any analysis script with
zero side effects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA_GEN = HERE.parent
ROOT = DATA_GEN.parent
DEFAULT_DATA_PATH = DATA_GEN / "data" / "2015_months_DebitDoseA.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MonthRecord:
    """One month of cleaned data plus its metadata."""

    name: str                       # column header, e.g. "24/02/2015 11:20"
    short_name: str                 # e.g. "Feb 2015"
    values: np.ndarray              # 1-D float array (already interpolated)
    sample_rate_hz: float = 1.0 / 60.0  # 1 sample / minute
    n_corrupt: int = 0              # number of NaN cells fixed in this record

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def duration_minutes(self) -> int:
        return self.n

    @property
    def duration_days(self) -> float:
        return self.n / 60.0 / 24.0


@dataclass
class RealData:
    """Canonical container exposed to all downstream analyses."""

    months: List[MonthRecord]
    source_path: Path

    def __getitem__(self, key) -> MonthRecord:
        if isinstance(key, int):
            return self.months[key]
        for m in self.months:
            if m.name == key or m.short_name == key:
                return m
        raise KeyError(key)

    def __iter__(self):
        return iter(self.months)

    def __len__(self) -> int:
        return len(self.months)

    def concat(self) -> np.ndarray:
        """Concatenate all months end-to-end."""
        return np.concatenate([m.values for m in self.months])

    @property
    def short_names(self) -> List[str]:
        return [m.short_name for m in self.months]


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────
_MONTH_ABBR = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}


def _short_name(column_header: str) -> str:
    """Turn '24/02/2015 11:20' into 'Feb 2015'."""
    try:
        day, month, rest = column_header.split("/")
        year = rest.split(" ")[0]
        return f"{_MONTH_ABBR.get(month, month)} {year}"
    except Exception:
        return column_header


def load_real_data(
    path: Optional[Path] = None,
    interpolate: bool = True,
    boundary_trim: int = 0,
) -> RealData:
    """
    Load the four months of 2015 gamma dose data.

    Parameters
    ----------
    path : path to the .txt file (default uses the project's location)
    interpolate : if True (default), linear-interpolate any corrupt/NaN
        values introduced by `pd.to_numeric(..., errors="coerce")`.
    boundary_trim : number of samples to drop from both ends of each
        record (useful when the first and last hour are unreliable).
    """
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Real data file not found: {path}")

    df = pd.read_csv(path)
    months: List[MonthRecord] = []
    for col in df.columns:
        raw_series = pd.to_numeric(df[col], errors="coerce")
        n_corrupt = int(raw_series.isna().sum())
        if interpolate:
            cleaned = raw_series.interpolate(method="linear", limit_direction="both")
        else:
            cleaned = raw_series.dropna()
        values = cleaned.values.astype(np.float64)
        if boundary_trim > 0:
            values = values[boundary_trim:-boundary_trim]
        months.append(
            MonthRecord(
                name=col,
                short_name=_short_name(col),
                values=values,
                n_corrupt=n_corrupt,
            )
        )
    return RealData(months=months, source_path=path)


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly masking helper — robust, statistics-only (no learning)
# ─────────────────────────────────────────────────────────────────────────────
def mad_outlier_mask(
    values: np.ndarray,
    baseline_window: int = 480,
    k: float = 5.0,
    bridge: int = 5,
) -> np.ndarray:
    """
    Boolean mask of *anomalous* samples, using the MAD around a slow
    moving-average baseline. This is purely statistical — no domain
    labels are used. Useful to remove visually obvious outliers
    before estimating background parameters.

    Parameters
    ----------
    values : 1-D signal.
    baseline_window : size of the centred moving-average window used
        to estimate the baseline.
    k : MAD multiplier; samples with |residual| > k * 1.4826 * MAD are
        flagged.
    bridge : single-sample gaps within k*MAD are bridged so that an
        event is not split into many tiny events.

    Returns
    -------
    mask : boolean array of the same length as `values`; True where
        the sample is considered anomalous.
    """
    n = len(values)
    kernel = np.ones(baseline_window) / baseline_window
    baseline_valid = np.convolve(values, kernel, mode="valid")
    pad_left = baseline_window // 2
    pad_right = n - len(baseline_valid) - pad_left
    baseline = np.concatenate([
        np.full(pad_left, baseline_valid[0]),
        baseline_valid,
        np.full(pad_right, baseline_valid[-1]),
    ])
    residual = values - baseline

    mad = np.median(np.abs(residual - np.median(residual)))
    scale = 1.4826 * mad + 1e-9
    raw_mask = np.abs(residual) > k * scale

    if bridge > 0 and raw_mask.any():
        idxs = np.flatnonzero(raw_mask)
        for i in range(1, len(idxs)):
            gap = idxs[i] - idxs[i - 1]
            if 1 < gap <= bridge:
                raw_mask[idxs[i - 1]:idxs[i] + 1] = True
    return raw_mask


def clean_background(
    values: np.ndarray,
    baseline_window: int = 480,
    k: float = 5.0,
) -> np.ndarray:
    """
    Return a copy of `values` with the MAD-flagged anomalies replaced
    by a linear interpolation between the surrounding clean samples.
    Used by Phase 2 so that the noise model is not contaminated by
    obvious peaks.
    """
    mask = mad_outlier_mask(values, baseline_window=baseline_window, k=k)
    if not mask.any():
        return values.copy()
    s = pd.Series(values.astype(float))
    s[mask] = np.nan
    s = s.interpolate(method="linear", limit_direction="both")
    return s.values


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_real_data()
    print(f"Loaded {len(data)} months from {data.source_path}")
    for m in data:
        clean = clean_background(m.values)
        n_anom = int(mad_outlier_mask(m.values).sum())
        print(
            f"  {m.short_name:<10}  n={m.n:>6}  "
            f"corrupt={m.n_corrupt:>3}  anom_samples={n_anom:>4}  "
            f"mean={m.values.mean():.2f}  std={m.values.std():.2f}  "
            f"clean_mean={clean.mean():.2f}  clean_std={clean.std():.2f}"
        )

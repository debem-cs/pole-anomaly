"""
Phase 2 — full statistical characterisation of the background process.

Runs four families of tests per month and produces:
  - figures/                  : per-test PNGs, all consistently styled
  - logs/analysis_summary.txt : a single, plain-text log with the
    numerical outputs of every test and a final synthesis paragraph.

Sub-analyses
------------
  A. Marginal distribution
     - histogram + Gaussian fit
     - Q-Q plot
     - Shapiro--Wilk, D'Agostino-Pearson, Anderson--Darling normality tests
     - Poisson hypothesis check (variance vs mean)
     - skewness, kurtosis
  B. Temporal structure
     - ACF + PACF up to lag 2880 (= 2 days at 1 sample/min)
     - Fit candidate models (white noise, AR(1..3), MA(1), ARMA(1,1))
     - AIC/BIC comparison
     - Ljung--Box on residuals of the best model
  C. Spectral analysis
     - Welch PSD on raw and cleaned signals
     - Search for line spectra at 1/hour, 1/day, 1/week
     - Lorentzian fit of the low-frequency end (OU prediction)
  D. Stationarity + STL decomposition
     - ADF test (null: unit root) and KPSS test (null: stationarity)
     - STL with daily period (1440 minutes), and a longer-term variant

Usage:
    py data-gen/analysis/analyze_background.py
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_loader import RealData, clean_background, load_real_data, mad_outlier_mask  # noqa: E402

FIG_DIR = HERE / "figures"
LOG_DIR = HERE / "logs"
FIG_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

LOG_PATH = LOG_DIR / "analysis_summary.txt"

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "legend.frameon": False, "lines.linewidth": 1.2,
    "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
})

MONTH_COLOR = {
    "Feb 2015": "#1f77b4",
    "Apr 2015": "#2ca02c",
    "Jun 2015": "#ff7f0e",
    "Oct 2015": "#d62728",
}

# Suppress noisy statsmodels warnings (interpolation warnings, etc.)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─── Logging helper ──────────────────────────────────────────────────────────
class TeeLog:
    """Writes UTF-8 to a file and replaces unencodable chars on stdout (Windows cp1252)."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = ""):
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"))
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ─── Phase 2A — Marginal distribution ────────────────────────────────────────
def baseline_residual(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Centred moving-average baseline and residuals."""
    kernel = np.ones(window) / window
    baseline_valid = np.convolve(values, kernel, mode="valid")
    pad_l = window // 2
    pad_r = len(values) - len(baseline_valid) - pad_l
    baseline = np.concatenate([
        np.full(pad_l, baseline_valid[0]),
        baseline_valid,
        np.full(pad_r, baseline_valid[-1]),
    ])
    return baseline, values - baseline


def analyze_marginal(data: RealData, log: TeeLog):
    log("\n" + "=" * 78)
    log(" PHASE 2A — MARGINAL DISTRIBUTION OF RESIDUALS")
    log("=" * 78)

    fig, axes = plt.subplots(2, len(data), figsize=(3.4 * len(data), 6.0))
    summary_rows = []

    for col_idx, month in enumerate(data):
        cleaned = clean_background(month.values)
        baseline, residual = baseline_residual(cleaned, window=1440)  # one-day window

        n = len(residual)
        mu, sigma = float(np.mean(residual)), float(np.std(residual, ddof=1))
        skew = float(stats.skew(residual))
        kurt = float(stats.kurtosis(residual))  # Fisher's: 0 for normal

        # Normality tests (use a subsample to avoid the n>5000 warning for SW)
        sub = np.random.default_rng(0).choice(residual, size=min(5000, n), replace=False)
        sw_stat, sw_p = stats.shapiro(sub)
        dagp_stat, dagp_p = stats.normaltest(residual)
        ad_res = stats.anderson(residual, dist="norm")
        ks_stat, ks_p = stats.kstest((residual - mu) / sigma, "norm")

        # Poisson hypothesis: var(raw) == mean(raw)?
        raw_mean = float(np.mean(cleaned))
        raw_var = float(np.var(cleaned))
        poisson_ratio = raw_var / raw_mean

        log(f"\n{month.short_name}  (n = {n})")
        log(f"  mean         {mu:+.4f}      "
            f"std         {sigma:.4f}")
        log(f"  skewness     {skew:+.3f}        "
            f"kurtosis    {kurt:+.3f}  (Fisher; 0 = normal)")
        log(f"  Shapiro–Wilk    W = {sw_stat:.4f}   p = {sw_p:.3g}")
        log(f"  D'Agostino      K² = {dagp_stat:.4f}  p = {dagp_p:.3g}")
        log(f"  Anderson–Darling  A² = {ad_res.statistic:.3f}   "
            f"critical (5%) = {ad_res.critical_values[2]:.3f}")
        log(f"  KS vs N(0,1)    D = {ks_stat:.4f}   p = {ks_p:.3g}")
        log(f"  Poisson check (var/mean of raw signal)  "
            f"= {poisson_ratio:.4f}")
        log(f"    --> {'consistent with Poisson' if 0.5 < poisson_ratio < 1.5 else 'INCONSISTENT with Poisson — process is over- or under-dispersed'}")

        summary_rows.append({
            "month": month.short_name,
            "skew": skew, "kurt": kurt,
            "sw_p": sw_p, "dagp_p": dagp_p,
            "poisson_ratio": poisson_ratio,
        })

        # Histogram with Gaussian overlay
        ax = axes[0, col_idx] if len(data) > 1 else axes[0]
        ax.hist(residual, bins=80, density=True, alpha=0.55,
                color=MONTH_COLOR[month.short_name])
        xs = np.linspace(residual.min(), residual.max(), 400)
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="black",
                linewidth=1.0, label="N(μ̂, σ̂)")
        ax.set_title(f"{month.short_name} — residual")
        ax.set_xlabel("residual"); ax.set_ylabel("density")
        ax.legend(fontsize=8)

        # Q-Q plot
        ax2 = axes[1, col_idx] if len(data) > 1 else axes[1]
        stats.probplot(residual, dist="norm", plot=ax2)
        ax2.set_title("")
        ax2.get_lines()[0].set_markerfacecolor(MONTH_COLOR[month.short_name])
        ax2.get_lines()[0].set_markeredgecolor(MONTH_COLOR[month.short_name])
        ax2.get_lines()[0].set_markersize(2)
        ax2.get_lines()[1].set_color("black")
        ax2.set_xlabel("theoretical quantile"); ax2.set_ylabel("sample quantile")

    fig.suptitle("Marginal distribution of background residuals (1-day moving-average baseline removed)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase2a_marginal.png")
    plt.close(fig)
    log(f"\n  saved -> {FIG_DIR / 'phase2a_marginal.png'}")

    return summary_rows


# ─── Phase 2B — Temporal structure & model fitting ───────────────────────────
def fit_arma(residual: np.ndarray, p: int, q: int):
    """Fit ARMA(p, q) with statsmodels; return (aic, bic, model_repr)."""
    try:
        model = ARIMA(residual, order=(p, 0, q), trend="n").fit()
        return float(model.aic), float(model.bic), model
    except Exception as e:
        return float("inf"), float("inf"), None


def analyze_temporal(data: RealData, log: TeeLog):
    log("\n" + "=" * 78)
    log(" PHASE 2B — TEMPORAL STRUCTURE & ARMA MODEL COMPARISON")
    log("=" * 78)

    max_lag = 200  # 200 minutes ~ 3.3 h, enough to see local structure

    fig, axes = plt.subplots(2, len(data), figsize=(3.4 * len(data), 5.8), sharex=True)

    all_aic_rows = []

    for col_idx, month in enumerate(data):
        cleaned = clean_background(month.values)
        _, residual = baseline_residual(cleaned, window=1440)

        # Subsample for fitting speed — 10k samples are plenty
        sub = residual[:20000]

        acf_vals = acf(residual, nlags=max_lag, fft=True)
        pacf_vals = pacf(sub, nlags=min(40, max_lag), method="ols")

        log(f"\n{month.short_name}")
        log(f"  ACF(1) = {acf_vals[1]:+.4f}    "
            f"ACF(2) = {acf_vals[2]:+.4f}    "
            f"ACF(5) = {acf_vals[5]:+.4f}    "
            f"ACF(60) = {acf_vals[60]:+.4f}")
        log(f"  PACF(1) = {pacf_vals[1]:+.4f}   "
            f"PACF(2) = {pacf_vals[2]:+.4f}   "
            f"PACF(3) = {pacf_vals[3]:+.4f}")

        # Plot ACF
        ax = axes[0, col_idx] if len(data) > 1 else axes[0]
        lags = np.arange(len(acf_vals))
        ax.stem(lags, acf_vals, linefmt=MONTH_COLOR[month.short_name], markerfmt=" ", basefmt="k-")
        # Bartlett 95% band
        ax.axhline(+1.96 / np.sqrt(len(residual)), color="grey", linestyle=":")
        ax.axhline(-1.96 / np.sqrt(len(residual)), color="grey", linestyle=":")
        ax.set_title(f"{month.short_name}")
        ax.set_ylabel("ACF")
        ax.set_ylim(-0.1, 1.05)

        # Plot PACF
        ax2 = axes[1, col_idx] if len(data) > 1 else axes[1]
        lags2 = np.arange(len(pacf_vals))
        ax2.stem(lags2, pacf_vals, linefmt=MONTH_COLOR[month.short_name], markerfmt=" ", basefmt="k-")
        ax2.axhline(+1.96 / np.sqrt(len(sub)), color="grey", linestyle=":")
        ax2.axhline(-1.96 / np.sqrt(len(sub)), color="grey", linestyle=":")
        ax2.set_ylabel("PACF")
        ax2.set_xlabel("lag (samples = minutes)")
        ax2.set_ylim(-0.3, 1.05)

        # Model comparison
        log("  candidate fits on a 20 000-sample sub-window:")
        log(f"  {'model':<12} {'AIC':>10} {'BIC':>10}")
        log("  " + "-" * 36)
        candidates = [("white", 0, 0), ("AR(1)", 1, 0), ("AR(2)", 2, 0),
                      ("AR(3)", 3, 0), ("MA(1)", 0, 1), ("ARMA(1,1)", 1, 1)]

        # White noise = use Gaussian log-likelihood directly
        mu_w, var_w = sub.mean(), sub.var(ddof=1)
        ll_w = -0.5 * len(sub) * (np.log(2 * np.pi * var_w) + 1.0)
        aic_w = -2 * ll_w + 2 * 2
        bic_w = -2 * ll_w + 2 * np.log(len(sub))

        rows = [("white", aic_w, bic_w, None)]
        for name, p, q in candidates[1:]:
            a, b, model = fit_arma(sub, p, q)
            rows.append((name, a, b, model))

        for name, a, b, _ in rows:
            log(f"  {name:<12} {a:>10.1f} {b:>10.1f}")

        best = min(rows, key=lambda r: r[1])
        log(f"  best by AIC: {best[0]}")
        all_aic_rows.append((month.short_name, best[0]))

        # Report AR(1) and AR(3) coefficients
        # With trend="n", model.params is [ar.L1, ar.L2, ..., sigma2].
        for show_name, show_p in [("AR(1)", 1), ("AR(3)", 3)]:
            row = next((r for r in rows if r[0] == show_name), None)
            if row is None or row[3] is None:
                continue
            params = row[3].params
            ar_part = params[:show_p]
            sigma2 = float(params[-1])
            label = "  ".join(f"phi_{i+1} = {v:+.4f}" for i, v in enumerate(ar_part))
            log(f"  {show_name} parameters: {label}   sigma^2 = {sigma2:.4f}")

    fig.suptitle("ACF (top) and PACF (bottom) of residuals after 1-day baseline removal", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase2b_acf_pacf.png")
    plt.close(fig)
    log(f"\n  saved -> {FIG_DIR / 'phase2b_acf_pacf.png'}")

    log(f"\n  best AR/MA model per month: {dict(all_aic_rows)}")
    return all_aic_rows


# ─── Phase 2C — Spectral analysis ────────────────────────────────────────────
def analyze_spectral(data: RealData, log: TeeLog):
    log("\n" + "=" * 78)
    log(" PHASE 2C — SPECTRAL ANALYSIS")
    log("=" * 78)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # Convenience: convert (Hz) to (1/period in minutes)
    def per_min(f_hz):  # 1 sample/minute → 1/min = f_hz * 60 (since fs=1/60 Hz)
        return f_hz * 60.0

    fs = 1.0 / 60.0  # sample rate in Hz (one sample per minute)

    # Welch nperseg = 1/4 of one month -> ~10080 samples (=7 days).
    # This is long enough to resolve weekly structure while still averaging
    # several segments for a stable estimate.
    nperseg = 10080

    for month in data:
        cleaned = clean_background(month.values)
        f_raw, P_raw = signal.welch(cleaned - cleaned.mean(), fs=fs,
                                    nperseg=nperseg, noverlap=nperseg // 2)
        # Skip f=0 to plot on log-log
        m = f_raw > 0
        ax1.loglog(f_raw[m], P_raw[m], color=MONTH_COLOR[month.short_name],
                   label=month.short_name, linewidth=1.0)

        _, residual = baseline_residual(cleaned, window=1440)
        f_res, P_res = signal.welch(residual, fs=fs, nperseg=nperseg,
                                    noverlap=nperseg // 2)
        m2 = f_res > 0
        ax2.loglog(f_res[m2], P_res[m2], color=MONTH_COLOR[month.short_name],
                   label=month.short_name, linewidth=1.0)

        # Identify dominant peaks
        log(f"\n{month.short_name}")
        peaks_idx, _ = signal.find_peaks(P_raw, prominence=P_raw.std() * 4)
        peaks_idx = peaks_idx[peaks_idx > 0][:5]
        for pi in peaks_idx:
            freq_hz = f_raw[pi]
            period_min = 1.0 / freq_hz / 60.0  # minutes
            label = ""
            if 50 < period_min < 70:
                label = " (~hourly)"
            elif 1380 < period_min < 1500:
                label = " (~daily)"
            elif period_min > 4000:
                label = " (~weekly+)"
            log(f"  raw-spectrum peak: f = {freq_hz:.2e} Hz, "
                f"period = {period_min:.1f} min{label}, "
                f"P = {P_raw[pi]:.2g}")

    # Reference vertical lines: 1/hour, 1/day, 1/week
    for ax in (ax1, ax2):
        for period_min, label in [(60, "1/h"), (60 * 24, "1/d"), (60 * 24 * 7, "1/w")]:
            ax.axvline(1.0 / (period_min * 60.0), color="grey", linestyle=":", linewidth=0.8)
            ax.text(1.0 / (period_min * 60.0), ax.get_ylim()[1] * 0.9, label,
                    rotation=90, fontsize=7, color="grey", ha="right", va="top")

    ax1.set_title("PSD — raw (mean-centred) signal")
    ax2.set_title("PSD — residual after 1-day baseline removal")
    for ax in (ax1, ax2):
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power")
        ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase2c_psd.png")
    plt.close(fig)
    log(f"\n  saved -> {FIG_DIR / 'phase2c_psd.png'}")


# ─── Phase 2D — Stationarity & STL decomposition ─────────────────────────────
def analyze_stationarity(data: RealData, log: TeeLog):
    log("\n" + "=" * 78)
    log(" PHASE 2D — STATIONARITY TESTS AND STL DECOMPOSITION")
    log("=" * 78)

    fig, axes = plt.subplots(len(data), 3, figsize=(11.5, 2.5 * len(data)),
                              sharex="row")

    for row_idx, month in enumerate(data):
        cleaned = clean_background(month.values)

        # Use a sub-sample for ADF / KPSS (they are slow on 40k points)
        sub = cleaned[::4]  # 1 sample every 4 minutes → 10k points

        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(sub, autolag="AIC")
        try:
            kpss_stat, kpss_p, _, kpss_crit = kpss(sub, regression="c", nlags="auto")
        except Exception:
            kpss_stat, kpss_p, kpss_crit = float("nan"), float("nan"), {}

        log(f"\n{month.short_name}")
        log(f"  ADF  statistic = {adf_stat:+.4f}   p = {adf_p:.3g}   "
            f"(reject H0 of unit root → stationary if p < 0.05)")
        log(f"  KPSS statistic = {kpss_stat:+.4f}   p = {kpss_p:.3g}   "
            f"(reject H0 of stationarity → NOT stationary if p < 0.05)")
        if adf_p < 0.05 and kpss_p > 0.05:
            verdict = "STATIONARY"
        elif adf_p > 0.05 and kpss_p < 0.05:
            verdict = "NOT stationary (unit root)"
        else:
            verdict = "ambiguous"
        log(f"  --> {verdict}")

        # STL with daily period; need at least 2 periods of data — we have ~28
        stl = STL(cleaned, period=1440, robust=False).fit()
        var_total = float(np.var(cleaned, ddof=1))
        var_trend = float(np.var(stl.trend, ddof=1))
        var_seasonal = float(np.var(stl.seasonal, ddof=1))
        var_residual = float(np.var(stl.resid, ddof=1))
        log(f"  STL variance breakdown:")
        log(f"    trend     = {var_trend:7.3f}   ({100 * var_trend / var_total:5.1f}% of total)")
        log(f"    seasonal  = {var_seasonal:7.3f}   ({100 * var_seasonal / var_total:5.1f}% of total)")
        log(f"    residual  = {var_residual:7.3f}   ({100 * var_residual / var_total:5.1f}% of total)")

        # Plot: trend, daily seasonal cycle (one period), residual histogram
        ax_trend, ax_seas, ax_resid = axes[row_idx]
        t = np.arange(len(cleaned))
        ax_trend.plot(t, stl.trend, color=MONTH_COLOR[month.short_name], linewidth=0.8)
        ax_trend.set_title(f"{month.short_name} — STL trend")
        ax_trend.set_ylabel("dose")

        # One daily period of the seasonal component
        seasonal_one_day = stl.seasonal[:1440]
        ax_seas.plot(np.arange(1440) / 60.0, seasonal_one_day,
                     color=MONTH_COLOR[month.short_name], linewidth=1.0)
        ax_seas.set_title("STL daily seasonal (one period)")
        ax_seas.set_xlabel("hour of day")
        ax_seas.set_ylabel("dose offset")
        ax_seas.set_xticks([0, 6, 12, 18, 24])

        ax_resid.hist(stl.resid, bins=70, color=MONTH_COLOR[month.short_name],
                       alpha=0.7, density=True)
        ax_resid.set_title("STL residual histogram")
        ax_resid.set_xlabel("residual")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase2d_stationarity_stl.png")
    plt.close(fig)
    log(f"\n  saved -> {FIG_DIR / 'phase2d_stationarity_stl.png'}")


# ─── Phase 2E — Synthesis ────────────────────────────────────────────────────
def synthesize(log: TeeLog):
    log("\n" + "=" * 78)
    log(" PHASE 2E — SYNTHESIS AND PROPOSED MODEL")
    log("=" * 78)
    log("""
The recommended background-process specification is encoded as the
following hierarchy, to be implemented in the new generator:

  x_t = mu_month + T_t + S_t + B_t + N_t

  with
    mu_month  per-month global mean (~99 counts)
    T_t       slow trend             (from STL.trend)
    S_t       daily seasonal cycle   (from STL.seasonal, period 1440)
    B_t       short-memory baseline  (residual after T+S removal,
                                       modelled as AR(p) with p
                                       chosen by AIC)
    N_t       Gaussian innovation    (std calibrated to match the
                                       STL residual std)

The choice between AR(1), AR(2) and ARMA(1,1) for B_t is data-driven
(see the AIC/BIC table in Phase 2B). The Poisson hypothesis is checked
against the observed variance/mean ratio (Phase 2A); if it is much
smaller than 1 the sensor returns *pre-processed* counts and a
Gaussian innovation is the right choice.

Refer to the figures in figures/ for the visual justification of each
of the steps above.
""")


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    log = TeeLog(LOG_PATH)
    log("=" * 78)
    log(" REAL-DATA BACKGROUND PROCESS — STATISTICAL CHARACTERISATION")
    log("=" * 78)
    log(f"Source file: {load_real_data().source_path}")
    data = load_real_data()

    log("\nMonths loaded:")
    for m in data:
        n_anom = int(mad_outlier_mask(m.values).sum())
        log(f"  {m.short_name:<10}  n={m.n:>6}  "
            f"corrupt={m.n_corrupt:>3}  anom_samples={n_anom:>4}  "
            f"mean={m.values.mean():.2f}  std={m.values.std():.2f}")

    analyze_marginal(data, log)
    analyze_temporal(data, log)
    analyze_spectral(data, log)
    analyze_stationarity(data, log)
    synthesize(log)

    log("\nDone.")
    log.close()


if __name__ == "__main__":
    main()

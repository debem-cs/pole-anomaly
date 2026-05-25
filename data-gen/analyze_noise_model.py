"""
Deeper characterisation of the background-noise model.

v2 fitted an AR(1) Gaussian noise model on the event-masked residual
and stopped there. This script pushes harder on the same data:

1. **AR-order selection.** Fit AR(1), AR(2), AR(3), AR(5), AR(10);
   compare AIC and BIC. If AR(1) is already best, an AR(1) generator
   is justified; if a higher order is needed, the generator has to
   match.

2. **Whitening quality.** Compute the innovations of each fitted AR
   model and test them for residual autocorrelation (Ljung-Box). A
   well-specified model leaves white innovations.

3. **Innovation distribution.** The marginal of the residual was
   already approximately Gaussian, but that hid the AR(1) structure.
   What does the marginal of the *innovations* look like? Is it
   Gaussian or heavy-tailed? Compare to a Student's t fit.

4. **Theoretical vs empirical PSD.** Plot the empirical periodogram
   against the AR(p) theoretical PSD. A good fit matches across all
   frequencies.

5. **Parameter stability.** Split each month in halves and quarters;
   refit AR(1) phi and sigma on each. Check that they don't drift.
   Also: compare phi and sigma across the four months.

6. **Innovation cross-correlation between months.** Sanity check:
   the four months should be independent realisations of the same
   process. The phi/sigma estimates should be close.

Outputs:
    data-gen/logs/noise_analysis/REPORT.md
    data-gen/logs/noise_analysis/<plots>.png
    data-gen/logs/noise_analysis/summary.json
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal as scipy_signal
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "2015_months_DebitDoseA.txt")
OUT_DIR = os.path.join(SCRIPT_DIR, "logs", "noise_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_WINDOW = 480
WIDE_BASELINE_WINDOW = 17280   # ~12 days, for the slow drift analysis
BOUNDARY_TRIM = 500
OUTLIER_Z_THRESHOLD = 4.0
OUTLIER_PAD = 60
AR_ORDERS_TO_TEST = [1, 2, 3, 5, 10]
LJUNG_BOX_LAGS = [10, 20, 50]


def centred_baseline(x, w):
    if w <= 1:
        return x.copy()
    kernel = np.ones(w) / w
    base = np.convolve(x, kernel, mode="valid")
    pad_left = w // 2
    pad_right = len(x) - len(base) - pad_left
    return np.concatenate([
        np.full(pad_left, base[0]),
        base,
        np.full(pad_right, base[-1]),
    ])


def find_runs(boolean_array):
    if len(boolean_array) == 0:
        return []
    diff = np.diff(boolean_array.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if boolean_array[0]:
        starts.insert(0, 0)
    if boolean_array[-1]:
        ends.append(len(boolean_array))
    return list(zip(starts, ends))


def robust_outlier_mask(series, baseline, w=BASELINE_WINDOW,
                       z_thresh=OUTLIER_Z_THRESHOLD, pad=OUTLIER_PAD):
    """Mask samples whose robust z-score exceeds threshold."""
    s = pd.Series(series - baseline)
    mad = s.rolling(window=w, center=True, min_periods=1).apply(
        lambda v: np.median(np.abs(v - np.median(v))), raw=True
    ).values
    sigma = 1.4826 * np.maximum(mad, 1e-6)
    z = (series - baseline) / sigma
    flagged = np.abs(z) > z_thresh
    mask = np.ones(len(series), dtype=bool)
    runs = find_runs(flagged)
    for s_, e_ in runs:
        s_ = max(0, s_ - pad)
        e_ = min(len(series), e_ + pad)
        mask[s_:e_] = False
    return mask


def longest_contiguous_run(mask):
    """Return (start, end) of the longest contiguous True run."""
    runs = find_runs(mask)
    if not runs:
        return None
    return max(runs, key=lambda r: r[1] - r[0])


def fit_ar_orders(clean_seg, orders):
    """Fit AR(p) for several p; return AIC/BIC/sigma/phi for each."""
    out = {}
    for p in orders:
        try:
            model = AutoReg(clean_seg, lags=p, old_names=False).fit()
            innovations = model.resid
            innov_std = float(np.std(innovations, ddof=1))
            out[p] = {
                "aic": float(model.aic),
                "bic": float(model.bic),
                "phi": [float(c) for c in model.params[1:p + 1]],
                "const": float(model.params[0]),
                "innovation_std": innov_std,
                "innovation_skew": float(stats.skew(innovations)),
                "innovation_kurtosis": float(stats.kurtosis(innovations,
                                                             fisher=True)),
                "innovations": innovations,
                "model": model,
            }
        except Exception as exc:
            out[p] = {"error": str(exc)}
    return out


def ljung_box_summary(innovations, lags=LJUNG_BOX_LAGS):
    """Return Ljung-Box Q-statistic and p-value at several lags."""
    try:
        lb = acorr_ljungbox(innovations, lags=lags, return_df=True)
        return {
            int(lag): {
                "Q": float(lb.loc[lag, "lb_stat"]),
                "p_value": float(lb.loc[lag, "lb_pvalue"]),
            }
            for lag in lags
        }
    except Exception as exc:
        return {"error": str(exc)}


def fit_t_distribution(samples):
    """ML fit of Student-t to innovations."""
    try:
        df, loc, scale = stats.t.fit(samples)
        return {
            "df": float(df),
            "loc": float(loc),
            "scale": float(scale),
        }
    except Exception as exc:
        return {"error": str(exc)}


def analyse_baseline_drift(series, w_wide=WIDE_BASELINE_WINDOW):
    """Linear-trend + ADF test on the ultra-wide baseline."""
    trend = centred_baseline(series, w_wide)
    t = np.arange(len(series))
    slope, intercept = np.polyfit(t, trend, 1)
    return {
        "linear_slope_per_sample": float(slope),
        "linear_intercept": float(intercept),
        "total_linear_drift": float(slope * len(series)),
        "wide_baseline_range": float(np.max(trend) - np.min(trend)),
        "wide_baseline_std": float(np.std(trend, ddof=1)),
        "adf_p_value": float(adfuller(trend, maxlag=20)[1]),
    }


def parameter_stability(series, baseline, mask, n_chunks=4):
    """Refit AR(1) on equal-size chunks; report phi and sigma stability."""
    n = len(series)
    chunk_size = n // n_chunks
    out = []
    for k in range(n_chunks):
        s = k * chunk_size
        e = (k + 1) * chunk_size if k < n_chunks - 1 else n
        sub_residual = (series - baseline)[s:e]
        sub_mask = mask[s:e]
        # Only fit on the contiguous longest run within the chunk
        sub_runs = find_runs(sub_mask)
        if not sub_runs:
            continue
        longest = max(sub_runs, key=lambda r: r[1] - r[0])
        if longest[1] - longest[0] < 200:
            continue
        seg = sub_residual[longest[0]:longest[1]]
        try:
            m = AutoReg(seg, lags=1, old_names=False).fit()
            out.append({
                "chunk": int(k),
                "start": int(s),
                "end": int(e),
                "n_clean_samples": int(longest[1] - longest[0]),
                "phi": float(m.params[1]),
                "innovation_std": float(np.std(m.resid, ddof=1)),
                "marginal_std": float(np.std(seg, ddof=1)),
            })
        except Exception:
            continue
    return out


def plot_innovation_diagnostics(name, ar_p, fit, out_path):
    innov = fit["innovations"]
    sigma = fit["innovation_std"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Histogram + Gaussian + t-fit
    bins = np.linspace(np.quantile(innov, 0.001),
                       np.quantile(innov, 0.999), 80)
    axes[0, 0].hist(innov, bins=bins, density=True, color="#1f77b4",
                    alpha=0.6, label="innovations")
    xs = np.linspace(bins[0], bins[-1], 400)
    axes[0, 0].plot(xs, stats.norm.pdf(xs, loc=0, scale=sigma),
                    color="#d62728", linewidth=1.8,
                    label=f"N(0, {sigma:.2f})")
    t_fit = fit_t_distribution(innov)
    if "df" in t_fit:
        axes[0, 0].plot(xs,
                        stats.t.pdf(xs, df=t_fit["df"],
                                    loc=t_fit["loc"], scale=t_fit["scale"]),
                        color="#2ca02c", linewidth=1.8,
                        linestyle="--",
                        label=f"t(df={t_fit['df']:.1f})")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title(
        f"AR({ar_p}) innovations: histogram\n"
        f"skew={fit['innovation_skew']:+.2f}, "
        f"exc kurt={fit['innovation_kurtosis']:+.2f}"
    )

    # Q-Q vs Normal
    stats.probplot(innov, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f"AR({ar_p}) innovations: Q-Q vs Normal")

    # ACF of innovations
    a = acf(innov, nlags=50, fft=True)
    axes[1, 0].stem(range(len(a)), a, basefmt=" ")
    axes[1, 0].axhline(0, color="black", linewidth=0.5)
    axes[1, 0].axhline(2 / np.sqrt(len(innov)), color="#d62728",
                       linestyle="--", linewidth=0.8, label="+/-2/sqrt(N)")
    axes[1, 0].axhline(-2 / np.sqrt(len(innov)), color="#d62728",
                       linestyle="--", linewidth=0.8)
    axes[1, 0].set_title(f"AR({ar_p}) innovations: ACF "
                          "(should be ~0 if model is correct)")
    axes[1, 0].set_xlabel("lag")
    axes[1, 0].legend(fontsize=8)

    # Ljung-Box stats
    lb = ljung_box_summary(innov)
    text_lines = ["Ljung-Box test (innovations are white if p > 0.05):"]
    for lag, res in lb.items():
        if isinstance(res, dict) and "p_value" in res:
            verdict = "white" if res["p_value"] > 0.05 else "NOT white"
            text_lines.append(
                f"  lag={lag}: Q={res['Q']:.1f}, p={res['p_value']:.3g} -> {verdict}"
            )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.05, 0.5, "\n".join(text_lines),
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment="center", fontfamily="monospace")
    axes[1, 1].set_title(f"AR({ar_p}) whitening quality")

    fig.suptitle(f"{name}: AR({ar_p}) innovation diagnostics",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_psd_comparison(name, residual, fits, out_path):
    """Compare empirical PSD to AR(p) theoretical PSD."""
    freqs_emp, psd_emp = scipy_signal.periodogram(residual, fs=1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freqs_emp[1:], psd_emp[1:], color="grey", alpha=0.5,
              linewidth=0.5, label="empirical periodogram")

    # Smoothed empirical PSD for clarity
    smoothed = pd.Series(psd_emp[1:]).rolling(50, min_periods=1).mean()
    ax.loglog(freqs_emp[1:], smoothed, color="#1f77b4", linewidth=1.5,
              label="empirical (smoothed)")

    # Theoretical AR(p) PSDs
    omega = 2 * np.pi * freqs_emp[1:]
    colours = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for (p, fit), c in zip(fits.items(), colours):
        if "phi" not in fit:
            continue
        # AR(p) PSD: sigma^2 / |1 - sum_k phi_k * exp(-i*k*omega)|^2
        phis = np.array(fit["phi"])
        sigma2 = fit["innovation_std"] ** 2
        denom = np.ones_like(omega, dtype=complex)
        for k, phi_k in enumerate(phis, start=1):
            denom = denom - phi_k * np.exp(-1j * k * omega)
        psd_theory = sigma2 / np.abs(denom) ** 2
        ax.loglog(freqs_emp[1:], psd_theory, color=c, linewidth=1.5,
                  linestyle="--",
                  label=f"AR({p}) theoretical "
                        f"(AIC={fit['aic']:.0f})")

    ax.set_xlabel("frequency [1/sample]")
    ax.set_ylabel("PSD")
    ax.set_title(f"{name}: residual PSD vs. AR(p) theoretical fits")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_param_stability(name, stab, out_path):
    if not stab:
        return
    chunks = [s["chunk"] for s in stab]
    phis = [s["phi"] for s in stab]
    sigmas = [s["innovation_std"] for s in stab]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(chunks, phis, "o-", color="#1f77b4")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("chunk (each = 1/4 of recording)")
    axes[0].set_ylabel("AR(1) phi")
    axes[0].set_title(f"{name}: phi stability across the month")
    axes[1].plot(chunks, sigmas, "o-", color="#1f77b4")
    axes[1].set_xlabel("chunk")
    axes[1].set_ylabel("innovation std")
    axes[1].set_title(f"{name}: sigma_eps stability across the month")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def analyse_column(name, series):
    print(f"\n=== {name} ===")
    out = {"name": name}

    baseline = centred_baseline(series, BASELINE_WINDOW)
    residual = series - baseline
    mask = robust_outlier_mask(series, baseline)
    longest = longest_contiguous_run(mask)
    if longest is None:
        out["error"] = "no clean segment"
        return out, None
    clean_seg = residual[longest[0]:longest[1]]
    out["clean_segment_length"] = int(len(clean_seg))
    out["clean_fraction"] = float(mask.sum() / len(mask))
    out["marginal_std"] = float(np.std(clean_seg, ddof=1))
    out["marginal_skew"] = float(stats.skew(clean_seg))
    out["marginal_kurtosis"] = float(stats.kurtosis(clean_seg, fisher=True))

    # 1. AR order selection
    fits = fit_ar_orders(clean_seg, AR_ORDERS_TO_TEST)
    out["ar_fits"] = {}
    for p, fit in fits.items():
        if "error" in fit:
            out["ar_fits"][p] = fit
            continue
        # Ljung-Box on innovations
        lb = ljung_box_summary(fit["innovations"])
        out["ar_fits"][p] = {
            "aic": fit["aic"],
            "bic": fit["bic"],
            "phi": fit["phi"],
            "innovation_std": fit["innovation_std"],
            "innovation_skew": fit["innovation_skew"],
            "innovation_kurtosis": fit["innovation_kurtosis"],
            "ljung_box": lb,
        }

    # 2. t-distribution fit on AR(1) innovations
    if "innovations" in fits[1]:
        t_fit = fit_t_distribution(fits[1]["innovations"])
        out["ar1_innovation_t_fit"] = t_fit

    # 3. Parameter stability across the month
    stab = parameter_stability(series, baseline, mask, n_chunks=4)
    out["parameter_stability"] = stab

    # 4. Baseline drift (wide MA + linear fit + ADF)
    out["baseline_drift"] = analyse_baseline_drift(series)

    print(f"  clean fraction: {out['clean_fraction']:.3f}, "
          f"marginal std: {out['marginal_std']:.2f}")
    print("  AR order selection:")
    print("    p   AIC      BIC      innov_std  LB(10)_p   LB(20)_p   LB(50)_p")
    for p in AR_ORDERS_TO_TEST:
        f = out["ar_fits"].get(p, {})
        if "aic" not in f:
            continue
        lb = f.get("ljung_box", {})
        lb10 = lb.get(10, {}).get("p_value", float("nan"))
        lb20 = lb.get(20, {}).get("p_value", float("nan"))
        lb50 = lb.get(50, {}).get("p_value", float("nan"))
        print(f"    {p:>2d}  {f['aic']:>7.0f}  {f['bic']:>7.0f}  "
              f"{f['innovation_std']:>9.3f}  {lb10:>9.3g}  {lb20:>9.3g}  "
              f"{lb50:>9.3g}")

    if "df" in out.get("ar1_innovation_t_fit", {}):
        t_fit = out["ar1_innovation_t_fit"]
        print(f"  AR(1) innov t-fit: df={t_fit['df']:.1f}, "
              f"scale={t_fit['scale']:.3f}")
    if stab:
        phis = [s["phi"] for s in stab]
        sigs = [s["innovation_std"] for s in stab]
        print(f"  per-quarter phi: {[f'{p:.3f}' for p in phis]}")
        print(f"  per-quarter sigma_innov: {[f'{s:.3f}' for s in sigs]}")

    return out, fits


def write_report(per_month, out_path):
    lines = []
    lines.append("# Background-noise model: deeper analysis\n")
    lines.append(
        "Follow-up to the v2 analysis, focused exclusively on the "
        "background-noise model. Anomaly detection is ignored.\n"
    )

    lines.append("## 1. AR order selection (AIC / BIC)\n")
    lines.append(
        "We fit AR(p) for p in 1, 2, 3, 5, 10 on the event-masked "
        "clean residual of each month. Lower AIC/BIC = better fit "
        "(BIC penalises higher orders more heavily).\n"
    )
    for name, info in per_month.items():
        if "ar_fits" not in info:
            continue
        lines.append(f"### {name}\n")
        lines.append("| p | AIC | BIC | innovation std | innov skew | innov exc kurt |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for p in AR_ORDERS_TO_TEST:
            f = info["ar_fits"].get(p, {})
            if "aic" not in f:
                continue
            lines.append(
                f"| {p} | {f['aic']:.0f} | {f['bic']:.0f} | "
                f"{f['innovation_std']:.3f} | "
                f"{f['innovation_skew']:+.2f} | "
                f"{f['innovation_kurtosis']:+.2f} |"
            )
        # Find best AR by BIC
        best_p_aic = min(
            (p for p in AR_ORDERS_TO_TEST
             if "aic" in info["ar_fits"].get(p, {})),
            key=lambda p: info["ar_fits"][p]["aic"],
        )
        best_p_bic = min(
            (p for p in AR_ORDERS_TO_TEST
             if "bic" in info["ar_fits"].get(p, {})),
            key=lambda p: info["ar_fits"][p]["bic"],
        )
        lines.append(
            f"\nBest by AIC: **AR({best_p_aic})**. "
            f"Best by BIC: **AR({best_p_bic})**.\n"
        )

    lines.append("## 2. Whitening quality (Ljung-Box on innovations)\n")
    lines.append(
        "Null hypothesis: innovations have zero autocorrelation up to "
        "the test lag (= the model captures all serial structure). "
        "p > 0.05 means we fail to reject -> the model is well "
        "specified.\n"
    )
    lines.append("| Month | order p | LB(10) p-value | LB(20) p-value | LB(50) p-value |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, info in per_month.items():
        for p in AR_ORDERS_TO_TEST:
            f = info.get("ar_fits", {}).get(p, {})
            lb = f.get("ljung_box", {})
            if not lb or "error" in lb:
                continue
            lines.append(
                f"| {name} | {p} | "
                f"{lb.get(10, {}).get('p_value', float('nan')):.3g} | "
                f"{lb.get(20, {}).get('p_value', float('nan')):.3g} | "
                f"{lb.get(50, {}).get('p_value', float('nan')):.3g} |"
            )
    lines.append("")

    lines.append("## 3. AR(1) innovation distribution\n")
    lines.append(
        "Is the innovation eps_t really Gaussian, or is it heavy-tailed? "
        "A Student-t with finite degrees of freedom would indicate heavy "
        "tails.\n"
    )
    lines.append("| Month | innov skew | innov exc kurt | t df | t scale |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, info in per_month.items():
        ar1 = info.get("ar_fits", {}).get(1, {})
        tf = info.get("ar1_innovation_t_fit", {})
        if "innovation_skew" not in ar1:
            continue
        df_str = f"{tf['df']:.1f}" if "df" in tf else "n/a"
        sc_str = f"{tf['scale']:.3f}" if "scale" in tf else "n/a"
        lines.append(
            f"| {name} | {ar1['innovation_skew']:+.2f} | "
            f"{ar1['innovation_kurtosis']:+.2f} | "
            f"{df_str} | {sc_str} |"
        )
    lines.append(
        "\n*Excess kurtosis* near zero = Gaussian. *t df* near infinity = "
        "Gaussian; t df near 4-6 = heavy-tailed.\n"
    )

    lines.append("## 4. Parameter stability across the month\n")
    lines.append(
        "Each month is split in 4 equal chunks; AR(1) is re-fitted on each."
        " If phi and sigma_innov drift across the month, the noise is "
        "non-stationary and a single AR(1) won't reproduce its full "
        "behaviour.\n"
    )
    for name, info in per_month.items():
        stab = info.get("parameter_stability", [])
        if not stab:
            continue
        lines.append(f"### {name}\n")
        lines.append("| chunk | n clean | phi | innov_std | marginal_std |")
        lines.append("|---:|---:|---:|---:|---:|")
        for s in stab:
            lines.append(
                f"| {s['chunk']} | {s['n_clean_samples']} | "
                f"{s['phi']:.3f} | {s['innovation_std']:.3f} | "
                f"{s['marginal_std']:.3f} |"
            )
        phis = [s["phi"] for s in stab]
        sigs = [s["innovation_std"] for s in stab]
        lines.append(
            f"\nRange: phi in [{min(phis):.3f}, {max(phis):.3f}] "
            f"(spread {max(phis) - min(phis):.3f}); "
            f"sigma_innov in [{min(sigs):.3f}, {max(sigs):.3f}] "
            f"(spread {max(sigs) - min(sigs):.3f}).\n"
        )

    lines.append("## 5. Cross-month consistency\n")
    lines.append(
        "If the same physical noise process produces all four months, "
        "AR(1) phi and sigma_innov should be similar across months.\n"
    )
    lines.append("| Month | AR(1) phi | innov std | marginal std |")
    lines.append("|---|---:|---:|---:|")
    for name, info in per_month.items():
        ar1 = info.get("ar_fits", {}).get(1, {})
        if "phi" not in ar1:
            continue
        lines.append(
            f"| {name} | {ar1['phi'][0]:.3f} | "
            f"{ar1['innovation_std']:.3f} | "
            f"{info['marginal_std']:.3f} |"
        )
    lines.append("")

    lines.append("## 6. Baseline drift\n")
    lines.append(
        "Linear fit on the ultra-wide MA-{} baseline; ADF tests the "
        "null hypothesis that the baseline is non-stationary "
        "(p<0.05 => stationary, p>0.05 => unit root / drift).\n".format(
            WIDE_BASELINE_WINDOW)
    )
    lines.append(
        "| Month | slope/sample | total drift | wide-base range | ADF p |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for name, info in per_month.items():
        d = info.get("baseline_drift")
        if not d:
            continue
        lines.append(
            f"| {name} | {d['linear_slope_per_sample']:+.2e} | "
            f"{d['total_linear_drift']:+.2f} | "
            f"{d['wide_baseline_range']:.2f} | "
            f"{d['adf_p_value']:.3g} |"
        )
    lines.append("")

    lines.append("## 7. Conclusions\n")
    lines.append(
        "1. **Best AR order.** See section 1. If AR(1) is consistently "
        "the AIC/BIC minimum, the v2 conclusion holds and the generator "
        "should use AR(1). If a higher order is preferred, the generator "
        "needs to match it.\n"
        "2. **Innovation distribution.** See section 3. If excess kurt "
        "is ~0 and t df is high, Gaussian innovations are fine.\n"
        "3. **Stationarity.** See section 4. If phi/sigma drift across "
        "the month, the generator should sample them per-recording "
        "from the empirical distribution rather than using point "
        "estimates.\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    per_month = {}
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")\
            .interpolate(method="linear", limit_direction="both")\
            .values.astype(np.float64)
        series = series[BOUNDARY_TRIM:-BOUNDARY_TRIM]
        info, fits = analyse_column(col, series)
        per_month[col] = info
        if fits is None:
            continue

        safe = col.replace("/", "_").replace(" ", "_").replace(":", "")
        # Per-month diagnostic plots
        if 1 in fits and "innovations" in fits[1]:
            plot_innovation_diagnostics(
                col, 1, fits[1],
                os.path.join(OUT_DIR, f"ar1_innov_{safe}.png"),
            )
        if 3 in fits and "innovations" in fits[3]:
            plot_innovation_diagnostics(
                col, 3, fits[3],
                os.path.join(OUT_DIR, f"ar3_innov_{safe}.png"),
            )
        # PSD comparison
        baseline = centred_baseline(series, BASELINE_WINDOW)
        residual = series - baseline
        # PSD on the longest clean segment
        mask = robust_outlier_mask(series, baseline)
        longest = longest_contiguous_run(mask)
        if longest:
            plot_psd_comparison(
                col, residual[longest[0]:longest[1]],
                fits, os.path.join(OUT_DIR, f"psd_cmp_{safe}.png"),
            )
        # Stability plot
        stab = per_month[col].get("parameter_stability", [])
        plot_param_stability(
            col, stab,
            os.path.join(OUT_DIR, f"stability_{safe}.png"),
        )

    # JSON-safe copy
    json_safe = {}
    for name, info in per_month.items():
        clean = {}
        for k, v in info.items():
            if k == "ar_fits":
                ar_clean = {}
                for p, f in v.items():
                    if isinstance(f, dict):
                        ar_clean[str(p)] = {kk: vv for kk, vv in f.items()
                                            if kk != "innovations"}
                clean["ar_fits"] = ar_clean
            else:
                clean[k] = v
        json_safe[name] = clean

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(json_safe, f, indent=2, default=float)

    write_report(per_month, os.path.join(OUT_DIR, "REPORT.md"))
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()

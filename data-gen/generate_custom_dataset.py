"""
Synthetic data generator (v2).

Compared to the v1 generator, this version aligns the synthetic
training distribution with the real 2015 recordings on the points the
reanalysis (Chapter 8) identified as the largest synthetic--real
mismatches:

1. **Background noise: AR(1), not IID Gaussian.**
   The fitted noise model is r_t = phi * r_{t-1} + eps_t with
   phi = 0.86 and eps_t ~ N(0, 1.12^2), giving marginal sigma ~ 2.30.
   The state r_t is propagated across chunks.

2. **Baseline: tuned OU process + global mean.**
   The OU parameters (theta = 2.5e-5, sigma_OU = 0.007) give a
   stationary baseline std of ~1 count and a typical 40k-sample drift
   range of ~4 counts, matching the empirical wide-scale baseline
   range observed in the four real months. The linear-trend term
   mentioned in the chapter is folded into the OU (rather than added
   on top) because a per-recording linear slope would diverge over
   the 10M-sample synthetic length. Global mean is sampled around
   99.3.

3. **Anomaly density: ~1.5 / 10k samples**, slightly above the
   empirical density (~1.05 / 10k from 17 hand labels across the four
   months) to give the network a few extra training examples. Old
   generator was at ~7 / 10k.

4. **Anomaly durations: log-uniform [30, 1000]**, matching the
   empirical p5-p95 range of the labels.

5. **Anomaly amplitudes: log-uniform [15, 40] counts above baseline**,
   absolute (no longer multiples of sigma_HF). The empirical range is
   [11.2, 36.8] counts; the lower bound was raised to 15 because
   amplitudes below that are not separable from the AR(1)+OU
   background texture and were degrading training rather than helping.

6. **Polarity: 100% positive.** All 17 hand-labelled real events are
   positive excursions.

7. **Templates: bell, fast_ascend, fast_descend.** The M, bell_sq and
   square templates were removed because no real event resembles them
   in the labelled data.

The CSV output schema is unchanged
(time_step, gamma_dose, is_anomaly) so the training pipeline does
not need to be modified.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
root_dir = script_dir
sys.path.append(root_dir)

from anomaly_generator import generate_anomaly

# ===========================================================================
#                       CONFIGURATION
# ===========================================================================
# All parameters fitted to the four months of real 2015 data --- see
# documentation/sections/08_real_data_reanalysis.tex for the derivation.
# ===========================================================================

CONFIG = {
    # --- Dataset size ---------------------------------------------------
    "TOTAL_POINTS":         10_000_000,
    "CHUNK_SIZE":           50_000,

    # --- Anomaly count -------------------------------------------------
    # Target density: ~1.5 events / 10k samples (slightly above the
    # empirical 1.05 / 10k from labels, to give the network more
    # training examples).
    "NUM_ANOMALIES_MIN":    1_300,
    "NUM_ANOMALIES_MAX":    1_700,
    "INJECT_JITTER":        400,    # +/- random jitter around grid

    # --- Anomaly shape -------------------------------------------------
    # Empirical amplitudes (peak above baseline) span 11.2-36.8 counts.
    # Log-uniform in counts, NOT multiples of HF noise std. The min is
    # held at 15 (not the empirical 11) because amplitudes below ~15 get
    # lost in the AR(1)+OU background texture and are not separable from
    # noise at training time.
    "AMPLITUDE_MIN":        15.0,
    "AMPLITUDE_MAX":        40.0,
    # Empirical durations span 32-1014 samples, log-uniform.
    "PERIOD_MIN":           30,
    "PERIOD_MAX":           1000,
    # Stochastic deformation of the template keypoints.
    "VARIANCE_MIN":         0.02,
    "VARIANCE_MAX":         0.10,
    # Per-sample threshold for labelling (in absolute counts).
    "ANOMALY_THRESHOLD":    0.5,

    # --- Background noise model (AR(1) Gaussian) ----------------------
    # r_t = phi * r_{t-1} + eps_t, eps_t ~ N(0, sigma_eps^2)
    # Fitted on event-masked clean residuals across all 4 months.
    "AR1_PHI":              0.86,
    "AR1_SIGMA_EPS":        1.12,   # innovation std

    # --- Baseline model ------------------------------------------------
    # Tuned OU process. Correlation time ~1/theta = 40k samples, so
    # the baseline drifts and reverts on the same timescale as real
    # recordings. Stationary std = sigma_OU / sqrt(2 theta) ~ 1 count,
    # typical 40k-sample drift range ~4 counts.
    "OU_THETA":             2.5e-5,
    "OU_SIGMA":             0.007,
    # Global mean around which the OU baseline oscillates.
    "BASELINE_MEAN":        99.3,
    "BASELINE_MEAN_JITTER": 1.5,    # per-recording offset jitter

    # --- Visualisation -------------------------------------------------
    "PLOT_PREVIEW_LIMIT":   100_000,
}


def generate_ar1_noise(n, phi, sigma_eps, init_state=0.0, rng=None):
    """Generate n samples of AR(1) noise with state continuation.

    Returns (noise_array, last_state)."""
    if rng is None:
        rng = np.random
    eps = rng.normal(scale=sigma_eps, size=n)
    noise = np.empty(n, dtype=float)
    state = init_state
    for i in range(n):
        state = phi * state + eps[i]
        noise[i] = state
    return noise, state


def create_synthetic_dataset(
    output_csv_path=None,
    total_points=None,
    num_anomalies=None,
    noise_multiplier=1.0,
    variance_override=None,
    return_df=False,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    data_dir = os.path.join(root_dir, "data")
    logs_dir = os.path.join(root_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-recording parameters
    # ------------------------------------------------------------------
    global_mean = CONFIG["BASELINE_MEAN"] + np.random.uniform(
        -CONFIG["BASELINE_MEAN_JITTER"], CONFIG["BASELINE_MEAN_JITTER"],
    )
    ar1_phi = CONFIG["AR1_PHI"]
    ar1_sigma = CONFIG["AR1_SIGMA_EPS"] * noise_multiplier
    ou_theta = CONFIG["OU_THETA"]
    ou_sigma = CONFIG["OU_SIGMA"]

    if not return_df:
        marginal_hf = ar1_sigma / np.sqrt(1.0 - ar1_phi ** 2)
        ou_stationary = ou_sigma / np.sqrt(2.0 * ou_theta)
        print("Background model")
        print(f"  global mean      = {global_mean:.3f}")
        print(f"  AR(1) phi        = {ar1_phi}")
        print(f"  AR(1) sigma_eps  = {ar1_sigma:.3f}")
        print(f"  -> marginal HF std = {marginal_hf:.3f}")
        print(f"  OU theta, sigma  = {ou_theta}, {ou_sigma}")
        print(f"  -> OU stationary std = {ou_stationary:.3f}")

    # ------------------------------------------------------------------
    # Load anomaly templates
    # ------------------------------------------------------------------
    anomalies_dir = os.path.join(root_dir, "anomalies")
    template_files = sorted(glob.glob(os.path.join(anomalies_dir, "*.csv")))
    if not template_files:
        print(f"Error: no templates in {anomalies_dir}")
        return None

    loaded_templates = {}
    for path in template_files:
        name = os.path.basename(path).replace(".csv", "")
        loaded_templates[name] = pd.read_csv(path)
    template_names = list(loaded_templates.keys())

    class_map = {name: i + 1 for i, name in enumerate(template_names)}
    if not return_df:
        with open(os.path.join(data_dir, "class_mapping.txt"), "w") as f:
            f.write("0: Normal Background\n")
            for name, cls_id in class_map.items():
                f.write(f"{cls_id}: {name}\n")

    # ------------------------------------------------------------------
    # Decide N, number of anomalies, injection grid
    # ------------------------------------------------------------------
    N = total_points if total_points is not None else CONFIG["TOTAL_POINTS"]
    chunk_size = CONFIG["CHUNK_SIZE"] if not return_df else N

    if num_anomalies is None:
        num_anomalies = np.random.randint(
            CONFIG["NUM_ANOMALIES_MIN"], CONFIG["NUM_ANOMALIES_MAX"] + 1,
        )

    spacing = N // (num_anomalies + 1)
    inject_points = [
        spacing * i + np.random.randint(
            -CONFIG["INJECT_JITTER"], CONFIG["INJECT_JITTER"] + 1,
        )
        for i in range(1, num_anomalies + 1)
    ]

    if not return_df:
        density = num_anomalies * 10_000 / N
        print(f"\nInjecting {num_anomalies} anomalies into {N:,} samples "
              f"(density = {density:.2f} / 10k).")
        print(f"Writing in chunks of {chunk_size:,}.")

    # ------------------------------------------------------------------
    # Pre-generate all anomaly waveforms
    # ------------------------------------------------------------------
    anomaly_injections = []
    for idx in inject_points:
        chosen = np.random.choice(template_names)
        df_template = loaded_templates[chosen]

        # Log-uniform amplitudes and durations in absolute units.
        amplitude = np.exp(np.random.uniform(
            np.log(CONFIG["AMPLITUDE_MIN"]),
            np.log(CONFIG["AMPLITUDE_MAX"]),
        ))
        period = int(np.exp(np.random.uniform(
            np.log(CONFIG["PERIOD_MIN"]),
            np.log(CONFIG["PERIOD_MAX"]),
        )))
        variance_level = (variance_override
                          if variance_override is not None
                          else np.random.uniform(
                              CONFIG["VARIANCE_MIN"],
                              CONFIG["VARIANCE_MAX"],
                          ))

        _, v_discrete = generate_anomaly(
            df_template,
            amplitude=amplitude,
            period=period,
            variance=variance_level,
        )

        anomaly_injections.append({
            "start": idx,
            "values": v_discrete,
            "template_name": chosen,
            "class_id": class_map[chosen],
        })

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if output_csv_path is None and not return_df:
        output_csv_path = os.path.join(data_dir, "synthetic_custom_dataset.csv")

    plot_limit = min(CONFIG["PLOT_PREVIEW_LIMIT"], N)
    plot_time, plot_gamma, plot_anomaly_vals = [], [], []
    plot_labels, plot_classes_text, plot_baseline = [], [], []
    all_time, all_gamma, all_labels = [], [], []

    csv_file = open(output_csv_path, "w") if not return_df else None
    if csv_file:
        csv_file.write("time_step,gamma_dose,is_anomaly\n")

    # State that must persist across chunks
    ar1_state = 0.0
    ou_state = 0.0

    try:
        num_chunks = (N + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            cs = chunk_idx * chunk_size
            ce = min(cs + chunk_size, N)
            clen = ce - cs

            # -- Baseline: global mean + OU process --------------------
            ou_chunk = np.empty(clen, dtype=float)
            ou_kicks = np.random.normal(scale=ou_sigma, size=clen)
            for i in range(clen):
                ou_state = ou_state * (1.0 - ou_theta) + ou_kicks[i]
                ou_chunk[i] = ou_state
            baseline_chunk = global_mean + ou_chunk

            # -- High-frequency AR(1) noise on top of baseline --------
            hf_chunk, ar1_state = generate_ar1_noise(
                clen, ar1_phi, ar1_sigma, init_state=ar1_state,
            )
            chunk_background = baseline_chunk + hf_chunk
            chunk_background = np.clip(chunk_background, 0.0, None)

            chunk_pure_anomalies = np.zeros(clen, dtype=float)
            chunk_labels = np.zeros(clen, dtype=int)
            chunk_classes_text = ["Normal Background"] * clen

            # -- Inject anomalies that overlap this chunk -------------
            for anom in anomaly_injections:
                a_s = anom["start"]
                a_e = a_s + len(anom["values"])
                if a_e <= cs or a_s >= ce:
                    continue
                o_s = max(a_s, cs)
                o_e = min(a_e, ce)
                for global_idx in range(o_s, o_e):
                    local_idx = global_idx - cs
                    anom_idx = global_idx - a_s
                    v = anom["values"][anom_idx]
                    chunk_background[local_idx] += v
                    chunk_pure_anomalies[local_idx] += v
                    if v > CONFIG["ANOMALY_THRESHOLD"]:
                        chunk_labels[local_idx] = anom["class_id"]
                        chunk_classes_text[local_idx] = (
                            f"Anomaly: {anom['template_name'].capitalize()}"
                        )

            # -- Write / collect --------------------------------------
            if return_df:
                all_time.extend(range(cs, ce))
                all_gamma.extend(chunk_background.tolist())
                all_labels.extend(chunk_labels.tolist())
            elif csv_file:
                for i in range(clen):
                    csv_file.write(
                        f"{cs + i},{chunk_background[i]},{chunk_labels[i]}\n",
                    )

            if not return_df and cs < plot_limit:
                pe = min(ce, plot_limit)
                pl = pe - cs
                plot_time.extend(range(cs, pe))
                plot_gamma.extend(chunk_background[:pl].tolist())
                plot_anomaly_vals.extend(chunk_pure_anomalies[:pl].tolist())
                plot_labels.extend(chunk_labels[:pl].tolist())
                plot_classes_text.extend(chunk_classes_text[:pl])
                plot_baseline.extend(baseline_chunk[:pl].tolist())

            if not return_df:
                print(f"  chunk {chunk_idx + 1}/{num_chunks} "
                      f"({cs:,}-{ce:,})")
    finally:
        if csv_file:
            csv_file.close()

    if return_df:
        return pd.DataFrame({
            "time_step": all_time,
            "gamma_dose": all_gamma,
            "is_anomaly": all_labels,
        })

    print(f"\nDataset saved -> {output_csv_path}")

    # ------------------------------------------------------------------
    # Preview plot
    # ------------------------------------------------------------------
    plot_time = np.array(plot_time)
    plot_gamma = np.array(plot_gamma)
    plot_anomaly_vals = np.array(plot_anomaly_vals)
    plot_labels = np.array(plot_labels)
    plot_baseline = np.array(plot_baseline)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(plot_time, plot_gamma, color="#1f77b4", linewidth=0.6,
            label="synthetic")
    ax.plot(plot_time, plot_baseline, color="#d62728", linewidth=1.2,
            linestyle="--", label="baseline (global mean + trend + OU)")
    ax.plot(plot_time, plot_anomaly_vals + plot_baseline, color="orange",
            linewidth=1.5, label="pure anomaly + baseline")
    ax.set_title("Synthetic dataset (preview): "
                 "AR(1) noise + linear trend + bumps")
    ax.set_xlabel("sample")
    ax.set_ylabel("gamma dose")
    ax.legend(loc="upper right")

    # Zoom to the first few anomalies for the static PNG
    is_anom = plot_labels > 0
    diffs = np.diff(is_anom.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if len(is_anom) and is_anom[0]:
        starts = np.insert(starts, 0, 0)
    if len(is_anom) and is_anom[-1]:
        ends = np.append(ends, len(is_anom))
    if len(starts) >= 3:
        pad = 2000
        ax.set_xlim(
            plot_time[max(0, starts[0] - pad)],
            plot_time[min(len(plot_time) - 1, ends[2] + pad)],
        )

    out_png = os.path.join(logs_dir, "synthetic_custom_dataset.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(
        x=plot_time, y=plot_gamma, mode="lines",
        name="synthetic", line=dict(color="teal", width=1),
        hovertext=plot_classes_text, hoverinfo="x+y+text",
    ))
    anomalous_idx = np.where(plot_labels > 0)[0]
    fig_plotly.add_trace(go.Scatter(
        x=plot_time[anomalous_idx], y=plot_gamma[anomalous_idx],
        mode="markers", name="anomalous points",
        marker=dict(color="red", size=4),
        hovertext=[plot_classes_text[i] for i in anomalous_idx],
        hoverinfo="x+y+text",
    ))
    fig_plotly.add_trace(go.Scatter(
        x=plot_time, y=plot_baseline, mode="lines",
        name="baseline", line=dict(color="orange", width=1.5, dash="dash"),
        hoverinfo="skip",
    ))
    fig_plotly.update_layout(
        title="Synthetic dataset preview",
        xaxis_title="sample", yaxis_title="gamma dose",
        hovermode="x unified", template="plotly_white",
    )
    out_html = os.path.join(logs_dir, "synthetic_custom_dataset.html")
    fig_plotly.write_html(out_html)

    print(f"Plot saved   -> {out_png}")
    print(f"HTML preview -> {out_html}")


if __name__ == "__main__":
    create_synthetic_dataset()

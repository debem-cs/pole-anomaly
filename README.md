# pole-anomaly — anomaly detection in radiological monitoring signals

Detection and classification of anomalies in gamma-dose time series from
environmental radiological monitoring beacons. Data-science *pôle projet* at
CentraleSupélec, second semester 2025-2026.

A beacon counts the gamma photons that reach a scintillation detector. The
signal is very noisy and drifts slowly. The events of interest (spikes,
plateaus, ramps) are rare and vary in shape, amplitude and duration. No labelled
set of real events exists, so supervised training on the real data is not
possible.

## Approach

1. **Real-data analysis.** Four months of 2015 recordings
   (`data-gen/data/2015_months_DebitDoseA.txt`) are used to characterise the
   baseline, the high-frequency noise and the plausible anomalies.
2. **Synthetic generator.** A signal model made of a drifting baseline
   (Ornstein–Uhlenbeck), high-frequency noise and injected anomaly templates,
   with random amplitude, duration and shape deformation. The model is
   calibrated on the real recordings.
3. **Three 1D architectures**, trained in PyTorch on sliding windows of the
   synthetic signal:
   - a compact 1D-CNN (baseline);
   - a 1D residual network (ResNet);
   - a hybrid CNN + Transformer encoder (CNN+Attention).
4. **Evaluation.** Window-level accuracy, then event-level evaluation on a
   continuous segment. Window probabilities are averaged into a per-sample
   trace, thresholded and merged into predicted events, which are then matched
   to ground-truth events. The models are also tested for robustness to more
   background noise and to stronger shape deformation, and compared on cost.

## Results (report, synthetic benchmark)

These results come from the first version of the generator: white Gaussian
noise and six anomaly morphologies plus background. The event-level test segment
has 59 events.

| Model | Parameters | Event-level macro F1 | Noise robustness (F1 < 50 % at) |
|---|---:|---:|---|
| 1D-CNN | 22,727 | 0.934 | ≈ 1.9× noise (collapses to one class) |
| ResNet 1D | 74,631 | **1.000** | never, up to 3× |
| CNN+Attention | 109,655 | 0.955 | never, up to 3× |

No model raised a false alarm on the test segment. The ResNet gave the best
trade-off: a perfect event-level score, the highest robustness, and a third of
the parameters of the attention model. Training took about 10 minutes on an RTX
5060 Laptop GPU.

**Synthetic-to-real gap.** On real recordings the models showed behaviour that
had not appeared in synthetic validation: long false positives, oscillation
between classes, and predictions collapsing to a single class. An exploratory
follow-up re-analysed the real noise and found it is better described by an
AR(1) process than by white noise. A small set of real events was hand-labelled,
and the generator was recalibrated (v2). Without expert-validated labels this
extension could not be evaluated quantitatively, so it was left out of the final
report's scope. Its code and results are still in the repository (see below).

## Repository structure

```text
data-gen/
├── data/                          # real 2015 recordings, class mapping, hand labels
├── anomalies/                     # anomaly templates (bell, fast_ascend, fast_descend)
├── anomaly_generator.py           # template-based anomaly with random deformation
├── generate_custom_dataset.py     # synthetic dataset generator (v2: AR(1) noise + OU baseline)
├── analyze_noise_model.py         # AR-order selection, whiteness, stationarity of real noise
├── labeling_view.py               # helper plots for hand-labelling real events
├── affichage_donnees.py, plot_dataset.py, visualize_anomalies_templates.py,
│   test_generator_deformation.py  # exploration / visual checks
└── logs/                          # generated HTML/PNG figures, noise-analysis report
architectures/
├── saved_models/                  # model definitions + checkpoints + normalisation stats
├── train.py                       # training on sliding windows (temporal train/test split)
├── inference_engine.py            # batched sliding-window inference
├── visualize_inference.py         # inference on a synthetic validation set
├── inference_real_data.py         # inference on the real 2015 recordings
├── robustness_test.py             # noise / deformation sweeps
├── model_selection.py             # model registry (1d_cnn, resnet, cnn_attention)
└── logs/<model>/                  # training, inference and robustness logs
documentation/
├── main.tex, sections/, figures/  # final report (LaTeX); figures/build_figures.py rebuilds the PNGs
└── presentation/                  # final presentation slides
```

The checkpoints and logs in `architectures/` come from the v2 generator (four
classes: background, `bell`, `fast_ascend`, `fast_descend`). The report's
figures and tables come from the v1 run (seven classes).

## Running

Requirements: Python 3.10+ and a CUDA GPU (recommended for training).

```bash
pip install -r requirements.txt
```

`requirements.txt` points pip at the PyTorch nightly CUDA 12.8 index. On other
hardware, install a matching `torch` build first.

```bash
# 1. Generate a synthetic dataset -> data-gen/data/synthetic_custom_dataset.csv
python data-gen/generate_custom_dataset.py

# 2. Copy it to where the training script expects it
mkdir -p architectures/data
cp data-gen/data/synthetic_custom_dataset.csv architectures/data/training_dataset.csv

# 3. Train / evaluate a model (--model 1d_cnn | resnet | cnn_attention; prompts if omitted)
python architectures/train.py --model resnet
python architectures/robustness_test.py --model resnet
python architectures/inference_real_data.py --model resnet
```

`visualize_inference.py` expects a separate synthetic set at
`architectures/data/validation_dataset.csv`. Generated CSV datasets are not
versioned.

To build the report, see [`documentation/README.md`](documentation/README.md).

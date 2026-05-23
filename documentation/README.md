# Final report — pole-anomaly

LaTeX source for the project's final report.

## Layout

```
documentation/
  main.tex              # entry point
  preamble.tex          # all packages and macros
  references.bib        # bibliography
  sections/
    00_abstract.tex
    01_introduction.tex
    02_theory.tex          # the big background chapter
    03_data.tex            # real data + synthetic generator
    04_architectures.tex   # the three networks + training pipeline
    05_results.tex         # results on synthetic test
    06_robustness.tex      # noise / deformation sweeps
    07_real_inference.tex  # real-data trial + failure analysis
    08_conclusion.tex
    09_appendix.tex
  figures/
    build_figures.py     # regenerates every PNG used in the report
    *.png                # 13 figures
```

## Build

A local TeX install (MiKTeX or TeX Live) with `pdflatex` and `bibtex` is
expected.

```
cd documentation
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Two final `pdflatex` runs are needed for the cross-references
(`\cref`, `\ref`, the table of contents, the list of figures) to
stabilise.

## Regenerating the figures

Every PNG in `figures/` is produced by `build_figures.py`, which reads
the training and inference logs under `architectures/logs/` plus the
raw data under `data-gen/data/`. To rebuild from scratch:

```
py documentation/figures/build_figures.py
```

The script also re-runs sliding-window inference on a 10 000-sample
real-data snapshot for each of the three trained models, so a saved
checkpoint is expected under
`architectures/saved_models/<model>/best_<model>.pth`. If a checkpoint
is missing, train it first with `architectures/train.py --model <model>`.

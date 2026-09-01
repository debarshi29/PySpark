# Report

LaTeX source for the theory-and-results report on the Superstore ML project.
Figures are pulled directly from `../outputs/figures/` (produced by the
notebooks), so the report and the notebooks can never show inconsistent
plots.

## Build

Requires a TeX distribution (MiKTeX / TeX Live) with `pdflatex`.

```bash
cd "EY Training/ml_superstore/report"
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex   # 2nd pass: TOC + cross-refs
cp build/main.pdf main.pdf                                          # tracked copy
```

`build/` (all LaTeX intermediates, including its own copy of `main.pdf`) is
git-ignored; only the top-level `report/main.pdf` is committed.

## Structure

```
report/
├── main.tex              title page, abstract, \input's every chapter
├── preamble.tex           shared packages/styling
├── chapters/
│   ├── 01_introduction.tex
│   ├── 02_data_eda.tex
│   ├── 03_feature_engineering.tex
│   ├── 04_forecasting.tex
│   ├── 05_supervised_models.tex
│   └── 06_conclusion.tex
└── build/main.pdf         compiled output (tracked)
```

Each chapter follows the same **Theory → Method → Results → Interpretation**
structure: the statistical/ML machinery is derived first, then applied,
then read for business meaning.

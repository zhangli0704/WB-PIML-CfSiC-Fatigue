# WB-PIML for C/SiC fatigue

This repository contains the tabular analysis for a Walker–Basquin physics-informed
machine-learning model of C/SiC fatigue life. The implementation combines a censored
accelerated failure-time physical model with an exact-fracture residual learner.
It includes nested model selection, comparator models, uncertainty intervals,
factor holdouts, ablations, and sensitivity analyses.

## Data and reference results

`data_v23.xlsx` is an unchanged copy of the analysis workbook. The primary analysis
uses `Plot_Data_Verified`: 222 unique records, comprising 160 exact fractures and
62 right-censored runouts. All primary likelihood records have unit weight.
Runouts contribute survival terms; exact-fracture records define point-error metrics.
Other workbook sheets provide provenance, label and campaign audits, and the
alternative datasets used in sensitivity analyses.

`reference_results/` contains the original 45-sheet results workbook, `protocol.json`,
and `README_results.md`. These files are retained as comparison references; running
the analysis computes results from the data workbook.

## Environment and execution

The reference protocol records Python **3.14.0**, NumPy **2.3.5**, pandas **3.0.1**,
SciPy **1.17.1**, and scikit-learn **1.8.0**. This implementation is checked with
those versions and openpyxl **3.1.5**.
Use Python 3.14.0 and the pinned dependencies in `requirements.txt`:

```sh
python -m venv .venv
```

Activate the environment (`.\.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS), then run:

```sh
python -m pip install -r requirements.txt
python run.py --data data_v23.xlsx --out results_v23_final --trees 400 --audit-trees 160 --repeats 10 --bootstrap 5000
```

These are the reference run settings, with seed `20260824` fixed in `config.py`.
The full run includes repeated and nested validation and can take substantial time.
Bootstrap calculations prepare each campaign once and reuse those inputs while
preserving the original sampling order and row-level numerical reductions.
Reducing these settings changes the analysis and does not reproduce the reference run.
Add `--export-csv` if individual table exports are needed.

The output is `results_v23_final/PIML_v23_final_results.xlsx` with all 45 tables,
together with `protocol.json` and `README_results.md`. Plot and PowerPoint generation
has been removed. Figure/PPT statements in the retained workbook README, protocol,
and results README describe the historical workflow; they remain unchanged for
cell-content compatibility and are not outputs of this implementation.

## Comparison

```sh
python verify_results.py reference_results results_v23_final --json verification.json
python verify_results.py reference_results results_v23_final --allow-roundoff --json verification_roundoff.json
```

The first command requires exact equality of every worksheet's order, dimensions,
cell types and values, plus the complete protocol JSON content and results README
bytes. It exits with a nonzero status if any difference exists. Workbook ZIP and
document timestamps are excluded because they are not worksheet data.

The second command permits only numeric differences within absolute or relative
tolerance `1e-12`; structural, text, boolean, and metadata comparisons stay strict.
Reports retain strict mismatch counts, maximum absolute/relative differences, and
examples. Seeded parallel tree predictions can differ in the final floating-point
digits even in the same environment. Tolerance agreement is therefore distinct
from exact agreement; inspect the generated reports before claiming reproduction.

The completed full run retained all 45 tables, selection settings, physical
coefficients, and nonnumeric content. Point predictions differed by at most
`4.44e-15`; AFT scale and likelihood-related fields differed by up to `6.39e-8`.
Consequently, this run passes neither strict numeric equality nor the default
`1e-12` check for the entire workbook. The original published results are preserved
byte-for-byte in `reference_results/`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
and `checks/` for the complete validation results.

## Code layout

`run.py` starts the analysis. The `wbpiml/` package separates configuration (`config`),
data preparation and audits (`data`), physical models (`physics`), metrics and
intervals (`metrics`), comparator models (`baselines`), residual learning (`residual`),
validation (`validation`), sensitivity analyses (`sensitivity`), workbook output
(`reporting`), and orchestration (`workflow`).

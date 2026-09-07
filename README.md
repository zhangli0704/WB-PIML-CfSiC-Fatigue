# WB-PIML for C/SiC fatigue

Walker–Basquin physics-informed machine learning for fatigue-life prediction in
C/SiC composites. The analysis combines a censored accelerated failure-time
physical model with a residual learner fitted to exact-fracture observations.

## Data

`data_v23.xlsx` contains the dataset and its provenance. The primary analysis uses
the `Plot_Data_Verified` worksheet: 222 unique records, comprising 160 exact
fractures and 62 right-censored runouts. Each record has unit weight in the joint
likelihood. Point-error metrics are evaluated on exact fractures and balanced
across experimental campaigns.

The manuscript's result tables are provided in
`reference_results/PIML_v23_final_results.xlsx`.

## Installation

Use Python 3.14.0. Package versions are specified in `requirements.txt`.
Install them in your Python environment:

```bash
python -m pip install -r requirements.txt
```

## Run

Open a terminal in this directory and run:

```bash
python run.py
```

The defaults use seed `20260824`, 400 trees, 160 trees for the sensitivity audit,
10 validation repeats, and 5,000 campaign-bootstrap draws. The complete analysis
includes nested model selection and factor-held-out validation and may take
about an hour, depending on the computer.

Results are written to `results_v23_final/`:

- `PIML_v23_final_results.xlsx`: all 45 result tables, including predictions,
  performance metrics, prediction intervals, model selection, and sensitivity analyses.
- `protocol.json`: analysis settings and software versions.
- `README_results.md`: a summary of the analysis and results.

Use `python run.py --help` to see the available options. Add `--export-csv` to
export each table as a separate CSV file.

## Source files

| File | Purpose |
| --- | --- |
| `data.py` | Settings, data preparation, metrics and statistical summaries |
| `models.py` | Physical models, comparator models and residual learning |
| `validation.py` | Nested validation, factor holdouts and sensitivity analyses |
| `run.py` | Analysis workflow and result export |


# Metrics Toolkit Batch Package

This package contains:

- metrics_toolkit.py
- batch_runner.py
- equations.md
- requirements.txt

## What it does

It computes these metrics for each patient:
- Mean difference (%)
- Std difference (%)
- KL divergence
- KDE area difference
- GIR_ED
- GIR_HD

It can also produce:
- a per-patient CSV
- a summary CSV with mean and std across patients

## Input folder structure

Put all patient files in one folder (or subfolders) with names like:

- SB-001_majority.xlsx
- SB-001_minority.xlsx
- SB-001_synthetic.xlsx

(or CSV equivalents)

## Install dependencies

pip install -r requirements.txt

## Run batch evaluation

python batch_runner.py --folder /path/to/data_folder --out_csv patient_metrics.csv --summary_csv summary_metrics.csv

## Output files

1. patient_metrics.csv
   One row per patient, with all metrics

2. summary_metrics.csv
   One row per metric, with:
   - mean across patients
   - std across patients

## Notes

- Features should already be min-max normalized to [0,1]
- Exact duplicates are excluded only when distance == 0
- GIR invalid rule is: d_MA(s) < d_MI(s)

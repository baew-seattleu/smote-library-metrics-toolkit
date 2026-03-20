
#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
import pandas as pd
from metrics_toolkit import compute_metrics

def find_triplets(folder):
    files = []
    for root, _, names in os.walk(folder):
        for f in names:
            if f.startswith("._"):
                continue
            if f.lower().endswith((".xlsx", ".xls", ".csv")):
                files.append(os.path.join(root, f))
    triplets = {}
    for path in files:
        name = os.path.basename(path).lower()
        if "_majority" in name:
            pid = os.path.basename(path)[:name.index("_majority")]
            triplets.setdefault(pid, {})["majority"] = path
        elif "_minority" in name and "syn" not in name:
            pid = os.path.basename(path)[:name.index("_minority")]
            triplets.setdefault(pid, {})["minority"] = path
        elif "_synthetic" in name or "_synminority" in name:
            idx = name.index("_synthetic") if "_synthetic" in name else name.index("_synminority")
            pid = os.path.basename(path)[:idx]
            triplets.setdefault(pid, {})["synthetic"] = path
    complete = {k:v for k,v in triplets.items() if {"majority","minority","synthetic"} <= set(v.keys())}
    return dict(sorted(complete.items()))

def summarize(df):
    metrics = ["MeanDiff_pct","StdDiff_pct","KL","KDE_AreaDiff","GIR_ED","GIR_HD"]
    rows = []
    for m in metrics:
        rows.append([m, df[m].mean(), df[m].std(ddof=1)])
    return pd.DataFrame(rows, columns=["Metric","Mean","Std"])

def main():
    ap = argparse.ArgumentParser(description="Batch-run oversampling metrics for all patients in a folder.")
    ap.add_argument("--folder", required=True, help="Folder containing patient files.")
    ap.add_argument("--out_csv", required=True, help="Output CSV for per-patient metrics.")
    ap.add_argument("--summary_csv", default="", help="Optional output CSV for mean±std summary.")
    args = ap.parse_args()

    triplets = find_triplets(args.folder)
    if not triplets:
        raise SystemExit("No complete patient triplets found.")

    rows = []
    for pid, paths in triplets.items():
        result = compute_metrics(paths["majority"], paths["minority"], paths["synthetic"])
        row = {"Patient": pid}
        row.update(result)
        rows.append(row)
        print(pid, "->", ", ".join(f"{k}={v:.6f}" for k,v in result.items()))

    df = pd.DataFrame(rows).sort_values("Patient")
    df.to_csv(args.out_csv, index=False)
    print(f"Saved per-patient CSV: {args.out_csv}")

    if args.summary_csv:
        summary_df = summarize(df)
        summary_df.to_csv(args.summary_csv, index=False)
        print(f"Saved summary CSV: {args.summary_csv}")

if __name__ == "__main__":
    main()

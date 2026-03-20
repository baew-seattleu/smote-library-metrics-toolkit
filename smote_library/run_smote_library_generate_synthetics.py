#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd

from smote_library_v1 import FUNCTION_REGISTRY


C_METHODS = {"C-GMC-Gamma", "C-GMC-SDD", "C-KMeans-Gamma", "C-KMeans-SDD"}
GC_METHODS = {"GC-GMC-Gamma", "GC-GMC-SDD", "GC-KMeans-Gamma", "GC-KMeans-SDD"}
NOCLUSTER_METHODS = {"Gamma-SMOTE", "SDD-SMOTE"}


def find_patient_pairs(input_dir: Path):

    pairs = []

    for maj in sorted(input_dir.glob("*_majority.xlsx")):

        patient_id = maj.name.replace("_majority.xlsx", "")

        minf = input_dir / f"{patient_id}_minority.xlsx"

        if minf.exists():
            pairs.append((patient_id, maj, minf))

    return pairs


def sanitize_method_name(method_name: str):

    return method_name.replace("/", "-").replace(" ", "_")


def build_call_kwargs(method_name, args):

    kwargs = {
        "random_state": args.random_state,
        "gamma_alpha": args.gamma_alpha,
        "neighbor_k": args.neighbor_k,
    }

    if method_name in C_METHODS:
        kwargs["k"] = args.k

    if method_name in GC_METHODS:
        kwargs["theta"] = args.theta
        kwargs["k"] = args.k

    if args.gamma_scale is not None:
        kwargs["gamma_scale"] = args.gamma_scale

    return kwargs


def save_result_xlsx(path, result):

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        result.synthetic_df.to_excel(
            writer,
            index=False,
            sheet_name="synthetic_data",
        )

        result.metadata_df.to_excel(
            writer,
            index=False,
            sheet_name="generation_metadata",
        )

        if result.cluster_df is not None:

            result.cluster_df.to_excel(
                writer,
                index=False,
                sheet_name="clusters",
            )

        pd.DataFrame(
            list(result.info.items()),
            columns=["metric", "value"],
        ).to_excel(
            writer,
            index=False,
            sheet_name="info",
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--method", default=None)

    parser.add_argument("--run_all", action="store_true")

    parser.add_argument("--theta", type=float, default=0.8)

    parser.add_argument("--k", type=int, default=8)

    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--gamma_alpha", type=float, default=2.0)

    parser.add_argument("--gamma_scale", type=float, default=None)

    parser.add_argument("--neighbor_k", type=int, default=3)

    args = parser.parse_args()

    if not args.run_all and not args.method:
        raise SystemExit("Provide --method or --run_all")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_patient_pairs(input_dir)

    if not pairs:
        raise SystemExit("No patient pairs found")

    if args.run_all:
        methods = list(FUNCTION_REGISTRY.keys())
    else:

        if args.method not in FUNCTION_REGISTRY:
            raise SystemExit(f"Unknown method {args.method}")

        methods = [args.method]

    errors = []

    print("pairs found:", pairs)

    print("methods:", methods)

    for method_name in methods:

        fn = FUNCTION_REGISTRY[method_name]

        method_dir = out_dir / sanitize_method_name(method_name)

        method_dir.mkdir(parents=True, exist_ok=True)

        for patient_id, maj_path, min_path in pairs:

            try:

                maj_df = pd.read_excel(maj_path)

                min_df = pd.read_excel(min_path)

                kwargs = build_call_kwargs(method_name, args)

                result = fn(
                    majority_df=maj_df,
                    minority_df=min_df,
                    **kwargs,
                )

                out_file = method_dir / f"{patient_id}_synthetic.xlsx"

                save_result_xlsx(out_file, result)

                print("Saved:", out_file)

            except Exception as e:

                print(
                    f"FAILED: method={method_name}, patient={patient_id}, error={e}"
                )

                errors.append(
                    {
                        "method": method_name,
                        "patient_id": patient_id,
                        "error": str(e),
                    }
                )

    if errors:

        err_df = pd.DataFrame(errors)

        err_file = out_dir / "generation_errors.xlsx"

        err_df.to_excel(err_file, index=False)

        print("Saved errors:", err_file)


if __name__ == "__main__":
    main()

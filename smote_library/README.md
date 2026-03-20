smote_library_v1

This package provides a standalone Python library with 10 callable oversampling functions.

Included functions
- gamma_smote
- sdd_smote
- c_gmc_gamma
- c_gmc_sdd
- c_kmeans_gamma
- c_kmeans_sdd
- gc_gmc_gamma
- gc_gmc_sdd
- gc_kmeans_gamma
- gc_kmeans_sdd

Return type
Each function returns a GenerationResult object with:
- synthetic_df
- metadata_df
- cluster_df
- info

Expected inputs
- majority_df: pandas DataFrame with numeric features
- minority_df: pandas DataFrame with numeric features

Basic example

from smote_library_v1 import gc_gmc_gamma
import pandas as pd

maj = pd.read_excel("SB-001_majority.xlsx")
mino = pd.read_excel("SB-001_minority.xlsx")

result = gc_gmc_gamma(majority_df=maj, minority_df=mino, theta=0.8, k=8, random_state=42)

synthetic = result.synthetic_df
metadata = result.metadata_df
clusters = result.cluster_df
info = result.info

Notes
- Euclidean rejection filtering is applied.
- Features are clipped to [0, 1] by default.
- If target_n is omitted, the default is len(majority_df) - len(minority_df).


Batch Synthetic Data Generation Script



This package also includes a helper script:

run_smote_library_generate_synthetics.py

This script generates synthetic minority samples for one or more patients using any SMOTE method implemented in smote_library_v1.py.

The script reads majority and minority data files from a data folder and writes the generated synthetic samples to method-specific folders.

Expected input files

Input files should follow the naming pattern:

SB-001_majority.xlsx
SB-001_minority.xlsx
SB-002_majority.xlsx
SB-002_minority.xlsx
...

All files should be placed in a single input directory.

Each file should contain numeric feature columns consistent across majority and minority datasets.

Output structure

For each method, the script creates a directory under the specified output folder.

Example:

outputs/

C-GMC-Gamma/
    SB-001_synthetic.xlsx

GC-GMC-Gamma/
    SB-001_synthetic.xlsx

Gamma-SMOTE/
    SB-001_synthetic.xlsx

Each output Excel file contains the following sheets:

synthetic_data
generation_metadata
clusters
info

Descriptions:

synthetic_data – generated synthetic minority samples

generation_metadata – metadata describing generation details

clusters – cluster statistics (for C-SMOTE and GC-SMOTE methods)

info – run parameters and summary statistics

Running the script

Run one SMOTE method

Example:

python run_smote_library_generate_synthetics.py \
--input_dir ./data \
--out_dir ./outputs \
--method C-GMC-Gamma \
--k 8
Run a GC-SMOTE method

Example:

python run_smote_library_generate_synthetics.py \
--input_dir ./data \
--out_dir ./outputs \
--method GC-GMC-Gamma \
--theta 0.8 \
--k 8

Parameters:

theta   minority density threshold for GC-SMOTE
k       number of clusters for C-SMOTE or GC-SMOTE
Run all methods

To generate synthetic data using all available methods:

python run_smote_library_generate_synthetics.py \
--input_dir ./data \
--out_dir ./outputs \
--run_all \
--theta 0.8 \
--k 8
Methods supported

The following oversampling methods are implemented in smote_library_v1.py:

Gamma-SMOTE
SDD-SMOTE

C-GMC-Gamma
C-GMC-SDD
C-KMeans-Gamma
C-KMeans-SDD

GC-GMC-Gamma
GC-GMC-SDD
GC-KMeans-Gamma
GC-KMeans-SDD

These correspond to:

SMOTE without clustering

C-SMOTE (clustering on minority samples)

GC-SMOTE (clustering on the full dataset with minority density filtering)

Error logging

If any patient-method run fails, the script records the error in:

generation_errors.xlsx

in the output directory.

Notes

All features should be normalized to [0,1] prior to generation.

The default number of generated synthetic samples equals:

len(majority_df) - len(minority_df)

unless overridden in the library functions.

Euclidean distance–based rejection filtering is applied to avoid generating samples closer to majority samples than minority samples.

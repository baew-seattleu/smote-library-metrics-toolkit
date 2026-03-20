
#!/usr/bin/env python3
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import entropy, gaussian_kde

def common_numeric_columns(df1, df2, df3):
    cols = sorted(set(df1.columns) & set(df2.columns) & set(df3.columns))
    valid = []
    for c in cols:
        try:
            pd.to_numeric(df1[c], errors="raise")
            pd.to_numeric(df2[c], errors="raise")
            pd.to_numeric(df3[c], errors="raise")
            valid.append(c)
        except Exception:
            pass
    return valid

def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def mean_difference_pct(mi, sm, eps=1e-12):
    return float(np.mean(np.abs((sm.mean(0) - mi.mean(0)) / np.maximum(np.abs(mi.mean(0)), eps)) * 100))

def std_difference_pct(mi, sm, eps=1e-12):
    return float(np.mean(np.abs((sm.std(0) - mi.std(0)) / np.maximum(np.abs(mi.std(0)), eps)) * 100))

def average_kl_divergence(mi, sm, bins=25):
    edges = np.linspace(0, 1, bins)
    vals = []
    for j in range(mi.shape[1]):
        hr, _ = np.histogram(mi[:, j], bins=edges, density=True)
        hs, _ = np.histogram(sm[:, j], bins=edges, density=True)
        hr += 1e-12
        hs += 1e-12
        vals.append(entropy(hr, hs))
    return float(np.mean(vals))

def average_kde_area_difference(mi, sm, grid_points=256, jitter=1e-8):
    """
    Robust KDE area difference.
    Handles singular covariance and constant features.
    """

    import numpy as np
    from scipy.stats import gaussian_kde

    vals = []

    for j in range(mi.shape[1]):

        x = np.asarray(mi[:, j], dtype=float)
        y = np.asarray(sm[:, j], dtype=float)

        xy = np.concatenate([x, y])

        lower = float(np.nanmin(xy))
        upper = float(np.nanmax(xy))
        span = upper - lower

        # constant column
        if span <= 1e-12:
            vals.append(0.0)
            continue

        pad = max(0.05 * span, 1e-6)
        xs = np.linspace(lower - pad, upper + pad, grid_points)

        # attempt KDE
        try:
            kx = gaussian_kde(x)
            ky = gaussian_kde(y)

            area = np.trapz(np.abs(kx(xs) - ky(xs)), xs)
            vals.append(float(area))
            continue

        except Exception:
            pass

        # retry with small jitter
        try:

            rng = np.random.default_rng(42 + j)

            xj = x + rng.normal(0.0, jitter, len(x))
            yj = y + rng.normal(0.0, jitter, len(y))

            kx = gaussian_kde(xj)
            ky = gaussian_kde(yj)

            area = np.trapz(np.abs(kx(xs) - ky(xs)), xs)

            vals.append(float(area))
            continue

        except Exception:
            pass

        # histogram fallback (never return false zero)
        edges = np.linspace(lower - pad, upper + pad, 200)

        hx, _ = np.histogram(x, bins=edges, density=True)
        hy, _ = np.histogram(y, bins=edges, density=True)

        mids = (edges[:-1] + edges[1:]) / 2

        area = np.trapz(np.abs(hx - hy), mids)

        vals.append(float(area))

    return float(np.mean(vals))



def hassanat_distance_matrix(A, B):
    A3 = A[:, None, :]
    B3 = B[None, :, :]
    mn = np.minimum(A3, B3)
    mx = np.maximum(A3, B3)
    D = np.where(
        mn >= 0,
        1 - (1 + mn) / (1 + mx),
        1 - (1 + mn + np.abs(mn)) / (1 + mx + np.abs(mn)),
    )
    return D.sum(axis=2)

def gir_euclidean(sm, mi, ma):
    d_sm_mi = cdist(sm, mi)
    d_sm_ma = cdist(sm, ma)
    d_sm_mi[d_sm_mi == 0] = np.inf
    d_sm_ma[d_sm_ma == 0] = np.inf
    dmi = d_sm_mi.min(axis=1)
    dma = d_sm_ma.min(axis=1)
    return float(np.mean(dma < dmi))

def gir_hassanat(sm, mi, ma):
    hd_mi = hassanat_distance_matrix(sm, mi)
    hd_ma = hassanat_distance_matrix(sm, ma)
    hd_mi[hd_mi == 0] = np.inf
    hd_ma[hd_ma == 0] = np.inf
    dmi = hd_mi.min(axis=1)
    dma = hd_ma.min(axis=1)
    return float(np.mean(dma < dmi))

def compute_metrics(majority_path, minority_path, synthetic_path):
    maj_df = load_table(majority_path)
    mi_df = load_table(minority_path)
    sm_df = load_table(synthetic_path)
    cols = common_numeric_columns(maj_df, mi_df, sm_df)
    if not cols:
        raise ValueError("No common numeric columns found.")
    maj = maj_df[cols].astype(float).to_numpy()
    mi = mi_df[cols].astype(float).to_numpy()
    sm = sm_df[cols].astype(float).to_numpy()
    return {
        "MeanDiff_pct": mean_difference_pct(mi, sm),
        "StdDiff_pct": std_difference_pct(mi, sm),
        "KL": average_kl_divergence(mi, sm),
        "KDE_AreaDiff": average_kde_area_difference(mi, sm),
        "GIR_ED": gir_euclidean(sm, mi, maj),
        "GIR_HD": gir_hassanat(sm, mi, maj),
    }

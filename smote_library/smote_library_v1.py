#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

@dataclass
class GenerationResult:
    synthetic_df: pd.DataFrame
    metadata_df: pd.DataFrame
    cluster_df: Optional[pd.DataFrame]
    info: Dict[str, float]

def _as_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(float).copy()

def _default_target_n(majority_df: pd.DataFrame, minority_df: pd.DataFrame) -> int:
    return max(0, len(majority_df) - len(minority_df))

def _gamma_step(anchor: np.ndarray, neighbor: np.ndarray, rng: np.random.Generator,
                gamma_alpha: float, gamma_scale: float) -> Tuple[np.ndarray, float]:
    mode = gamma_scale * (gamma_alpha - 1.0)
    t = float(rng.gamma(shape=gamma_alpha, scale=gamma_scale))
    return anchor + (t - mode) * (neighbor - anchor), t

def _sdd_step(anchor: np.ndarray, neighbor: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    lam = float(rng.random())
    return anchor + lam * (neighbor - anchor), lam

def _make_rejection_helpers(X_ma: np.ndarray, X_mi: np.ndarray):
    nn_majority = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_ma)
    nn_minority = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_mi)
    def d_majority(batch: np.ndarray) -> np.ndarray:
        return nn_majority.kneighbors(batch, return_distance=True)[0][:, 0]
    def d_minority(batch: np.ndarray) -> np.ndarray:
        d = nn_minority.kneighbors(batch, return_distance=True)[0]
        return np.where(d[:, 0] <= 1e-12, d[:, 1], d[:, 0])
    return d_majority, d_minority

def _allocate_quotas(weights: np.ndarray, target_n: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    raw = target_n * weights
    quotas = np.floor(raw).astype(int)
    remainder = target_n - int(quotas.sum())
    if remainder > 0:
        frac = raw - quotas
        for i in np.argsort(-frac)[:remainder]:
            quotas[i] += 1
    return quotas

def _cluster_gmc(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    model = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state, reg_covar=1e-6, max_iter=200)
    return model.fit_predict(X)

def _cluster_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    return model.fit_predict(X)

def _generate_no_cluster(majority_df: pd.DataFrame, minority_df: pd.DataFrame, generator_name: str,
    target_n: Optional[int] = None, random_state: int = 42, gamma_alpha: float = 2.0,
    gamma_scale: float = 0.125, neighbor_k: int = 3, clip_min: float = 0.0, clip_max: float = 1.0) -> GenerationResult:
    maj_df = _as_numeric_df(majority_df); min_df = _as_numeric_df(minority_df)
    X_ma = maj_df.to_numpy(float); X_mi = min_df.to_numpy(float)
    if len(X_mi) < 2: raise ValueError('At least 2 minority samples are required.')
    if target_n is None: target_n = _default_target_n(maj_df, min_df)
    if target_n <= 0: raise ValueError('target_n must be positive, or majority count must exceed minority count.')
    rng = np.random.default_rng(random_state)
    n_neighbors = min(neighbor_k + 1, len(X_mi))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(X_mi)
    d_majority, d_minority = _make_rejection_helpers(X_ma, X_mi)
    synthetic_rows, meta_rows = [], []
    accepted = 0; attempts = 0; max_attempts = max(2000, target_n * 80)
    while accepted < target_n and attempts < max_attempts:
        batch_n = min(256, max(32, target_n - accepted + 16))
        batch_rows, batch_meta = [], []
        for _ in range(batch_n):
            i = int(rng.integers(0, len(X_mi)))
            inds = nbrs.kneighbors(X_mi[i].reshape(1, -1), return_distance=False)[0]
            pool = inds[1:] if len(inds) > 1 else inds
            j = int(rng.choice(pool))
            if generator_name == 'Gamma':
                q, gen_param = _gamma_step(X_mi[i], X_mi[j], rng, gamma_alpha, gamma_scale); meta_name = 'gamma_t'
            else:
                q, gen_param = _sdd_step(X_mi[i], X_mi[j], rng); meta_name = 'lambda'
            q = np.clip(q, clip_min, clip_max)
            batch_rows.append(q); batch_meta.append((i, j, gen_param, meta_name))
        batch = np.asarray(batch_rows, dtype=float)
        dmj = d_majority(batch); dmi = d_minority(batch); keep = dmj >= dmi
        for row, (ai, nj, gen_param, meta_name), a, b, flag in zip(batch, batch_meta, dmj, dmi, keep):
            attempts += 1
            if not flag: continue
            synthetic_rows.append(row)
            md = {'anchor_local_index': int(ai), 'neighbor_local_index': int(nj), meta_name: float(gen_param),
                  'nearest_majority_ed': float(a), 'nearest_minority_ed': float(b), 'margin_ed': float(a - b)}
            if meta_name == 'lambda': md['gamma_t'] = np.nan
            else: md['lambda'] = np.nan
            meta_rows.append(md); accepted += 1
            if accepted >= target_n: break
    if accepted < target_n: raise RuntimeError(f'Only generated {accepted}/{target_n} accepted samples.')
    syn_df = pd.DataFrame(np.asarray(synthetic_rows), columns=min_df.columns)
    meta_df = pd.DataFrame(meta_rows)
    info = {'target_n': int(target_n), 'overall_attempts': int(attempts), 'overall_acceptance_rate': float(accepted / attempts) if attempts else np.nan,
            'gamma_alpha': float(gamma_alpha), 'gamma_scale': float(gamma_scale), 'neighbor_k': int(neighbor_k)}
    return GenerationResult(synthetic_df=syn_df, metadata_df=meta_df, cluster_df=None, info=info)

def _generate_c_smote(majority_df: pd.DataFrame, minority_df: pd.DataFrame, k: int,
    clusterer: Callable[[np.ndarray, int, int], np.ndarray], generator_name: str, target_n: Optional[int] = None,
    random_state: int = 42, gamma_alpha: float = 2.0, gamma_scale: Optional[float] = None,
    neighbor_k: int = 3, clip_min: float = 0.0, clip_max: float = 1.0) -> GenerationResult:
    maj_df = _as_numeric_df(majority_df); min_df = _as_numeric_df(minority_df)
    X_ma = maj_df.to_numpy(float); X_mi = min_df.to_numpy(float)
    if target_n is None: target_n = _default_target_n(maj_df, min_df)
    if target_n <= 0: raise ValueError('target_n must be positive, or majority count must exceed minority count.')
    if k < 2: raise ValueError('k must be at least 2.')
    if k > len(X_mi): raise ValueError(f'k={k} is larger than the number of minority samples ({len(X_mi)}).')
    if gamma_scale is None: gamma_scale = 1.0 / k
    labels = clusterer(X_mi, k, random_state)
    clusters = []
    for cid in range(int(np.max(labels)) + 1):
        idx = np.where(labels == cid)[0]
        if len(idx) >= 2:
            clusters.append({'cluster_id': int(cid), 'n_all': int(len(idx)), 'n_min': int(len(idx)), 'n_maj': 0, 'minority_prop': 1.0, 'minority_indices': idx})
    if not clusters: raise RuntimeError(f'No minority clusters with at least 2 samples found for k={k}.')
    quotas = _allocate_quotas(np.array([c['n_min'] for c in clusters], dtype=float), target_n)
    d_majority, d_minority = _make_rejection_helpers(X_ma, X_mi)
    rng = np.random.default_rng(random_state)
    synthetic_rows, meta_rows, cluster_rows = [], [], []
    for cluster, quota in zip(clusters, quotas):
        idx_min = cluster['minority_indices']; Xc = X_mi[idx_min]
        n_neighbors = min(neighbor_k + 1, len(Xc))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(Xc)
        accepted = 0; attempts = 0; max_attempts = max(1000, quota * 60)
        while accepted < quota and attempts < max_attempts:
            batch_n = min(256, max(32, quota - accepted + 16))
            batch_rows, batch_meta = [], []
            for _ in range(batch_n):
                i = int(rng.integers(0, len(Xc)))
                inds = nbrs.kneighbors(Xc[i].reshape(1, -1), return_distance=False)[0]
                pool = inds[1:] if len(inds) > 1 else inds
                j = int(rng.choice(pool))
                if generator_name == 'Gamma':
                    q, gen_param = _gamma_step(Xc[i], Xc[j], rng, gamma_alpha, gamma_scale); meta_name = 'gamma_t'
                else:
                    q, gen_param = _sdd_step(Xc[i], Xc[j], rng); meta_name = 'lambda'
                q = np.clip(q, clip_min, clip_max)
                batch_rows.append(q); batch_meta.append((i, j, gen_param, meta_name))
            batch = np.asarray(batch_rows, dtype=float)
            dmj = d_majority(batch); dmi = d_minority(batch); keep = dmj >= dmi
            for row, (ai, nj, gen_param, meta_name), a, b, flag in zip(batch, batch_meta, dmj, dmi, keep):
                attempts += 1
                if not flag: continue
                synthetic_rows.append(row)
                md = {'cluster_id': cluster['cluster_id'], 'cluster_minority_prop': cluster['minority_prop'],
                      'anchor_local_index': int(ai), 'neighbor_local_index': int(nj), meta_name: float(gen_param),
                      'nearest_majority_ed': float(a), 'nearest_minority_ed': float(b), 'margin_ed': float(a - b)}
                if meta_name == 'lambda': md['gamma_t'] = np.nan
                else: md['lambda'] = np.nan
                meta_rows.append(md); accepted += 1
                if accepted >= quota: break
        cluster_rows.append({'cluster_id': cluster['cluster_id'], 'n_all': cluster['n_all'], 'n_min': cluster['n_min'],
                             'n_maj': cluster['n_maj'], 'minority_prop': cluster['minority_prop'], 'quota': int(quota),
                             'accepted': int(accepted), 'attempts': int(attempts),
                             'acceptance_rate': float(accepted / attempts) if attempts else np.nan})
        if accepted < quota: raise RuntimeError(f"Cluster {cluster['cluster_id']} only generated {accepted}/{quota} accepted samples.")
    syn_df = pd.DataFrame(np.asarray(synthetic_rows), columns=min_df.columns)
    meta_df = pd.DataFrame(meta_rows); cluster_df = pd.DataFrame(cluster_rows)
    info = {'k': int(k), 'target_n': int(target_n), 'cluster_count': int(len(clusters)),
            'overall_attempts': int(cluster_df['attempts'].sum()),
            'overall_acceptance_rate': float(cluster_df['accepted'].sum() / cluster_df['attempts'].sum()),
            'gamma_alpha': float(gamma_alpha), 'gamma_scale': float(gamma_scale), 'neighbor_k': int(neighbor_k)}
    return GenerationResult(synthetic_df=syn_df, metadata_df=meta_df, cluster_df=cluster_df, info=info)

def _generate_gc_smote(majority_df: pd.DataFrame, minority_df: pd.DataFrame, theta: float, k: int,
    clusterer: Callable[[np.ndarray, int, int], np.ndarray], generator_name: str, target_n: Optional[int] = None,
    random_state: int = 42, gamma_alpha: float = 2.0, gamma_scale: Optional[float] = None,
    neighbor_k: int = 3, clip_min: float = 0.0, clip_max: float = 1.0) -> GenerationResult:
    maj_df = _as_numeric_df(majority_df); min_df = _as_numeric_df(minority_df)
    X_ma = maj_df.to_numpy(float); X_mi = min_df.to_numpy(float); X_all = np.vstack([X_ma, X_mi])
    y_all = np.array([0] * len(X_ma) + [1] * len(X_mi))
    if target_n is None: target_n = _default_target_n(maj_df, min_df)
    if target_n <= 0: raise ValueError('target_n must be positive, or majority count must exceed minority count.')
    if k < 2: raise ValueError('k must be at least 2.')
    if gamma_scale is None: gamma_scale = 1.0 / k
    labels = clusterer(X_all, k, random_state)
    clusters = []
    for cid in range(int(np.max(labels)) + 1):
        idx = np.where(labels == cid)[0]; n_all = len(idx)
        if n_all == 0: continue
        n_min = int(np.sum(y_all[idx] == 1)); prop = n_min / n_all
        if n_min >= 2 and prop > theta:
            idx_min_local = idx[y_all[idx] == 1] - len(X_ma)
            clusters.append({'cluster_id': int(cid), 'n_all': int(n_all), 'n_min': int(n_min),
                             'n_maj': int(n_all - n_min), 'minority_prop': float(prop), 'minority_indices': idx_min_local})
    if not clusters: raise RuntimeError(f'No eligible clusters found for theta={theta}, k={k}.')
    quotas = _allocate_quotas(np.array([c['n_min'] for c in clusters], dtype=float), target_n)
    d_majority, d_minority = _make_rejection_helpers(X_ma, X_mi)
    rng = np.random.default_rng(random_state)
    synthetic_rows, meta_rows, cluster_rows = [], [], []
    for cluster, quota in zip(clusters, quotas):
        idx_min = cluster['minority_indices']; Xc = X_mi[idx_min]
        n_neighbors = min(neighbor_k + 1, len(Xc))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(Xc)
        accepted = 0; attempts = 0; max_attempts = max(1000, quota * 60)
        while accepted < quota and attempts < max_attempts:
            batch_n = min(256, max(32, quota - accepted + 16))
            batch_rows, batch_meta = [], []
            for _ in range(batch_n):
                i = int(rng.integers(0, len(Xc)))
                inds = nbrs.kneighbors(Xc[i].reshape(1, -1), return_distance=False)[0]
                pool = inds[1:] if len(inds) > 1 else inds
                j = int(rng.choice(pool))
                if generator_name == 'Gamma':
                    q, gen_param = _gamma_step(Xc[i], Xc[j], rng, gamma_alpha, gamma_scale); meta_name = 'gamma_t'
                else:
                    q, gen_param = _sdd_step(Xc[i], Xc[j], rng); meta_name = 'lambda'
                q = np.clip(q, clip_min, clip_max)
                batch_rows.append(q); batch_meta.append((i, j, gen_param, meta_name))
            batch = np.asarray(batch_rows, dtype=float)
            dmj = d_majority(batch); dmi = d_minority(batch); keep = dmj >= dmi
            for row, (ai, nj, gen_param, meta_name), a, b, flag in zip(batch, batch_meta, dmj, dmi, keep):
                attempts += 1
                if not flag: continue
                synthetic_rows.append(row)
                md = {'cluster_id': cluster['cluster_id'], 'cluster_minority_prop': cluster['minority_prop'],
                      'anchor_local_index': int(ai), 'neighbor_local_index': int(nj), meta_name: float(gen_param),
                      'nearest_majority_ed': float(a), 'nearest_minority_ed': float(b), 'margin_ed': float(a - b)}
                if meta_name == 'lambda': md['gamma_t'] = np.nan
                else: md['lambda'] = np.nan
                meta_rows.append(md); accepted += 1
                if accepted >= quota: break
        cluster_rows.append({'cluster_id': cluster['cluster_id'], 'n_all': cluster['n_all'], 'n_min': cluster['n_min'],
                             'n_maj': cluster['n_maj'], 'minority_prop': cluster['minority_prop'], 'quota': int(quota),
                             'accepted': int(accepted), 'attempts': int(attempts),
                             'acceptance_rate': float(accepted / attempts) if attempts else np.nan})
        if accepted < quota: raise RuntimeError(f"Cluster {cluster['cluster_id']} only generated {accepted}/{quota} accepted samples.")
    syn_df = pd.DataFrame(np.asarray(synthetic_rows), columns=min_df.columns)
    meta_df = pd.DataFrame(meta_rows); cluster_df = pd.DataFrame(cluster_rows)
    info = {'theta': float(theta), 'k': int(k), 'target_n': int(target_n), 'eligible_cluster_count': int(len(clusters)),
            'overall_attempts': int(cluster_df['attempts'].sum()),
            'overall_acceptance_rate': float(cluster_df['accepted'].sum() / cluster_df['attempts'].sum()),
            'gamma_alpha': float(gamma_alpha), 'gamma_scale': float(gamma_scale), 'neighbor_k': int(neighbor_k)}
    return GenerationResult(synthetic_df=syn_df, metadata_df=meta_df, cluster_df=cluster_df, info=info)

def gamma_smote(majority_df: pd.DataFrame, minority_df: pd.DataFrame, **kwargs) -> GenerationResult:
    return _generate_no_cluster(majority_df, minority_df, generator_name='Gamma', **kwargs)

def sdd_smote(majority_df: pd.DataFrame, minority_df: pd.DataFrame, **kwargs) -> GenerationResult:
    return _generate_no_cluster(majority_df, minority_df, generator_name='SDD', **kwargs)

def c_gmc_gamma(majority_df: pd.DataFrame, minority_df: pd.DataFrame, k: int, **kwargs) -> GenerationResult:
    return _generate_c_smote(majority_df, minority_df, k=k, clusterer=_cluster_gmc, generator_name='Gamma', **kwargs)

def c_gmc_sdd(majority_df: pd.DataFrame, minority_df: pd.DataFrame, k: int, **kwargs) -> GenerationResult:
    return _generate_c_smote(majority_df, minority_df, k=k, clusterer=_cluster_gmc, generator_name='SDD', **kwargs)

def c_kmeans_gamma(majority_df: pd.DataFrame, minority_df: pd.DataFrame, k: int, **kwargs) -> GenerationResult:
    return _generate_c_smote(majority_df, minority_df, k=k, clusterer=_cluster_kmeans, generator_name='Gamma', **kwargs)

def c_kmeans_sdd(majority_df: pd.DataFrame, minority_df: pd.DataFrame, k: int, **kwargs) -> GenerationResult:
    return _generate_c_smote(majority_df, minority_df, k=k, clusterer=_cluster_kmeans, generator_name='SDD', **kwargs)

def gc_gmc_gamma(majority_df: pd.DataFrame, minority_df: pd.DataFrame, theta: float, k: int, **kwargs) -> GenerationResult:
    return _generate_gc_smote(majority_df, minority_df, theta=theta, k=k, clusterer=_cluster_gmc, generator_name='Gamma', **kwargs)

def gc_gmc_sdd(majority_df: pd.DataFrame, minority_df: pd.DataFrame, theta: float, k: int, **kwargs) -> GenerationResult:
    return _generate_gc_smote(majority_df, minority_df, theta=theta, k=k, clusterer=_cluster_gmc, generator_name='SDD', **kwargs)

def gc_kmeans_gamma(majority_df: pd.DataFrame, minority_df: pd.DataFrame, theta: float, k: int, **kwargs) -> GenerationResult:
    return _generate_gc_smote(majority_df, minority_df, theta=theta, k=k, clusterer=_cluster_kmeans, generator_name='Gamma', **kwargs)

def gc_kmeans_sdd(majority_df: pd.DataFrame, minority_df: pd.DataFrame, theta: float, k: int, **kwargs) -> GenerationResult:
    return _generate_gc_smote(majority_df, minority_df, theta=theta, k=k, clusterer=_cluster_kmeans, generator_name='SDD', **kwargs)

FUNCTION_REGISTRY = {
    'Gamma-SMOTE': gamma_smote,
    'SDD-SMOTE': sdd_smote,
    'C-GMC-Gamma': c_gmc_gamma,
    'C-GMC-SDD': c_gmc_sdd,
    'C-KMeans-Gamma': c_kmeans_gamma,
    'C-KMeans-SDD': c_kmeans_sdd,
    'GC-GMC-Gamma': gc_gmc_gamma,
    'GC-GMC-SDD': gc_gmc_sdd,
    'GC-KMeans-Gamma': gc_kmeans_gamma,
    'GC-KMeans-SDD': gc_kmeans_sdd,
}

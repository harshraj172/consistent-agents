from __future__ import annotations
 
import argparse
import json
import math
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
 
import numpy as np
from scipy.stats import norm as _norm
from sklearn.cluster import SpectralClustering
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as _plt

from consistent_agents.trajectory_metrics import (  
        _action_sequence,
        _levenshtein_distance,
        _read_json,
        _write_json,
    )


def _normalized_levenshtein(a: Sequence[str], b: Sequence[str]) -> float:
    """Normalized Levenshtein distance d_lev(tau, tau') / max(T, T') in [0, 1]."""
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return _levenshtein_distance(a, b) / float(denom)

def _log_gak(
    a: Sequence[str],
    b: Sequence[str],
    lam: float,
) -> float:
    """
    Log of the unnormalized Global Alignment Kernel between two action
    sequences using "halved-and-normalized" form:
 
        kappa(x, y) = exp(-lam * 1[x != y])      (1 if equal, e^{-lam} else)
 
    Returns log K(a, b). Empty-vs-empty returns log 1 = 0; exactly-one-empty
    returns -inf so the normalization can fall back cleanly.
    """
    Ta, Tb = len(a), len(b)
    if Ta == 0 and Tb == 0:
        return 0.0
    if Ta == 0 or Tb == 0:
        return -math.inf
 
    NEG_INF = -math.inf
 
    log_kappa_eq = 0.0  
    e_neg_lam = math.exp(-lam)
    log_kappa_neq = math.log(e_neg_lam / (2.0 - e_neg_lam))
 
    # log_M with boundary: M[0,0] = 1, M[0, j>=1] = M[i>=1, 0] = 0.
    prev = [NEG_INF] * (Tb + 1)
    prev[0] = 0.0
 
    for i in range(1, Ta + 1):
        curr = [NEG_INF] * (Tb + 1)
        ai = a[i - 1]
        for j in range(1, Tb + 1):
            log_k = log_kappa_eq if ai == b[j - 1] else log_kappa_neq
            v1 = prev[j - 1]   
            v2 = prev[j]       
            v3 = curr[j - 1]   
            m = max(v1, v2, v3)
            if m == NEG_INF:
                lse = NEG_INF
            else:
                s = 0.0
                if v1 != NEG_INF:
                    s += math.exp(v1 - m)
                if v2 != NEG_INF:
                    s += math.exp(v2 - m)
                if v3 != NEG_INF:
                    s += math.exp(v3 - m)
                lse = m + math.log(s)
            curr[j] = log_k + lse
        prev = curr
    return prev[Tb]
 
 
def _normalized_gak_matrix(
    sequences: Sequence[Sequence[str]],
    lam: float,
) -> np.ndarray:
    """
    Normalized GAK kernel matrix:
        Kbar[i, j] = K(t_i, t_j) / sqrt(K(t_i, t_i) * K(t_j, t_j))
    """
    n = len(sequences)
    log_self = np.full(n, -math.inf, dtype=np.float64)
    for i in range(n):
        log_self[i] = _log_gak(sequences[i], sequences[i], lam)
 
    K = np.eye(n, dtype=np.float64)
    for i, j in combinations(range(n), 2):
        si, sj = sequences[i], sequences[j]
        if len(si) == 0 and len(sj) == 0:
            K[i, j] = K[j, i] = 1.0
            continue
        if len(si) == 0 or len(sj) == 0:
            K[i, j] = K[j, i] = 0.0
            continue
 
        log_kij = _log_gak(si, sj, lam)
        log_kbar = log_kij - 0.5 * (log_self[i] + log_self[j])
        kbar = math.exp(log_kbar) if log_kbar != -math.inf else 0.0
        kbar = max(0.0, min(1.0, kbar))  # clamp tiny FP overshoot
        K[i, j] = K[j, i] = kbar
    return K

#############Spectral Clustering##################

def _eigengap_num_clusters(
    K: np.ndarray,
    *,
    max_clusters: Optional[int] = None,
    gap_tol: float = 1e-9,
) -> Tuple[int, List[float]]:
    """
    Pick K_hat by the eigengap heuristic on the symmetric normalized Laplacian
    L_sym = I - D^{-1/2} K D^{-1/2}.
 
    Returns (K_hat, eigenvalues_used).
    """
    n = K.shape[0]
    if n <= 1:
        return n, [0.0] * n

    A = np.maximum((K + K.T) / 2.0, 0.0)
 
    d = A.sum(axis=1)
    d_safe = np.where(d > 0, d, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    L_sym = np.eye(n) - (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    L_sym = (L_sym + L_sym.T) / 2.0  # FP symmetrize
 
    eigvals = np.linalg.eigvalsh(L_sym)  # ascending
    eigvals = np.clip(eigvals, 0.0, None)
 
    if max_clusters is None:
        max_clusters = n
    max_clusters = max(1, min(max_clusters, n))
 
    upper = min(max_clusters, n - 1)
    if upper < 1:
        return 1, eigvals.tolist()
 
    gaps = np.diff(eigvals[: upper + 1])
    if gaps.size == 0 or float(np.max(gaps)) < gap_tol:
        return 1, eigvals.tolist()
 
    # Gap at index k is between eigval[k] and eigval[k+1], implying K = k+1.
    k_hat = int(np.argmax(gaps)) + 1
    return k_hat, eigvals.tolist()
 
 
def _spectral_cluster(
    K: np.ndarray,
    k_hat: int,
    *,
    random_state: int = 0,
) -> np.ndarray:
    """Spectral clustering on a precomputed affinity matrix."""
    n = K.shape[0]
    if k_hat <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int64)
    if k_hat >= n:
        return np.arange(n, dtype=np.int64)
 
    A = np.maximum((K + K.T) / 2.0, 0.0)
    sc = SpectralClustering(
        n_clusters=k_hat,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return sc.fit_predict(A).astype(np.int64)
    

#########Observe within cluster consistency###############

def _within_cluster_metrics(
    sequences: Sequence[Sequence[str]],
    K: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    For one instance, compute:
      - per-cluster d_bar_k^m            (eq. 8) : mean pairwise normalized Lev
      - per-cluster V_hat_k^{m,(k)}      (eq. 9) : 1 - mean within-cluster kbar
      - mixture weights w_k^m = n_k / (n+1)
      - aggregated V_hat_k^{m,within}    (eq. 10)
      - aggregated d_bar^{m,within}              : mixture-weighted mean of d_bar_k
 
    Singleton clusters contribute 0 to within-cluster summaries (no pairs);
    their mixture weight still applies.
    """
    n = len(sequences)
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)
 
    per_cluster: Dict[int, Dict[str, Any]] = {}
    v_within = 0.0
    d_within = 0.0
 
    for c_id, members in sorted(clusters.items()):
        n_k = len(members)
        w_k = n_k / float(n)
        if n_k < 2:
            d_bar_k = 0.0
            v_hat_k = 0.0
            num_pairs = 0
        else:
            pairs = list(combinations(members, 2))
            num_pairs = len(pairs)
            sum_lev = 0.0
            sum_kbar = 0.0
            for i, j in pairs:
                sum_lev += _normalized_levenshtein(sequences[i], sequences[j])
                sum_kbar += float(K[i, j])
            d_bar_k = sum_lev / num_pairs
            v_hat_k = 1.0 - (sum_kbar / num_pairs)
 
        per_cluster[c_id] = {
            "size": n_k,
            "weight": w_k,
            "num_pairs": num_pairs,
            "d_bar": d_bar_k,
            "V_hat": v_hat_k,
            "members": members,
        }
        v_within += w_k * v_hat_k
        d_within += w_k * d_bar_k
 
    return {
        "K_hat": len(clusters),
        "per_cluster": per_cluster,
        "V_within": v_within,
        "d_within": d_within,
    }

def _median_heuristic_lambda(
    sequences_by_instance: Sequence[Sequence[Sequence[str]]],
    *,
    max_pairs: int = 2000,
    rng_seed: int = 0,
) -> float:
    """
    lambda = 1 / median(d_lev) where the median is taken over a random
    sample of within-instance pairwise normalized Levenshtein distances.
    """
    rng = np.random.default_rng(rng_seed)
    pool: List[float] = []
    for seqs in sequences_by_instance:
        if len(seqs) < 2:
            continue
        for i, j in combinations(range(len(seqs)), 2):
            pool.append(_normalized_levenshtein(seqs[i], seqs[j]))
    if not pool:
        return 1.0
    pool_arr = np.asarray(pool, dtype=np.float64)
    if len(pool_arr) > max_pairs:
        idx = rng.choice(len(pool_arr), size=max_pairs, replace=False)
        pool_arr = pool_arr[idx]
    med = float(np.median(pool_arr))
    if med <= 0:
        return 1.0  # all-identical -> arbitrary
    return 1.0 / med


#########Between-cluster separation##################

def _between_cluster_metrics(
    sequences: Sequence[Sequence[str]],
    K: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Between-cluster variance: mixture-weighted mean dissimilarity FROM each
    cluster TO all other clusters.

    For cluster k define:
        kbar_k_to_other = mean_{i in C_k, j in C_l, l != k} kbar_GAK(t_i, t_j)
        V_between^{m,(k)} = 1 - kbar_k_to_other

    Aggregate via mixture weights:
        V_between^m = sum_k w_k * V_between^{m,(k)}
        d_between^m = sum_k w_k * d_bar_k_to_other^{m}    
    """
    n = len(sequences)
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)

    if len(clusters) < 2:
        return {"V_between": 0.0, "d_between": 0.0}

    v_between = 0.0
    d_between = 0.0
    for c_id, members in clusters.items():
        n_k = len(members)
        w_k = n_k / float(n)

        # Sum kbar and Lev between this cluster's points and all OTHER clusters' points.
        sum_kbar = 0.0
        sum_lev = 0.0
        n_pairs = 0
        for i in members:
            for c_other, others in clusters.items():
                if c_other == c_id:
                    continue
                for j in others:
                    sum_kbar += float(K[i, j])
                    sum_lev += _normalized_levenshtein(sequences[i], sequences[j])
                    n_pairs += 1

        if n_pairs == 0:
            v_k_to_other = 0.0
            d_k_to_other = 0.0
        else:
            v_k_to_other = 1.0 - (sum_kbar / n_pairs)
            d_k_to_other = sum_lev / n_pairs

        v_between += w_k * v_k_to_other
        d_between += w_k * d_k_to_other

    return {"V_between": v_between, "d_between": d_between}


#########Silhouette analysis##################

def _silhouette_for_instance(
    sequences: Sequence[Sequence[str]],
    K: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Rousseeuw silhouette per run i in one instance.

      a(i) = mean distance from i to other runs in i's cluster (cohesion)
      b(i) = min over other clusters of (mean distance from i to that cluster) (separation)
      s(i) = (b(i) - a(i)) / max(a(i), b(i))     in [-1, 1]
    """
    n = len(sequences)
    if n < 2:
        return {
            "per_run": [],
            "mean_silhouette_kernel": None,
            "mean_silhouette_lev": None,
        }

    # Pairwise distance matrices (symmetric, zero diagonal).
    D_ker = 1.0 - K
    np.fill_diagonal(D_ker, 0.0)
    D_lev = np.zeros((n, n), dtype=np.float64)
    for i, j in combinations(range(n), 2):
        d = _normalized_levenshtein(sequences[i], sequences[j])
        D_lev[i, j] = D_lev[j, i] = d

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)

    per_run: List[Dict[str, Any]] = []
    s_ker_list: List[float] = []
    s_lev_list: List[float] = []

    for i in range(n):
        own = int(labels[i])
        own_members = [j for j in clusters[own] if j != i]

        # a(i): mean dist to own cluster (excluding self). Singleton => s = 0.
        if not own_members:
            entry = {
                "run_index": i,
                "cluster": own,
                "a_kernel": None, "b_kernel": None, "s_kernel": 0.0,
                "a_lev": None,    "b_lev": None,    "s_lev": 0.0,
            }
            per_run.append(entry)
            s_ker_list.append(0.0)
            s_lev_list.append(0.0)
            continue

        a_ker = float(np.mean([D_ker[i, j] for j in own_members]))
        a_lev = float(np.mean([D_lev[i, j] for j in own_members]))

        # b(i): min over other clusters of mean dist to that cluster.
        b_ker_candidates: List[float] = []
        b_lev_candidates: List[float] = []
        for c_other, others in clusters.items():
            if c_other == own or not others:
                continue
            b_ker_candidates.append(float(np.mean([D_ker[i, j] for j in others])))
            b_lev_candidates.append(float(np.mean([D_lev[i, j] for j in others])))

        if not b_ker_candidates:
            # Only one cluster overall; silhouette undefined -> 0.
            s_ker = 0.0
            s_lev = 0.0
            b_ker = None
            b_lev = None
        else:
            b_ker = min(b_ker_candidates)
            b_lev = min(b_lev_candidates)
            s_ker = (b_ker - a_ker) / max(a_ker, b_ker) if max(a_ker, b_ker) > 0 else 0.0
            s_lev = (b_lev - a_lev) / max(a_lev, b_lev) if max(a_lev, b_lev) > 0 else 0.0

        per_run.append({
            "run_index": i,
            "cluster": own,
            "a_kernel": a_ker, "b_kernel": b_ker, "s_kernel": s_ker,
            "a_lev": a_lev,    "b_lev": b_lev,    "s_lev": s_lev,
        })
        s_ker_list.append(s_ker)
        s_lev_list.append(s_lev)

    return {
        "per_run": per_run,
        "mean_silhouette_kernel": float(np.mean(s_ker_list)),
        "mean_silhouette_lev": float(np.mean(s_lev_list)),
    }


#########K-sweep for elbow plot##################

def _sweep_K_for_instance(
    sequences: Sequence[Sequence[str]],
    K: np.ndarray,
    *,
    random_state: int = 0,
) -> Dict[str, List[float]]:
    """
    For one instance, force K = 1, 2, ..., n and record:
        V_within(K)   eq. (within_vk_instance) at forced K
        V_between(K)  mixture-weighted mean dist to other clusters at forced K
        d_within(K)   Lev. analogue
        d_between(K)  Lev. analogue
        silhouette(K) mean silhouette at forced K (kernel distance)
    """
    n = K.shape[0]
    out = {
        "K_values": list(range(1, n + 1)),
        "V_within": [],
        "V_between": [],
        "d_within": [],
        "d_between": [],
        "mean_silhouette_kernel": [],
    }
    if n < 2:
        return out

    for k_target in range(1, n + 1):
        labels = _spectral_cluster(K, k_target, random_state=random_state)
        wc = _within_cluster_metrics(sequences, K, labels)
        bc = _between_cluster_metrics(sequences, K, labels)
        sil = _silhouette_for_instance(sequences, K, labels)
        out["V_within"].append(wc["V_within"])
        out["V_between"].append(bc["V_between"])
        out["d_within"].append(wc["d_within"])
        out["d_between"].append(bc["d_between"])
        out["mean_silhouette_kernel"].append(
            sil["mean_silhouette_kernel"] if sil["mean_silhouette_kernel"] is not None else 0.0
        )
    return out



plt = None  


def _find_elbow(K_values: Sequence[int], curve: Sequence[float]) -> Optional[int]:
    """
    When multiple K values are
    near-tied for the max distance (within 5% of the best), prefer the
    smallest K.
    """
    K = np.asarray(K_values, dtype=np.float64)
    y = np.asarray(curve, dtype=np.float64)
    if len(K) < 3:
        return None
    p1 = np.array([K[0], y[0]])
    p2 = np.array([K[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return None
    line_unit = line_vec / line_len

    distances = []
    for i in range(len(K)):
        pt = np.array([K[i], y[i]]) - p1
        proj_len = np.dot(pt, line_unit)
        proj = proj_len * line_unit
        perp = pt - proj
        distances.append(np.linalg.norm(perp))
    distances = np.asarray(distances)
    max_d = distances.max()
    if max_d == 0:
        return None
    # Prefer smallest K within 5% of the best distance.
    threshold = 0.95 * max_d
    near_max = np.where(distances >= threshold)[0]
    elbow_idx = int(near_max[0])
    return int(K_values[elbow_idx])


def plot_v_vs_k(result: Dict[str, Any], out_path: Path) -> None:
    """
    V_within and V_between vs K, mean across instances. Elbow marked.
    """
    sweep = result.get("aggregate", {}).get("K_sweep_mean")
    if not sweep:
        return

    K_vals = sweep["K_values"]
    V_within = sweep["V_within_mean"]
    V_between = sweep["V_between_mean"]
    d_within = sweep["d_within_mean"]
    d_between = sweep["d_between_mean"]
    sil = sweep["mean_silhouette_kernel"]

    elbow_within = _find_elbow(K_vals, V_within)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: kernel-distance variance decomposition.
    ax = axes[0]
    ax.plot(K_vals, V_within,  marker="o", label=r"$\overline{V}^{\,\mathrm{within}}_k(K)$ (kernel)")
    ax.plot(K_vals, V_between, marker="s", label=r"$\overline{V}^{\,\mathrm{between}}_k(K)$ (kernel)")
    if elbow_within is not None:
        ax.axvline(elbow_within, color="grey", linestyle="--", alpha=0.6,
                   label=f"elbow @ K={elbow_within}")
    K_bar = result.get("aggregate", {}).get("K_bar")
    if K_bar is not None:
        ax.axvline(K_bar, color="C3", linestyle=":", alpha=0.7,
                   label=fr"$\bar K$ (eigengap) = {K_bar:.2f}")
    ax.set_xlabel("K (forced number of clusters)")
    ax.set_ylabel("Variance (kernel)")
    ax.set_title("Within- and between-cluster variance vs K\n(mean across instances)")
    ax.set_xticks(K_vals)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Levenshtein analogue + silhouette overlay (right axis).
    ax = axes[1]
    ax.plot(K_vals, d_within,  marker="o", color="C0", label=r"$\bar d^{\,\mathrm{within}}(K)$ (Lev.)")
    ax.plot(K_vals, d_between, marker="s", color="C1", label=r"$\bar d^{\,\mathrm{between}}(K)$ (Lev.)")
    ax.set_xlabel("K (forced number of clusters)")
    ax.set_ylabel("Normalized Levenshtein distance")
    ax.set_xticks(K_vals)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(K_vals, sil, marker="^", color="C2", linestyle="--",
             label="mean silhouette (kernel)")
    ax2.set_ylabel("Mean silhouette")
    ax2.axhline(0, color="grey", alpha=0.3)

    # Combined legend.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax.set_title("Levenshtein analogue + mean silhouette vs K")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_silhouette(result: Dict[str, Any], out_path: Path) -> None:
    """
    Aggregated silhouette plot (Rousseeuw style) at each instance's K_hat^m.
    """
    per_inst = result.get("per_example_id", {})

    # Bucket by the instance's K_hat. Each entry: (instance_local_cluster, s).
    by_khat: Dict[int, List[float]] = {}
    by_khat_clusters: Dict[int, List[int]] = {}
    for inst in per_inst.values():
        if inst.get("skipped") or inst.get("silhouette") is None:
            continue
        k_hat = inst["K_hat"]
        for run in inst["silhouette"]["per_run"]:
            by_khat.setdefault(k_hat, []).append(run["s_kernel"])
            by_khat_clusters.setdefault(k_hat, []).append(run["cluster"])

    if not by_khat:
        return

    khats = sorted(by_khat.keys())
    n_panels = len(khats)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, k in zip(axes, khats):
        s_vals = np.array(by_khat[k])
        clust = np.array(by_khat_clusters[k])

        # Sort: cluster ascending, then silhouette descending within cluster.
        order = np.lexsort((-s_vals, clust))
        s_sorted = s_vals[order]
        c_sorted = clust[order]

        # Color by cluster id.
        cmap = plt.get_cmap("tab10")
        colors = [cmap(c % 10) for c in c_sorted]

        y = np.arange(len(s_sorted))
        ax.barh(y, s_sorted, color=colors, edgecolor="none", height=1.0)
        ax.axvline(s_sorted.mean(), color="red", linestyle="--",
                   label=f"mean = {s_sorted.mean():.3f}")
        ax.axvline(0, color="black", alpha=0.4, linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("silhouette s(i)  (kernel distance)")
        ax.set_ylabel(f"runs (n={len(s_sorted)})")
        ax.set_title(f"Instances with $\\hat K^m = {k}$")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_yticks([])

        if k == 1:
            ax.text(
                0.5, 0.5,
                "silhouette undefined for K=1\n(no other cluster to compare to)\nall s(i) := 0",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="grey", style="italic",
            )

    fig.suptitle(
        "Silhouette per run, grouped by instance's $\\hat K^m$\n"
        "a(i) = mean dist to own cluster, b(i) = mean dist to nearest other cluster",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_examples(result: Dict[str, Any], out_path: Path) -> None:
    """
    Per-sample grid: one panel per non-skipped instance showing
    V_within(K), V_between(K), and the eigengap-selected K_hat^m.
    Auto-layout: ceil(sqrt(M)) cols.
    """
    per_inst = result.get("per_example_id", {})
    items = [
        (eid, inst) for eid, inst in per_inst.items()
        if not inst.get("skipped") and inst.get("K_sweep")
    ]
    if not items:
        return

    n = len(items)
    ncols = max(1, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, (ex_id, inst) in zip(axes_flat, items):
        sw = inst["K_sweep"]
        ax.plot(sw["K_values"], sw["V_within"],  marker="o", markersize=3,
                label=r"$V^{\,\mathrm{within}}(K)$")
        ax.plot(sw["K_values"], sw["V_between"], marker="s", markersize=3,
                label=r"$V^{\,\mathrm{between}}(K)$")
        ax.axvline(inst["K_hat"], color="grey", linestyle="--", alpha=0.7,
                   label=fr"$\hat K^m={inst['K_hat']}$")
        ax.set_xlabel("K (forced)", fontsize=9)
        ax.set_ylabel("Variance (kernel)", fontsize=9)
        ax.set_title(f"{ex_id} (n_runs={inst['n_runs']})", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Hide any unused subplot cells.
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Per-sample K-sweep curves", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_overlay(result: Dict[str, Any], out_path: Path) -> None:
    """
    Overlay summary: all per-instance V_within(K) curves in faint grey,
    mean curve in bold, +/- 1 SD shaded band. Same for V_between(K) on
    the right panel.
    """
    per_inst = result.get("per_example_id", {})
    curves_within: List[List[float]] = []
    curves_between: List[List[float]] = []
    K_axis: Optional[List[int]] = None
    for inst in per_inst.values():
        if inst.get("skipped") or not inst.get("K_sweep"):
            continue
        sw = inst["K_sweep"]
        if K_axis is None:
            K_axis = list(sw["K_values"])
        # Pad shorter sweeps with NaN so means align.
        if len(sw["V_within"]) == len(K_axis):
            curves_within.append(list(sw["V_within"]))
            curves_between.append(list(sw["V_between"]))

    if not curves_within or K_axis is None:
        return

    A = np.asarray(curves_within, dtype=np.float64)
    B = np.asarray(curves_between, dtype=np.float64)
    mean_w = np.nanmean(A, axis=0); sd_w = np.nanstd(A, axis=0)
    mean_b = np.nanmean(B, axis=0); sd_b = np.nanstd(B, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, curves, mean_c, sd_c, label in [
        (axes[0], A, mean_w, sd_w, r"$V^{\,\mathrm{within}}(K)$"),
        (axes[1], B, mean_b, sd_b, r"$V^{\,\mathrm{between}}(K)$"),
    ]:
        for row in curves:
            ax.plot(K_axis, row, color="grey", alpha=0.25, linewidth=0.8)
        ax.plot(K_axis, mean_c, color="C0", linewidth=2.2, marker="o", label="mean across samples")
        ax.fill_between(K_axis, mean_c - sd_c, mean_c + sd_c, color="C0", alpha=0.2,
                        label=r"$\pm 1$ SD")
        ax.set_xlabel("K (forced)")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: per-sample curves and mean (n={len(curves)})")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def make_all_plots(result: Dict[str, Any], plot_dir: Path) -> None:
    """Lazily import matplotlib and produce the three diagnostic plots."""
    global plt
    if plt is None:
        plt = _plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_v_vs_k(result,    plot_dir / "v_vs_k.png")
    plot_silhouette(result, plot_dir / "silhouette.png")
    plot_examples(result,   plot_dir / "per_sample_grid.png")
    plot_overlay(result,    plot_dir / "overlay.png")


def cluster_trajectories_for_run(
    trajectory_payload: Dict[str, Any],
    *,
    epsilon: float = 0.0,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Run the two-stage clustering + within-cluster consistency analysis on a
    single experiment run.
    """
    entries = trajectory_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("trajectory.json missing 'entries' list")
 
    by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if not isinstance(e, dict):
            continue
        ex_id = e.get("example_id")
        if ex_id is None:
            continue
        by_example[str(ex_id)].append(e)
 
    per_instance: Dict[str, Dict[str, Any]] = {}
    instance_V: List[float] = []
    instance_d: List[float] = []
    K_hats: List[int] = []

    all_seqs = [
        [_action_sequence(r) for r in runs]
        for runs in by_example.values()
        if isinstance(runs, list) and len(runs) >= 2
    ]
    lam = _median_heuristic_lambda(all_seqs)
 
    for ex_id, runs in sorted(by_example.items()):
        sequences = [_action_sequence(r) for r in runs]
        variants = [r.get("variant") for r in runs]
        n = len(sequences)
 
        if n < 2:
            per_instance[ex_id] = {
                "n_runs": n,
                "variants": variants,
                "K_hat": n,
                "eigvals": [],
                "labels": [0] * n if n == 1 else [],
                "sequences": [],
                "per_cluster": {},
                "V_within": None,
                "d_within": None,
                "V_between": None,
                "d_between": None,
                "silhouette": None,
                "K_sweep": None,
                "skipped": True,
                "skip_reason": "fewer than 2 runs",
            }
            continue
 
        # Stage 1
        Kmat = _normalized_gak_matrix(sequences, lam=lam)
        k_hat, eigvals = _eigengap_num_clusters(Kmat)
        labels = _spectral_cluster(Kmat, k_hat, random_state=random_state)
 
        # Stage 2: within (existing) + between (new) + silhouette (new)
        wc = _within_cluster_metrics(sequences, Kmat, labels)
        bc = _between_cluster_metrics(sequences, Kmat, labels)
        sil = _silhouette_for_instance(sequences, Kmat, labels)

        # K-sweep for elbow plot (forces K = 1..n)
        sweep = _sweep_K_for_instance(sequences, Kmat, random_state=random_state)
 
        per_instance[ex_id] = {
            "n_runs": n,
            "variants": variants,
            "K_hat": wc["K_hat"],
            "eigvals": eigvals,
            "labels": labels.tolist(),
            "sequences":  [list(s) for s in sequences],
            "per_cluster": {
                str(c): {
                    "size": info["size"],
                    "weight": info["weight"],
                    "num_pairs": info["num_pairs"],
                    "d_bar": info["d_bar"],
                    "V_hat": info["V_hat"],
                    "members": info["members"],
                    "member_variants": [variants[i] for i in info["members"]],
                }
                for c, info in wc["per_cluster"].items()
            },
            "V_within": wc["V_within"],
            "d_within": wc["d_within"],
            "V_between": bc["V_between"],
            "d_between": bc["d_between"],
            "silhouette": sil,
            "K_sweep": sweep,
            "skipped": False,
        }
        instance_V.append(wc["V_within"])
        instance_d.append(wc["d_within"])
        K_hats.append(wc["K_hat"])
 
    # Aggregate (eqs. 11, 12, 13) + the new between/silhouette/sweep aggregates
    M = len(instance_V)
    if M == 0:
        aggregate: Dict[str, Any] = {
            "M": 0,
            "K_bar": None,
            "V_bar_within": None,
            "d_bar_within": None,
            "V_bar_between": None,
            "d_bar_between": None,
            "mean_silhouette_kernel": None,
            "mean_silhouette_lev": None,
            "sigma_V_within": None,
            "T_conc_within": None,
            "p_value_one_sided": None,
            "epsilon": epsilon,
            "lambda_GAK": lam,
            "K_sweep_mean": None,
        }
    else:
        # Re-collect values from per_instance (so order matches and we get the new ones).
        v_within_list: List[float] = []
        d_within_list: List[float] = []
        v_between_list: List[float] = []
        d_between_list: List[float] = []
        sil_ker_list: List[float] = []
        sil_lev_list: List[float] = []
        sweep_v_within: Dict[int, List[float]] = defaultdict(list)
        sweep_v_between: Dict[int, List[float]] = defaultdict(list)
        sweep_d_within: Dict[int, List[float]] = defaultdict(list)
        sweep_d_between: Dict[int, List[float]] = defaultdict(list)
        sweep_sil: Dict[int, List[float]] = defaultdict(list)

        for inst in per_instance.values():
            if inst.get("skipped"):
                continue
            v_within_list.append(inst["V_within"])
            d_within_list.append(inst["d_within"])
            v_between_list.append(inst["V_between"])
            d_between_list.append(inst["d_between"])
            if inst["silhouette"] is not None:
                if inst["silhouette"]["mean_silhouette_kernel"] is not None:
                    sil_ker_list.append(inst["silhouette"]["mean_silhouette_kernel"])
                if inst["silhouette"]["mean_silhouette_lev"] is not None:
                    sil_lev_list.append(inst["silhouette"]["mean_silhouette_lev"])
            sw = inst.get("K_sweep")
            if sw:
                for k_val, vw in zip(sw["K_values"], sw["V_within"]):
                    sweep_v_within[k_val].append(vw)
                for k_val, vb in zip(sw["K_values"], sw["V_between"]):
                    sweep_v_between[k_val].append(vb)
                for k_val, dw in zip(sw["K_values"], sw["d_within"]):
                    sweep_d_within[k_val].append(dw)
                for k_val, db in zip(sw["K_values"], sw["d_between"]):
                    sweep_d_between[k_val].append(db)
                for k_val, s in zip(sw["K_values"], sw["mean_silhouette_kernel"]):
                    sweep_sil[k_val].append(s)

        v_arr = np.asarray(v_within_list, dtype=np.float64)
        v_bar = float(v_arr.mean())
        d_bar = float(np.mean(d_within_list))
        v_bar_between = float(np.mean(v_between_list)) if v_between_list else None
        d_bar_between = float(np.mean(d_between_list)) if d_between_list else None
        sil_ker = float(np.mean(sil_ker_list)) if sil_ker_list else None
        sil_lev = float(np.mean(sil_lev_list)) if sil_lev_list else None
        k_bar = float(np.mean(K_hats))
 
        if M >= 2:
            sigma2 = float(((v_arr - v_bar) ** 2).sum() / (M - 1))
            sigma = math.sqrt(sigma2) if sigma2 > 0 else 0.0
            if sigma > 0:
                T = math.sqrt(M) * (v_bar - epsilon) / sigma
                p_val = 1.0 - _norm.cdf(T)
            else:
                T = math.inf if v_bar > epsilon else (
                    -math.inf if v_bar < epsilon else 0.0
                )
                p_val = 0.0 if v_bar > epsilon else 1.0
        else:
            sigma = None
            T = None
            p_val = None

        # Mean K-sweep curves (only K values present in at least one instance).
        sweep_keys = sorted(sweep_v_within.keys())
        K_sweep_mean = {
            "K_values": sweep_keys,
            "V_within_mean":  [float(np.mean(sweep_v_within[k]))  for k in sweep_keys],
            "V_between_mean": [float(np.mean(sweep_v_between[k])) for k in sweep_keys],
            "d_within_mean":  [float(np.mean(sweep_d_within[k]))  for k in sweep_keys],
            "d_between_mean": [float(np.mean(sweep_d_between[k])) for k in sweep_keys],
            "mean_silhouette_kernel": [float(np.mean(sweep_sil[k])) for k in sweep_keys],
            "n_instances_per_K": [len(sweep_v_within[k]) for k in sweep_keys],
        }
 
        aggregate = {
            "M": M,
            "K_bar": k_bar,
            "V_bar_within": v_bar,
            "d_bar_within": d_bar,
            "V_bar_between": v_bar_between,
            "d_bar_between": d_bar_between,
            "mean_silhouette_kernel": sil_ker,
            "mean_silhouette_lev": sil_lev,
            "sigma_V_within": sigma,
            "T_conc_within": T,
            "p_value_one_sided": p_val,
            "epsilon": epsilon,
            "lambda_GAK": lam,
            "K_sweep_mean": K_sweep_mean,
        }
 
    # Distribution of K_hat across instances (diagnostic).
    k_hat_dist: Dict[int, int] = defaultdict(int)
    for k in K_hats:
        k_hat_dist[int(k)] += 1
 
    return {
        "config": {
            "lambda_GAK": lam,
            "epsilon": epsilon,
            "random_state": random_state,
        },
        "aggregate": aggregate,
        "K_hat_distribution": dict(sorted(k_hat_dist.items())),
        "per_example_id": per_instance,
    }

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Two-stage trajectory clustering + within-cluster consistency "
            "analysis from a single trajectory.json."
        ),
    )
    ap.add_argument(
        "outputs_path",
        type=str,
        help="Run directory containing trajectory.json (or path to the json itself).",
    )
    ap.add_argument(
        "--lambda-gak",
        type=float,
        default=500,
        help="GAK bandwidth lambda. Larger -> closer to hard alignment / edit distance.",
    )
    ap.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Concentration tolerance epsilon for the within-cluster test (eq. 13).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state for spectral clustering's k-means stage.",
    )
    ap.add_argument(
        "--output-name",
        type=str,
        default="trajectory_clusters.json",
        help="Filename to write into outputs_path.",
    )
    ap.add_argument(
        "--make-plots",
        action="store_true",
        help="Also produce diagnostic plots (V vs K elbow, silhouette).",
    )
    args = ap.parse_args(argv)
 
    run_dir = Path(args.outputs_path).resolve()
    if run_dir.is_file() and run_dir.name.endswith(".json"):
        trajectory_json = run_dir
        run_dir = run_dir.parent
    elif run_dir.is_dir():
        trajectory_json = run_dir / "trajectory.json"
        if not trajectory_json.is_file():
            raise SystemExit(f"Missing trajectory.json in: {run_dir}")
    else:
        raise SystemExit(f"Not a directory or json file: {run_dir}")
 
    payload = _read_json(trajectory_json)
    result = cluster_trajectories_for_run(
        payload,
        # max_clusters_per_instance=args.max_clusters,
        epsilon=args.epsilon,
        random_state=args.random_state,
    )
 
    agg = result["aggregate"]
    M = agg["M"]
    print(f"[{run_dir.name}] M={M}  K_bar={agg['K_bar']}")
    print(f"  K_hat distribution: {result['K_hat_distribution']}")
    print()
    print("  Per-sample stats:")
    print(f"    {'example_id':<28} {'K_hat':>5} {'sizes':<14} "
          f"{'V_within':>9} {'V_between':>10} {'d_within':>9} {'d_between':>10} "
          f"{'silh_ker':>9} {'V_pool(K=1)':>12}")
    print("    " + "-" * 116)
    for ex_id, inst in result["per_example_id"].items():
        if inst.get("skipped"):
            print(f"    {ex_id:<28} SKIPPED ({inst.get('skip_reason','')})")
            continue
        sizes = sorted(
            (info["size"] for info in inst["per_cluster"].values()),
            reverse=True,
        )
        sizes_str = "/".join(map(str, sizes))
        sil = (inst.get("silhouette") or {}).get("mean_silhouette_kernel") or 0.0
        v_pool = inst["K_sweep"]["V_within"][0] if inst.get("K_sweep") else float("nan")
        print(f"    {ex_id:<28} {inst['K_hat']:>5} {sizes_str:<14} "
              f"{inst['V_within']:>9.4f} {inst['V_between']:>10.4f} "
              f"{inst['d_within']:>9.4f} {inst['d_between']:>10.4f} "
              f"{sil:>9.4f} {v_pool:>12.4f}")
    print()
    print("  Aggregate summary (mean across samples):")
    print(
        f"    V_bar_within  = {agg['V_bar_within']}   "
        f"d_bar_within  = {agg['d_bar_within']}"
    )
    print(
        f"    V_bar_between = {agg['V_bar_between']}   "
        f"d_bar_between = {agg['d_bar_between']}"
    )
    print(
        f"    silhouette (kernel) = {agg['mean_silhouette_kernel']}   "
        f"silhouette (lev) = {agg['mean_silhouette_lev']}"
    )
    if M is not None and M < 30:
        print(f"  Aggregate test (M={M} is small — interpret with caution):")
    else:
        print("  Aggregate test:")
    print(
        f"    sigma_V_within = {agg['sigma_V_within']}   "
        f"T_conc_within  = {agg['T_conc_within']}   "
        f"p (one-sided) = {agg['p_value_one_sided']}"
    )
    print(f"  lambda_GAK = {result['config']['lambda_GAK']}")
    # Per-sample stats as CSV for easy import.
    csv_path = run_dir / "per_sample_stats.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("example_id,K_hat,n_runs,cluster_sizes,V_within,V_between,"
                "d_within,d_between,silhouette_kernel,silhouette_lev,V_pool_K1,d_pool_K1\n")
        for ex_id, inst in result["per_example_id"].items():
            if inst.get("skipped"):
                f.write(f"{ex_id},SKIPPED,,,,,,,,,,\n")
                continue
            sizes = sorted(
                (info["size"] for info in inst["per_cluster"].values()),
                reverse=True,
            )
            sizes_str = "|".join(map(str, sizes))
            sil_k = (inst.get("silhouette") or {}).get("mean_silhouette_kernel") or 0.0
            sil_l = (inst.get("silhouette") or {}).get("mean_silhouette_lev") or 0.0
            sweep = inst.get("K_sweep") or {}
            v_pool = sweep.get("V_within", [float("nan")])[0]
            d_pool = sweep.get("d_within", [float("nan")])[0]
            f.write(
                f"{ex_id},{inst['K_hat']},{inst['n_runs']},{sizes_str},"
                f"{inst['V_within']:.6f},{inst['V_between']:.6f},"
                f"{inst['d_within']:.6f},{inst['d_between']:.6f},"
                f"{sil_k:.6f},{sil_l:.6f},{v_pool:.6f},{d_pool:.6f}\n"
            )
    print(f"  wrote: {csv_path}")

    out_path = run_dir / "clusters.json"
    _write_json(out_path, result)
    print(f"  wrote: {out_path}")

    if args.make_plots:
        plot_dir = run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        make_all_plots(result, plot_dir)
        print(f"  wrote plots to: {plot_dir}")
 
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
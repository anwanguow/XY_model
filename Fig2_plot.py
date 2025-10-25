#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load & plot robust XY-model helicity modulus (Upsilon) vs T.

Reads the NPZ saved by compute_xy_upsilon_robust.py, applies robust aggregation:
  - Outlier removal via median/MAD (|z_MAD| > cutoff)
  - 10% trimmed mean (configurable)
  - Nonparametric bootstrap CI on the trimmed mean

Then plots, for each lattice size L, the two samplers (Metropolis / randomized twist)
on the same panel, with CI bands, plus the 2T/π reference line and an algorithm-
independence z-check (fraction within 2σ, reported to stdout).

Usage:
    python Fig2_plot.py

Output:
    Fig2.png
"""

import sys, json, math
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- Global font size -------------------------------
plt.rcParams.update({
    "font.size": 14, 
    "axes.titlesize": 14,   
    "axes.labelsize": 14,  
    "xtick.labelsize": 14,  
    "ytick.labelsize": 14,  
    "legend.fontsize": 14,  
})

# ------------------------------- File Address -------------------------------

IN_DATA_DEFAULT = "Fig2_results.npz"
OUT_FIG_DEFAULT = "Fig2.png"

MAD_Z_CUTOFF_DEFAULT = 4.0
TRIM_FRAC_DEFAULT = 0.10
BOOT_SAMPLES_DEFAULT = 500

# ------------------------------- Doing -------------------------------

def median_mad(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    sigma = 1.4826 * mad
    z = (x - med) / sigma
    return med, sigma, z

def trimmed_mean_sem(x, trim_frac=0.10):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    k = int(math.floor(trim_frac * n))
    if 2*k >= n:
        return x.mean(), x.std(ddof=1)/math.sqrt(max(1, n-1)), (x.min(), x.max())
    xt = x[k:n-k]
    m = xt.mean()
    sem = xt.std(ddof=1) / math.sqrt(len(xt))
    return m, sem, (xt.min(), xt.max())

def bootstrap_ci(x, nboot=500, alpha=0.05, trim_frac=0.10, rng=None):
    if rng is None: rng = np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    k = int(math.floor(trim_frac * n))
    stats = []
    for _ in range(nboot):
        b = x[rng.integers(0, n, size=n)]
        b.sort()
        if 2*k < n:
            b = b[k:n-k]
        stats.append(b.mean())
    lo = np.percentile(stats, 2.5)
    hi = np.percentile(stats, 97.5)
    return float(lo), float(hi)

# ------------------------------- Loading -------------------------------

def load_npz(path: str):
    z = np.load(path, allow_pickle=False)
    Ls = z["Ls"].astype(int)
    Ts = z["Ts"].astype(float)
    samplers = z["samplers"].astype(str)
    U_values = z["U_values"].astype(float)
    meta_json = z["meta_json"].item() if z["meta_json"].ndim == 0 else str(z["meta_json"])
    meta = json.loads(meta_json)
    return Ls, Ts, samplers, U_values, meta

# ------------------------------- Aggregation + plotting -------------------------------

def aggregate_and_plot(Ls, Ts, samplers, U_values, meta, out_path):
    MAD_Z_CUTOFF = meta.get("MAD_Z_CUTOFF", MAD_Z_CUTOFF_DEFAULT)
    TRIM_FRAC = meta.get("TRIM_FRAC", TRIM_FRAC_DEFAULT)
    BOOT_SAMPLES = meta.get("BOOT_SAMPLES", BOOT_SAMPLES_DEFAULT)
    SIZES = meta.get("SIZES", sorted(list(set(map(int, Ls)))))
    SAMPLERS = meta.get("SAMPLERS", ["metropolis", "twist"])
    T_MIN = meta.get("T_MIN", float(np.min(Ts)))
    T_MAX = meta.get("T_MAX", float(np.max(Ts)))

    groups: Dict[Tuple[int,float,str], List[float]] = {}
    for L, T, s, U in zip(Ls, Ts, samplers, U_values):
        groups.setdefault((int(L), float(T), str(s)), []).append(float(U))

    nL = len(SIZES)
    fig, axes = plt.subplots(1, nL, figsize=(6.3*nL, 5.0), sharey=True)
    if nL == 1:
        axes = [axes]

    for ax, L in zip(axes, SIZES):
        curves = {}
        Ts_unique = sorted(set([float(T) for (LL, T, s) in groups.keys() if LL == L]))
        for sampler in SAMPLERS:
            T_list, mean_list, sem_list, lo_list, hi_list = [], [], [], [], []
            for T in Ts_unique:
                vals = groups.get((L, T, sampler), [])
                if not vals:
                    continue
                med, sigma, z = median_mad(vals)
                mask = (np.abs(z) <= MAD_Z_CUTOFF)
                kept = np.asarray(vals)[mask]
                removed = len(vals) - kept.size
                if removed > 0:
                    print(f"[L={L}, T={T:.3f}, {sampler}] removed {removed}/{len(vals)} outliers (|z_MAD|>{MAD_Z_CUTOFF}).")
                m, sem, _rng = trimmed_mean_sem(kept, TRIM_FRAC)
                lo, hi = bootstrap_ci(kept, BOOT_SAMPLES, 0.05, TRIM_FRAC)
                T_list.append(T); mean_list.append(m); sem_list.append(sem); lo_list.append(lo); hi_list.append(hi)

            T_arr = np.array(T_list); mean_arr = np.array(mean_list); sem_arr = np.array(sem_list)
            lo_arr = np.array(lo_list); hi_arr = np.array(hi_list)
            curves[sampler] = (T_arr, mean_arr, sem_arr, lo_arr, hi_arr)

            lbl = "Local Metropolis" if sampler=="metropolis" else "Metropolis + global rotation"
            style = "-" if sampler=="metropolis" else "--"
            marker= "o" if sampler=="metropolis" else "s"
            ax.plot(T_arr, mean_arr, style, marker=marker, lw=2, ms=4, label=lbl)
            ax.fill_between(T_arr, lo_arr, hi_arr, alpha=0.18)

        Ts_line = np.linspace(T_MIN, T_MAX, 200)
        ax.plot(Ts_line, 2.0*Ts_line/np.pi, lw=1.6, alpha=0.9, label=r"$2T/\pi$")
        ax.set_xlabel("Temperature $T$")
        ax.grid(True, alpha=0.25)

        # Algorithm-independence z-check
        if "metropolis" in curves and "twist" in curves:
            Tm, m1, s1, _, _ = curves["metropolis"]
            Tt, m2, s2, _, _ = curves["twist"]
            common = np.intersect1d(Tm, Tt)
            ok=0; tot=len(common); maxz=0.0
            for T in common:
                i = np.where(np.isclose(Tm, T))[0][0]
                j = np.where(np.isclose(Tt, T))[0][0]
                delta = abs(m1[i]-m2[j])
                z = delta / math.sqrt(s1[i]**2 + s2[j]**2 + 1e-16)
                maxz = max(maxz, z)
                if z <= 2.0: ok += 1
            print(f"[L={L}] alg-indep check: {ok}/{tot} points within 2σ; max z={maxz:.2f}")

    axes[0].set_ylabel(r"Helicity modulus  $\Upsilon(T)$")
    axes[-1].legend(frameon=False, loc="lower left")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_path, dpi=220)
    print(f"Saved figure to: {out_path}")

# ------------------------------- Main -------------------------------

def main():
    in_path  = sys.argv[1] if len(sys.argv) > 1 else IN_DATA_DEFAULT
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUT_FIG_DEFAULT

    Ls, Ts, samplers, U_values, meta = load_npz(in_path)
    aggregate_and_plot(Ls, Ts, samplers, U_values, meta, out_path)

if __name__ == "__main__":
    main()

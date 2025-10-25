#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load saved XY magnetization-angle results and PLOT:
  ECDF distance (Cramér–von Mises ω²) vs number of samples n

Inputs:
  - Fig1_results.npz  (from Fig1_calc.py)

Outputs:
  - figure: Fig1.png
"""

import math, sys
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- I/O -------------------------------

IN_RESULTS     = "Fig1_results.npz"
OUT_FIG        = "Fig1.png"
N_CHECKPOINTS  = 24
PLOT_N_OMEGA2  = False      # True -> plot n*ω²; False -> plot ω² (PRB ω²)

# ------------------------------- ω² -------------------------------

def cramervonmises_omega2_from_sorted_u(u_sorted: np.ndarray) -> float:
    n = u_sorted.size
    if n == 0:
        return 0.0
    i = np.arange(1, n+1, dtype=np.float64)
    term = u_sorted - (2.0*i - 1.0) / (2.0*n)
    return (1.0 / (12.0*n)) + float(np.sum(term*term))

def omega2_prefix_curve_from_phi(phi_series: np.ndarray) -> np.ndarray:
    N = phi_series.size
    if N == 0:
        return np.zeros(0, dtype=np.float64)
    u_all = (phi_series / (2.0*np.pi)).astype(np.float64) % 1.0
    omega2 = np.empty(N, dtype=np.float64)
    for n in range(1, N+1):
        u_sorted = np.sort(u_all[:n], kind="quicksort")
        omega2[n-1] = cramervonmises_omega2_from_sorted_u(u_sorted)
    return omega2

def resample_prefix_curve(curve: np.ndarray, n_checkpoints: int) -> Tuple[np.ndarray, np.ndarray]:
    N = curve.size
    if N <= n_checkpoints:
        idx = np.arange(N)
    else:
        start = max(8, N // n_checkpoints)
        idx = np.unique(np.linspace(start-1, N-1, n_checkpoints, dtype=int))
    n_grid = idx + 1
    return n_grid, curve[idx]

# ------------------------------- Plotting -------------------------------

def aggregate_and_plot(records, out_fig: str, n_checkpoints: int):
    by_group: Dict[Tuple[float,str], Dict[str, List[np.ndarray]]] = {}
    temps = set()
    for rec in records:
        T = float(rec["T"]); sampler = rec["sampler"]
        temps.add(T)
        d = by_group.setdefault((T, sampler), {"phi": [], "w2": []})
        for phi in rec["phi_series_list"]:
            d["phi"].append(np.asarray(phi, dtype=np.float64))
        # use precomputed ω² if present
        if "omega2_list" in rec:
            for w2 in rec["omega2_list"]:
                d["w2"].append(np.asarray(w2, dtype=np.float64))

    series = {}
    for (T, sampler), d in by_group.items():
        w2_curves = d["w2"]
        if len(w2_curves) != len(d["phi"]) or len(w2_curves) == 0:
            w2_curves = [omega2_prefix_curve_from_phi(phi) for phi in d["phi"]]
        Xs, Ys = [], []
        for w2 in w2_curves:
            x, y = resample_prefix_curve(w2, n_checkpoints)
            if PLOT_N_OMEGA2:
                y = y * x  # n * ω²
            Xs.append(x); Ys.append(y)
        m = min(len(x) for x in Xs)
        grid = Xs[0][:m]
        Y = np.vstack([y[:m] for y in Ys])
        mean = Y.mean(axis=0)
        sem  = Y.std(axis=0, ddof=1) / math.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mean)
        series[(T, sampler)] = (grid, mean, sem)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4', '#ff7f0e'])
    color_by_alg = {"metropolis": "#1f77b4", "twist": "#ff7f0e"}
    color_cycle_idx = 0
    temps_sorted = sorted(list(temps))
    base_styles = ['-', '--', '-.', ':']
    style_by_T = {T: base_styles[i % len(base_styles)] for i, T in enumerate(temps_sorted)}


    for (T, sampler), (grid, mean, sem) in sorted(series.items()):
        if sampler not in color_by_alg:
            color_by_alg[sampler] = default_colors[color_cycle_idx % len(default_colors)]
            color_cycle_idx += 1
        label_alg = "Local Metropolis" if sampler == "metropolis" else ("Metropolis + global rotation" if sampler == "twist" else sampler)
        label = f"{label_alg}, T={T:.2f}"
        ax.plot(grid, mean, linestyle=style_by_T[T], lw=2.6, color=color_by_alg[sampler], label=label)
        ax.fill_between(grid, mean-1.96*sem, mean+1.96*sem, alpha=0.12, color=color_by_alg[sampler])

    ax.set_xlabel("Observation window n (samples)", fontsize=20)
    ylab = r"$n\,\omega^2$" if PLOT_N_OMEGA2 else r"Cramér–von Mises distance $\omega^2$"
    ax.set_ylabel(ylab + " to $U[0,2\pi)$", fontsize=20)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=20)
    ax.legend(loc="best", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="#dddddd",
              fontsize=20, title=None)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    print(f"Saved figure to: {out_fig}")

def main():
    data = np.load(IN_RESULTS, allow_pickle=True)
    records = list(data["records"])
    aggregate_and_plot(records, OUT_FIG, N_CHECKPOINTS)

if __name__ == "__main__":
    main()

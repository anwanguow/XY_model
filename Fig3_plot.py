#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, pickle
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

L = 64
J = 1.0
T_LOW  = 0.85
T_HIGH = 1.05

OUT_FIG = "Fig3.png"
RESULTS_FILE = "Fig3_results.npz"
EPS_LOG = 1e-12

BASE_LOW_LO  = lambda L: max(2, L//8)
BASE_LOW_HI  = lambda L: L//3
BASE_HIGH_LO = 5
BASE_HIGH_HI = lambda L: L//3


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})

# ------------------------------- Fitting -------------------------------

def auto_fit_window(r, mean, sem, base_lo, base_hi, floor=EPS_LOG):
    mask = (r>=base_lo) & (r<=base_hi) & (mean > np.maximum(3.0*sem, floor))
    if not np.any(mask):
        lo = base_lo; hi = max(base_lo+5, min(base_hi, base_lo+12))
        return lo, hi
    idx = np.where(mask)[0]
    lo = int(max(r[idx[0]], base_lo)); hi = int(min(r[idx[-1]], base_hi))
    if hi - lo < 5: hi = min(base_hi, lo+5)
    return lo, hi

def fit_power_law(r, g, r_lo, r_hi):
    m = (r>=r_lo) & (r<=r_hi) & (g>0)
    x = np.log(r[m]); y = np.log(g[m])
    if x.size < 3: return np.nan, (np.array([]), np.array([]))
    b, a = np.polyfit(x, y, 1)
    eta = -b
    xx = np.linspace(x.min(), x.max(), 200)
    yy = a + b*xx
    return float(eta), (np.exp(xx), np.exp(yy))

def fit_highT_linearized(r, g, r_lo, r_hi):
    m = (r>=r_lo) & (r<=r_hi) & (g>0)
    x = r[m].astype(float)
    y = np.log(g[m]*np.sqrt(x))
    if x.size < 3: return np.nan, (np.array([]), np.array([]))
    b, a = np.polyfit(x, y, 1)
    xi = -1.0/b if b!=0 else np.nan
    xx = np.linspace(x.min(), x.max(), 200)
    y_trans = np.exp(a + b*xx)
    return float(xi), (xx, y_trans)

def first_descending_window(r, y_trans, base_lo, base_hi, k=7, eps=1e-4):
    for lo in range(base_lo, base_hi - k + 2):
        xs = r[lo-1:lo-1+k].astype(float)
        ys = np.log(np.maximum(y_trans[lo-1:lo-1+k], EPS_LOG))
        b, a = np.polyfit(xs, ys, 1)
        if b < -eps:
            return lo, base_hi
    return None

# ------------------------------- Plotting -------------------------------

def aggregate_and_plot(results: Dict[float, List[Tuple[np.ndarray,np.ndarray]]]):
    Rmax = L//2
    r = np.arange(1, Rmax+1, dtype=int)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8, 6))
    styles = {"metropolis": ("o-",2), "twist": ("s--",2)}
    labels = {"metropolis": "Local Metropolis", "twist": "Metropolis + global rotation"}

    # ----- low T -----
    T = T_LOW
    pairs = results.get(T, [])
    G0 = np.vstack([p[0] for p in pairs])
    G1 = np.vstack([p[1] for p in pairs])
    mean0, mean1 = G0.mean(axis=0), G1.mean(axis=0)
    sem0  = G0.std(axis=0, ddof=1)/math.sqrt(G0.shape[0])
    sem1  = G1.std(axis=0, ddof=1)/math.sqrt(G1.shape[0])

    r_lo, r_hi = auto_fit_window(r, mean0, sem0, BASE_LOW_LO(L), BASE_LOW_HI(L))
    eta, (xf, yf) = fit_power_law(r, np.maximum(mean0, EPS_LOG), r_lo, r_hi)

    ax = axA
    ax.set_title(f"$T = {T:.2f}$ (log–log of $G(r)$)")
    ax.loglog(r, np.maximum(mean0, EPS_LOG), styles["metropolis"][0], lw=2, ms=4, label=labels["metropolis"])
    ax.fill_between(r, np.maximum(mean0-1.96*sem0, EPS_LOG), np.maximum(mean0+1.96*sem0, EPS_LOG), alpha=0.20)
    ax.loglog(r, np.maximum(mean1, EPS_LOG), styles["twist"][0], lw=2, ms=4, label=labels["twist"])
    ax.fill_between(r, np.maximum(mean1-1.96*sem1, EPS_LOG), np.maximum(mean1+1.96*sem1, EPS_LOG), alpha=0.12)
    if xf.size>0:
        ax.loglog(
            xf, np.maximum(yf, EPS_LOG),
            "-", lw=1.5, alpha=0.85,
            color="tab:green",
            label=r"Power-law fit ($r^{-\eta}$)"
        )
    ax.set_xlabel("$r$", labelpad=0)
    ax.set_ylabel(r"$G(r)=\langle\cos(\theta_0-\theta_r)\rangle$")
    ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, loc="upper right")
    ax.text(0.03,0.05, rf"$\eta \approx {eta:.3f}$  (horizontal axis range {r_lo} to {r_hi})",
            transform=ax.transAxes, fontsize=13, bbox=dict(boxstyle="round", fc="w", alpha=0.7))

    # ----- high T -----
    T = T_HIGH
    pairs = results.get(T, [])
    G0 = np.vstack([p[0] for p in pairs])
    G1 = np.vstack([p[1] for p in pairs])
    mean0, mean1 = G0.mean(axis=0), G1.mean(axis=0)
    sem0  = G0.std(axis=0, ddof=1)/math.sqrt(G0.shape[0])
    sem1  = G1.std(axis=0, ddof=1)/math.sqrt(G1.shape[0])

    trans0 = mean0*np.sqrt(r); trans1 = mean1*np.sqrt(r)
    ax = axB
    ax.set_title(f"$T = {T:.2f}$ (semi–log of  $G(r)\\,\\sqrt{{r}}$)")
    ax.semilogy(r, np.maximum(trans0, EPS_LOG), styles["metropolis"][0], lw=2, ms=4, label=labels["metropolis"])
    ax.fill_between(r, np.maximum(trans0-1.96*(sem0*np.sqrt(r)), EPS_LOG),
                       np.maximum(trans0+1.96*(sem0*np.sqrt(r)), EPS_LOG), alpha=0.20)
    ax.semilogy(r, np.maximum(trans1, EPS_LOG), styles["twist"][0], lw=2, ms=4, label=labels["twist"])
    ax.fill_between(r, np.maximum(trans1-1.96*(sem1*np.sqrt(r)), EPS_LOG),
                       np.maximum(trans1+1.96*(sem1*np.sqrt(r)), EPS_LOG), alpha=0.12)

    win = first_descending_window(r, trans0, BASE_HIGH_LO, BASE_HIGH_HI(L), k=7, eps=1e-4)
    xi_text = "n/a"
    if win is not None:
        r_lo, r_hi = win
        xi, (xf, y_trans) = fit_highT_linearized(r, np.maximum(mean0, EPS_LOG), r_lo, r_hi)
        if not np.isnan(xi):
            xi_text = rf"$\xi \approx {xi:.2f}$  (horizontal axis range: {r_lo} to {r_hi})"
            ax.semilogy(
                xf, np.maximum(y_trans, EPS_LOG),
                "-", lw=1.5, alpha=0.85,
                color="tab:green",
                label=r"Asymptotic fit ($r^{-1/2} e^{-r/\xi}$)"
            )
    else:
        ax.text(0.03,0.90,"no descending window\n(not in asymptotic regime)",
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round", fc="w", alpha=0.7))

    ax.set_xlabel("$r$", labelpad=2)
    ax.set_ylabel(r"$G(r)\,\sqrt{r}$")
    ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, loc="upper right")
    ax.text(0.03,0.05, xi_text, transform=ax.transAxes, fontsize=13,
            bbox=dict(boxstyle="round", fc="w", alpha=0.7))

    plt.tight_layout(h_pad=0.5)
    plt.savefig(OUT_FIG, dpi=220)
    print(f"Saved figure to: {OUT_FIG}")

# ------------------------------- I/O -------------------------------

def load_results(path: str) -> Dict[float, List[Tuple[np.ndarray, np.ndarray]]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {float(k): v for k, v in data.items()}

# ------------------------------- main -------------------------------

def main():
    results = load_results(RESULTS_FILE)
    aggregate_and_plot(results)

if __name__ == "__main__":
    main()

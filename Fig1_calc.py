#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute XY model magnetization-angle series (multiple runs) and save results.

Outputs:
  - results npz:  Fig1_results.npz
      · records: list[ dict{
            'T': float,
            'sampler': str,
            'phi_series_list': list[np.ndarray],   # magnetization-angle series
            'omega2_list':     list[np.ndarray],   # Cramér–von Mises ω^2 for prefixes n=1..N
        } ]
      · meta:    dict with all key params for reproducibility

Run:
  python Fig1_calc.py
"""

import math, time, sys, json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from multiprocessing import Process, Queue, cpu_count, set_start_method
from pathlib import Path

# ------------------------------- User params -------------------------------

L = 64
J = 1.0
T_LOW, T_HIGH = 0.85, 1.05
TEMPS = [T_LOW, T_HIGH]

THERM_SWEEPS = 4000
MEAS_SWEEPS  = 6000
SAMPLE_EVERY = 10
ACCEPT_STEP  = 0.30
TWIST_EVERY  = 50

N_REP = 8


# How many cores of CPU do you want to use.
# N_PROCESSES = min(max(1, cpu_count()//2), 8)
N_PROCESSES = 8
RNG_BASE_SEED = 20251019

RESULTS_FILE = "Fig1_results.npz"

# ------------------------------- Progress bar -------------------------------

def print_progress(done, total, prefix="Progress", length=42):
    q = min(max(done / max(1,total), 0.0), 1.0)
    bar = "█" * int(q*length) + "·" * (length - int(q*length))
    sys.stdout.write(f"\r{prefix} |{bar}| {done}/{total} ({100*q:5.1f}%)")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n"); sys.stdout.flush()

@dataclass
class Task:
    task_id: int
    L: int
    J: float
    T: float
    beta: float
    therm_sweeps: int
    meas_sweeps: int
    sample_every: int
    accept_step: float
    sampler: str         # "metropolis" or "twist"
    twist_every: int
    seed: int

# ------------------------------- XY kernels -------------------------------

def xy_metropolis_sweep(theta: np.ndarray, beta: float, J: float,
                        rng: np.random.Generator, accept_step: float) -> int:
    L = theta.shape[0]; acc = 0
    for i in range(L):
        im1 = (i - 1) % L; ip1 = (i + 1) % L
        for j in range(L):
            jm1 = (j - 1) % L; jp1 = (j + 1) % L
            old = theta[i, j]
            dth = rng.uniform(-accept_step, accept_step)
            new = old + dth
            n0, n1, n2, n3 = theta[im1,j], theta[ip1,j], theta[i,jm1], theta[i,jp1]
            dE = -J*(math.cos(new-n0)+math.cos(new-n1)+math.cos(new-n2)+math.cos(new-n3)
                     -math.cos(old-n0)-math.cos(old-n1)-math.cos(old-n2)-math.cos(old-n3))
            if dE <= 0.0 or rng.random() < math.exp(-beta*dE):
                theta[i, j] = new; acc += 1
    return acc

def global_twist(theta: np.ndarray, rng: np.random.Generator):
    theta += rng.uniform(0.0, 2.0*math.pi)
    theta %= (2.0*math.pi)

def magnetization_angle(theta: np.ndarray) -> float:
    s = float(np.sin(theta, dtype=np.float64).sum())
    c = float(np.cos(theta, dtype=np.float64).sum())
    phi = math.atan2(s, c)
    return phi if phi >= 0 else (phi + 2.0*math.pi)

# ------------------------------- ω^2 from φ-series ------------------------

def cramervonmises_omega2(u_sorted: np.ndarray) -> float:
    n = u_sorted.size
    if n == 0:
        return np.nan
    i = np.arange(1, n+1, dtype=np.float64)
    term = u_sorted - (2.0*i - 1.0) / (2.0*n)
    return (1.0 / (12.0*n)) + float(np.sum(term*term))

def omega2_curve_from_phi_series(phi_series: np.ndarray) -> np.ndarray:
    N = phi_series.size
    omega2 = np.empty(N, dtype=np.float64)
    u_all = (phi_series / (2.0*np.pi)).astype(np.float64)  # map to [0,1)
    for n in range(1, N+1):
        u_sorted = np.sort(u_all[:n], kind="quicksort")
        omega2[n-1] = cramervonmises_omega2(u_sorted)
    return omega2

# ------------------------------- Worker & orchestrator ---------------------

def worker(task: Task, pq: Queue, rq: Queue):
    try:
        rng = np.random.default_rng(task.seed)
        theta = rng.uniform(0.0, 2.0*math.pi, size=(task.L, task.L))
        total = task.therm_sweeps + task.meas_sweeps
        stride = max(1, total // 240)

        # thermalize
        for t in range(1, task.therm_sweeps+1):
            xy_metropolis_sweep(theta, task.beta, task.J, rng, task.accept_step)
            if task.sampler == "twist" and (t % task.twist_every) == 0:
                global_twist(theta, rng)
            if (t % stride) == 0:
                pq.put(("prog", task.task_id, stride))

        # measure φ_m
        n_samples = task.meas_sweeps // task.sample_every
        phi_series = np.zeros(n_samples, dtype=np.float64)
        idx = 0
        for t in range(1, task.meas_sweeps+1):
            xy_metropolis_sweep(theta, task.beta, task.J, rng, task.accept_step)
            if task.sampler == "twist" and (t % task.twist_every) == 0:
                global_twist(theta, rng)
            if (t % task.sample_every) == 0:
                phi_series[idx] = magnetization_angle(theta); idx += 1
            if (t % stride) == 0:
                pq.put(("prog", task.task_id, stride))

        rq.put(("result", task.task_id, task.T, task.sampler, phi_series))
    except Exception as e:
        rq.put(("error", task.task_id, str(e)))

def build_tasks() -> List[Task]:
    tasks=[]; tid=0
    for T in TEMPS:
        beta = 1.0 / T
        for sampler in ["metropolis", "twist"]:
            for rep in range(N_REP):
                seed = RNG_BASE_SEED + (tid+1)*9973
                tasks.append(Task(
                    task_id=tid, L=L, J=J, T=T, beta=beta,
                    therm_sweeps=THERM_SWEEPS, meas_sweeps=MEAS_SWEEPS,
                    sample_every=SAMPLE_EVERY, accept_step=ACCEPT_STEP,
                    sampler=sampler, twist_every=TWIST_EVERY, seed=seed
                ))
                tid += 1
    return tasks

def run_all(tasks: List[Task]):
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    n_tasks=len(tasks); per=THERM_SWEEPS+MEAS_SWEEPS; total=n_tasks*per
    pq, rq = Queue(), Queue()
    procs=[Process(target=worker, args=(t,pq,rq), daemon=True) for t in tasks]

    active=set(); idx=0
    while idx<n_tasks and len(active)<N_PROCESSES:
        procs[idx].start(); active.add(idx); idx+=1

    done=0; got=0
    results: Dict[Tuple[float,str], List[np.ndarray]] = {}
    print(f"Running {n_tasks} tasks on {N_PROCESSES} processes "
          f"(each {per} MCS; total ~{total} MCS)...")

    t0=time.time()
    try:
        while got<n_tasks:
            progressed=False
            if not pq.empty():
                _, tid, inc = pq.get(); done+=inc; progressed=True; print_progress(done,total)
            if not progressed and not rq.empty():
                msg=rq.get(); tag=msg[0]
                if tag=="result":
                    _, tid, T, sampler, phi_series = msg
                    results.setdefault((T, sampler), []).append(phi_series)
                    got+=1; print_progress(done,total)
                    if idx<n_tasks: procs[idx].start(); active.add(idx); idx+=1
                elif tag=="error":
                    _, tid, err = msg; print(f"\n[ERROR] Task {tid}: {err}")
                    got+=1
                    if idx<n_tasks: procs[idx].start(); active.add(idx); idx+=1
            if not progressed and pq.empty() and rq.empty():
                time.sleep(0.05)
        for p in procs:
            p.join(timeout=0.1)
    finally:
        print(f"\nDone in {time.time()-t0:.1f}s")
    return results

def save_results(results: Dict[Tuple[float,str], List[np.ndarray]], out_path: str):
    # pack to a simple records list to ease loading later
    records = []
    for (T, sampler), arrs in results.items():
        omega2_curves = []
        for series in arrs:
            w2 = omega2_curve_from_phi_series(np.asarray(series, dtype=np.float64))
            omega2_curves.append(w2)

        records.append({
            "T": float(T),
            "sampler": str(sampler),
            "phi_series_list": [np.asarray(a, dtype=np.float64) for a in arrs],
            "omega2_list":     [np.asarray(w, dtype=np.float64) for w in omega2_curves],
        })

    meta = {
        "L": L, "J": J, "TEMPS": TEMPS,
        "THERM_SWEEPS": THERM_SWEEPS, "MEAS_SWEEPS": MEAS_SWEEPS,
        "SAMPLE_EVERY": SAMPLE_EVERY, "ACCEPT_STEP": ACCEPT_STEP, "TWIST_EVERY": TWIST_EVERY,
        "N_REP": N_REP, "N_PROCESSES": N_PROCESSES, "RNG_BASE_SEED": RNG_BASE_SEED,
        "distance_metric": "cramer-von-mises",
        "omega2_definition": "omega^2 = 1/(12n) + sum_{i=1}^n (u_(i) - (2i-1)/(2n))^2 with u=phi/(2pi)",
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        records=np.array(records, dtype=object),
                        meta=np.array([meta], dtype=object))
    print(f"Saved results to: {out_path}")
    print(f"[info] groups: {len(records)}  "
          f"(per group mean #series: {np.mean([len(r['phi_series_list']) for r in records]):.1f})")

def main():
    tasks = build_tasks()
    results = run_all(tasks)
    save_results(results, RESULTS_FILE)

if __name__ == "__main__":
    main()

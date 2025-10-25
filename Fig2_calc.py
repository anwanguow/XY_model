#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute & save robust XY-model helicity modulus (Upsilon) data.

This script simulates the 2D XY model for multiple temperatures and two samplers:
  - "metropolis"
  - "twist" (Metropolis + randomized global rotations; geometric gaps + random delay)

Output (compressed NPZ):
  - Ls:        int array (one per replicate)
  - Ts:        float array
  - samplers:  string array in {"metropolis","twist"}
  - U_values:  float array (helicity modulus per replicate)
  - meta_json: JSON string with all key parameters for reproducibility

Run:
    python Fig2_calc.py
"""

import math, time, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from multiprocessing import Process, Queue, cpu_count, set_start_method

# ------------------------------- User parameters -------------------------------

SIZES = [64]
J = 1.0
T_MIN, T_MAX, T_POINTS = 0.70, 1.10, 16
TEMPS = np.linspace(T_MIN, T_MAX, T_POINTS)
THERM_SWEEPS = 60_000
MEAS_SWEEPS  = 120_000
MEAS_EVERY   = 3             # prime; reduces aliasing with rotation
ADAPT_SWEEPS = 5_000
ACCEPT_STEP0 = 0.30

N_REP = 12

# How many cores of CPU do you want to use.
# N_PROCESSES = min(max(1, cpu_count()//2), 8)
N_PROCESSES = 8

SAMPLERS = ["metropolis", "twist"]
TWIST_MEAN  = 50
RNG_BASE_SEED = 20251019

OUT_DATA = "Fig2_results.npz"

# ------------------------------- Utilities -------------------------------

def print_progress(done, total, prefix="Progress", length=44):
    q = min(max(done / max(1, total), 0.0), 1.0)
    bar = "█" * int(q*length) + "·" * (length - int(q*length))
    sys.stdout.write(f"\r{prefix} |{bar}| {done}/{total} ({100*q:5.1f}%)")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n"); sys.stdout.flush()

@dataclass
class Task:
    task_id: int
    L: int
    T: float
    beta: float
    therm_sweeps: int
    meas_sweeps: int
    meas_every: int
    adapt_sweeps: int
    accept_step0: float
    sampler: str      # "metropolis" or "twist"
    twist_mean: int
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

def dir_means(theta: np.ndarray):
    dth_x = theta - np.roll(theta, -1, axis=0)
    dth_y = theta - np.roll(theta, -1, axis=1)
    cbar_x = float(np.cos(dth_x, dtype=np.float64).mean())
    sbar_x = float(np.sin(dth_x, dtype=np.float64).mean())
    cbar_y = float(np.cos(dth_y, dtype=np.float64).mean())
    sbar_y = float(np.sin(dth_y, dtype=np.float64).mean())
    return cbar_x, sbar_x, cbar_y, sbar_y

# ------------------------------- Worker -------------------------------

def worker(task: Task, pq: Queue, rq: Queue):
    try:
        rng = np.random.default_rng(task.seed)
        L = task.L; beta = task.beta
        theta = rng.uniform(0.0, 2.0*math.pi, size=(L, L))

        accept_step = task.accept_step0
        total = task.therm_sweeps + task.meas_sweeps
        stride = max(1, total // 240)

        if task.sampler == "twist":
            p_twist = 1.0 / float(task.twist_mean)
            t_delay = rng.integers(0, task.twist_mean)

        for t in range(1, task.therm_sweeps+1):
            acc = xy_metropolis_sweep(theta, beta, J, rng, accept_step)
            if t <= task.adapt_sweeps:
                rate = acc / (L*L)
                if rate < 0.30: accept_step *= 0.9
                elif rate > 0.70: accept_step *= 1.1
                accept_step = float(min(max(accept_step, 0.05), 1.25))
            if task.sampler == "twist":
                if t > t_delay and (rng.random() < p_twist):
                    global_twist(theta, rng)
            if (t % stride) == 0:
                pq.put(("prog", task.task_id, stride))

        cnt = 0
        ex_sum = ey_sum = 0.0
        px2_sum = py2_sum = 0.0

        for t in range(1, task.meas_sweeps+1):
            xy_metropolis_sweep(theta, beta, J, rng, accept_step)
            if task.sampler == "twist":
                if rng.random() < (1.0 / float(task.twist_mean)):
                    global_twist(theta, rng)
            if (t % task.meas_every) == 0:
                cbar_x, sbar_x, cbar_y, sbar_y = dir_means(theta)
                ex_sum += cbar_x; ey_sum += cbar_y
                px2_sum += sbar_x*sbar_x; py2_sum += sbar_y*sbar_y
                cnt += 1
            if (t % stride) == 0:
                pq.put(("prog", task.task_id, stride))

        if cnt == 0: raise RuntimeError("No measurements collected.")

        ex, ey = ex_sum/cnt, ey_sum/cnt
        px2, py2 = px2_sum/cnt, py2_sum/cnt
        Ux = ex - beta*(L*L)*px2
        Uy = ey - beta*(L*L)*py2
        U  = 0.5*(Ux+Uy)

        rq.put(("result", task.task_id, L, task.T, task.sampler, U))
    except Exception as e:
        rq.put(("error", task.task_id, str(e)))

# ------------------------------- Orchestration -------------------------------

def build_tasks() -> List[Task]:
    tasks=[]; tid=0
    for L in SIZES:
        for T in TEMPS:
            beta=1.0/T
            for sampler in SAMPLERS:
                for rep in range(N_REP):
                    seed = RNG_BASE_SEED + (tid+1)*10007
                    tasks.append(Task(tid,L,T,beta,THERM_SWEEPS,MEAS_SWEEPS,MEAS_EVERY,
                                      ADAPT_SWEEPS,ACCEPT_STEP0,sampler,TWIST_MEAN,seed))
                    tid+=1
    return tasks

def run_all(tasks: List[Task]):
    try: set_start_method("spawn")
    except RuntimeError: pass
    n_tasks=len(tasks); per=THERM_SWEEPS+MEAS_SWEEPS; total=n_tasks*per
    pq, rq = Queue(), Queue()
    procs=[Process(target=worker, args=(t,pq,rq), daemon=True) for t in tasks]

    active=set(); idx=0
    while idx<n_tasks and len(active)<N_PROCESSES:
        procs[idx].start(); active.add(idx); idx+=1

    done=0; got=0
    # store raw replicate scalars: U_values + labels
    L_list: List[int] = []
    T_list: List[float] = []
    sampler_list: List[str] = []
    U_list: List[float] = []

    print(f"Running {n_tasks} tasks on {N_PROCESSES} processes (each {per} MCS; total ~{total} MCS)...")
    t0=time.time()
    try:
        while got<n_tasks:
            progressed=False
            if not pq.empty():
                _, tid, inc = pq.get(); done+=inc; progressed=True; print_progress(done,total)
            if not progressed and not rq.empty():
                msg=rq.get(); tag=msg[0]
                if tag=="result":
                    _, tid, L, T, sampler, U = msg
                    L_list.append(int(L)); T_list.append(float(T))
                    sampler_list.append(str(sampler)); U_list.append(float(U))
                    got+=1; print_progress(done,total)
                    if idx<n_tasks: procs[idx].start(); active.add(idx); idx+=1
                elif tag=="error":
                    _, tid, err = msg
                    print(f"\n[ERROR] Task {tid}: {err}")
                    got+=1
                    if idx<n_tasks: procs[idx].start(); active.add(idx); idx+=1
            if not progressed and pq.empty() and rq.empty():
                time.sleep(0.05)
        for p in procs: p.join(timeout=0.1)
    finally:
        print(f"\nDone in {time.time()-t0:.1f}s")

    return L_list, T_list, sampler_list, U_list

# ------------------------------- Save results -------------------------------

def save_results_npz(L_list, T_list, sampler_list, U_list, out_path: str):
    import json, time
    Ls = np.array(L_list, dtype=np.int64)
    Ts = np.array(T_list, dtype=np.float64)
    samplers = np.array(sampler_list, dtype=np.unicode_)
    U_values = np.array(U_list, dtype=np.float64)

    meta = {
        "SIZES": SIZES, "J": J,
        "T_MIN": float(T_MIN), "T_MAX": float(T_MAX), "T_POINTS": int(T_POINTS),
        "THERM_SWEEPS": int(THERM_SWEEPS), "MEAS_SWEEPS": int(MEAS_SWEEPS),
        "MEAS_EVERY": int(MEAS_EVERY), "ADAPT_SWEEPS": int(ADAPT_SWEEPS),
        "ACCEPT_STEP0": float(ACCEPT_STEP0),
        "SAMPLERS": SAMPLERS, "TWIST_MEAN": int(TWIST_MEAN),
        "N_REP": int(N_REP), "N_PROCESSES": int(N_PROCESSES),
        "RNG_BASE_SEED": int(RNG_BASE_SEED),
        "created": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    meta_json = json.dumps(meta)

    np.savez_compressed(
        out_path,
        Ls=Ls, Ts=Ts, samplers=samplers, U_values=U_values,
        meta_json=meta_json
    )
    print(f"Saved results to: {out_path}")
    print(f"  replicates: {len(U_values)}")

# ------------------------------- Main -------------------------------

def main():
    tasks = build_tasks()
    L_list, T_list, sampler_list, U_list = run_all(tasks)
    save_results_npz(L_list, T_list, sampler_list, U_list, OUT_DATA)

if __name__ == "__main__":
    main()

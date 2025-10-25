#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, time, sys, pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from multiprocessing import Process, Queue, cpu_count, set_start_method


L = 64
J = 1.0
T_LOW  = 0.85
T_HIGH = 1.05  

THERM_SWEEPS = 40_000
MEAS_SWEEPS  = 60_000
MEAS_EVERY   = 50
ACCEPT_STEP0 = 0.40

ADAPT_SWEEPS = min(20_000, THERM_SWEEPS//3)
ACC_LOW, ACC_HIGH = 0.35, 0.65
STEP_MIN, STEP_MAX = 0.05, 1.20
N_REP = 8

# How many cores of CPU do you want to use.
# N_PROCESSES = min(max(1, cpu_count()//2), 8)
N_PROCESSES = 8
RNG_BASE_SEED = 20251019

TWIST_EVERY = 50
RESULTS_FILE = "Fig3_results.npz"

# ------------------------------- Utils -------------------------------

def print_bar(done, total, prefix="Progress", length=44):
    q = min(max(done / max(total,1), 0.0), 1.0)
    bar = "█"*int(q*length) + "·"*(length-int(q*length))
    end = "" if done < total else "\n"
    sys.stdout.write(f"\r{prefix} |{bar}| {done}/{total} ({q*100:5.1f}%)")
    sys.stdout.flush()
    if end: sys.stdout.write(end); sys.stdout.flush()

@dataclass
class Task:
    tid: int
    T: float
    beta: float
    L: int
    therm_sweeps: int
    meas_sweeps: int
    meas_every: int
    accept_step0: float
    twist_every: int
    seed_local: int  
    seed_twist: int 

# ------------------------------- XY Model -------------------------------

def one_site_update_pair(theta0: np.ndarray, theta1: np.ndarray,
                         i: int, j: int, beta: float, J: float,
                         rng_local: np.random.Generator, accept_step: float):
    L = theta0.shape[0]
    im1 = (i-1) % L; ip1 = (i+1) % L
    jm1 = (j-1) % L; jp1 = (j+1) % L
    dth = rng_local.uniform(-accept_step, accept_step)
    u   = rng_local.random()

    # chain 0
    old0 = theta0[i,j]
    n00, n01, n02, n03 = theta0[im1,j], theta0[ip1,j], theta0[i,jm1], theta0[i,jp1]
    dE0 = -J * ( math.cos(old0+dth - n00) + math.cos(old0+dth - n01)
               + math.cos(old0+dth - n02) + math.cos(old0+dth - n03)
               - math.cos(old0 - n00) - math.cos(old0 - n01)
               - math.cos(old0 - n02) - math.cos(old0 - n03) )
    if dE0 <= 0.0 or u < math.exp(-beta*dE0):
        theta0[i,j] = old0 + dth

    # chain 1
    old1 = theta1[i,j]
    n10, n11, n12, n13 = theta1[im1,j], theta1[ip1,j], theta1[i,jm1], theta1[i,jp1]
    dE1 = -J * ( math.cos(old1+dth - n10) + math.cos(old1+dth - n11)
               + math.cos(old1+dth - n12) + math.cos(old1+dth - n13)
               - math.cos(old1 - n10) - math.cos(old1 - n11)
               - math.cos(old1 - n12) - math.cos(old1 - n13) )
    if dE1 <= 0.0 or u < math.exp(-beta*dE1):
        theta1[i,j] = old1 + dth

def sweep_pair(theta0: np.ndarray, theta1: np.ndarray,
               beta: float, J: float, rng_local: np.random.Generator,
               accept_step: float) -> int:
    L = theta0.shape[0]
    acc_count = 0
    for i in range(L):
        for j in range(L):
            before = theta0[i,j]
            one_site_update_pair(theta0, theta1, i, j, beta, J, rng_local, accept_step)
            if theta0[i,j] != before:
                acc_count += 1
    return acc_count

def global_twist_in_place(theta: np.ndarray, rng: np.random.Generator):
    theta += rng.uniform(0.0, 2.0*math.pi)
    theta %= (2.0*math.pi)

def measure_G_xy(theta: np.ndarray) -> np.ndarray:
    L = theta.shape[0]; Rmax = L//2
    g = np.empty(Rmax, dtype=np.float64)
    for r in range(1, Rmax+1):
        dth_x = theta - np.roll(theta, -r, axis=0)
        dth_y = theta - np.roll(theta, -r, axis=1)
        g[r-1] = 0.5*(np.cos(dth_x, dtype=np.float64).mean()
                      + np.cos(dth_y, dtype=np.float64).mean())
    return g

# ------------------------------- Worker -------------------------------

def worker(task: Task, pq: Queue, rq: Queue):
    try:
        L = task.L; beta=task.beta
        rng_local = np.random.default_rng(task.seed_local)
        rng_twist = np.random.default_rng(task.seed_twist)
        theta0 = rng_local.uniform(0.0, 2.0*math.pi, size=(L, L))
        theta1 = theta0.copy()

        accept_step = task.accept_step0
        total = task.therm_sweeps + task.meas_sweeps
        stride = max(1, total // 240)

        # --- thermalization (paired) ---
        for t in range(1, task.therm_sweeps+1):
            acc = sweep_pair(theta0, theta1, beta, J, rng_local, accept_step)
            rate = acc / (L*L)
            if t <= ADAPT_SWEEPS:
                if rate < ACC_LOW:  accept_step = max(STEP_MIN, accept_step*0.90)
                elif rate > ACC_HIGH: accept_step = min(STEP_MAX, accept_step*1.10)
            if (t % task.twist_every) == 0:
                global_twist_in_place(theta1, rng_twist)
            if (t % stride) == 0:
                pq.put(("prog", task.tid, stride))

        # --- smoke test (same for both) ---
        dth_x = theta0 - np.roll(theta0, -1, axis=0)
        dth_y = theta0 - np.roll(theta0, -1, axis=1)
        c1 = 0.5*(np.cos(dth_x).mean() + np.cos(dth_y).mean())
        e_site = -2.0*J*c1
        pq.put(("log", task.tid, f"[T={task.T:.3f}] c1={c1:.3f} e/site={e_site:.3f} step={accept_step:.3f}"))

        # --- measurement (paired) ---
        Rmax = L//2
        g0_sum = np.zeros(Rmax, dtype=np.float64)
        g1_sum = np.zeros(Rmax, dtype=np.float64)
        cnt = 0

        for t in range(1, task.meas_sweeps+1):
            sweep_pair(theta0, theta1, beta, J, rng_local, accept_step)
            if (t % task.twist_every) == 0:
                global_twist_in_place(theta1, rng_twist)
            if (t % task.meas_every) == 0:
                g0_sum += measure_G_xy(theta0)
                g1_sum += measure_G_xy(theta1)
                cnt += 1
            if (t % stride) == 0:
                pq.put(("prog", task.tid, stride))

        g0_mean = g0_sum / max(1, cnt)
        g1_mean = g1_sum / max(1, cnt)
        rq.put(("result", task.tid, task.T, g0_mean, g1_mean))
    except Exception as e:
        rq.put(("error", task.tid, str(e)))

# ------------------------------- run & collect -------------------------------

def build_tasks() -> List[Task]:
    tasks=[]; tid=0
    for T in [T_LOW, T_HIGH]:
        beta=1.0/T
        for rep in range(N_REP):
            seed_local = RNG_BASE_SEED + (tid+1)*10007
            seed_twist  = RNG_BASE_SEED + (tid+1)*7919
            tasks.append(Task(tid,T,beta,L,THERM_SWEEPS,MEAS_SWEEPS,MEAS_EVERY,
                              ACCEPT_STEP0,TWIST_EVERY,seed_local,seed_twist))
            tid+=1
    return tasks

def run_all(tasks: List[Task]):
    try: set_start_method("spawn")
    except RuntimeError: pass

    n=len(tasks); per=THERM_SWEEPS+MEAS_SWEEPS; total=n*per
    pq, rq = Queue(), Queue()
    procs=[Process(target=worker, args=(t,pq,rq), daemon=True) for t in tasks]

    active=set(); idx=0
    while idx<n and len(active)<N_PROCESSES:
        procs[idx].start(); active.add(idx); idx+=1

    done=0; got=0
    results: Dict[float, List[Tuple[np.ndarray,np.ndarray]]] = {}

    print(f"Running {n} paired tasks on {N_PROCESSES} processes (each {per} MCS; total ~{total} MCS)...")
    t0=time.time()
    try:
        while got<n:
            progressed=False
            if not pq.empty():
                tag, tid, val = pq.get()
                if tag=="prog": done+=val; print_bar(done,total); progressed=True
                elif tag=="log": sys.stdout.write("\n"+val+"\n"); progressed=True
            if not progressed and not rq.empty():
                msg=rq.get(); tag=msg[0]
                if tag=="result":
                    _, tid, T, g0, g1 = msg
                    results.setdefault(float(T), []).append((g0, g1))
                    got+=1; print_bar(done,total)
                    if idx<n: procs[idx].start(); active.add(idx); idx+=1
                elif tag=="error":
                    _, tid, err = msg; print(f"\n[ERROR] Task {tid}: {err}")
                    got+=1
                    if idx<n: procs[idx].start(); active.add(idx); idx+=1
            if not progressed and pq.empty() and rq.empty():
                time.sleep(0.05)
        for p in procs: p.join(timeout=0.1)
    finally:
        print(f"\nDone in {time.time()-t0:.1f}s")

    return results

# ------------------------------- main -------------------------------

def main():
    tasks = build_tasks()
    results = run_all(tasks)
    with open(RESULTS_FILE, "wb") as f:
        pickle.dump(dict(results), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved results to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()

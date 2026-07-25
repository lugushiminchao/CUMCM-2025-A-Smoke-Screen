# -*- coding: utf-8 -*-
"""
点目标模型：问题1–5 统一优化 + 与圆柱结果对比（遮蔽时长 / 运行速度）

判据: 云团球心到导弹-真目标中心(0,200,5) 视线段距离 <= 10 m，s∈[0,1]
与圆柱严格全遮蔽共用 common.smoke_geom，仅 mode='point'
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    from openpyxl import Workbook, load_workbook
except Exception:
    Workbook = None
    load_workbook = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.smoke_geom import (  # noqa: E402
    DROP_GAP,
    FY0,
    MISSILES,
    T_HIT,
    UAVS,
    _MISSILE_CACHE,
    device_info,
    heading_deg,
    kinematics,
    kinematics_batch,
    shield_duration,
    union_duration_batch,
    union_duration_single_path,
)

MODE = "point"
OUT_ROOT = ROOT / "点模型对比"

def display_path(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()

OUT_ROOT.mkdir(exist_ok=True)

UAV_NAMES = ["FY1", "FY2", "FY3", "FY4", "FY5"]
MIS_NAMES = ["M1", "M2", "M3"]

# 圆柱侧已有结果（用于对比表）
CYL = {
    "Q1": {"duration": 1.362100, "elapsed_s": 446.24},
    "Q2": {"duration": 4.542008, "elapsed_s": 1778.92},
    "Q3": {"duration": 7.608993, "elapsed_s": 2658.26},
    "Q4": {"duration": 10.612404, "elapsed_s": 1753.23},
    "Q5": {
        "duration": 21.980761,
        "M1": 7.609280,
        "M2": 10.649877,
        "M3": 3.721603,
        "elapsed_s": 2312.94,
    },
}


def aim_theta(fy0, target_xy=(0.0, 0.0)):
    dxy = np.array(target_xy, dtype=float) - np.asarray(fy0, dtype=float)[:2]
    return float(np.arctan2(dxy[1], dxy[0]))


def mis_info(name):
    m0 = MISSILES[name]
    u, v, t_hit, dist = _MISSILE_CACHE[name]
    return m0, v, t_hit, dist


def highres(bombs, missile="M1", n=150000):
    if not bombs:
        return 0.0, [], []
    return union_duration_single_path(
        bombs, mode=MODE, pts=None, n=n, refine=True, missile=missile
    )


def bomb_info(uav, missile, th, v, td, tau, bomb_id=1):
    fy0 = UAVS[uav]
    v_fy, p_drop, t_det, p_det = kinematics(th, v, td, tau, fy0=fy0)
    dur, segs = shield_duration(
        p_det, t_det, mode=MODE, pts=None, n=80000, refine=True, missile=missile
    )
    return {
        "uav": uav,
        "missile": missile,
        "theta": float(th),
        "heading_deg": heading_deg(th),
        "v": float(v),
        "t_drop": float(td),
        "tau": float(tau),
        "t_det": float(t_det),
        "P_drop": p_drop.tolist(),
        "P_det": p_det.tolist(),
        "duration": float(dur),
        "segments": segs,
        "bomb_id": bomb_id,
    }


# ---------------- Q1: 固定策略 ----------------
def run_q1():
    t0 = time.perf_counter()
    # 题面固定: v=120 朝假目标, t_drop=1.5, tau=3.6
    th = aim_theta(FY0, (0.0, 0.0))
    v, td, tau = 120.0, 1.5, 3.6
    v_fy, p_drop, t_det, p_det = kinematics(th, v, td, tau, fy0=FY0)
    dur, segs = shield_duration(
        p_det, t_det, mode=MODE, n=200000, refine=True, missile="M1"
    )
    elapsed = time.perf_counter() - t0
    res = {
        "question": "Q1",
        "mode": MODE,
        "duration": float(dur),
        "segments": segs,
        "theta": th,
        "heading_deg": heading_deg(th),
        "v": v,
        "t_drop": td,
        "tau": tau,
        "t_det": float(t_det),
        "P_drop": p_drop.tolist(),
        "P_det": p_det.tolist(),
        "elapsed_s": elapsed,
        "device": device_info(),
    }
    print(f"[Q1-point] dur={dur:.6f}s segs={segs} elapsed={elapsed:.2f}s")
    return res


# ---------------- 单弹优化（通用） ----------------
def optimize_single(uav, missile, n_rand=2200, n_top=6, use_gpu=True, warms=None, seed=0):
    fy0 = UAVS[uav]
    m0, _, t_hit, _ = mis_info(missile)
    th0 = aim_theta(fy0)
    th_m = aim_theta(fy0, m0[:2])
    th_t = aim_theta(fy0, (0.0, 200.0))
    rng = np.random.default_rng((abs(hash((uav, missile, seed))) % (2**32)))

    seeds = list(warms or [])
    seeds.extend(
        [
            [th0, 120, 0.0, 3.0],
            [th0, 140, 0.0, 2.5],
            [th0, 70, 0.0, 2.5],
            [th_m, 120, 1.0, 4.0],
            [th_m, 140, 0.0, 3.0],
            [th_t, 110, 2.0, 5.0],
            [th_t, 140, 1.0, 3.5],
            [np.pi, 140, 0.0, 3.5],
            [np.pi, 70, 0.0, 2.5],
        ]
    )
    # known point Q2-ish / cylinder seeds as warm
    if uav == "FY1" and missile == "M1":
        seeds.extend(
            [
                [3.0880757565, 71.889, 0.0, 2.50324],
                [3.0825785063, 70.0, 0.0, 2.48412],
                [3.13544, 140.0, 0.0, 3.611],
            ]
        )
    if uav == "FY2" and missile == "M1":
        seeds.append([5.34404, 136.85, 8.551, 3.986])
    if uav == "FY2" and missile == "M2":
        seeds.extend([[5.035, 136.1, 5.60, 1.95], [3.963, 134.6, 5.0, 3.0]])
    if uav == "FY3" and missile == "M1":
        seeds.append([2.13758, 92.96, 31.733, 7.646])
    if uav == "FY3" and missile == "M2":
        seeds.append([1.515, 133.4, 24.29, 0.30])
    if uav == "FY4" and missile == "M2":
        seeds.append([4.147, 81.6, 12.54, 11.86])
    if uav == "FY5" and missile == "M3":
        seeds.extend([[2.010, 140.0, 12.44, 0.30], [2.03, 140.0, 12.0, 1.0]])
    if uav == "FY5" and missile == "M1":
        seeds.append([1.780, 136.7, 12.55, 2.64])

    ths = (
        [th0 + d for d in np.linspace(-1.2, 1.2, 11)]
        + [th_m + d for d in np.linspace(-0.8, 0.8, 7)]
        + [th_t + d for d in np.linspace(-0.8, 0.8, 7)]
        + [s[0] for s in seeds]
    )
    vs = [70, 85, 100, 120, 140]
    tds = list(np.linspace(0.0, min(48.0, t_hit * 0.85), 14))
    taus = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    grid = []
    for th in ths[::1]:
        for v in vs:
            for td in tds[::2]:
                for tau in taus[::2]:
                    grid.append([th, v, td, tau])

    Xr = np.column_stack(
        [
            (th0 + (rng.random(n_rand) - 0.5) * 2.5) % (2 * np.pi),
            70 + rng.random(n_rand) * 70,
            rng.random(n_rand) * min(48.0, t_hit * 0.85),
            0.4 + rng.random(n_rand) * 12,
        ]
    )
    Xi = []
    for _ in range(800):
        th = th0 + (rng.random() - 0.5) * 1.2
        v = 70 + rng.random() * 70
        t_det_g = 2.0 + rng.random() * min(42.0, t_hit * 0.75)
        tau = 0.5 + rng.random() * 10
        td = max(0.0, t_det_g - tau)
        Xi.append([th, v, td, tau])

    X = np.vstack(
        [np.asarray(seeds, float), np.asarray(grid, float), Xr, np.asarray(Xi, float)]
    )
    X[:, 0] %= 2 * np.pi
    X[:, 1] = np.clip(X[:, 1], 70, 140)
    X[:, 2] = np.clip(X[:, 2], 0, 55)
    X[:, 3] = np.clip(X[:, 3], 0.3, 16)
    X = np.unique(np.round(X, 5), axis=0)

    _, pdet, tdet = kinematics_batch(X[:, 0], X[:, 1], X[:, 2], X[:, 3], fy0=fy0)
    durs, _ = union_duration_batch(
        pdet[:, None, :],
        tdet[:, None],
        mode=MODE,
        n_time=4000,
        use_gpu=use_gpu,
        missile=missile,
    )
    order = np.argsort(-durs)
    best_x = X[order[0]].copy()
    best_d = float(durs[order[0]])

    lb = np.array([0.0, 70.0, 0.0, 0.3])
    ub = np.array([2 * np.pi, 140.0, 55.0, 16.0])
    for k in range(min(n_top, len(order))):
        if durs[order[k]] <= 0:
            break
        x = X[order[k]].copy()
        span = np.array([0.1, 10.0, 1.5, 1.5])
        val = float(durs[order[k]])
        for _ in range(24):
            improved = False
            for d in range(4):
                for sgn in (-1.0, 1.0):
                    trial = x.copy()
                    trial[d] += sgn * span[d]
                    trial[0] %= 2 * np.pi
                    trial = np.clip(trial, lb, ub)
                    _, pd, td_ = kinematics_batch(
                        [trial[0]], [trial[1]], [trial[2]], [trial[3]], fy0=fy0
                    )
                    dv, _ = union_duration_batch(
                        pd[:, None, :],
                        td_[:, None],
                        mode=MODE,
                        n_time=4500,
                        use_gpu=use_gpu,
                        missile=missile,
                    )
                    if dv[0] > val + 1e-9:
                        val = float(dv[0])
                        x = trial
                        improved = True
            noise = (rng.random((28, 4)) - 0.5) * 2 * span
            trials = np.clip(x[None, :] + noise, lb, ub)
            trials[:, 0] %= 2 * np.pi
            _, pd, td_ = kinematics_batch(
                trials[:, 0], trials[:, 1], trials[:, 2], trials[:, 3], fy0=fy0
            )
            dvs, _ = union_duration_batch(
                pd[:, None, :],
                td_[:, None],
                mode=MODE,
                n_time=4000,
                use_gpu=use_gpu,
                missile=missile,
            )
            j = int(np.argmax(dvs))
            if dvs[j] > val + 1e-9:
                val = float(dvs[j])
                x = trials[j]
                improved = True
            span *= 0.9 if improved else 1.05
            span = np.minimum(span, np.array([0.15, 15, 2.5, 2.5]))
        if val > best_d:
            best_d = val
            best_x = x

    info = bomb_info(uav, missile, *best_x, bomb_id=1)
    return float(info["duration"]), best_x, info


def optimize_multi(uav, missile, n_bombs, base_x, use_gpu=True, also_heading=False):
    fy0 = UAVS[uav]
    rng = np.random.default_rng(abs(hash((uav, missile, n_bombs, "pt"))) % (2**32))
    cands = [(float(base_x[0]), float(base_x[1]))]
    if also_heading:
        th0, v0 = float(base_x[0]), float(base_x[1])
        for dth in [-0.1, 0.0, 0.1]:
            for dv in [-15, 0, 15]:
                cands.append((th0 + dth, float(np.clip(v0 + dv, 70, 140))))
    if uav == "FY1" and missile == "M1":
        cands.append((3.1354404025, 139.9982757891))
        cands.append((3.0880757565, 71.889))
        cands.append((np.pi, 140.0))

    seen, uniq = set(), []
    for th, v in cands:
        key = (round(th, 5), round(v, 3))
        if key not in seen:
            seen.add(key)
            uniq.append((th, v))

    best_total, best_bombs = -1.0, []
    for th, v in uniq:
        dim = 2 * n_bombs
        lb = np.array([0.0, 0.3] * n_bombs)
        ub = np.array([40.0, 14.0] * n_bombs)

        def fix(x):
            x = np.asarray(x, dtype=float).copy()
            ts = []
            for i in range(n_bombs):
                t = max(0.0, x[2 * i])
                if i > 0:
                    t = max(t, ts[-1] + DROP_GAP)
                ts.append(t)
                x[2 * i] = t
                x[2 * i + 1] = max(0.3, min(16.0, x[2 * i + 1]))
            return x

        def eval_x(xs, n_time=4000):
            xs = np.atleast_2d(xs)
            N = xs.shape[0]
            dets, tds = [], []
            for i in range(n_bombs):
                _, pd, td_ = kinematics_batch(
                    np.full(N, th),
                    np.full(N, v),
                    xs[:, 2 * i],
                    xs[:, 2 * i + 1],
                    fy0=fy0,
                )
                dets.append(pd)
                tds.append(td_)
            P = np.stack(dets, axis=1)
            T = np.stack(tds, axis=1)
            return union_duration_batch(
                P, T, mode=MODE, n_time=n_time, use_gpu=use_gpu, missile=missile
            )

        rows = []
        btd, btau = float(base_x[2]), float(base_x[3])
        if n_bombs == 2:
            for dt in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
                for dtau in [-1.0, 0.0, 1.0, 2.0, 3.0]:
                    rows.append([btd, btau, btd + dt, max(0.5, btau + dtau)])
            if uav == "FY1" and missile == "M1":
                rows.append([0.003, 3.611, 3.702, 5.337])
                rows.append([0.0, 2.5, 3.5, 5.0])
        else:
            for dt1 in [1.2, 2.0, 3.0, 4.0]:
                for dt2 in [1.2, 2.0, 3.5, 5.0]:
                    rows.append(
                        [
                            btd,
                            btau,
                            btd + dt1,
                            max(0.5, btau + 1.0),
                            btd + dt1 + dt2,
                            max(0.5, btau + 2.0),
                        ]
                    )
            if uav == "FY1" and missile == "M1":
                rows.append([0.00298, 3.61113, 3.70246, 5.33749, 5.56948, 6.04053])
                rows.append([0.0, 2.5, 3.5, 4.5, 6.0, 5.5])

        n_rand = 1000 if n_bombs == 2 else 800
        Xr = lb + rng.random((n_rand, dim)) * (ub - lb)
        X = np.array([fix(r) for r in np.vstack([np.asarray(rows, float), Xr])])
        durs, _ = eval_x(X, n_time=3500)
        x_best = X[int(np.argmax(durs))].copy()
        best = float(np.max(durs))

        span = np.array([1.0, 1.0] * n_bombs)
        for _ in range(18):
            improved = False
            for d in range(dim):
                for sgn in (-1.0, 1.0):
                    trial = fix(
                        np.clip(x_best + sgn * span * np.eye(1, dim, d).ravel(), lb, ub)
                    )
                    val = float(eval_x(trial[None, :], n_time=4000)[0][0])
                    if val > best + 1e-9:
                        best, x_best, improved = val, trial, True
            noise = (rng.random((20, dim)) - 0.5) * 2 * span
            trials = np.array([fix(np.clip(x_best + n, lb, ub)) for n in noise])
            vals = eval_x(trials, n_time=3500)[0]
            j = int(np.argmax(vals))
            if vals[j] > best + 1e-9:
                best, x_best, improved = float(vals[j]), trials[j], True
            span *= 0.9 if improved else 1.04

        bombs = []
        for i in range(n_bombs):
            bombs.append(
                bomb_info(
                    uav,
                    missile,
                    th,
                    v,
                    float(x_best[2 * i]),
                    float(x_best[2 * i + 1]),
                    bomb_id=i + 1,
                )
            )
        pairs = [(np.array(b["P_det"]), b["t_det"]) for b in bombs]
        total, per, segs = highres(pairs, missile=missile, n=100000)
        for i, b in enumerate(bombs):
            if i < len(per):
                b["duration"] = float(per[i])
        if total > best_total:
            best_total, best_bombs = total, bombs
    return best_total, best_bombs


# ---------------- Q2 ----------------
def run_q2(use_gpu=True):
    t0 = time.perf_counter()
    d, x, info = optimize_single("FY1", "M1", n_rand=2500, n_top=7, use_gpu=use_gpu)
    elapsed = time.perf_counter() - t0
    res = {
        "question": "Q2",
        "mode": MODE,
        "duration": d,
        "info": info,
        "x": x.tolist(),
        "elapsed_s": elapsed,
        "device": device_info(),
    }
    print(
        f"[Q2-point] dur={d:.6f}s heading={info['heading_deg']:.4f} "
        f"v={info['v']:.4f} td={info['t_drop']:.4f} tau={info['tau']:.4f} "
        f"elapsed={elapsed:.2f}s"
    )
    return res


# ---------------- Q3 ----------------
def run_q3(use_gpu=True):
    t0 = time.perf_counter()
    d1, x1, info1 = optimize_single("FY1", "M1", n_rand=2200, n_top=6, use_gpu=use_gpu)
    print(f"  single={d1:.4f}s")
    # Q3 known multi seed evaluate
    th, v = 3.1354404025, 139.9982757891
    seed_bombs = [
        bomb_info("FY1", "M1", th, v, 0.0029830233, 3.6111304684, 1),
        bomb_info("FY1", "M1", th, v, 3.7024597386, 5.3374867068, 2),
        bomb_info("FY1", "M1", th, v, 5.5694839986, 6.0405298954, 3),
    ]
    pairs = [(np.array(b["P_det"]), b["t_det"]) for b in seed_bombs]
    d_seed, per_seed, segs_seed = highres(pairs, "M1", n=150000)
    print(f"  Q3-seed multi={d_seed:.4f}s")

    d2, b2 = optimize_multi("FY1", "M1", 2, x1, use_gpu=use_gpu, also_heading=True)
    print(f"  double={d2:.4f}s")
    d3, b3 = optimize_multi("FY1", "M1", 3, x1, use_gpu=use_gpu, also_heading=True)
    print(f"  triple={d3:.4f}s")

    cands = [(d1, [info1]), (d_seed, seed_bombs), (d2, b2), (d3, b3)]
    cands = [(d, b) for d, b in cands if b and d > 0]
    cands.sort(key=lambda t: -t[0])
    best_d, best_b = cands[0]
    for i, b in enumerate(best_b, 1):
        b["bomb_id"] = i
    pairs = [(np.array(b["P_det"]), b["t_det"]) for b in best_b]
    total, per, segs = highres(pairs, "M1", n=200000)
    for i, b in enumerate(best_b):
        if i < len(per):
            b["duration"] = float(per[i])
    elapsed = time.perf_counter() - t0
    res = {
        "question": "Q3",
        "mode": MODE,
        "duration": float(total),
        "per_bomb": list(per),
        "segments": segs,
        "bombs": best_b,
        "elapsed_s": elapsed,
        "device": device_info(),
    }
    print(f"[Q3-point] union={total:.6f}s n={len(best_b)} elapsed={elapsed:.2f}s")
    return res


# ---------------- Q4 ----------------
def run_q4(use_gpu=True):
    t0 = time.perf_counter()
    solos = {}
    for name, seed in [
        ("FY1", 1),
        ("FY2", 2),
        ("FY3", 3),
    ]:
        d, x, info = optimize_single(name, "M1", n_rand=2000, n_top=5, use_gpu=use_gpu, seed=seed)
        solos[name] = (d, x, info)
        print(f"  {name} solo={d:.4f}s th={info['heading_deg']:.2f} v={info['v']:.1f}")

    # joint: concatenate independent optima then local refine on 12-D
    x0 = np.concatenate([solos[n][1] for n in ["FY1", "FY2", "FY3"]])
    fy_list = [UAVS[n] for n in ["FY1", "FY2", "FY3"]]

    def eval_joint(X, n_time=4000):
        X = np.atleast_2d(X)
        dets, tds = [], []
        for i, fy0 in enumerate(fy_list):
            _, pd, td_ = kinematics_batch(
                X[:, 4 * i],
                X[:, 4 * i + 1],
                X[:, 4 * i + 2],
                X[:, 4 * i + 3],
                fy0=fy0,
            )
            dets.append(pd)
            tds.append(td_)
        P = np.stack(dets, axis=1)
        T = np.stack(tds, axis=1)
        return union_duration_batch(
            P, T, mode=MODE, n_time=n_time, use_gpu=use_gpu, missile="M1"
        )[0]

    def clip_joint(x):
        x = np.asarray(x, float).copy()
        for i in range(3):
            x[4 * i] %= 2 * np.pi
            x[4 * i + 1] = np.clip(x[4 * i + 1], 70, 140)
            x[4 * i + 2] = np.clip(x[4 * i + 2], 0, 50)
            x[4 * i + 3] = np.clip(x[4 * i + 3], 0.3, 16)
        return x

    x = clip_joint(x0)
    best = float(eval_joint(x[None, :])[0])
    span = np.array([0.08, 8.0, 1.2, 1.2] * 3)
    lb = np.array([0.0, 70.0, 0.0, 0.3] * 3)
    ub = np.array([2 * np.pi, 140.0, 50.0, 16.0] * 3)
    rng = np.random.default_rng(42)
    for _ in range(30):
        improved = False
        for d in range(12):
            for sgn in (-1.0, 1.0):
                trial = clip_joint(x + sgn * span * np.eye(1, 12, d).ravel())
                val = float(eval_joint(trial[None, :], n_time=4500)[0])
                if val > best + 1e-9:
                    best, x, improved = val, trial, True
        noise = (rng.random((24, 12)) - 0.5) * 2 * span
        trials = np.array([clip_joint(x + n) for n in noise])
        vals = eval_joint(trials, n_time=3500)
        j = int(np.argmax(vals))
        if vals[j] > best + 1e-9:
            best, x, improved = float(vals[j]), trials[j], True
        span *= 0.9 if improved else 1.04

    bombs = []
    for i, name in enumerate(["FY1", "FY2", "FY3"]):
        bombs.append(
            bomb_info(
                name,
                "M1",
                x[4 * i],
                x[4 * i + 1],
                x[4 * i + 2],
                x[4 * i + 3],
                bomb_id=1,
            )
        )
    pairs = [(np.array(b["P_det"]), b["t_det"]) for b in bombs]
    total, per, segs = highres(pairs, "M1", n=200000)
    for i, b in enumerate(bombs):
        if i < len(per):
            b["duration"] = float(per[i])
    elapsed = time.perf_counter() - t0
    res = {
        "question": "Q4",
        "mode": MODE,
        "duration": float(total),
        "per_bomb": list(per),
        "segments": segs,
        "bombs": bombs,
        "solo": {n: solos[n][0] for n in solos},
        "elapsed_s": elapsed,
        "device": device_info(),
    }
    print(f"[Q4-point] union={total:.6f}s per={per} elapsed={elapsed:.2f}s")
    return res


# ---------------- Q5 ----------------
def run_q5(use_gpu=True):
    t0 = time.perf_counter()
    assignment = {
        "M1": ["FY1"],
        "M2": ["FY2", "FY4", "FY3"],
        "M3": ["FY5"],
    }
    print("分派:", assignment)
    details = {}
    totals = {}
    used = set()

    for missile, uavs in assignment.items():
        print(f"\n---- {missile} {uavs} ----")
        bombs_m = []
        for ui, uav in enumerate(uavs):
            try_triple = ui == 0 and missile in ("M1",)
            try_double = ui == 0 or (missile == "M2" and uav == "FY2")
            if missile == "M3":
                try_double = True
                try_triple = True
            d1, x1, info1 = optimize_single(
                uav, missile, n_rand=1800, n_top=5, use_gpu=use_gpu, seed=ui + 10
            )
            print(
                f"  {uav} 单弹: {d1:.4f}s th={info1['heading_deg']:.2f} v={info1['v']:.1f}"
            )
            cands = [(d1, [info1] if d1 > 1e-9 else [])]
            if d1 > 1e-6 and try_double:
                d2, b2 = optimize_multi(
                    uav, missile, 2, x1, use_gpu=use_gpu, also_heading=(uav == "FY1")
                )
                print(f"  {uav} 双弹: {d2:.4f}s")
                if d2 > 0:
                    cands.append((d2, b2))
            if d1 > 1e-6 and try_triple:
                d3, b3 = optimize_multi(
                    uav, missile, 3, x1, use_gpu=use_gpu, also_heading=(uav == "FY1")
                )
                print(f"  {uav} 三弹: {d3:.4f}s")
                if d3 > 0:
                    cands.append((d3, b3))
            # FY1-M1 also try known Q3 multi
            if uav == "FY1" and missile == "M1":
                th, v = 3.1354404025, 139.9982757891
                sb = [
                    bomb_info("FY1", "M1", th, v, 0.0029830233, 3.6111304684, 1),
                    bomb_info("FY1", "M1", th, v, 3.7024597386, 5.3374867068, 2),
                    bomb_info("FY1", "M1", th, v, 5.5694839986, 6.0405298954, 3),
                ]
                pr = [(np.array(b["P_det"]), b["t_det"]) for b in sb]
                ds, _, _ = highres(pr, "M1", n=100000)
                print(f"  FY1 Q3种子: {ds:.4f}s")
                cands.append((ds, sb))

            cands = [(d, b) for d, b in cands if b]
            if not cands:
                print(f"  {uav} 无效")
                continue
            cands.sort(key=lambda t: -t[0])
            bd, bb = cands[0]
            bb = [b for b in bb if b.get("duration", 0) > 1e-9] or bb[:1]
            for i, b in enumerate(bb, 1):
                b["bomb_id"] = i
            print(f"  => 选用 {len(bb)} 弹 ~{bd:.4f}s")
            bombs_m.extend(bb)
            used.add(uav)

        pairs = [(np.array(b["P_det"]), b["t_det"]) for b in bombs_m]
        total, per, segs = highres(pairs, missile, n=180000)
        for i, b in enumerate(bombs_m):
            if i < len(per):
                b["duration"] = float(per[i])
        bombs_m = [b for b in bombs_m if b.get("duration", 0) > 1e-9] or bombs_m
        totals[missile] = total
        details[missile] = {"segments": segs, "per": list(per), "bombs": bombs_m}
        print(f"  {missile} 并集={total:.6f}s segs={segs}")

    # assemble
    all_bombs = []
    for m in MIS_NAMES:
        all_bombs.extend(details[m]["bombs"])
    by_uav = {n: [] for n in UAV_NAMES}
    for b in all_bombs:
        by_uav[b["uav"]].append(b)
    final = []
    for name in UAV_NAMES:
        bl = sorted(by_uav[name], key=lambda x: x["t_drop"])[:3]
        if not bl:
            continue
        th0, v0 = bl[0]["theta"], bl[0]["v"]
        for i, b in enumerate(bl, 1):
            b = dict(b)
            b["bomb_id"] = i
            if abs(b["theta"] - th0) > 1e-9 or abs(b["v"] - v0) > 1e-9:
                _, p_drop, t_det, p_det = kinematics(
                    th0, v0, b["t_drop"], b["tau"], fy0=UAVS[name]
                )
                b["theta"] = th0
                b["heading_deg"] = heading_deg(th0)
                b["v"] = v0
                b["P_drop"] = p_drop.tolist()
                b["P_det"] = p_det.tolist()
                b["t_det"] = float(t_det)
            final.append(b)
    all_bombs = final

    missile_totals = {}
    missile_details = {}
    for m in MIS_NAMES:
        bombs_m = [b for b in all_bombs if b["missile"] == m]
        pairs = [(np.array(b["P_det"]), b["t_det"]) for b in bombs_m]
        total, per, segs = highres(pairs, m, n=200000)
        for i, b in enumerate(bombs_m):
            if i < len(per):
                b["duration"] = float(per[i])
        missile_totals[m] = total
        missile_details[m] = {
            "segments": segs,
            "per": list(per),
            "n_bombs": len(bombs_m),
        }
        print(f"终算 {m}: {total:.6f}s n={len(bombs_m)}")

    total_sum = float(sum(missile_totals.values()))
    elapsed = time.perf_counter() - t0
    res = {
        "question": "Q5",
        "mode": MODE,
        "total_sum": total_sum,
        "missile_totals": missile_totals,
        "missile_details": missile_details,
        "bombs": all_bombs,
        "elapsed_s": elapsed,
        "device": device_info(),
        "assignment": assignment,
    }
    print(f"[Q5-point] sum={total_sum:.6f}s elapsed={elapsed:.2f}s")
    return res


def conv(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, list):
        return [conv(x) for x in o]
    if isinstance(o, tuple):
        return [conv(x) for x in o]
    if isinstance(o, dict):
        return {k: conv(v) for k, v in o.items()}
    return o


def write_comparison(results):
    lines = []
    lines.append("==== 点模型 vs 圆柱严格全遮蔽 对比 ====")
    lines.append(f"device: {device_info()}")
    lines.append("")
    lines.append("判据差异:")
    lines.append("  点模型: 仅遮蔽真目标几何中心 (0,200,5)")
    lines.append("  圆柱:   严格全遮蔽圆柱表面全部采样点")
    lines.append("")
    lines.append(
        f"{'问题':<6} {'点时长(s)':>12} {'圆柱时长(s)':>12} {'Δ时长':>10} "
        f"{'点耗时(s)':>12} {'圆柱耗时(s)':>12} {'加速比':>8}"
    )
    lines.append("-" * 80)

    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        r = results[q]
        if q == "Q5":
            pt = float(r["total_sum"])
            cy = CYL[q]["duration"]
            pe = float(r["elapsed_s"])
            ce = CYL[q]["elapsed_s"]
        else:
            pt = float(r["duration"])
            cy = CYL[q]["duration"]
            pe = float(r["elapsed_s"])
            ce = CYL[q]["elapsed_s"]
        d_dur = pt - cy
        speedup = ce / pe if pe > 1e-9 else float("inf")
        lines.append(
            f"{q:<6} {pt:12.6f} {cy:12.6f} {d_dur:+10.6f} "
            f"{pe:12.2f} {ce:12.2f} {speedup:8.2f}x"
        )
        rows.append(
            {
                "question": q,
                "point_duration": pt,
                "cylinder_duration": cy,
                "delta_duration": d_dur,
                "point_elapsed_s": pe,
                "cylinder_elapsed_s": ce,
                "speedup": speedup,
            }
        )

    lines.append("")
    lines.append("问题5 分导弹:")
    r5 = results["Q5"]
    for m in MIS_NAMES:
        pt = float(r5["missile_totals"][m])
        cy = CYL["Q5"][m]
        lines.append(f"  {m}: 点={pt:.6f}s  圆柱={cy:.6f}s  Δ={pt-cy:+.6f}s")
    lines.append("")
    lines.append("结论摘要:")
    lines.append("  1) 点模型遮蔽时长普遍 ≥ 圆柱严格全遮蔽（判据更宽松）。")
    lines.append("  2) 点模型优化/评估显著更快（目标点数 1 vs 数百）。")
    lines.append("  3) 两者最优策略参数往往接近，圆柱可作为稳健性校验。")
    text = "\n".join(lines) + "\n"
    (OUT_ROOT / "point_vs_cylinder_comparison.txt").write_text(text, encoding="utf-8")
    with open(OUT_ROOT / "point_vs_cylinder_comparison.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "table": rows,
                "point_results": conv(results),
                "cylinder_reference": CYL,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(text)
    return text


def bench_eval_throughput(use_gpu=True, n_samples=800):
    """同预算下点/圆柱批量评估吞吐对比（公平速度基准）。"""
    from common.smoke_geom import PTS_FAST, PTS_OPT

    rng = np.random.default_rng(0)
    th = rng.random(n_samples) * 2 * np.pi
    v = 70 + rng.random(n_samples) * 70
    td = rng.random(n_samples) * 20
    tau = 0.5 + rng.random(n_samples) * 8
    _, P, T = kinematics_batch(th, v, td, tau, fy0=FY0)
    P = P[:, None, :]
    T = T[:, None]

    # warmup
    union_duration_batch(P[:64], T[:64], mode="point", n_time=3000, use_gpu=use_gpu)
    union_duration_batch(
        P[:64], T[:64], mode="cylinder", pts=PTS_FAST, n_time=3000, use_gpu=use_gpu
    )

    t0 = time.perf_counter()
    union_duration_batch(P, T, mode="point", n_time=3000, use_gpu=use_gpu)
    t_point = time.perf_counter() - t0

    t0 = time.perf_counter()
    union_duration_batch(P, T, mode="cylinder", pts=PTS_FAST, n_time=3000, use_gpu=use_gpu)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    union_duration_batch(P, T, mode="cylinder", pts=PTS_OPT, n_time=3000, use_gpu=use_gpu)
    t_opt = time.perf_counter() - t0

    bench = {
        "n_samples": n_samples,
        "n_time": 3000,
        "point_s": t_point,
        "cylinder_fast_s": t_fast,
        "cylinder_opt_s": t_opt,
        "speedup_vs_fast": t_fast / t_point if t_point > 0 else None,
        "speedup_vs_opt": t_opt / t_point if t_point > 0 else None,
        "pts_fast": int(PTS_FAST.shape[0]),
        "pts_opt": int(PTS_OPT.shape[0]),
    }
    print(
        f"[bench] N={n_samples} point={t_point:.3f}s  "
        f"cyl_fast({bench['pts_fast']}pts)={t_fast:.3f}s ({bench['speedup_vs_fast']:.1f}x)  "
        f"cyl_opt({bench['pts_opt']}pts)={t_opt:.3f}s ({bench['speedup_vs_opt']:.1f}x)"
    )
    with open(OUT_ROOT / "eval_throughput_bench.json", "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2)
    return bench


def main():
    print("==== 点模型 Q1–Q5 优化 + 圆柱对比 ====")
    print("device:", device_info())
    use_gpu = True
    results = {}

    print("\n---- 评估吞吐基准（同样本数）----")
    results["_bench"] = bench_eval_throughput(use_gpu=use_gpu, n_samples=800)

    results["Q1"] = run_q1()
    with open(OUT_ROOT / "q1_point_result.json", "w", encoding="utf-8") as f:
        json.dump(conv(results["Q1"]), f, ensure_ascii=False, indent=2)

    results["Q2"] = run_q2(use_gpu=use_gpu)
    with open(OUT_ROOT / "q2_point_result.json", "w", encoding="utf-8") as f:
        json.dump(conv(results["Q2"]), f, ensure_ascii=False, indent=2)

    results["Q3"] = run_q3(use_gpu=use_gpu)
    with open(OUT_ROOT / "q3_point_result.json", "w", encoding="utf-8") as f:
        json.dump(conv(results["Q3"]), f, ensure_ascii=False, indent=2)

    results["Q4"] = run_q4(use_gpu=use_gpu)
    with open(OUT_ROOT / "q4_point_result.json", "w", encoding="utf-8") as f:
        json.dump(conv(results["Q4"]), f, ensure_ascii=False, indent=2)

    results["Q5"] = run_q5(use_gpu=use_gpu)
    with open(OUT_ROOT / "q5_point_result.json", "w", encoding="utf-8") as f:
        json.dump(conv(results["Q5"]), f, ensure_ascii=False, indent=2)

    write_comparison(results)
    print(f"全部结果目录: {display_path(OUT_ROOT)}")


if __name__ == "__main__":
    main()

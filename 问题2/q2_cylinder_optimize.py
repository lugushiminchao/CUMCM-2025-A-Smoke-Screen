# -*- coding: utf-8 -*-
"""
问题2：圆柱严格全遮蔽下，FY1 单弹最大有效遮蔽时长
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common.smoke_geom import (  # noqa: E402
    PTS_FAST,
    PTS_FINE,
    PTS_OPT,
    T_HIT,
    device_info,
    kinematics,
    kinematics_batch,
    shield_duration,
    union_duration_batch,
)

OUT = ROOT / "结果"

def display_path(path):
    return Path(path).resolve().relative_to(ROOT.parent.resolve()).as_posix()

OUT.mkdir(exist_ok=True)

MODE = "cylinder"
PTS_COARSE = PTS_FAST
PTS_MID = PTS_OPT
PTS_HI = PTS_FINE


def objective_batch(X, n_time=5000, use_gpu=True, pts=None):
    """X (N,4): theta,v,t_drop,tau"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    theta, v, td, tau = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    _, p_det, t_det = kinematics_batch(theta, v, td, tau)
    P = p_det[:, None, :]  # (N,1,3)
    T = t_det[:, None]
    if pts is None:
        pts = PTS_COARSE
    dur, per = union_duration_batch(
        P, T, mode=MODE, pts=pts, n_time=n_time, use_gpu=use_gpu
    )
    return dur


def objective_one(x, n=40000, refine=False, pts=None):
    theta, v, td, tau = map(float, x)
    _, p_drop, t_det, p_det = kinematics(theta, v, td, tau)
    if t_det >= T_HIT:
        return 0.0, p_drop, t_det, p_det
    if pts is None:
        pts = PTS_MID
    dur, segs = shield_duration(
        p_det, t_det, mode=MODE, pts=pts, n=n, refine=refine
    )
    return dur, p_drop, t_det, p_det


def local_refine(x0, lb, ub, n_time=4000, steps=60, spans=None, use_gpu=True, pts=None):
    if spans is None:
        spans = np.array([0.08, 8.0, 1.2, 1.2])
    if pts is None:
        pts = PTS_MID
    x = np.array(x0, dtype=float)
    best = float(objective_batch(x[None, :], n_time=n_time, use_gpu=use_gpu, pts=pts)[0])
    span = spans.copy()
    rng = np.random.default_rng(2025)
    for it in range(steps):
        improved = False
        # 坐标轮换
        for d in range(4):
            for sgn in (-1.0, 1.0):
                trial = x.copy()
                trial[d] += sgn * span[d]
                trial = np.clip(trial, lb, ub)
                val = float(
                    objective_batch(trial[None, :], n_time=n_time, use_gpu=use_gpu, pts=pts)[0]
                )
                if val > best + 1e-9:
                    best = val
                    x = trial
                    improved = True
        # 随机扰动
        noise = (rng.random((32, 4)) - 0.5) * 2.0 * span[None, :]
        trials = np.clip(x[None, :] + noise, lb, ub)
        vals = objective_batch(trials, n_time=n_time, use_gpu=use_gpu, pts=pts)
        j = int(np.argmax(vals))
        if vals[j] > best + 1e-9:
            best = float(vals[j])
            x = trials[j].copy()
            improved = True
        if improved:
            span *= 0.92
        else:
            span = np.minimum(span * 1.08, spans * 1.3)
        if (it + 1) % 15 == 0:
            print(f"    refine it={it+1} best={best:.4f} x={x}")
    return x, best


def main():
    t0 = time.perf_counter()
    use_gpu = True
    print("==== 问题2 圆柱严格全遮蔽 优化 ====")
    print("device:", device_info())
    print(f"t_hit={T_HIT:.6f}")
    print(f"采样: coarse={PTS_COARSE.shape[0]}, mid={PTS_MID.shape[0]}, hi={PTS_HI.shape[0]}")

    # Q1 基准
    x_q1 = np.array([np.pi, 120.0, 1.5, 3.6])
    dur_q1, _, td1, pd1 = objective_one(x_q1, n=120000, refine=True, pts=PTS_HI)
    print(f"[Q1 cylinder] {dur_q1:.6f} s")

    # 点模型最优 warm-start
    x_pt = np.array([3.0880757565, 71.8890217683, 0.0, 2.5032397513])
    dur_pt_as_cyl, *_ = objective_one(x_pt, n=80000, refine=True, pts=PTS_MID)
    print(f"[point-opt as cylinder] {dur_pt_as_cyl:.6f} s")

    lb = np.array([0.0, 70.0, 0.0, 0.3])
    ub = np.array([2 * np.pi, 140.0, 25.0, 16.0])

    th_list = np.pi + np.linspace(-0.35, 0.35, 15)
    v_list = np.array([70, 80, 90, 100, 110, 120, 130, 140], dtype=float)
    td_list = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    tau_list = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0])
    grid = np.array(np.meshgrid(th_list, v_list, td_list, tau_list, indexing="ij"))
    Xg = grid.reshape(4, -1).T

    rng = np.random.default_rng(2025)
    n_rand = 2500
    Xr = lb + rng.random((n_rand, 4)) * (ub - lb)
    n_band = 1200
    Xb = np.column_stack(
        [
            (np.pi + (rng.random(n_band) - 0.5) * 0.6) % (2 * np.pi),
            70 + rng.random(n_band) * 70,
            rng.random(n_band) * 12,
            0.5 + rng.random(n_band) * 10,
        ]
    )
    seeds = np.array(
        [
            x_q1,
            x_pt,
            [np.pi, 100, 0.0, 2.5],
            [np.pi, 140, 0.0, 2.5],
            [np.pi, 140, 0.5, 3.0],
            [np.pi, 90, 1.0, 4.0],
            [np.pi, 120, 1.5, 3.6],
            [np.pi, 70, 0.0, 3.0],
            [np.pi + 0.05, 100, 0.5, 3.5],
            [np.pi - 0.05, 100, 0.5, 3.5],
            [np.pi, 110, 0.0, 2.8],
            [np.pi, 130, 0.2, 3.2],
        ],
        dtype=float,
    )
    X = np.vstack([seeds, Xg, Xr, Xb])
    # 去重近似
    X = np.unique(np.round(X, 6), axis=0)
    n_all = len(X)
    print(f"粗搜样本 {n_all}")

    t_c = time.perf_counter()
    # 分批评估
    chunk = 2000
    durs = np.zeros(n_all)
    for i0 in range(0, n_all, chunk):
        i1 = min(n_all, i0 + chunk)
        durs[i0:i1] = objective_batch(
            X[i0:i1], n_time=4000, use_gpu=use_gpu, pts=PTS_COARSE
        )
        print(
            f"  coarse {i1}/{n_all} best={durs[:i1].max():.4f} "
            f"({time.perf_counter()-t_c:.1f}s)"
        )

    order = np.argsort(-durs)
    Xs, ds = X[order], durs[order]
    print("Top-15 粗搜:")
    for k in range(min(15, n_all)):
        xk = Xs[k]
        print(
            f"  #{k+1:2d} {ds[k]:.4f} th={np.degrees(xk[0])%360:.2f} "
            f"v={xk[1]:.1f} td={xk[2]:.3f} tau={xk[3]:.3f}"
        )

    n_top = 18
    print(f"---- 局部精修 Top-{n_top} (mid pts) ----")
    refined = []
    t_r = time.perf_counter()
    for k in range(min(n_top, n_all)):
        xk, dk = local_refine(
            Xs[k],
            lb,
            ub,
            n_time=5000,
            steps=45,
            use_gpu=use_gpu,
            pts=PTS_MID,
            spans=np.array([0.06, 6.0, 0.9, 0.9]),
        )
        # 高精度 refine 评分
        d_hi, *_ = objective_one(xk, n=60000, refine=False, pts=PTS_MID)
        refined.append((d_hi, xk, dk))
        print(f"  {k+1}/{n_top}: coarse={ds[k]:.4f} -> mid={dk:.4f} hi={d_hi:.4f}")
    refined.sort(key=lambda t: -t[0])
    x_best = refined[0][1]
    print(f"精修最优(中) {refined[0][0]:.6f}  耗时 {time.perf_counter()-t_r:.1f}s")

    # 二次精修
    x_best, _ = local_refine(
        x_best,
        lb,
        ub,
        n_time=8000,
        steps=40,
        use_gpu=use_gpu,
        pts=PTS_MID,
        spans=np.array([0.03, 3.0, 0.4, 0.4]),
    )

    # 高精度终算
    dur_ref, p_drop, t_det, p_det = objective_one(
        x_best, n=250000, refine=True, pts=PTS_HI
    )
    # ultra 复核
    dur_ultra, segs_u = shield_duration(
        p_det, t_det, mode=MODE, pts=PTS_FINE, n=300000, refine=True
    )
    v_fy, _, _, _ = kinematics(*x_best)
    heading = float(np.degrees(x_best[0]) % 360)

    print("\n==== 问题2 圆柱最优策略 ====")
    print(f"有效遮蔽时长 = {dur_ref:.6f} s  (ultra/fine {dur_ultra:.6f})")
    print(f"segments = {segs_u}")
    print(f"theta={x_best[0]:.10f} rad = {heading:.6f} deg")
    print(f"v={x_best[1]:.10f}")
    print(f"t_drop={x_best[2]:.10f}  tau={x_best[3]:.10f}")
    print(f"t_det={t_det:.10f}")
    print(f"P_drop={p_drop}")
    print(f"P_det={p_det}")
    print(f"Q1 {dur_q1:.4f} -> {dur_ref:.4f} (Δ={dur_ref-dur_q1:.4f})")

    elapsed = time.perf_counter() - t0
    result = {
        "duration": dur_ref,
        "duration_fine": dur_ultra,
        "segments": segs_u,
        "theta": float(x_best[0]),
        "heading_deg": heading,
        "v": float(x_best[1]),
        "t_drop": float(x_best[2]),
        "tau": float(x_best[3]),
        "t_det": float(t_det),
        "P_drop": p_drop.tolist(),
        "P_det": p_det.tolist(),
        "v_fy": v_fy.tolist(),
        "q1_duration": dur_q1,
        "mode": "cylinder_strict_full_cover",
        "elapsed_s": elapsed,
        "device": device_info(),
    }

    with open(OUT / "q2_cylinder_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(OUT / "q2_cylinder_result.txt", "w", encoding="utf-8") as f:
        f.write("==== 2025国赛A题 问题2 圆柱严格全遮蔽 结果 ====\n")
        f.write("判据: 云团到导弹-圆柱表面采样点视线段距离均 <= 10 m\n")
        f.write(f"有效遮蔽时长 = {dur_ref:.6f} s\n")
        f.write(f"高精度复核 = {dur_ultra:.6f} s\n")
        f.write(f"有效区间 = {segs_u}\n")
        f.write(f"航向角 theta = {x_best[0]:.10f} rad = {heading:.6f} deg\n")
        f.write(f"飞行速度 v   = {x_best[1]:.10f} m/s\n")
        f.write(f"投放时刻 t_drop = {x_best[2]:.10f} s\n")
        f.write(f"引信延时 tau    = {x_best[3]:.10f} s\n")
        f.write(f"起爆时刻 t_det  = {t_det:.10f} s\n")
        f.write(f"投放点 P_drop = {p_drop.tolist()}\n")
        f.write(f"起爆点 P_det  = {p_det.tolist()}\n")
        f.write(f"问题1圆柱时长 = {dur_q1:.6f} s\n")
        f.write(f"提升 Delta = {dur_ref - dur_q1:.6f} s\n")
        f.write(f"总运行时间 = {elapsed:.2f} s\n")
        f.write(f"device = {device_info()}\n")

    print(f"总耗时 {elapsed:.1f}s  结果: {display_path(OUT / 'q2_cylinder_result.txt')}")


if __name__ == "__main__":
    main()

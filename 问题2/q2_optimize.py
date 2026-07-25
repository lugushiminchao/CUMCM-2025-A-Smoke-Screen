# -*- coding: utf-8 -*-
"""2025国赛A题 问题2：FY1 单弹干扰 M1 最大遮蔽时长（NumPy 向量化）"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "结果"

def display_path(path):
    return Path(path).resolve().relative_to(ROOT.parent.resolve()).as_posix()

OUT.mkdir(exist_ok=True)

G = 9.8
V_MISSILE = 300.0
V_SINK = 3.0
R = 10.0
T_LIFE = 20.0
M0 = np.array([20000.0, 0.0, 2000.0])
FY0 = np.array([17800.0, 0.0, 1800.0])
TGT = np.array([0.0, 200.0, 5.0])
DIST_M0 = float(np.linalg.norm(M0))
U_M = -M0 / DIST_M0
V_M = V_MISSILE * U_M
T_HIT = DIST_M0 / V_MISSILE


def kinematics(theta, v, t_drop, tau):
    v_fy = np.array([v * np.cos(theta), v * np.sin(theta), 0.0])
    p_drop = FY0 + v_fy * t_drop
    t_det = t_drop + tau
    p_det = p_drop + np.array([v_fy[0] * tau, v_fy[1] * tau, -0.5 * G * tau * tau])
    return v_fy, p_drop, t_det, p_det


def los_dist(ts, p_det, t_det):
    ts = np.asarray(ts, dtype=float).ravel()
    M = M0[None, :] + V_M[None, :] * ts[:, None]
    C = np.empty_like(M)
    C[:, 0] = p_det[0]
    C[:, 1] = p_det[1]
    C[:, 2] = p_det[2] - V_SINK * (ts - t_det)
    AB = TGT[None, :] - M
    AC = C - M
    L2 = np.sum(AB * AB, axis=1)
    s = np.sum(AC * AB, axis=1) / np.maximum(L2, 1e-18)
    sc = np.clip(s, 0.0, 1.0)
    P = M + sc[:, None] * AB
    d = np.linalg.norm(C - P, axis=1)
    return d, s


def shield_duration(p_det, t_det, n=20000, refine=False):
    t0 = float(t_det)
    t1 = min(t_det + T_LIFE, T_HIT)
    if t1 <= t0:
        return 0.0, np.nan, np.nan
    ts = np.linspace(t0, t1, n)
    d, s = los_dist(ts, p_det, t_det)
    ok = (d <= R) & (s >= 0.0) & (s <= 1.0)
    dt = ts[1] - ts[0]
    dur = float(np.sum(ok) * dt)
    if not np.any(ok):
        return 0.0, np.nan, np.nan
    jump = np.where(np.diff(ok.astype(np.int8)) != 0)[0]
    if jump.size == 0:
        return (t1 - t0 if refine else dur), t0, t1

    edges = []
    for j in jump:
        lo, hi = float(ts[j]), float(ts[j + 1])
        left_ok = bool(ok[j])
        if refine:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                dm, sm = los_dist(np.array([mid]), p_det, t_det)
                mid_ok = bool((dm[0] <= R) and (0.0 <= sm[0] <= 1.0))
                if mid_ok == left_ok:
                    lo = mid
                else:
                    hi = mid
        edges.append(0.5 * (lo + hi))

    segs = []
    state = bool(ok[0])
    cur_in = t0
    for e in edges:
        if state:
            segs.append((cur_in, e))
            state = False
        else:
            cur_in = e
            state = True
    if state:
        segs.append((cur_in, t1))
    if not segs:
        return 0.0, np.nan, np.nan
    if refine:
        dur = float(sum(b - a for a, b in segs))
    return dur, segs[0][0], segs[0][1]


def objective(x, n=20000):
    theta, v, t_drop, tau = map(float, x)
    _, p_drop, t_det, p_det = kinematics(theta, v, t_drop, tau)
    if t_det >= T_HIT:
        return 0.0, p_drop, t_det, p_det
    dur, _, _ = shield_duration(p_det, t_det, n=n, refine=False)
    return dur, p_drop, t_det, p_det


def local_refine(x0, lb, ub, n=40000, steps=80, spans=None):
    """简易坐标轮换 + 随机爬山，无依赖 scipy。"""
    if spans is None:
        spans = np.array([0.08, 8.0, 1.2, 1.2])
    x = np.array(x0, dtype=float)
    best, *_ = objective(x, n=n)
    span = spans.copy()
    for it in range(steps):
        improved = False
        for j in range(4):
            for sgn in (-1.0, 1.0):
                trial = x.copy()
                trial[j] += sgn * span[j]
                trial = np.minimum(np.maximum(trial, lb), ub)
                if j == 0:
                    trial[0] = trial[0] % (2 * np.pi)
                val, *_ = objective(trial, n=n)
                if val > best + 1e-8:
                    best = val
                    x = trial
                    improved = True
        # 随机扰动
        for _ in range(6):
            trial = x + (np.random.rand(4) - 0.5) * 2.0 * span
            trial = np.minimum(np.maximum(trial, lb), ub)
            trial[0] %= 2 * np.pi
            val, *_ = objective(trial, n=n)
            if val > best + 1e-8:
                best = val
                x = trial
                improved = True
        if not improved:
            span *= 0.55
            if np.all(span < np.array([1e-4, 0.02, 1e-3, 1e-3])):
                break
        else:
            span = np.minimum(span * 1.05, spans)
    return x, best


def main():
    t0 = time.perf_counter()
    print("==== 问题2 Python 优化 ====")
    print(f"t_hit={T_HIT:.6f}")

    # Q1 基准
    x_q1 = np.array([np.pi, 120.0, 1.5, 3.6])
    dur_q1, _, t_det_q1, p_det_q1 = objective(x_q1, n=100000)
    dur_q1r, tin1, tout1 = shield_duration(p_det_q1, t_det_q1, n=200000, refine=True)
    print(f"[Q1] coarse={dur_q1:.6f} refine={dur_q1r:.6f} [{tin1:.6f},{tout1:.6f}]")

    lb = np.array([0.0, 70.0, 0.0, 0.3])
    ub = np.array([2 * np.pi, 140.0, 25.0, 16.0])
    n_coarse = 10000

    th_list = np.pi + np.linspace(-0.30, 0.30, 13)
    v_list = np.array([70, 85, 100, 110, 120, 130, 140], dtype=float)
    td_list = np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    tau_list = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0])

    grid = np.array(np.meshgrid(th_list, v_list, td_list, tau_list, indexing="ij"))
    Xg = grid.reshape(4, -1).T

    rng = np.random.default_rng(2025)
    n_rand = 1500
    Xr = lb + rng.random((n_rand, 4)) * (ub - lb)
    n_band = 800
    Xb = np.column_stack(
        [
            (np.pi + (rng.random(n_band) - 0.5) * 0.5) % (2 * np.pi),
            70 + rng.random(n_band) * 70,
            0.1 + rng.random(n_band) * 12,
            0.5 + rng.random(n_band) * 10,
        ]
    )
    seeds = np.array(
        [
            [np.pi, 120, 1.5, 3.6],
            [np.pi, 100, 2.0, 4.0],
            [np.pi, 140, 0.5, 3.0],
            [np.pi, 90, 2.0, 5.0],
            [np.pi, 110, 1.5, 4.0],
            [np.pi, 100, 1.5, 4.5],
            [np.pi, 130, 2.5, 3.5],
            [np.pi + 0.08, 100, 2.0, 4.0],
            [np.pi - 0.08, 100, 2.0, 4.0],
            [np.pi, 70, 1.0, 5.0],
            [np.pi, 140, 2.0, 3.5],
            [np.pi, 120, 3.0, 4.0],
            [np.pi, 100, 2.5, 3.8],
            [np.pi, 95, 1.8, 4.2],
        ],
        dtype=float,
    )
    X = np.vstack([seeds, Xg, Xr, Xb])
    n_all = len(X)
    print(f"粗搜样本 {n_all}, n_time={n_coarse}")

    durs = np.zeros(n_all)
    best = -1.0
    best_x = X[0].copy()
    t_c = time.perf_counter()
    report = max(1, n_all // 20)
    for i in range(n_all):
        durs[i], *_ = objective(X[i], n=n_coarse)
        if durs[i] > best:
            best = durs[i]
            best_x = X[i].copy()
        if (i + 1) % report == 0 or i + 1 == n_all:
            print(
                f"  {i+1}/{n_all} best={best:.4f} "
                f"th={np.degrees(best_x[0])%360:.2f} v={best_x[1]:.1f} "
                f"td={best_x[2]:.2f} tau={best_x[3]:.2f} "
                f"({time.perf_counter()-t_c:.1f}s)"
            )

    order = np.argsort(-durs)
    Xs = X[order]
    ds = durs[order]
    print("Top-12 粗搜:")
    for k in range(min(12, n_all)):
        xk = Xs[k]
        print(
            f"  #{k+1:2d} {ds[k]:.4f} th={np.degrees(xk[0])%360:.2f} "
            f"v={xk[1]:.1f} td={xk[2]:.3f} tau={xk[3]:.3f}"
        )

    # 局部精修
    n_top = 20
    print(f"---- 局部精修 Top-{n_top} ----")
    refined = []
    t_r = time.perf_counter()
    for k in range(min(n_top, n_all)):
        xk, dk = local_refine(Xs[k], lb, ub, n=40000, steps=70)
        refined.append((dk, xk))
        print(f"  {k+1}/{n_top}: {ds[k]:.4f} -> {dk:.4f}")
    refined.sort(key=lambda t: -t[0])
    best_mid, x_best = refined[0]
    print(f"精修最优(中精度) {best_mid:.6f}  耗时 {time.perf_counter()-t_r:.1f}s")

    # 再精修一轮更高分辨率
    x_best, best_mid2 = local_refine(
        x_best, lb, ub, n=80000, steps=50, spans=np.array([0.03, 3.0, 0.5, 0.5])
    )
    print(f"二次精修 {best_mid2:.6f}")

    # 高精度
    dur_hi, p_drop, t_det, p_det = objective(x_best, n=300000)
    dur_ref, t_in, t_out = shield_duration(p_det, t_det, n=400000, refine=True)
    v_fy, _, _, _ = kinematics(*x_best)
    heading = float(np.degrees(x_best[0]) % 360)

    print("\n==== 问题2 最优策略 ====")
    print(f"有效遮蔽时长 = {dur_ref:.6f} s")
    print(f"有效区间 = [{t_in:.6f}, {t_out:.6f}] s")
    print(f"theta = {x_best[0]:.10f} rad = {heading:.6f} deg")
    print(f"v = {x_best[1]:.10f} m/s")
    print(f"t_drop = {x_best[2]:.10f} s")
    print(f"tau = {x_best[3]:.10f} s")
    print(f"t_det = {t_det:.10f} s")
    print(f"v_fy = {v_fy}")
    print(f"P_drop = {p_drop}")
    print(f"P_det = {p_det}")
    print(f"Q1 {dur_q1r:.4f} -> {dur_ref:.4f} (Δ={dur_ref-dur_q1r:.4f})")

    elapsed = time.perf_counter() - t0
    txt = OUT / "q2_result.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("==== 2025国赛A题 问题2 结果 ====\n")
        f.write("判据: 云团球心到导弹-真目标中心视线段距离 <= 10 m\n")
        f.write(f"有效遮蔽时长 = {dur_ref:.6f} s\n")
        f.write(f"有效区间(首段) = [{t_in:.6f}, {t_out:.6f}] s\n")
        f.write("\n--- 决策变量 ---\n")
        f.write(f"航向角 theta = {x_best[0]:.10f} rad = {heading:.6f} deg\n")
        f.write(f"飞行速度 v   = {x_best[1]:.10f} m/s\n")
        f.write(f"投放时刻 t_drop = {x_best[2]:.10f} s\n")
        f.write(f"引信延时 tau    = {x_best[3]:.10f} s\n")
        f.write("\n--- 导出量 ---\n")
        f.write(f"速度向量 v_fy = [{v_fy[0]:.10f}, {v_fy[1]:.10f}, {v_fy[2]:.10f}] m/s\n")
        f.write(f"投放点 P_drop = [{p_drop[0]:.10f}, {p_drop[1]:.10f}, {p_drop[2]:.10f}] m\n")
        f.write(f"起爆时刻 t_det = {t_det:.10f} s\n")
        f.write(f"起爆点 P_det  = [{p_det[0]:.10f}, {p_det[1]:.10f}, {p_det[2]:.10f}] m\n")
        f.write("\n--- 对照 ---\n")
        f.write(f"问题1固定策略时长 = {dur_q1r:.6f} s\n")
        f.write(f"提升 Delta = {dur_ref - dur_q1r:.6f} s\n")
        f.write(f"粗搜样本数 = {n_all}, 精修候选 = {n_top}\n")
        f.write(f"总运行时间 = {elapsed:.2f} s\n")
        f.write("求解器 = Python/NumPy 结构化粗搜 + 坐标轮换爬山\n")

    csv = OUT / "q2_result.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write(
            "theta_rad,heading_deg,v_mps,t_drop_s,tau_s,t_det_s,"
            "Pdrop_x,Pdrop_y,Pdrop_z,Pdet_x,Pdet_y,Pdet_z,duration_s,t_in,t_out\n"
        )
        f.write(
            f"{x_best[0]:.10f},{heading:.6f},{x_best[1]:.10f},{x_best[2]:.10f},"
            f"{x_best[3]:.10f},{t_det:.10f},{p_drop[0]:.10f},{p_drop[1]:.10f},"
            f"{p_drop[2]:.10f},{p_det[0]:.10f},{p_det[1]:.10f},{p_det[2]:.10f},"
            f"{dur_ref:.10f},{t_in:.10f},{t_out:.10f}\n"
        )

    meta = {
        "duration": dur_ref,
        "t_in": t_in,
        "t_out": t_out,
        "theta": float(x_best[0]),
        "heading_deg": heading,
        "v": float(x_best[1]),
        "t_drop": float(x_best[2]),
        "tau": float(x_best[3]),
        "t_det": float(t_det),
        "P_drop": p_drop.tolist(),
        "P_det": p_det.tolist(),
        "v_fy": v_fy.tolist(),
        "q1_duration": dur_q1r,
        "elapsed_s": elapsed,
        "top_coarse": [
            {
                "dur": float(ds[k]),
                "theta": float(Xs[k, 0]),
                "v": float(Xs[k, 1]),
                "t_drop": float(Xs[k, 2]),
                "tau": float(Xs[k, 3]),
            }
            for k in range(min(12, n_all))
        ],
    }
    with open(OUT / "q2_result.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n结果已写入: {display_path(txt)}")
    print(f"总耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    main()

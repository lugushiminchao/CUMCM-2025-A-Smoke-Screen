# -*- coding: utf-8 -*-
"""问题1：向量化高精度计算有效遮蔽时长"""
import math
from pathlib import Path

import numpy as np

g = 9.8
v_drone = 120.0
v_missile = 300.0
v_sink = 3.0
R = 10.0
t_drop = 1.5
tau = 3.6
t_det = t_drop + tau
T_life = 20.0

M0 = np.array([20000.0, 0.0, 2000.0])
FY0 = np.array([17800.0, 0.0, 1800.0])
dist_M0 = float(np.linalg.norm(M0))
u_m = -M0 / dist_M0
v_m = v_missile * u_m
t_hit = dist_M0 / v_missile

v_fy = np.array([-v_drone, 0.0, 0.0])
P_drop = FY0 + v_fy * t_drop
P_det = P_drop + np.array([v_fy[0] * tau, v_fy[1] * tau, -0.5 * g * tau * tau])


def dist_and_s_batch(ts, T):
    """向量化：对每个t计算云团中心到 导弹位置-目标点 线段的距离和参数s"""
    ts = np.asarray(ts, dtype=float)
    # M(t)
    Mx = M0[0] + v_m[0] * ts
    My = M0[1] + v_m[1] * ts
    Mz = M0[2] + v_m[2] * ts
    # C(t)
    Cx = np.full_like(ts, P_det[0])
    Cy = np.full_like(ts, P_det[1])
    Cz = P_det[2] - v_sink * (ts - t_det)

    ABx = T[0] - Mx
    ABy = T[1] - My
    ABz = T[2] - Mz
    ACx = Cx - Mx
    ACy = Cy - My
    ACz = Cz - Mz
    L2 = ABx * ABx + ABy * ABy + ABz * ABz
    s = (ACx * ABx + ACy * ABy + ACz * ABz) / np.maximum(L2, 1e-18)
    sc = np.clip(s, 0.0, 1.0)
    px = Mx + sc * ABx
    py = My + sc * ABy
    pz = Mz + sc * ABz
    d = np.sqrt((Cx - px) ** 2 + (Cy - py) ** 2 + (Cz - pz) ** 2)
    return d, s


def duration_for_target(T, n=2_000_001):
    t0 = t_det
    t1 = min(t_det + T_life, t_hit)
    ts = np.linspace(t0, t1, n)
    d, s = dist_and_s_batch(ts, T)
    ok = (d <= R) & (s >= 0.0) & (s <= 1.0)
    dt = float(ts[1] - ts[0])
    total = float(ok.sum()) * dt

    # 边界细化
    edges = []
    diff = np.diff(ok.astype(np.int8))
    idxs = np.where(diff != 0)[0]
    for i in idxs:
        lo, hi = float(ts[i]), float(ts[i + 1])
        want_left = bool(ok[i])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            dm, sm = dist_and_s_batch(np.array([mid]), T)
            mid_ok = bool((dm[0] <= R) and (0.0 <= sm[0] <= 1.0))
            if mid_ok == want_left:
                lo = mid
            else:
                hi = mid
        edges.append(0.5 * (lo + hi))
    return total, edges, t0, t1


def full_cylinder_duration(n_time=400001, n_theta=48, n_z=9):
    """对圆柱全部采样点同时被遮挡"""
    pts = []
    Rc, H = 7.0, 10.0
    for iz in range(n_z):
        z = H * iz / (n_z - 1)
        pts.append([0.0, 200.0, z])
        for it in range(n_theta):
            th = 2 * math.pi * it / n_theta
            pts.append([Rc * math.cos(th), 200 + Rc * math.sin(th), z])
    for r_frac in (0.5, 1.0):
        for it in range(n_theta):
            th = 2 * math.pi * it / n_theta
            x = r_frac * Rc * math.cos(th)
            y = 200 + r_frac * Rc * math.sin(th)
            pts.append([x, y, 0.0])
            pts.append([x, y, H])
    pts = np.array(pts)

    t0 = t_det
    t1 = min(t_det + T_life, t_hit)
    ts = np.linspace(t0, t1, n_time)
    ok_all = np.ones(len(ts), dtype=bool)
    for P in pts:
        d, s = dist_and_s_batch(ts, P)
        ok_all &= (d <= R) & (s >= 0.0) & (s <= 1.0)
    dt = float(ts[1] - ts[0])
    total = float(ok_all.sum()) * dt

    edges = []
    diff = np.diff(ok_all.astype(np.int8))
    idxs = np.where(diff != 0)[0]
    for i in idxs:
        lo, hi = float(ts[i]), float(ts[i + 1])
        want_left = bool(ok_all[i])
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            mid_ok = True
            for P in pts:
                dm, sm = dist_and_s_batch(np.array([mid]), P)
                if not ((dm[0] <= R) and (0.0 <= sm[0] <= 1.0)):
                    mid_ok = False
                    break
            if mid_ok == want_left:
                lo = mid
            else:
                hi = mid
        edges.append(0.5 * (lo + hi))
    return total, edges, len(pts)


def main():
    out_lines = []

    def p(*a):
        s = " ".join(str(x) for x in a)
        out_lines.append(s)
        print(s, flush=True)

    p("==== 问题1 关键几何量 ====")
    p(f"|M0|={dist_M0:.9f} m, t_hit={t_hit:.9f} s")
    p(f"u_m={u_m.tolist()}")
    p(f"v_m={v_m.tolist()}")
    p(f"P_drop={P_drop.tolist()}")
    p(f"t_det={t_det}, P_det={P_det.tolist()}")
    p(f"cloud z(t) = {P_det[2]} - 3*(t-{t_det})")
    p("")

    targets = {
        "真目标几何中心 (0,200,5)": np.array([0.0, 200.0, 5.0]),
        "真目标底心 (0,200,0)": np.array([0.0, 200.0, 0.0]),
        "真目标顶心 (0,200,10)": np.array([0.0, 200.0, 10.0]),
        "假目标原点 (0,0,0)": np.array([0.0, 0.0, 0.0]),
    }
    for name, T in targets.items():
        total, edges, t0, t1 = duration_for_target(T)
        if len(edges) >= 2:
            span = edges[1] - edges[0]
        else:
            span = total
        p(f"[{name}]")
        p(f"  扫描窗口 [{t0:.4f},{t1:.4f}]")
        p(f"  有效遮蔽时长 = {total:.6f} s")
        p(f"  边界时刻 = {edges}")
        if len(edges) >= 2:
            p(f"  区间长度(边界差) = {span:.6f} s")

    p("")
    p("==== 全圆柱严格遮挡 ====")
    total, edges, npts = full_cylinder_duration()
    p(f"采样点数={npts}, 时长={total:.6f} s, 边界={edges}")
    if len(edges) >= 2:
        p(f"区间长度={edges[1]-edges[0]:.6f} s")

    # 诊断表
    p("")
    p("==== 中心视线距离诊断 ====")
    T = targets["真目标几何中心 (0,200,5)"]
    for t in np.linspace(7.8, 9.5, 35):
        d, s = dist_and_s_batch(np.array([t]), T)
        ok = (d[0] <= R) and (0 <= s[0] <= 1)
        p(f"t={t:.5f} d={d[0]:.6f} s={s[0]:.6f} ok={ok}")

    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "q1_result.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    p(f"\n结果已保存: {path.relative_to(repo_root).as_posix()}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""问题1：解析/高精度计算有效遮蔽时长，并输出关键几何量"""
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

M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)
FY0 = np.array([17800.0, 0.0, 1800.0], dtype=float)
# 真目标：圆柱底心 (0,200,0), r=7, h=10
# 问题1常用代表点：几何中心 / 底心 / 顶心
targets = {
    "center": np.array([0.0, 200.0, 5.0]),
    "bottom": np.array([0.0, 200.0, 0.0]),
    "top": np.array([0.0, 200.0, 10.0]),
    "fake": np.array([0.0, 0.0, 0.0]),
}

dist_M0 = float(np.linalg.norm(M0))
u_m = -M0 / dist_M0
v_m = v_missile * u_m
t_hit = dist_M0 / v_missile

# 无人机朝向假目标（水平）等高度
v_fy = np.array([-v_drone, 0.0, 0.0])
P_drop = FY0 + v_fy * t_drop
# 弹体脱离后水平匀速、竖直自由落体（初竖速=0）
P_det = P_drop + np.array([v_fy[0] * tau, v_fy[1] * tau, -0.5 * g * tau * tau])


def missile(t):
    return M0 + v_m * t


def cloud(t):
    return np.array([P_det[0], P_det[1], P_det[2] - v_sink * (t - t_det)])


def dist_to_seg(C, A, B):
    """点C到线段AB距离，以及未夹紧参数s"""
    AB = B - A
    L2 = float(np.dot(AB, AB))
    if L2 < 1e-18:
        return float(np.linalg.norm(C - A)), 0.0
    s = float(np.dot(C - A, AB) / L2)
    sc = min(1.0, max(0.0, s))
    return float(np.linalg.norm(C - (A + sc * AB))), s


def shielded(t, T):
    if t < t_det or t > t_det + T_life or t > t_hit:
        return False
    d, s = dist_to_seg(cloud(t), missile(t), T)
    # 云团必须在导弹与目标之间：s∈[0,1]，且到视线距离≤R
    return (0.0 <= s <= 1.0) and (d <= R + 1e-12)


def duration_and_roots(T, n=400000):
    t0, t1 = t_det, min(t_det + T_life, t_hit)
    ts = np.linspace(t0, t1, n)
    dt = float(ts[1] - ts[0])
    flags = np.array([shielded(float(t), T) for t in ts])
    total = float(flags.sum()) * dt

    # 找进入/离开时刻（二分细化）
    edges = []
    for i in range(len(flags) - 1):
        if flags[i] != flags[i + 1]:
            lo, hi = float(ts[i]), float(ts[i + 1])
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if shielded(mid, T) == flags[i]:
                    lo = mid
                else:
                    hi = mid
            edges.append(0.5 * (lo + hi))
    return total, edges, flags, ts


def sample_cylinder(n_theta=90, n_z=15):
    pts = []
    Rc, H = 7.0, 10.0
    for iz in range(n_z):
        z = H * iz / (n_z - 1)
        pts.append(np.array([0.0, 200.0, z]))
        for it in range(n_theta):
            th = 2 * math.pi * it / n_theta
            pts.append(np.array([Rc * math.cos(th), 200 + Rc * math.sin(th), z]))
    for r_frac in (0.0, 0.5, 1.0):
        for it in range(n_theta):
            th = 2 * math.pi * it / n_theta
            x = r_frac * Rc * math.cos(th)
            y = 200 + r_frac * Rc * math.sin(th)
            pts.append(np.array([x, y, 0.0]))
            pts.append(np.array([x, y, H]))
    return pts


def full_cyl_shielded(t, pts):
    if t < t_det or t > t_det + T_life or t > t_hit:
        return False
    M = missile(t)
    C = cloud(t)
    for P in pts:
        d, s = dist_to_seg(C, M, P)
        if not ((0.0 <= s <= 1.0) and (d <= R + 1e-12)):
            return False
    return True


def main():
    lines = []
    def p(*a):
        s = " ".join(str(x) for x in a)
        lines.append(s)
        print(s)

    p("==== 问题1 关键量 ====")
    p(f"导弹初距假目标 |M0| = {dist_M0:.6f} m, 到达假目标 t_hit = {t_hit:.6f} s")
    p(f"导弹单位方向 u_m = {u_m.tolist()}")
    p(f"无人机速度 v_fy = {v_fy.tolist()}")
    p(f"投放时刻 t_drop = {t_drop} s, 投放点 P_drop = {P_drop.tolist()}")
    p(f"起爆时刻 t_det  = {t_det} s, 起爆点 P_det  = {P_det.tolist()}")
    p(f"云团有效窗口 [{t_det}, {t_det + T_life}] s")
    p("")

    for name, T in targets.items():
        total, edges, _, _ = duration_and_roots(T)
        p(f"[点目标 {name} {T.tolist()}] 有效遮蔽时长 = {total:.6f} s, 边界 = {edges}")

    # 全圆柱严格遮挡
    pts = sample_cylinder()
    t0, t1 = t_det, min(t_det + T_life, t_hit)
    ts = np.linspace(t0, t1, 200001)
    dt = float(ts[1] - ts[0])
    flags = np.array([full_cyl_shielded(float(t), pts) for t in ts])
    total = float(flags.sum()) * dt
    edges = []
    for i in range(len(flags) - 1):
        if flags[i] != flags[i + 1]:
            lo, hi = float(ts[i]), float(ts[i + 1])
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if full_cyl_shielded(mid, pts) == flags[i]:
                    lo = mid
                else:
                    hi = mid
            edges.append(0.5 * (lo + hi))
    p(f"[全圆柱严格遮挡 采样{len(pts)}点] 时长 = {total:.6f} s, 边界 = {edges}")

    # 几何解释：何时云团穿过导弹-真目标中心视线
    p("")
    p("==== 视线距离随时间（真目标中心）====")
    T = targets["center"]
    for t in np.linspace(7.5, 9.6, 22):
        d, s = dist_to_seg(cloud(t), missile(t), T)
        ok = (0 <= s <= 1) and (d <= R)
        p(f"t={t:.4f} d={d:.6f} s={s:.6f} shielded={ok}")

    repo_root = Path(__file__).resolve().parent.parent
    out = repo_root / "q1_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    p(f"\n已写入 {out.relative_to(repo_root).as_posix()}")


if __name__ == "__main__":
    main()

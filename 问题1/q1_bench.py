# -*- coding: utf-8 -*-
import time
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
T = np.array([0.0, 200.0, 5.0])


def run(n):
    t0 = time.perf_counter()
    ts = np.linspace(t_det, min(t_det + T_life, t_hit), n)
    Mx = M0[0] + v_m[0] * ts
    My = M0[1] + v_m[1] * ts
    Mz = M0[2] + v_m[2] * ts
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
    d = np.sqrt((Cx - (Mx + sc * ABx)) ** 2 + (Cy - (My + sc * ABy)) ** 2 + (Cz - (Mz + sc * ABz)) ** 2)
    ok = (d <= R) & (s >= 0.0) & (s <= 1.0)
    dt = float(ts[1] - ts[0])
    total = float(ok.sum()) * dt
    elapsed = time.perf_counter() - t0
    return elapsed, total


# pure python version for comparison
def run_python(n):
    t0 = time.perf_counter()
    t_end = min(t_det + T_life, t_hit)
    dt = (t_end - t_det) / (n - 1)
    cnt = 0
    for i in range(n):
        t = t_det + i * dt
        Mx = M0[0] + v_m[0] * t
        My = M0[1] + v_m[1] * t
        Mz = M0[2] + v_m[2] * t
        Cx, Cy = P_det[0], P_det[1]
        Cz = P_det[2] - v_sink * (t - t_det)
        ABx, ABy, ABz = T[0] - Mx, T[1] - My, T[2] - Mz
        ACx, ACy, ACz = Cx - Mx, Cy - My, Cz - Mz
        L2 = ABx * ABx + ABy * ABy + ABz * ABz
        s = (ACx * ABx + ACy * ABy + ACz * ABz) / (L2 if L2 > 1e-18 else 1e-18)
        sc = 0.0 if s < 0 else (1.0 if s > 1 else s)
        dx = Cx - (Mx + sc * ABx)
        dy = Cy - (My + sc * ABy)
        dz = Cz - (Mz + sc * ABz)
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        if d <= R and 0 <= s <= 1:
            cnt += 1
    total = cnt * dt
    return time.perf_counter() - t0, total


if __name__ == "__main__":
    print("=== 计时基准 ===")
    for n in [50001, 200001, 2000001]:
        e, tot = run(n)
        print(f"numpy  n={n:>9,}: {e*1000:8.2f} ms, 时长={tot:.6f}s")

    e, tot = run_python(50001)
    print(f"python n={50001:>9,}: {e*1000:8.2f} ms, 时长={tot:.6f}s")

    # full script-like: 4 targets + cylinder
    t_all = time.perf_counter()
    for _ in range(4):
        run(2_000_001)
    # light cylinder
    import math
    pts = []
    for iz in range(9):
        z = 10 * iz / 8
        pts.append([0, 200, z])
        for it in range(48):
            th = 2 * math.pi * it / 48
            pts.append([7 * math.cos(th), 200 + 7 * math.sin(th), z])
    t0 = time.perf_counter()
    ts = np.linspace(t_det, min(t_det + T_life, t_hit), 400001)
    ok_all = np.ones(len(ts), dtype=bool)
    for P in pts:
        P = np.array(P, float)
        Mx = M0[0] + v_m[0] * ts
        My = M0[1] + v_m[1] * ts
        Mz = M0[2] + v_m[2] * ts
        Cx = np.full_like(ts, P_det[0])
        Cy = np.full_like(ts, P_det[1])
        Cz = P_det[2] - v_sink * (ts - t_det)
        ABx = P[0] - Mx
        ABy = P[1] - My
        ABz = P[2] - Mz
        ACx = Cx - Mx
        ACy = Cy - My
        ACz = Cz - Mz
        L2 = ABx * ABx + ABy * ABy + ABz * ABz
        s = (ACx * ABx + ACy * ABy + ACz * ABz) / np.maximum(L2, 1e-18)
        sc = np.clip(s, 0.0, 1.0)
        d = np.sqrt((Cx - (Mx + sc * ABx)) ** 2 + (Cy - (My + sc * ABy)) ** 2 + (Cz - (Mz + sc * ABz)) ** 2)
        ok_all &= (d <= R) & (s >= 0) & (s <= 1)
    cyl_t = time.perf_counter() - t0
    all_t = time.perf_counter() - t_all
    print(f"4个点目标(各200万点): 含在 all 中")
    print(f"圆柱扫描: {cyl_t*1000:.1f} ms")
    print(f"整包估算: {all_t:.3f} s")

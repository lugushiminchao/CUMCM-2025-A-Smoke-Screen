# -*- coding: utf-8 -*-
"""2025 CUMCM A题 问题1：给定策略下有效遮蔽时长计算"""
import math
import numpy as np

OUT = r"C:\数模\2025国赛A题\q1_calc.txt"


def log(fp, *a):
    s = " ".join(str(x) for x in a)
    fp.write(s + "\n")
    print(s, flush=True)


def main():
    g = 9.8
    v_drone = 120.0
    v_missile = 300.0
    v_sink = 3.0
    R_smoke = 10.0
    t_drop = 1.5
    dt_fuse = 3.6
    t_det = t_drop + dt_fuse
    t_smoke_life = 20.0

    M0 = np.array([20000.0, 0.0, 2000.0])
    FY0 = np.array([17800.0, 0.0, 1800.0])
    T_center = np.array([0.0, 200.0, 5.0])
    T_bottom = np.array([0.0, 200.0, 0.0])
    R_cyl = 7.0
    H_cyl = 10.0

    dist_M0 = float(np.linalg.norm(M0))
    u_m = -M0 / dist_M0
    v_m = v_missile * u_m

    with open(OUT, "w", encoding="utf-8") as fp:
        log(fp, "missile dist", dist_M0, "t_hit_fake", dist_M0 / v_missile)
        log(fp, "u_m", u_m.tolist())
        log(fp, "v_m", v_m.tolist())

        # 朝向假目标方向：水平指向原点，等高度
        v_fy = np.array([-v_drone, 0.0, 0.0])
        P_drop = FY0 + v_fy * t_drop
        log(fp, "P_drop", P_drop.tolist())

        tau = dt_fuse
        P_det = np.array(
            [
                P_drop[0] + v_fy[0] * tau,
                P_drop[1] + v_fy[1] * tau,
                P_drop[2] - 0.5 * g * tau ** 2,
            ]
        )
        log(fp, "t_det", t_det, "P_det", P_det.tolist())
        log(fp, "smoke window", [t_det, t_det + t_smoke_life])

        def missile_pos(t):
            return M0 + v_m * t

        def cloud_center(t):
            return np.array(
                [P_det[0], P_det[1], P_det[2] - v_sink * (t - t_det)]
            )

        def dist_point_to_segment(C, A, B):
            AB = B - A
            AC = C - A
            L2 = float(np.dot(AB, AB))
            if L2 < 1e-12:
                return float(np.linalg.norm(AC)), 0.0
            s = float(np.dot(AC, AB) / L2)
            s_clamped = max(0.0, min(1.0, s))
            closest = A + s_clamped * AB
            return float(np.linalg.norm(C - closest)), s

        def is_shielded_point(t, T):
            if t < t_det or t > t_det + t_smoke_life:
                return False
            # 导弹到达假目标后不再考虑（或到达真目标附近）
            M = missile_pos(t)
            if np.linalg.norm(M) < 1e-6:
                return False
            C = cloud_center(t)
            d, s = dist_point_to_segment(C, M, T)
            return d <= R_smoke + 1e-9

        # 圆柱采样
        pts = []
        n_theta, n_z = 72, 11
        for iz in range(n_z):
            z = H_cyl * iz / (n_z - 1)
            for it in range(n_theta):
                th = 2 * math.pi * it / n_theta
                pts.append(
                    np.array(
                        [R_cyl * math.cos(th), 200 + R_cyl * math.sin(th), z]
                    )
                )
            pts.append(np.array([0.0, 200.0, z]))
        # 上下底面内部点
        for r_frac in [0.5, 1.0]:
            for it in range(n_theta):
                th = 2 * math.pi * it / n_theta
                x = r_frac * R_cyl * math.cos(th)
                y = 200 + r_frac * R_cyl * math.sin(th)
                pts.append(np.array([x, y, 0.0]))
                pts.append(np.array([x, y, H_cyl]))
        log(fp, "n_pts", len(pts))

        def is_full(t):
            if t < t_det or t > t_det + t_smoke_life:
                return False
            M = missile_pos(t)
            C = cloud_center(t)
            for P in pts:
                d, s = dist_point_to_segment(C, M, P)
                if d > R_smoke + 1e-9:
                    return False
            return True

        # 导弹到达假目标时间
        t_hit = dist_M0 / v_missile
        t_end = min(t_det + t_smoke_life, t_hit)
        log(fp, "t_hit", t_hit, "scan_end", t_end)

        ts = np.linspace(t_det, t_end, 50001)
        dt = float(ts[1] - ts[0])

        for name, fn in [
            ("center(0,200,5)", lambda t: is_shielded_point(t, T_center)),
            ("bottom(0,200,0)", lambda t: is_shielded_point(t, T_bottom)),
            ("top(0,200,10)", lambda t: is_shielded_point(t, np.array([0.0, 200.0, 10.0]))),
            ("full_cyl_samples", is_full),
        ]:
            flags = np.array([fn(float(t)) for t in ts])
            total = float(flags.sum()) * dt
            intervals = []
            in_int = False
            start = 0.0
            for i, f in enumerate(flags):
                if f and not in_int:
                    start = float(ts[i])
                    in_int = True
                elif (not f) and in_int:
                    intervals.append((start, float(ts[i - 1])))
                    in_int = False
            if in_int:
                intervals.append((start, float(ts[-1])))
            log(fp, "===", name, "duration", f"{total:.6f}")
            log(fp, "intervals", intervals)

        log(fp, "--- diagnostics (center LOS) ---")
        for t in np.linspace(t_det, t_end, 31):
            M = missile_pos(t)
            C = cloud_center(t)
            d, s = dist_point_to_segment(C, M, T_center)
            log(
                fp,
                f"t={t:.4f} d={d:.4f} s={s:.6f} "
                f"M=({M[0]:.1f},{M[1]:.1f},{M[2]:.2f}) "
                f"C=({C[0]:.1f},{C[1]:.1f},{C[2]:.2f})",
            )

        # 解析：求 d(t) <= 10 的区间（点目标中心）
        # 用更密扫描 + 二分找边界
        def find_duration(fn, lo, hi, n=200000):
            ts2 = np.linspace(lo, hi, n)
            dt2 = float(ts2[1] - ts2[0])
            flags = np.array([fn(float(t)) for t in ts2])
            return float(flags.sum()) * dt2, flags, ts2

        dur_c, flags_c, ts_c = find_duration(
            lambda t: is_shielded_point(t, T_center), t_det, t_end
        )
        log(fp, "highres center duration", f"{dur_c:.6f}")

        # 也计算：云团遮挡导弹到真目标圆柱“视线锥”的更严格/宽松变体
        # 宽松：只要遮挡圆柱任一关键点（中心）
        # 严格：遮挡全部采样点

        # 云团中心相对导弹-假目标连线的关系
        log(fp, "--- relative geometry at det ---")
        M_det = missile_pos(t_det)
        log(fp, "M_det", M_det.tolist())
        log(fp, "C_det", P_det.tolist())
        d0, s0 = dist_point_to_segment(P_det, M_det, T_center)
        log(fp, "d0", d0, "s0", s0)
        d1, s1 = dist_point_to_segment(P_det, M_det, np.array([0.0, 0.0, 0.0]))
        log(fp, "to fake target d", d1, "s", s1)


if __name__ == "__main__":
    main()

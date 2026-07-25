# -*- coding: utf-8 -*-
"""高精度并集时长：时间掩码 + 边界二分，消除网格噪声。"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from q3_optimize import (  # noqa
    FY0,
    G,
    M0,
    OUT,
    R,
    T_HIT,
    T_LIFE,
    TGT,
    V_M,
    V_SINK,
    device_info,
    pack_strategy,
    write_result1_xlsx,
)


def los_ok_at_t(t, P_dets, t_dets):
    """任一云团在时刻 t 是否遮蔽 LOS。"""
    t = float(t)
    M = M0 + V_M * t
    AB = TGT - M
    L2 = float(np.dot(AB, AB))
    if L2 < 1e-18:
        return False
    for P, td in zip(P_dets, t_dets):
        if t < td or t > td + T_LIFE:
            continue
        C = np.array([P[0], P[1], P[2] - V_SINK * (t - td)], dtype=float)
        AC = C - M
        s = float(np.dot(AC, AB) / L2)
        if s < 0.0 or s > 1.0:
            continue
        sc = min(max(s, 0.0), 1.0)
        P_closest = M + sc * AB
        if float(np.linalg.norm(C - P_closest)) <= R + 1e-12:
            return True
    return False


def los_ok_mask(ts, P_dets, t_dets):
    """向量化：ts (T,) -> ok (T,) 并集遮蔽掩码。"""
    ts = np.asarray(ts, dtype=float).ravel()
    M = M0[None, :] + V_M[None, :] * ts[:, None]
    AB = TGT[None, :] - M
    L2 = np.maximum(np.sum(AB * AB, axis=1), 1e-18)
    any_ok = np.zeros(ts.shape[0], dtype=bool)
    for P, td in zip(P_dets, t_dets):
        alive = (ts >= td) & (ts <= td + T_LIFE)
        C = np.empty_like(M)
        C[:, 0] = P[0]
        C[:, 1] = P[1]
        C[:, 2] = P[2] - V_SINK * (ts - td)
        AC = C - M
        s = np.sum(AC * AB, axis=1) / L2
        sc = np.clip(s, 0.0, 1.0)
        Pc = M + sc[:, None] * AB
        d = np.linalg.norm(C - Pc, axis=1)
        any_ok |= alive & (d <= R) & (s >= 0.0) & (s <= 1.0)
    return any_ok


def refined_union(x, n_scan=200000, bisect_iters=50):
    info = pack_strategy(x)
    P_dets = [np.array(b["P_det"], dtype=float) for b in info["bombs"]]
    t_dets = [float(b["t_det"]) for b in info["bombs"]]
    t0 = max(0.0, min(t_dets) - 0.2)
    t1 = min(T_HIT, max(t_dets) + T_LIFE)
    ts = np.linspace(t0, t1, n_scan)
    ok = los_ok_mask(ts, P_dets, t_dets)
    if not np.any(ok):
        return 0.0, np.zeros(3), info, []

    # 找段
    jumps = np.where(np.diff(ok.astype(np.int8)) != 0)[0]
    edges = []
    for j in jumps:
        lo, hi = float(ts[j]), float(ts[j + 1])
        left_ok = bool(ok[j])
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            mid_ok = los_ok_at_t(mid, P_dets, t_dets)
            if mid_ok == left_ok:
                lo = mid
            else:
                hi = mid
        edges.append(0.5 * (lo + hi))

    segs = []
    state = bool(ok[0])
    cur = t0
    for e in edges:
        if state:
            segs.append((cur, e))
            state = False
        else:
            cur = e
            state = True
    if state:
        segs.append((cur, t1))
    total = float(sum(b - a for a, b in segs))

    # 单弹时长（同样精修）
    per = []
    for k in range(3):
        Pk = [P_dets[k]]
        tk = [t_dets[k]]
        t0k = max(0.0, t_dets[k] - 0.05)
        t1k = min(T_HIT, t_dets[k] + T_LIFE)
        tsk = np.linspace(t0k, t1k, max(50000, n_scan // 3))
        okk = los_ok_mask(tsk, Pk, tk)
        if not np.any(okk):
            per.append(0.0)
            continue
        jumps = np.where(np.diff(okk.astype(np.int8)) != 0)[0]
        edges = []
        for j in jumps:
            lo, hi = float(tsk[j]), float(tsk[j + 1])
            left_ok = bool(okk[j])
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                mid_ok = los_ok_at_t(mid, Pk, tk)
                if mid_ok == left_ok:
                    lo = mid
                else:
                    hi = mid
            edges.append(0.5 * (lo + hi))
        segs_k = []
        state = bool(okk[0])
        cur = t0k
        for e in edges:
            if state:
                segs_k.append((cur, e))
                state = False
            else:
                cur = e
                state = True
        if state:
            segs_k.append((cur, t1k))
        per.append(float(sum(b - a for a, b in segs_k)))

    return total, np.array(per, dtype=float), info, segs


def main():
    cur = json.load(open(OUT / "q3_result.json", encoding="utf-8"))
    x = np.array(cur["info"]["x"], dtype=float)

    # 轻微规范化：v 贴 140，t1 贴 0 若很接近
    x2 = x.copy()
    if abs(x2[1] - 140) < 0.05:
        x2[1] = 140.0
    if abs(x2[2]) < 0.02:
        x2[2] = 0.0

    for label, xx in [("raw", x), ("norm", x2)]:
        total, per, info, segs = refined_union(xx, n_scan=250000, bisect_iters=55)
        print(label, "union", total)
        print(" per", per)
        print(" segs", [(round(a, 6), round(b, 6), round(b - a, 6)) for a, b in segs])
        print(" heading", info["heading_deg"], "v", info["v"])

    # 取更好的
    t1, p1, i1, s1 = refined_union(x, n_scan=300000, bisect_iters=60)
    t2, p2, i2, s2 = refined_union(x2, n_scan=300000, bisect_iters=60)
    if t2 >= t1 - 1e-9:
        total, per, info, segs = t2, p2, i2, s2
        xbest = x2
    else:
        total, per, info, segs = t1, p1, i1, s1
        xbest = x

    # 若 norm 后略差则用 raw
    info = pack_strategy(xbest)
    total, per, info, segs = refined_union(xbest, n_scan=400000, bisect_iters=60)

    print("\nFINAL refined", total)
    print("per", per)
    print("segments", segs)
    print("sum per", per.sum(), "union", total, "overlap", per.sum() - total)

    with open(OUT / "q3_result.txt", "w", encoding="utf-8") as f:
        f.write("==== 2025国赛A题 问题3 结果 ====\n")
        f.write("FY1 投放 3 枚烟幕干扰弹，干扰 M1；遮蔽取并集（可不连续）\n")
        f.write(f"backend = {device_info()}\n")
        f.write(f"并集有效遮蔽时长 = {total:.6f} s\n")
        f.write(
            f"单弹有效时长(各自，可重叠) = [{per[0]:.6f}, {per[1]:.6f}, {per[2]:.6f}] s\n"
        )
        f.write("有效遮蔽区间(并集分段):\n")
        for a, b in segs:
            f.write(f"  [{a:.6f}, {b:.6f}]  长={b-a:.6f} s\n")
        f.write("\n--- 无人机 ---\n")
        f.write(
            f"航向角 theta = {info['theta']:.10f} rad = {info['heading_deg']:.6f} deg\n"
        )
        f.write(f"飞行速度 v   = {info['v']:.10f} m/s\n")
        f.write("航向定义: 以 x 轴正向为基准逆时针，取值 0~360°\n")
        for j, b in enumerate(info["bombs"]):
            f.write(f"\n--- 烟幕干扰弹 {j+1} ---\n")
            f.write(f"投放时刻 t_drop = {b['t_drop']:.10f} s\n")
            f.write(f"引信延时 tau    = {b['tau']:.10f} s\n")
            f.write(f"起爆时刻 t_det  = {b['t_det']:.10f} s\n")
            f.write(
                f"投放点 P_drop = [{b['P_drop'][0]:.10f}, {b['P_drop'][1]:.10f}, {b['P_drop'][2]:.10f}]\n"
            )
            f.write(
                f"起爆点 P_det  = [{b['P_det'][0]:.10f}, {b['P_det'][1]:.10f}, {b['P_det'][2]:.10f}]\n"
            )
            f.write(f"单弹有效时长 = {per[j]:.6f} s\n")
        f.write(f"\n决策向量 x = {info['x']}\n")
        f.write("求解 = Python + PyTorch CUDA 批量评估 + 多阶段粗搜/坐标轮换精修\n")
        f.write(
            "判据: 云团球心到导弹-真目标中心视线段距离 <= 10 m；三枚取并集（可不连续）\n"
        )
        f.write("约束: 投放间隔 >= 1 s；无人机速度 70~140 m/s；等高匀速直线\n")
        f.write("最终时长: 时间扫描 + 边界二分精修\n")

    json.dump(
        {
            "total_union": float(total),
            "per_bomb": per.tolist(),
            "segments": segs,
            "info": info,
            "backend": device_info(),
            "method": "scan+bisection",
        },
        open(OUT / "q3_result.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    path = write_result1_xlsx(info, per, total, OUT / "result1.xlsx")
    print("saved", Path(path).resolve().relative_to(THIS_DIR.parent.resolve()).as_posix())

    # Q2 check refined
    xq2 = np.array([3.0880757565, 71.8890217683, 0.0, 2.5032397513, 50, 1, 52, 1])
    tq, pq, _, sq = refined_union(xq2, n_scan=200000, bisect_iters=50)
    print("Q2 refined check", tq, pq, sq)


if __name__ == "__main__":
    main()

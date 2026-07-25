# -*- coding: utf-8 -*-
"""
问题1：圆柱严格全遮蔽基准时长（固定策略 FY1）
策略: v=120, 朝向假目标, t_drop=1.5, tau=3.6
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
    PTS_ULTRA,
    T_HIT,
    cylinder_ok_circle_max,
    device_info,
    kinematics,
    shield_duration,
)

OUT = ROOT / "结果"
OUT.mkdir(exist_ok=True)


def main():
    t0 = time.perf_counter()
    print("==== 问题1 圆柱严格全遮蔽基准 ====")
    print("device:", device_info())
    print(f"t_hit={T_HIT:.9f}")

    _, p_drop, t_det, p_det = kinematics(np.pi, 120.0, 1.5, 3.6)
    print(f"P_drop={p_drop}")
    print(f"t_det={t_det}, P_det={p_det}")

    results = {}
    # 点模型对照
    dur_pt, segs_pt = shield_duration(p_det, t_det, mode="point", n=200000, refine=True)
    results["point"] = {"duration": dur_pt, "segments": segs_pt}
    print(f"[point]      {dur_pt:.6f} s  segs={segs_pt}")

    for name, pts, n in [
        ("cyl_fast", PTS_FAST, 100000),
        ("cyl_opt", PTS_OPT, 150000),
        ("cyl_fine", PTS_FINE, 200000),
        ("cyl_ultra", PTS_ULTRA, 250000),
    ]:
        t1 = time.perf_counter()
        dur, segs = shield_duration(p_det, t_det, mode="cylinder", pts=pts, n=n, refine=True)
        el = time.perf_counter() - t1
        results[name] = {
            "duration": dur,
            "segments": segs,
            "n_pts": int(pts.shape[0]),
            "elapsed_s": el,
        }
        print(f"[{name:9s}] {dur:.6f} s  n_pts={pts.shape[0]:4d}  segs={segs}  ({el:.2f}s)")

    # 圆周 max 精修扫边界（终算）
    print("---- 圆周最坏点复核 (n_z=7, n_theta_coarse=180) ----")
    t_lo = float(t_det)
    t_hi = min(float(t_det) + 20.0, T_HIT)
    ts = np.linspace(t_lo, t_hi, 80000)
    ok = np.array(
        [cylinder_ok_circle_max(t, p_det, t_det, n_coarse=128, n_z_side=7) for t in ts],
        dtype=bool,
    )
    # 粗时长
    dt = ts[1] - ts[0]
    dur_circle_coarse = float(np.sum(ok) * dt)
    # 二分边界
    jumps = np.where(np.diff(ok.astype(np.int8)) != 0)[0]
    edges = []
    for j in jumps:
        lo, hi = float(ts[j]), float(ts[j + 1])
        left_ok = bool(ok[j])
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            mid_ok = cylinder_ok_circle_max(mid, p_det, t_det, n_coarse=180, n_z_side=9)
            if mid_ok == left_ok:
                lo = mid
            else:
                hi = mid
        edges.append(0.5 * (lo + hi))
    segs_c = []
    state = bool(ok[0])
    cur = t_lo
    for e in edges:
        if state:
            segs_c.append((cur, e))
            state = False
        else:
            cur = e
            state = True
    if state:
        segs_c.append((cur, t_hi))
    dur_circle = float(sum(b - a for a, b in segs_c)) if segs_c else 0.0
    results["circle_max"] = {
        "duration": dur_circle,
        "duration_coarse": dur_circle_coarse,
        "segments": segs_c,
    }
    print(f"[circle_max] {dur_circle:.6f} s  (coarse grid {dur_circle_coarse:.6f}) segs={segs_c}")

    # 推荐采用 fine 采样 + circle 复核一致者
    rec = results["cyl_fine"]["duration"]
    results["recommended"] = {
        "duration": rec,
        "note": "圆柱表面密采样(PTS_FINE) + refine；circle_max 作对照",
        "circle_max": dur_circle,
    }

    elapsed = time.perf_counter() - t0
    txt = OUT / "q1_cylinder_result.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("==== 2025国赛A题 问题1 圆柱严格全遮蔽 ====\n")
        f.write("判据: 云团球心到导弹-圆柱表面采样点视线段距离均 <= 10 m\n")
        f.write("策略: FY1 v=120 朝向假目标, t_drop=1.5, tau=3.6\n")
        f.write(f"P_drop = {p_drop.tolist()}\n")
        f.write(f"t_det  = {t_det}\n")
        f.write(f"P_det  = {p_det.tolist()}\n\n")
        f.write(f"点模型(中心)时长 = {dur_pt:.6f} s  segs={segs_pt}\n")
        for k in ("cyl_fast", "cyl_opt", "cyl_fine", "cyl_ultra"):
            r = results[k]
            f.write(
                f"{k}: {r['duration']:.6f} s  n_pts={r['n_pts']}  segs={r['segments']}\n"
            )
        f.write(f"circle_max: {dur_circle:.6f} s  segs={segs_c}\n")
        f.write(f"\n【推荐报告值】有效遮蔽时长 = {rec:.6f} s\n")
        f.write(f"运行时间 = {elapsed:.2f} s\n")

    with open(OUT / "q1_cylinder_result.json", "w", encoding="utf-8") as f:
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
                return {kk: conv(vv) for kk, vv in o.items()}
            return o

        json.dump(conv(results), f, ensure_ascii=False, indent=2)

    print(f"\n推荐 Q1 圆柱时长 = {rec:.6f} s")
    print(f"已写入 {txt}")
    print(f"总耗时 {elapsed:.2f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""问题3 第二阶段：更强多起点精修 + 结构化时间轴搜索"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from q3_optimize import (
    DROP_GAP,
    OUT,
    device_info,
    eval_strategies,
    highres_union,
    local_refine,
    sample_candidates,
    write_result1_xlsx,
)

ROOT = Path(__file__).resolve().parent
def display_path(path):
    return Path(path).resolve().relative_to(ROOT.parent.resolve()).as_posix()



def structured_time_candidates(n_each=2000, rng=None):
    """围绕 Q2 窗口 [~2.5, ~7.2] 构造“接力”时间轴候选。"""
    if rng is None:
        rng = np.random.default_rng(123)
    rows = []
    # 航向/速度：Q2 附近 + 180 附近 + 全周少量
    for _ in range(n_each):
        mode = rng.integers(0, 4)
        if mode == 0:
            th = 3.0880757565 + rng.normal(0, 0.03)
            v = float(np.clip(71.889 + rng.normal(0, 8), 70, 140))
        elif mode == 1:
            th = np.pi + rng.normal(0, 0.08)
            v = float(rng.uniform(90, 140))
        elif mode == 2:
            th = np.pi + rng.normal(0, 0.2)
            v = float(rng.uniform(70, 140))
        else:
            th = float(rng.uniform(0, 2 * np.pi))
            v = float(rng.uniform(70, 140))

        # 接力：希望三段遮蔽尽量不重叠
        # t_det ≈ t_drop + tau
        # 第一弹靠近 Q2
        t1 = float(max(0.0, rng.uniform(0.0, 1.5)))
        tau1 = float(rng.uniform(1.8, 4.5))
        # 第二弹：起爆更晚
        t2 = t1 + DROP_GAP + float(rng.uniform(0.0, 4.0))
        tau2 = float(rng.uniform(2.0, 7.0))
        # 第三弹
        t3 = t2 + DROP_GAP + float(rng.uniform(0.0, 5.0))
        tau3 = float(rng.uniform(2.0, 9.0))
        rows.append([th, v, t1, tau1, t2, tau2, t3, tau3])

    # 固定航向速度网格 + 时间轴（控制规模）
    ths = np.array([3.0880757565, np.pi, 3.05, 3.12])
    vs = np.array([70, 85, 100, 115, 130, 140], dtype=float)
    for th in ths:
        for v in vs:
            for t1 in [0.0, 0.3]:
                for tau1 in [2.0, 2.5, 3.0, 3.5]:
                    for g1 in [1.0, 1.5, 2.5, 3.5]:
                        for tau2 in [2.5, 3.5, 4.5, 6.0]:
                            for g2 in [1.0, 2.0, 3.5]:
                                for tau3 in [3.0, 4.5, 6.0, 8.0]:
                                    t2 = t1 + g1
                                    t3 = t2 + g2
                                    rows.append([th, v, t1, tau1, t2, tau2, t3, tau3])
    return np.asarray(rows, dtype=np.float64)


def main():
    t_all = time.perf_counter()
    print("==== Q3 stage2 ====")
    print("backend =", device_info())

    with open(OUT / "q3_result.json", "r", encoding="utf-8") as f:
        cur = json.load(f)
    x_best0 = np.array(cur["info"]["x"], dtype=float)
    print("prev best", cur["total_union"])

    rng = np.random.default_rng(20250724)
    Xs = structured_time_candidates(4000, rng)
    Xr = sample_candidates(15000, rng)
    # 围绕当前最优扰动
    noise = rng.normal(0, 1, (8000, 8)) * np.array(
        [0.08, 12, 1.0, 1.2, 1.5, 1.5, 2.0, 2.0]
    )
    Xn = x_best0[None, :] + noise
    Xn[:, 1] = np.clip(Xn[:, 1], 70, 140)
    for j in (3, 5, 7):
        Xn[:, j] = np.clip(Xn[:, j], 0.3, 14)
    Xn[:, 2] = np.clip(Xn[:, 2], 0, 20)
    Xn[:, 0] = Xn[:, 0] % (2 * np.pi)

    X = np.vstack([x_best0[None, :], Xs, Xr, Xn])
    print(f"candidates {len(X)}  struct={len(Xs)}")

    t0 = time.perf_counter()
    durs, _, Xfix, _, _, _ = eval_strategies(X, n_time=7000, use_gpu=True)
    print(f"coarse {time.perf_counter()-t0:.2f}s best={durs.max():.4f}")
    order = np.argsort(-durs)
    for k in range(12):
        i = order[k]
        x = Xfix[i]
        print(
            f"  #{k+1:2d} {durs[i]:.4f} th={np.degrees(x[0])%360:.2f} v={x[1]:.1f} "
            f"t=[{x[2]:.2f},{x[4]:.2f},{x[6]:.2f}] tau=[{x[3]:.2f},{x[5]:.2f},{x[7]:.2f}]"
        )

    # 中精度精修 top-50
    refined = []
    n_top = 50
    t1 = time.perf_counter()
    for k in range(n_top):
        xf, df = local_refine(Xfix[order[k]], n_time=10000, steps=45, use_gpu=True)
        refined.append((df, xf))
        if (k + 1) % 5 == 0 or k == 0:
            print(
                f"  refine {k+1}/{n_top}: last={df:.4f} best={max(r[0] for r in refined):.4f}"
            )
    refined.sort(key=lambda z: -z[0])
    print(f"mid refine done {time.perf_counter()-t1:.1f}s best={refined[0][0]:.6f}")

    # 对前 8 做高精度精修
    finals = []
    for k in range(8):
        xf, df = local_refine(
            refined[k][1],
            n_time=22000,
            steps=55,
            spans=np.array([0.025, 2.5, 0.3, 0.3, 0.35, 0.35, 0.4, 0.4]),
            use_gpu=True,
        )
        tot, per, info = highres_union(xf, n_time=300000)
        finals.append((tot, per, info, xf))
        print(f"  final#{k+1} mid={df:.4f} high={tot:.6f} per={np.round(per,3)}")

    finals.sort(key=lambda z: -z[0])
    total, per, info, x_best = finals[0]

    # 再压一轮
    x_best, _ = local_refine(
        x_best,
        n_time=35000,
        steps=40,
        spans=np.array([0.015, 1.5, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25]),
        use_gpu=True,
    )
    total, per, info = highres_union(x_best, n_time=500000)

    elapsed = time.perf_counter() - t_all
    print("\n==== BEST ====")
    print(f"union = {total:.6f} s")
    print(f"per   = {per}")
    print(f"heading={info['heading_deg']:.6f} deg  v={info['v']:.6f}")
    for i, b in enumerate(info["bombs"]):
        print(
            f" bomb{i+1}: drop={b['t_drop']:.4f} tau={b['tau']:.4f} det={b['t_det']:.4f} "
            f"P_det={np.round(b['P_det'],3)}"
        )

    txt = OUT / "q3_result.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("==== 2025国赛A题 问题3 结果 ====\n")
        f.write("FY1 投放 3 枚烟幕干扰弹，干扰 M1；遮蔽取并集（可不连续）\n")
        f.write(f"backend = {device_info()}\n")
        f.write(f"并集有效遮蔽时长 = {total:.6f} s\n")
        f.write(
            f"单弹有效时长(各自，可重叠) = [{per[0]:.6f}, {per[1]:.6f}, {per[2]:.6f}] s\n"
        )
        f.write("\n--- 无人机 ---\n")
        f.write(
            f"航向角 theta = {info['theta']:.10f} rad = {info['heading_deg']:.6f} deg\n"
        )
        f.write(f"飞行速度 v   = {info['v']:.10f} m/s\n")
        for i, b in enumerate(info["bombs"]):
            f.write(f"\n--- 烟幕干扰弹 {i+1} ---\n")
            f.write(f"投放时刻 t_drop = {b['t_drop']:.10f} s\n")
            f.write(f"引信延时 tau    = {b['tau']:.10f} s\n")
            f.write(f"起爆时刻 t_det  = {b['t_det']:.10f} s\n")
            f.write(
                f"投放点 P_drop = [{b['P_drop'][0]:.10f}, {b['P_drop'][1]:.10f}, {b['P_drop'][2]:.10f}]\n"
            )
            f.write(
                f"起爆点 P_det  = [{b['P_det'][0]:.10f}, {b['P_det'][1]:.10f}, {b['P_det'][2]:.10f}]\n"
            )
            f.write(f"单弹有效时长 = {per[i]:.6f} s\n")
        f.write(f"\n总运行时间 = {elapsed:.2f} s\n")
        f.write(f"决策向量 x = {info['x']}\n")
        f.write("求解 = GPU torch batch + stage2 multi-start refine\n")

    with open(OUT / "q3_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_union": total,
                "per_bomb": per.tolist(),
                "info": info,
                "elapsed_s": elapsed,
                "backend": device_info(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    xlsx = write_result1_xlsx(info, per, total, OUT / "result1.xlsx")
    print("saved", display_path(txt))
    print("excel", display_path(xlsx))
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()

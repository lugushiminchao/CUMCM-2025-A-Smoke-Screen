# -*- coding: utf-8 -*-
"""问题3 第三阶段：固定 v=140、航向近 180° 的接力时间轴精细搜索"""
from __future__ import annotations

import json
import time

import numpy as np

from q3_optimize import (
    OUT,
    device_info,
    eval_strategies,
    highres_union,
    local_refine,
    write_result1_xlsx,
)


def main():
    t_all = time.perf_counter()
    print("==== Q3 stage3 relay search ====")
    print(device_info())

    with open(OUT / "q3_result.json", "r", encoding="utf-8") as f:
        cur = json.load(f)
    x0 = np.array(cur["info"]["x"], dtype=float)
    print("start", cur["total_union"], x0)

    rng = np.random.default_rng(999)
    rows = [x0.tolist()]

    # 航向、速度网格（控制规模，避免组合爆炸）
    ths = np.linspace(np.pi - 0.06, np.pi + 0.06, 9)
    vs = np.array([125, 135, 140], dtype=float)

    # 起爆时刻接力：det1 ~ 2.5~5, det2 更晚, det3 再晚
    for th in ths:
        for v in vs:
            for det1 in np.linspace(2.8, 4.8, 6):
                for det2 in np.linspace(det1 + 2.0, det1 + 5.0, 5):
                    for det3 in np.linspace(det2 + 1.5, det2 + 4.5, 4):
                        for t1 in [0.0, 0.4]:
                            tau1 = det1 - t1
                            if tau1 < 0.5 or tau1 > 14:
                                continue
                            for t2 in [t1 + 1.0, t1 + 2.0, t1 + 3.0]:
                                if t2 >= det2 - 0.4:
                                    continue
                                tau2 = det2 - t2
                                if tau2 < 0.5 or tau2 > 14:
                                    continue
                                for t3 in [t2 + 1.0, t2 + 2.0, t2 + 3.5]:
                                    if t3 >= det3 - 0.4:
                                        continue
                                    tau3 = det3 - t3
                                    if tau3 < 0.5 or tau3 > 14:
                                        continue
                                    rows.append([th, v, t1, tau1, t2, tau2, t3, tau3])

    # 随机接力
    for _ in range(20000):
        th = float(np.pi + rng.normal(0, 0.05))
        v = float(rng.choice([125, 130, 135, 140]))
        det1 = float(rng.uniform(2.8, 4.8))
        det2 = float(rng.uniform(det1 + 2.0, det1 + 5.5))
        det3 = float(rng.uniform(det2 + 1.5, det2 + 5.0))
        t1 = float(rng.uniform(0.0, 1.2))
        t2 = float(rng.uniform(t1 + 1.0, t1 + 4.5))
        t3 = float(rng.uniform(t2 + 1.0, t2 + 4.5))
        tau1, tau2, tau3 = det1 - t1, det2 - t2, det3 - t3
        if min(tau1, tau2, tau3) < 0.5 or max(tau1, tau2, tau3) > 14:
            continue
        rows.append([th, v, t1, tau1, t2, tau2, t3, tau3])

    # 邻域
    noise = rng.normal(0, 1, (10000, 8)) * np.array(
        [0.04, 5, 0.6, 0.8, 0.8, 0.8, 1.0, 1.0]
    )
    Xn = x0[None, :] + noise
    Xn[:, 1] = np.clip(Xn[:, 1], 70, 140)
    for j in (3, 5, 7):
        Xn[:, j] = np.clip(Xn[:, j], 0.4, 14)
    Xn[:, 2] = np.clip(Xn[:, 2], 0, 15)
    Xn[:, 0] = Xn[:, 0] % (2 * np.pi)

    X = np.vstack([np.asarray(rows, dtype=float), Xn])
    # 去重抽样若太大
    if len(X) > 120000:
        idx = rng.choice(len(X), 120000, replace=False)
        X = X[idx]
        X = np.vstack([x0[None, :], X])
    print("candidates", len(X))

    t0 = time.perf_counter()
    durs, _, Xfix, _, _, _ = eval_strategies(X, n_time=8000, use_gpu=True)
    print(f"eval {time.perf_counter()-t0:.2f}s best={durs.max():.4f}")
    order = np.argsort(-durs)
    for k in range(10):
        i = order[k]
        x = Xfix[i]
        print(
            f" #{k+1} {durs[i]:.4f} th={np.degrees(x[0])%360:.2f} v={x[1]:.1f} "
            f"t=[{x[2]:.2f},{x[4]:.2f},{x[6]:.2f}] tau=[{x[3]:.2f},{x[5]:.2f},{x[7]:.2f}]"
        )

    refined = []
    for k in range(30):
        xf, df = local_refine(
            Xfix[order[k]],
            n_time=14000,
            steps=60,
            spans=np.array([0.04, 4.0, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7]),
            use_gpu=True,
        )
        refined.append((df, xf))
        if (k + 1) % 5 == 0:
            print(f" refine {k+1}/30 best={max(r[0] for r in refined):.4f}")
    refined.sort(key=lambda z: -z[0])

    finals = []
    for k in range(min(10, len(refined))):
        xf, df = local_refine(
            refined[k][1],
            n_time=28000,
            steps=50,
            spans=np.array([0.02, 2.0, 0.25, 0.25, 0.3, 0.3, 0.35, 0.35]),
            use_gpu=True,
        )
        tot, per, info = highres_union(xf, n_time=400000)
        finals.append((tot, per, info, xf))
        print(f" final#{k+1} {tot:.6f} per={np.round(per,3)}")

    finals.sort(key=lambda z: -z[0])
    total, per, info, xb = finals[0]

    # polish
    xb, _ = local_refine(
        xb,
        n_time=40000,
        steps=35,
        spans=np.array([0.01, 1.0, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2]),
        use_gpu=True,
    )
    total, per, info = highres_union(xb, n_time=600000)

    elapsed = time.perf_counter() - t_all
    print("BEST", total, "per", per)
    print("heading", info["heading_deg"], "v", info["v"])
    for i, b in enumerate(info["bombs"]):
        print(i + 1, b)

    # 若不如旧解则保留旧解
    if total < cur["total_union"] - 1e-4:
        print("keep previous better", cur["total_union"])
        total = cur["total_union"]
        per = np.array(cur["per_bomb"])
        info = cur["info"]
    else:
        with open(OUT / "q3_result.txt", "w", encoding="utf-8") as f:
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
            f.write(f"\n总运行时间(stage3) = {elapsed:.2f} s\n")
            f.write(f"决策向量 x = {info['x']}\n")
            f.write("求解 = GPU torch batch + stage3 relay refine\n")

        with open(OUT / "q3_result.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_union": float(total),
                    "per_bomb": np.asarray(per).tolist(),
                    "info": info,
                    "elapsed_s": elapsed,
                    "backend": device_info(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        write_result1_xlsx(info, per, total, OUT / "result1.xlsx")

    print("done", total, f"{elapsed:.1f}s")


if __name__ == "__main__":
    main()

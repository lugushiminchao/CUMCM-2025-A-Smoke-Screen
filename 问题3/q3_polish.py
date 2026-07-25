# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from q3_optimize import (  # noqa
    OUT,
    eval_strategies,
    highres_union,
    local_refine,
    write_result1_xlsx,
    device_info,
)

log = open(OUT / "polish_log.txt", "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


# Q2 sanity
x = np.array([[3.0880757565, 71.8890217683, 0.0, 2.5032397513, 50, 1, 52, 1]])
d, pe, _, _, _, _ = eval_strategies(x, n_time=50000, use_gpu=True)
p("Q2 via union", float(d[0]), pe[0].tolist())
tot, per, _ = highres_union(
    np.array([3.0880757565, 71.8890217683, 0.0, 2.5032397513, 50, 1, 52, 1]),
    n_time=300000,
)
p("Q2 highres", tot, per.tolist())

cur = json.load(open(OUT / "q3_result.json", encoding="utf-8"))
x = np.array(cur["info"]["x"], dtype=float)
tot, per, info = highres_union(x, n_time=800000)
p("current", tot, per.tolist())

# multi polish seeds
seeds = [x.copy()]
# fix t1=0 v=140
s = x.copy()
s[1] = 140
s[2] = 0.0
seeds.append(s)
# slightly different
for th in [3.13, 3.135, 3.14, np.pi]:
    for det_shift in [0.0, -0.3, 0.3]:
        s = x.copy()
        s[0] = th
        s[1] = 140
        s[2] = 0.0
        s[3] = max(0.5, s[3] + det_shift)
        s[5] = max(0.5, s[5] + det_shift * 0.5)
        s[7] = max(0.5, s[7] + det_shift * 0.5)
        seeds.append(s)

best = (tot, per, info, x)
for i, s0 in enumerate(seeds):
    xf, df = local_refine(
        s0,
        n_time=40000,
        steps=70,
        spans=np.array([0.03, 3.0, 0.4, 0.5, 0.6, 0.6, 0.7, 0.7]),
        use_gpu=True,
    )
    t2, p2, i2 = highres_union(xf, n_time=500000)
    p(f"seed{i}", "mid", df, "high", t2, "per", np.round(p2, 3).tolist())
    if t2 > best[0]:
        best = (t2, p2, i2, xf)

# final polish
xf, _ = local_refine(
    best[3],
    n_time=60000,
    steps=40,
    spans=np.array([0.01, 1.0, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2]),
    use_gpu=True,
)
total, per, info = highres_union(xf, n_time=1000000)
p("FINAL", total, per.tolist())
p("heading", info["heading_deg"], "v", info["v"])
for j, b in enumerate(info["bombs"]):
    p("bomb", j + 1, b)

# save if better or equal
if total + 1e-6 >= cur["total_union"]:
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
        f.write("求解 = GPU torch batch multi-stage refine\n")
        f.write("判据: 云团球心到导弹-真目标中心视线段距离 <= 10 m；三枚取并集\n")
    json.dump(
        {
            "total_union": float(total),
            "per_bomb": per.tolist(),
            "info": info,
            "backend": device_info(),
        },
        open(OUT / "q3_result.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    path = write_result1_xlsx(info, per, total, OUT / "result1.xlsx")
    p("saved", Path(path).resolve().relative_to(THIS_DIR.parent.resolve()).as_posix())
else:
    p("kept old", cur["total_union"])

log.close()

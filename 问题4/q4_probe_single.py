# -*- coding: utf-8 -*-
"""Probe single-UAV best cover vs M1 under cylinder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common.smoke_geom import (  # noqa: E402
    PTS_FAST,
    PTS_OPT,
    UAVS,
    device_info,
    kinematics,
    kinematics_batch,
    shield_duration,
    union_duration_batch,
)

print("device:", device_info())


def aim_theta(fy0, target_xy=(0.0, 0.0)):
    dxy = np.array(target_xy, dtype=float) - fy0[:2]
    return float(np.arctan2(dxy[1], dxy[0]))


def best_single(name, n_rand=3000):
    fy0 = UAVS[name]
    th0 = aim_theta(fy0)
    thm = aim_theta(fy0, (20000.0, 0.0))
    # intercept heading: fly toward a point near origin-target corridor
    th_t = aim_theta(fy0, (0.0, 200.0))
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    rows = []
    for th in [th0, thm, th_t, np.pi, 0.0] + list(th0 + np.linspace(-1.2, 1.2, 11)) + list(
        thm + np.linspace(-0.8, 0.8, 7)
    ) + list(th_t + np.linspace(-0.8, 0.8, 7)):
        for v in [70, 85, 100, 120, 140]:
            for td in [0, 0.5, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]:
                for tau in [1.0, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12]:
                    rows.append([th, v, td, tau])
    Xr = np.column_stack(
        [
            (th0 + (rng.random(n_rand) - 0.5) * 2.5) % (2 * np.pi),
            70 + rng.random(n_rand) * 70,
            rng.random(n_rand) * 35,
            0.4 + rng.random(n_rand) * 12,
        ]
    )
    # bias toward low altitude detonation near LOS: larger tau to sink, or early drop
    X = np.vstack([np.asarray(rows, float), Xr])
    X[:, 0] %= 2 * np.pi
    X[:, 1] = np.clip(X[:, 1], 70, 140)
    X[:, 2] = np.clip(X[:, 2], 0, 45)
    X[:, 3] = np.clip(X[:, 3], 0.3, 16)
    X = np.unique(np.round(X, 5), axis=0)

    best = -1.0
    bestx = None
    chunk = 800
    for i0 in range(0, len(X), chunk):
        i1 = min(len(X), i0 + chunk)
        th, v, td, tau = X[i0:i1, 0], X[i0:i1, 1], X[i0:i1, 2], X[i0:i1, 3]
        _, P, Td = kinematics_batch(th, v, td, tau, fy0=fy0)
        d, _ = union_duration_batch(
            P[:, None, :],
            Td[:, None],
            mode="cylinder",
            pts=PTS_FAST,
            n_time=3500,
            use_gpu=True,
            missile="M1",
        )
        j = int(np.argmax(d))
        if d[j] > best:
            best = float(d[j])
            bestx = X[i0 + j].copy()
    # refine
    lb = np.array([0.0, 70.0, 0.0, 0.3])
    ub = np.array([2 * np.pi, 140.0, 45.0, 16.0])
    x = bestx.copy()
    val = best
    span = np.array([0.1, 10.0, 1.5, 1.5])
    for it in range(40):
        improved = False
        for d in range(4):
            for sgn in (-1.0, 1.0):
                trial = x.copy()
                trial[d] += sgn * span[d]
                trial[0] %= 2 * np.pi
                trial = np.clip(trial, lb, ub)
                _, pd, td_ = kinematics_batch(
                    [trial[0]], [trial[1]], [trial[2]], [trial[3]], fy0=fy0
                )
                dv, _ = union_duration_batch(
                    pd[:, None, :],
                    td_[:, None],
                    mode="cylinder",
                    pts=PTS_OPT,
                    n_time=4500,
                    use_gpu=True,
                    missile="M1",
                )
                if dv[0] > val + 1e-9:
                    val = float(dv[0])
                    x = trial
                    improved = True
        noise = (rng.random((40, 4)) - 0.5) * 2 * span
        trials = np.clip(x[None, :] + noise, lb, ub)
        trials[:, 0] %= 2 * np.pi
        _, pd, td_ = kinematics_batch(
            trials[:, 0], trials[:, 1], trials[:, 2], trials[:, 3], fy0=fy0
        )
        dvs, _ = union_duration_batch(
            pd[:, None, :],
            td_[:, None],
            mode="cylinder",
            pts=PTS_OPT,
            n_time=4000,
            use_gpu=True,
            missile="M1",
        )
        j = int(np.argmax(dvs))
        if dvs[j] > val + 1e-9:
            val = float(dvs[j])
            x = trials[j]
            improved = True
        span *= 0.9 if improved else 1.06
        span = np.minimum(span, np.array([0.15, 15, 2.5, 2.5]))

    v_fy, p_drop, t_det, p_det = kinematics(*x, fy0=fy0)
    dur_hi, segs = shield_duration(
        p_det, t_det, mode="cylinder", pts=PTS_OPT, n=100000, refine=True, missile="M1"
    )
    print(
        f"{name}: coarse={best:.4f} refine={val:.4f} hi={dur_hi:.4f} "
        f"th={np.degrees(x[0])%360:.2f} v={x[1]:.2f} td={x[2]:.3f} tau={x[3]:.3f} "
        f"t_det={t_det:.3f} P_det={p_det.tolist()} segs={segs}"
    )
    return x, float(dur_hi)


if __name__ == "__main__":
    for n in ["FY1", "FY2", "FY3"]:
        best_single(n)

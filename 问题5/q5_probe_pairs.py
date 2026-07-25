# -*- coding: utf-8 -*-
"""Probe UAV-missile pairings under cylinder model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common.smoke_geom import (  # noqa: E402
    MISSILES,
    PTS_FAST,
    PTS_OPT,
    UAVS,
    _MISSILE_CACHE,
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


def best_pair(uav, missile, n_rand=3500):
    fy0 = UAVS[uav]
    m0 = MISSILES[missile]
    _, _, t_hit, _ = _MISSILE_CACHE[missile]
    th0 = aim_theta(fy0)
    thm = aim_theta(fy0, m0[:2])
    tht = aim_theta(fy0, (0.0, 200.0))
    rng = np.random.default_rng(abs(hash((uav, missile))) % (2**32))

    rows = []
    # denser time grids up to near hit
    tds = np.linspace(0, min(50, t_hit * 0.85), 18)
    taus = [1.0, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12]
    ths = (
        [th0, thm, tht, np.pi, 0.0]
        + list(th0 + np.linspace(-1.5, 1.5, 13))
        + list(thm + np.linspace(-1.0, 1.0, 9))
        + list(tht + np.linspace(-1.0, 1.0, 9))
    )
    for th in ths[::1]:
        for v in [70, 90, 110, 130, 140]:
            for td in tds[::2]:
                for tau in taus[::2]:
                    rows.append([th, v, td, tau])

    Xr = np.column_stack(
        [
            (th0 + (rng.random(n_rand) - 0.5) * 2.8) % (2 * np.pi),
            70 + rng.random(n_rand) * 70,
            rng.random(n_rand) * min(50.0, t_hit * 0.9),
            0.4 + rng.random(n_rand) * 12,
        ]
    )
    # intercept-biased: drop so det near missile path time
    for _ in range(800):
        # aim roughly toward origin corridor
        th = th0 + (rng.random() - 0.5) * 1.0
        v = 70 + rng.random() * 70
        # det time roughly when missile is near UAV x
        t_det_guess = max(1.0, (fy0[0] - 0) / 300.0)  # rough
        t_det_guess = max(2.0, min(t_hit * 0.8, t_det_guess + (rng.random() - 0.5) * 15))
        tau = 1.0 + rng.random() * 10
        td = max(0.0, t_det_guess - tau)
        rows.append([th, v, td, tau])

    X = np.vstack([np.asarray(rows, float), Xr])
    X[:, 0] %= 2 * np.pi
    X[:, 1] = np.clip(X[:, 1], 70, 140)
    X[:, 2] = np.clip(X[:, 2], 0, 55)
    X[:, 3] = np.clip(X[:, 3], 0.3, 16)
    X = np.unique(np.round(X, 5), axis=0)

    best = -1.0
    bestx = None
    chunk = 900
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
            missile=missile,
        )
        j = int(np.argmax(d))
        if d[j] > best:
            best = float(d[j])
            bestx = X[i0 + j].copy()

    if best <= 0 or bestx is None:
        print(f"{uav}->{missile}: best=0 (no cover found in coarse)")
        return None, 0.0

    # refine
    lb = np.array([0.0, 70.0, 0.0, 0.3])
    ub = np.array([2 * np.pi, 140.0, 55.0, 16.0])
    x = bestx.copy()
    val = best
    span = np.array([0.12, 12.0, 2.0, 2.0])
    for _ in range(40):
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
                    n_time=4000,
                    use_gpu=True,
                    missile=missile,
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
            n_time=3500,
            use_gpu=True,
            missile=missile,
        )
        j = int(np.argmax(dvs))
        if dvs[j] > val + 1e-9:
            val = float(dvs[j])
            x = trials[j]
            improved = True
        span *= 0.9 if improved else 1.06
        span = np.minimum(span, np.array([0.2, 18, 3.0, 3.0]))

    v_fy, p_drop, t_det, p_det = kinematics(*x, fy0=fy0)
    dur_hi, segs = shield_duration(
        p_det, t_det, mode="cylinder", pts=PTS_OPT, n=80000, refine=True, missile=missile
    )
    print(
        f"{uav}->{missile}: hi={dur_hi:.4f} th={np.degrees(x[0])%360:.1f} "
        f"v={x[1]:.1f} td={x[2]:.2f} tau={x[3]:.2f} t_det={t_det:.2f} segs={segs}"
    )
    return x, float(dur_hi)


if __name__ == "__main__":
    pairs = [
        ("FY1", "M1"),
        ("FY1", "M2"),
        ("FY1", "M3"),
        ("FY2", "M1"),
        ("FY2", "M2"),
        ("FY2", "M3"),
        ("FY3", "M1"),
        ("FY3", "M2"),
        ("FY3", "M3"),
        ("FY4", "M1"),
        ("FY4", "M2"),
        ("FY4", "M3"),
        ("FY5", "M1"),
        ("FY5", "M2"),
        ("FY5", "M3"),
    ]
    results = {}
    for u, m in pairs:
        x, d = best_pair(u, m)
        results[(u, m)] = d
    print("\n==== pairing matrix (s) ====")
    print(f"{'':6s}", end="")
    for m in ["M1", "M2", "M3"]:
        print(f"{m:>8s}", end="")
    print()
    for u in ["FY1", "FY2", "FY3", "FY4", "FY5"]:
        print(f"{u:6s}", end="")
        for m in ["M1", "M2", "M3"]:
            print(f"{results.get((u,m),0):8.3f}", end="")
        print()

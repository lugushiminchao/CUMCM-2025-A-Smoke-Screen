# -*- coding: utf-8 -*-
"""
2025国赛A题 公共几何：烟幕云团对真目标的有效遮蔽判据

支持两种真目标模型:
  - point:  几何中心 (0,200,5)
  - cylinder: 严格全遮蔽 —— 导弹到圆柱表面采样点的视线段均被云团挡住

判据统一:
  云团球心 C(t) 到视线段 M(t)–P 的距离 d <= R(=10),
  且垂足参数 s 落在 [0,1]（与问题1/2 点模型一致）。
圆柱严格全遮蔽: 对所有采样点 P 同时成立。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch

    TORCH_OK = True
    CUDA_OK = bool(torch.cuda.is_available())
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_OK = False
    CUDA_OK = False

# ---------------- 场景常量 ----------------
G = 9.8
V_MISSILE = 300.0
V_SINK = 3.0
R_SMOKE = 10.0
R2_SMOKE = R_SMOKE * R_SMOKE
T_LIFE = 20.0
DROP_GAP = 1.0

# 默认导弹 / 无人机：问题1-3 的 M1 / FY1
M0 = np.array([20000.0, 0.0, 2000.0], dtype=np.float64)
FY0 = np.array([17800.0, 0.0, 1800.0], dtype=np.float64)
TGT_CENTER = np.array([0.0, 200.0, 5.0], dtype=np.float64)
CYL_CENTER_XY = np.array([0.0, 200.0], dtype=np.float64)
R_CYL = 7.0
H_CYL = 10.0
Z_BOT = 0.0
Z_TOP = 10.0

# 全部导弹 / 无人机初值
MISSILES = {
    "M1": np.array([20000.0, 0.0, 2000.0], dtype=np.float64),
    "M2": np.array([19000.0, 600.0, 2100.0], dtype=np.float64),
    "M3": np.array([18000.0, -600.0, 1900.0], dtype=np.float64),
}
UAVS = {
    "FY1": np.array([17800.0, 0.0, 1800.0], dtype=np.float64),
    "FY2": np.array([12000.0, 1400.0, 1400.0], dtype=np.float64),
    "FY3": np.array([6000.0, -3000.0, 700.0], dtype=np.float64),
    "FY4": np.array([11000.0, 2000.0, 1800.0], dtype=np.float64),
    "FY5": np.array([13000.0, -2000.0, 1300.0], dtype=np.float64),
}


def missile_kinematics(m0: np.ndarray):
    """导弹指向假目标原点 (0,0,0)，返回 (u_m, v_m, t_hit, dist)."""
    m0 = np.asarray(m0, dtype=np.float64)
    dist = float(np.linalg.norm(m0))
    u = -m0 / dist
    v = V_MISSILE * u
    t_hit = dist / V_MISSILE
    return u, v, t_hit, dist


DIST_M0 = float(np.linalg.norm(M0))
U_M = -M0 / DIST_M0
V_M = V_MISSILE * U_M
T_HIT = DIST_M0 / V_MISSILE

# 预计算各导弹运动学
_MISSILE_CACHE = {k: missile_kinematics(v) for k, v in MISSILES.items()}


def device_info() -> str:
    if TORCH_OK and CUDA_OK:
        return f"torch-cuda:{torch.cuda.get_device_name(0)}"
    if TORCH_OK:
        return "torch-cpu"
    return "numpy-cpu"


def heading_deg(theta: float) -> float:
    return float(np.degrees(theta) % 360.0)


# ---------------- 圆柱采样点 ----------------
def cylinder_sample_points(
    n_theta: int = 24,
    n_z: int = 5,
    include_caps: bool = True,
    cap_rings: Sequence[float] = (0.5, 1.0),
    include_axis: bool = True,
) -> np.ndarray:
    """
    生成圆柱表面采样点 (K,3)。
    - 侧面: n_theta × n_z
    - 轴心线: n_z（可选）
    - 上下底: 若干半径环 × n_theta（可选）
    """
    pts: List[np.ndarray] = []
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(Z_BOT, Z_TOP, n_z)

    for z in zs:
        for th in thetas:
            pts.append(
                np.array(
                    [
                        CYL_CENTER_XY[0] + R_CYL * np.cos(th),
                        CYL_CENTER_XY[1] + R_CYL * np.sin(th),
                        z,
                    ],
                    dtype=np.float64,
                )
            )
        if include_axis:
            pts.append(np.array([CYL_CENTER_XY[0], CYL_CENTER_XY[1], z], dtype=np.float64))

    if include_caps:
        for z in (Z_BOT, Z_TOP):
            for r_frac in cap_rings:
                rr = r_frac * R_CYL
                for th in thetas:
                    pts.append(
                        np.array(
                            [
                                CYL_CENTER_XY[0] + rr * np.cos(th),
                                CYL_CENTER_XY[1] + rr * np.sin(th),
                                z,
                            ],
                            dtype=np.float64,
                        )
                    )

    arr = np.unique(np.asarray(pts, dtype=np.float64), axis=0)
    return arr


# 预置几套密度（优化 / 终算）
PTS_FAST = cylinder_sample_points(n_theta=16, n_z=4, include_caps=True, cap_rings=(1.0,))
PTS_OPT = cylinder_sample_points(n_theta=24, n_z=5, include_caps=True, cap_rings=(0.5, 1.0))
PTS_FINE = cylinder_sample_points(n_theta=48, n_z=9, include_caps=True, cap_rings=(0.5, 1.0))
PTS_ULTRA = cylinder_sample_points(n_theta=72, n_z=11, include_caps=True, cap_rings=(0.35, 0.7, 1.0))


# ---------------- 运动学 ----------------
def kinematics(theta: float, v: float, t_drop: float, tau: float, fy0: Optional[np.ndarray] = None):
    if fy0 is None:
        fy0 = FY0
    fy0 = np.asarray(fy0, dtype=np.float64)
    v_fy = np.array([v * np.cos(theta), v * np.sin(theta), 0.0], dtype=np.float64)
    p_drop = fy0 + v_fy * t_drop
    t_det = t_drop + tau
    p_det = p_drop + np.array(
        [v_fy[0] * tau, v_fy[1] * tau, -0.5 * G * tau * tau], dtype=np.float64
    )
    return v_fy, p_drop, t_det, p_det


def kinematics_batch(theta, v, t_drop, tau, fy0: Optional[np.ndarray] = None):
    if fy0 is None:
        fy0 = FY0
    fy0 = np.asarray(fy0, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t_drop = np.asarray(t_drop, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    p_drop = np.stack(
        [fy0[0] + vx * t_drop, fy0[1] + vy * t_drop, np.full_like(t_drop, fy0[2])],
        axis=-1,
    )
    t_det = t_drop + tau
    p_det = np.stack(
        [
            p_drop[..., 0] + vx * tau,
            p_drop[..., 1] + vy * tau,
            p_drop[..., 2] - 0.5 * G * tau * tau,
        ],
        axis=-1,
    )
    return p_drop, p_det, t_det


def _resolve_missile(missile: Optional[str] = None, m0=None, v_m=None):
    """返回 (m0, v_m, t_hit)。默认 M1。"""
    if m0 is not None and v_m is not None:
        m0 = np.asarray(m0, dtype=np.float64)
        v_m = np.asarray(v_m, dtype=np.float64)
        t_hit = float(np.linalg.norm(m0) / V_MISSILE)
        return m0, v_m, t_hit
    if missile is not None and missile in _MISSILE_CACHE:
        u, v, t_hit, _ = _MISSILE_CACHE[missile]
        return MISSILES[missile], v, t_hit
    return M0, V_M, T_HIT


# ---------------- 点目标 LOS ----------------
def los_dist_point(ts, p_det, t_det, tgt=TGT_CENTER, missile: Optional[str] = None, m0=None, v_m=None):
    """云团中心到导弹-单点目标视线段距离。返回 d,s shape (T,)"""
    ts = np.asarray(ts, dtype=np.float64).ravel()
    m0, v_m, _ = _resolve_missile(missile, m0, v_m)
    M = m0[None, :] + v_m[None, :] * ts[:, None]
    C = np.empty_like(M)
    C[:, 0] = p_det[0]
    C[:, 1] = p_det[1]
    C[:, 2] = p_det[2] - V_SINK * (ts - t_det)
    AB = np.asarray(tgt, dtype=np.float64)[None, :] - M
    AC = C - M
    L2 = np.maximum(np.sum(AB * AB, axis=1), 1e-18)
    s = np.sum(AC * AB, axis=1) / L2
    sc = np.clip(s, 0.0, 1.0)
    P = M + sc[:, None] * AB
    d = np.linalg.norm(C - P, axis=1)
    return d, s


def shield_mask_point(ts, p_det, t_det, tgt=TGT_CENTER, missile: Optional[str] = None, m0=None, v_m=None):
    ts = np.asarray(ts, dtype=np.float64).ravel()
    d, s = los_dist_point(ts, p_det, t_det, tgt=tgt, missile=missile, m0=m0, v_m=v_m)
    alive = (ts >= t_det) & (ts <= t_det + T_LIFE)
    return alive & (d <= R_SMOKE) & (s >= 0.0) & (s <= 1.0)


# ---------------- 圆柱严格全遮蔽 ----------------
def los_max_dist_cylinder(ts, p_det, t_det, pts: np.ndarray, missile: Optional[str] = None, m0=None, v_m=None):
    """
    对每个时刻，计算 max_k d(C, M–P_k)。
    返回:
      dmax (T,), 以及是否全部点 s∈[0,1] 且 d<=R 的 mask 所需中间量
    实际返回 ok_all, dmax
    """
    ts = np.asarray(ts, dtype=np.float64).ravel()
    pts = np.asarray(pts, dtype=np.float64)
    T = ts.shape[0]
    K = pts.shape[0]
    m0, v_m, _ = _resolve_missile(missile, m0, v_m)
    M = m0[None, :] + v_m[None, :] * ts[:, None]  # (T,3)
    C = np.empty((T, 3), dtype=np.float64)
    C[:, 0] = p_det[0]
    C[:, 1] = p_det[1]
    C[:, 2] = p_det[2] - V_SINK * (ts - t_det)

    # 分块点，控制内存
    dmax = np.full(T, -np.inf, dtype=np.float64)
    all_ok = np.ones(T, dtype=bool)
    alive = (ts >= t_det) & (ts <= t_det + T_LIFE)
    chunk = 64
    for i0 in range(0, K, chunk):
        P = pts[i0 : i0 + chunk]  # (p,3)
        AB = P[None, :, :] - M[:, None, :]  # (T,p,3)
        AC = C[:, None, :] - M[:, None, :]  # (T,p,3)
        L2 = np.maximum(np.sum(AB * AB, axis=2), 1e-18)
        s = np.sum(AC * AB, axis=2) / L2
        sc = np.clip(s, 0.0, 1.0)
        closest = M[:, None, :] + sc[:, :, None] * AB
        d = np.linalg.norm(C[:, None, :] - closest, axis=2)  # (T,p)
        dmax = np.maximum(dmax, np.max(d, axis=1))
        ok_chunk = (d <= R_SMOKE) & (s >= 0.0) & (s <= 1.0)
        all_ok &= np.all(ok_chunk, axis=1)

    all_ok &= alive
    return all_ok, dmax


def shield_mask_cylinder(ts, p_det, t_det, pts: np.ndarray = None, missile: Optional[str] = None, m0=None, v_m=None):
    if pts is None:
        pts = PTS_OPT
    ok, _ = los_max_dist_cylinder(ts, p_det, t_det, pts, missile=missile, m0=m0, v_m=v_m)
    return ok


def shield_mask(
    ts,
    p_det,
    t_det,
    mode: str = "cylinder",
    pts: Optional[np.ndarray] = None,
    missile: Optional[str] = None,
    m0=None,
    v_m=None,
):
    if mode == "point":
        return shield_mask_point(ts, p_det, t_det, missile=missile, m0=m0, v_m=v_m)
    if pts is None:
        pts = PTS_OPT
    return shield_mask_cylinder(ts, p_det, t_det, pts, missile=missile, m0=m0, v_m=v_m)


# ---------------- 单弹时长（扫描 + 可选二分） ----------------
def _mask_to_segments(ts, ok, bisect_fn=None, bisect_iters=40):
    ts = np.asarray(ts, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool)
    t0, t1 = float(ts[0]), float(ts[-1])
    if not np.any(ok):
        return 0.0, []

    jumps = np.where(np.diff(ok.astype(np.int8)) != 0)[0]
    edges = []
    for j in jumps:
        lo, hi = float(ts[j]), float(ts[j + 1])
        left_ok = bool(ok[j])
        if bisect_fn is not None:
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                mid_ok = bool(bisect_fn(mid))
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
    dur = float(sum(b - a for a, b in segs))
    return dur, segs


def shield_duration(
    p_det,
    t_det,
    mode: str = "cylinder",
    pts: Optional[np.ndarray] = None,
    n: int = 50000,
    refine: bool = True,
    bisect_iters: int = 40,
    missile: Optional[str] = None,
    m0=None,
    v_m=None,
    t_hit: Optional[float] = None,
):
    """单云团有效遮蔽时长。返回 (dur, segs)"""
    if pts is None:
        pts = PTS_OPT if mode == "cylinder" else None
    m0, v_m, t_hit_def = _resolve_missile(missile, m0, v_m)
    if t_hit is None:
        t_hit = t_hit_def
    t0 = float(t_det)
    t1 = min(float(t_det) + T_LIFE, float(t_hit))
    if t1 <= t0:
        return 0.0, []
    ts = np.linspace(t0, t1, n)
    ok = shield_mask(ts, p_det, t_det, mode=mode, pts=pts, missile=missile, m0=m0, v_m=v_m)

    def bisect_fn(t):
        return bool(
            shield_mask(np.array([t]), p_det, t_det, mode=mode, pts=pts, missile=missile, m0=m0, v_m=v_m)[0]
        )

    if refine:
        return _mask_to_segments(ts, ok, bisect_fn=bisect_fn, bisect_iters=bisect_iters)
    dt = ts[1] - ts[0]
    return float(np.sum(ok) * dt), []


def union_duration_single_path(
    bombs: Sequence[Tuple[np.ndarray, float]],
    mode: str = "cylinder",
    pts: Optional[np.ndarray] = None,
    n: int = 100000,
    refine: bool = True,
    bisect_iters: int = 40,
    missile: Optional[str] = None,
    m0=None,
    v_m=None,
    t_hit: Optional[float] = None,
):
    """
    多云团并集时长。
    bombs: list of (p_det, t_det)
    返回 total, per_bomb_durs, segs
    """
    if pts is None and mode == "cylinder":
        pts = PTS_OPT
    if not bombs:
        return 0.0, [], []
    m0, v_m, t_hit_def = _resolve_missile(missile, m0, v_m)
    if t_hit is None:
        t_hit = t_hit_def
    t_dets = [float(td) for _, td in bombs]
    t0 = max(0.0, min(t_dets) - 0.2)
    t1 = min(float(t_hit), max(t_dets) + T_LIFE)
    if t1 <= t0:
        return 0.0, [0.0] * len(bombs), []
    ts = np.linspace(t0, t1, n)
    any_ok = np.zeros(ts.shape[0], dtype=bool)
    per_masks = []
    for p_det, t_det in bombs:
        mk = shield_mask(ts, p_det, t_det, mode=mode, pts=pts, missile=missile, m0=m0, v_m=v_m)
        per_masks.append(mk)
        any_ok |= mk

    def bisect_union(t):
        tt = np.array([t])
        for p_det, t_det in bombs:
            if bool(
                shield_mask(tt, p_det, t_det, mode=mode, pts=pts, missile=missile, m0=m0, v_m=v_m)[0]
            ):
                return True
        return False

    if refine:
        total, segs = _mask_to_segments(ts, any_ok, bisect_fn=bisect_union, bisect_iters=bisect_iters)
    else:
        dt = ts[1] - ts[0]
        total = float(np.sum(any_ok) * dt)
        segs = []

    per = []
    for (p_det, t_det), mk in zip(bombs, per_masks):
        if refine:
            d, _ = shield_duration(
                p_det,
                t_det,
                mode=mode,
                pts=pts,
                n=max(30000, n // max(1, len(bombs))),
                refine=True,
                missile=missile,
                m0=m0,
                v_m=v_m,
                t_hit=t_hit,
            )
        else:
            dt = ts[1] - ts[0]
            d = float(np.sum(mk) * dt)
        per.append(d)
    return total, per, segs


# ---------------- 批量评估（优化用，圆柱） ----------------
def _union_duration_numpy_targets(
    P_det: np.ndarray,
    t_det: np.ndarray,
    targets: np.ndarray,
    n_time: int = 6000,
    m0=None,
    v_m=None,
    t_hit: Optional[float] = None,
):
    """
    点目标批量并集: 每个 target 独立判据，圆柱严格 = 全部 target 同时遮蔽。
    P_det: (N,B,3), t_det: (N,B), targets: (P,3)
    返回 dur(N,), per(N,B)
    实现: 对每个采样点算 bomb 遮蔽，再对点取 AND，对弹取 OR。
    """
    P_det = np.asarray(P_det, dtype=np.float64)
    t_det = np.asarray(t_det, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    N, B, _ = P_det.shape
    P = targets.shape[0]
    if m0 is None:
        m0 = M0
    if v_m is None:
        v_m = V_M
    if t_hit is None:
        t_hit = T_HIT
    m0 = np.asarray(m0, dtype=np.float64)
    v_m = np.asarray(v_m, dtype=np.float64)

    t0 = max(0.0, float(np.nanmin(t_det)) - 0.5)
    t1 = min(float(t_hit), float(np.nanmax(t_det)) + T_LIFE)
    if t1 <= t0:
        return np.zeros(N), np.zeros((N, B))

    ts = np.linspace(t0, t1, n_time, dtype=np.float64)
    dt = ts[1] - ts[0]
    M = m0[None, :] + v_m[None, :] * ts[:, None]  # (T,3)

    # bomb_ok_all_pts: (N,B,T) 初始 True，对每个点 AND
    bomb_ok = np.ones((N, B, n_time), dtype=bool)
    alive = (ts[None, None, :] >= t_det[:, :, None]) & (
        ts[None, None, :] <= t_det[:, :, None] + T_LIFE
    )

    Cx = P_det[:, :, 0:1]
    Cy = P_det[:, :, 1:2]
    Cz = P_det[:, :, 2:3] - V_SINK * (ts[None, None, :] - t_det[:, :, None])

    # 按目标点分块
    p_chunk = 16
    for p0 in range(0, P, p_chunk):
        tgt = targets[p0 : p0 + p_chunk]  # (p,3)
        # AB: (T,p,3)
        AB = tgt[None, :, :] - M[:, None, :]
        L2 = np.maximum(np.sum(AB * AB, axis=2), 1e-18)  # (T,p)

        for j in range(tgt.shape[0]):
            ABx = AB[:, j, 0]
            ABy = AB[:, j, 1]
            ABz = AB[:, j, 2]
            L2j = L2[:, j]
            ACxj = Cx - M[None, None, :, 0]
            ACyj = Cy - M[None, None, :, 1]
            ACzj = Cz - M[None, None, :, 2]
            s = (ACxj * ABx + ACyj * ABy + ACzj * ABz) / L2j
            sc = np.clip(s, 0.0, 1.0)
            px = M[None, None, :, 0] + sc * ABx
            py = M[None, None, :, 1] + sc * ABy
            pz = M[None, None, :, 2] + sc * ABz
            d2 = (Cx - px) ** 2 + (Cy - py) ** 2 + (Cz - pz) ** 2
            ok_j = (d2 <= R2_SMOKE) & (s >= 0.0) & (s <= 1.0)
            bomb_ok &= ok_j

    bomb_ok &= alive
    any_ok = np.any(bomb_ok, axis=1)
    dur = np.sum(any_ok, axis=1) * dt
    per = np.sum(bomb_ok, axis=2) * dt
    return dur.astype(np.float64), per.astype(np.float64)


def _union_duration_torch_targets(
    P_det: np.ndarray,
    t_det: np.ndarray,
    targets: np.ndarray,
    n_time: int = 6000,
    device: str = None,
    m0=None,
    v_m=None,
    t_hit: Optional[float] = None,
):
    if device is None:
        device = "cuda" if CUDA_OK else "cpu"
    if m0 is None:
        m0 = M0
    if v_m is None:
        v_m = V_M
    if t_hit is None:
        t_hit = T_HIT
    P_det_t = torch.as_tensor(P_det, dtype=torch.float32, device=device)
    t_det_t = torch.as_tensor(t_det, dtype=torch.float32, device=device)
    targets_t = torch.as_tensor(targets, dtype=torch.float32, device=device)
    N, B, _ = P_det_t.shape
    P = targets_t.shape[0]

    t0 = float(torch.clamp(t_det_t.min() - 0.5, min=0.0).item())
    t1 = float(min(float(t_hit), float(t_det_t.max().item()) + T_LIFE))
    if t1 <= t0:
        z = torch.zeros(N, device=device)
        return z.detach().cpu().numpy(), torch.zeros(N, B, device=device).detach().cpu().numpy()

    ts = torch.linspace(t0, t1, n_time, device=device, dtype=torch.float32)
    dt = float(ts[1] - ts[0])
    M0_t = torch.as_tensor(np.asarray(m0, dtype=np.float64), dtype=torch.float32, device=device)
    VM_t = torch.as_tensor(np.asarray(v_m, dtype=np.float64), dtype=torch.float32, device=device)
    M = M0_t[None, :] + VM_t[None, :] * ts[:, None]

    bomb_ok = torch.ones((N, B, n_time), dtype=torch.bool, device=device)
    alive = (ts[None, None, :] >= t_det_t[:, :, None]) & (
        ts[None, None, :] <= t_det_t[:, :, None] + T_LIFE
    )
    Cx = P_det_t[:, :, 0:1]
    Cy = P_det_t[:, :, 1:2]
    Cz = P_det_t[:, :, 2:3] - V_SINK * (ts[None, None, :] - t_det_t[:, :, None])

    # 逐点 AND，避免 (N,B,T,P) 爆显存
    for j in range(P):
        tgt = targets_t[j]
        AB = tgt[None, :] - M  # (T,3)
        L2 = torch.clamp((AB * AB).sum(dim=1), min=1e-18)
        ACx = Cx - M[None, None, :, 0]
        ACy = Cy - M[None, None, :, 1]
        ACz = Cz - M[None, None, :, 2]
        s = (ACx * AB[None, None, :, 0] + ACy * AB[None, None, :, 1] + ACz * AB[None, None, :, 2]) / L2
        sc = torch.clamp(s, 0.0, 1.0)
        px = M[None, None, :, 0] + sc * AB[None, None, :, 0]
        py = M[None, None, :, 1] + sc * AB[None, None, :, 1]
        pz = M[None, None, :, 2] + sc * AB[None, None, :, 2]
        d2 = (Cx - px) ** 2 + (Cy - py) ** 2 + (Cz - pz) ** 2
        ok_j = (d2 <= R2_SMOKE) & (s >= 0.0) & (s <= 1.0)
        bomb_ok &= ok_j

    bomb_ok &= alive
    any_ok = bomb_ok.any(dim=1)
    dur = any_ok.sum(dim=1).to(torch.float64) * dt
    per = bomb_ok.sum(dim=2).to(torch.float64) * dt
    return dur.detach().cpu().numpy(), per.detach().cpu().numpy()


def union_duration_batch(
    P_det,
    t_det,
    mode: str = "cylinder",
    pts: Optional[np.ndarray] = None,
    n_time: int = 6000,
    use_gpu: bool = True,
    missile: Optional[str] = None,
    m0=None,
    v_m=None,
    t_hit: Optional[float] = None,
):
    """
    P_det (N,B,3), t_det (N,B) -> dur(N,), per(N,B)
    mode=point 时 targets 仅中心一点。
    """
    P_det = np.asarray(P_det, dtype=np.float64)
    t_det = np.asarray(t_det, dtype=np.float64)
    if P_det.ndim == 2:
        P_det = P_det[None, ...]
        t_det = t_det[None, ...]
    if mode == "point":
        targets = TGT_CENTER[None, :]
    else:
        targets = PTS_FAST if pts is None else np.asarray(pts, dtype=np.float64)

    m0_r, v_m_r, t_hit_r = _resolve_missile(missile, m0, v_m)
    if t_hit is None:
        t_hit = t_hit_r

    N = P_det.shape[0]
    if use_gpu and TORCH_OK and CUDA_OK:
        # 圆柱点多时减小 batch
        chunk = 256 if targets.shape[0] > 80 else 512
        if n_time >= 10000:
            chunk = max(64, chunk // 2)
        durs, pers = [], []
        for i0 in range(0, N, chunk):
            i1 = min(N, i0 + chunk)
            d, p = _union_duration_torch_targets(
                P_det[i0:i1],
                t_det[i0:i1],
                targets,
                n_time=n_time,
                m0=m0_r,
                v_m=v_m_r,
                t_hit=t_hit,
            )
            durs.append(d)
            pers.append(p)
        return np.concatenate(durs), np.concatenate(pers)

    chunk = 128 if targets.shape[0] > 80 else 256
    durs, pers = [], []
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        d, p = _union_duration_numpy_targets(
            P_det[i0:i1],
            t_det[i0:i1],
            targets,
            n_time=n_time,
            m0=m0_r,
            v_m=v_m_r,
            t_hit=t_hit,
        )
        durs.append(d)
        pers.append(p)
    return np.concatenate(durs), np.concatenate(pers)


# ---------------- 上下圆周最坏点精修（终算可选） ----------------
def max_los_dist_on_circle(M, C, z, n_coarse=128, refine_local=True):
    """
    固定时刻，在 z 高度圆周上求 max d(C, M–P(θ))。
    返回 dmax, theta_star
    """
    M = np.asarray(M, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_coarse, endpoint=False)

    def d_of(th):
        P = np.array(
            [
                CYL_CENTER_XY[0] + R_CYL * np.cos(th),
                CYL_CENTER_XY[1] + R_CYL * np.sin(th),
                z,
            ],
            dtype=np.float64,
        )
        AB = P - M
        AC = C - M
        L2 = float(np.dot(AB, AB))
        if L2 < 1e-18:
            return float(np.linalg.norm(AC)), th
        s = float(np.dot(AC, AB) / L2)
        sc = min(max(s, 0.0), 1.0)
        closest = M + sc * AB
        # 若 s 不在 [0,1]，按主判据视为未遮挡该点：用一个大距离惩罚
        d = float(np.linalg.norm(C - closest))
        if s < 0.0 or s > 1.0:
            d = max(d, R_SMOKE + 1.0 + abs(s - 0.5))
        return d, th

    ds = np.array([d_of(th)[0] for th in thetas])
    i = int(np.argmax(ds))
    th_best = float(thetas[i])
    d_best = float(ds[i])

    if refine_local:
        # 邻域三点二次/网格细化
        span = 2.0 * np.pi / n_coarse
        for _ in range(3):
            grid = np.linspace(th_best - span, th_best + span, 21)
            vals = [d_of(th)[0] for th in grid]
            j = int(np.argmax(vals))
            th_best = float(grid[j] % (2.0 * np.pi))
            d_best = float(vals[j])
            span *= 0.35
    return d_best, th_best


def cylinder_ok_circle_max(t, p_det, t_det, n_coarse=128, n_z_side=5):
    """
    更连续的圆柱严格判据近似：
    - 上下圆周 max d
    - 侧面若干高度圆周 max d
    全部 <= R 且云存活。
    """
    t = float(t)
    if t < t_det or t > t_det + T_LIFE:
        return False
    M = M0 + V_M * t
    C = np.array([p_det[0], p_det[1], p_det[2] - V_SINK * (t - t_det)], dtype=np.float64)
    zs = np.linspace(Z_BOT, Z_TOP, n_z_side)
    for z in zs:
        dmax, _ = max_los_dist_on_circle(M, C, z, n_coarse=n_coarse)
        if dmax > R_SMOKE + 1e-12:
            return False
    return True


@dataclass
class Strategy1:
    theta: float
    v: float
    t_drop: float
    tau: float

    def pack(self):
        v_fy, p_drop, t_det, p_det = kinematics(self.theta, self.v, self.t_drop, self.tau)
        return {
            "theta": self.theta,
            "heading_deg": float(np.degrees(self.theta) % 360.0),
            "v": self.v,
            "t_drop": self.t_drop,
            "tau": self.tau,
            "t_det": t_det,
            "v_fy": v_fy,
            "P_drop": p_drop,
            "P_det": p_det,
        }


if __name__ == "__main__":
    print("device:", device_info())
    print("PTS_FAST", PTS_FAST.shape[0], "PTS_OPT", PTS_OPT.shape[0], "PTS_FINE", PTS_FINE.shape[0])
    # Q1 固定策略自检
    _, _, t_det, p_det = kinematics(np.pi, 120.0, 1.5, 3.6)
    for mode, pts, name in [
        ("point", None, "point"),
        ("cylinder", PTS_FAST, "cyl_fast"),
        ("cylinder", PTS_OPT, "cyl_opt"),
        ("cylinder", PTS_FINE, "cyl_fine"),
    ]:
        dur, segs = shield_duration(p_det, t_det, mode=mode, pts=pts, n=80000, refine=True)
        print(f"Q1 {name}: {dur:.6f} segs={segs[:3]}")

# -*- coding: utf-8 -*-
"""驱动 MATLAB 分块串行跑问题2（不开并行池）。

R2026a 在 -batch 退出时常 Access Violation，但计算往往已完成。
因此以 stdout/日志中的 TOKEN 与状态文件为准，不依赖 exit code。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

Q2 = Path(__file__).resolve().parent
OUT = Q2 / "结果"


def find_matlab() -> str:
    configured = os.environ.get("MATLAB_EXE") or os.environ.get("MATLAB_EXECUTABLE")
    if configured:
        executable = Path(configured).expanduser()
        if not executable.is_file():
            raise FileNotFoundError(f"MATLAB_EXE is not a file: {executable}")
        return str(executable)
    found = shutil.which("matlab")
    if found:
        return found
    raise FileNotFoundError(
        "MATLAB executable not found. Set MATLAB_EXE to the full MATLAB "
        "executable path, or add matlab to PATH."
    )


MATLAB = find_matlab()
CHUNK = 200  # 每段粗搜点数
REFINE_CHUNK = 2
TOKEN_FILE = OUT / "q2_last_token.txt"


def run_matlab(expr: str, timeout: int = 300) -> str:
    # 不用 -nojvm：本机 R2026a 在 nojvm 下更容易中途崩
    cmd = [
        MATLAB,
        "-nodesktop",
        "-nosplash",
        "-batch",
        expr,
    ]
    print(f"\n>>> {expr}", flush=True)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            cmd,
            cwd=str(Q2),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        out = (r.stdout or "") + (r.stderr or "")
        rc = r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        rc = -1
        print(out)
        print(f"[TIMEOUT after {timeout}s]", flush=True)
        return out or ""

    print(out, end="" if out.endswith("\n") else "\n", flush=True)
    # 0xc0000005 = 3221225477：退出时 AV，计算可能已成功
    note = ""
    if rc == 3221225477:
        note = " (exit AV, check token)"
    print(f"[elapsed {time.perf_counter()-t0:.1f}s exit={rc}{note}]", flush=True)
    return out


def has_token(out: str, token: str) -> bool:
    if token in out:
        return True
    if TOKEN_FILE.exists():
        try:
            return token in TOKEN_FILE.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    log = OUT / "q2_matlab_serial.log"
    if log.exists():
        try:
            return token in log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return False


def main() -> int:
    OUT.mkdir(exist_ok=True)
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()

    # 1) init
    out = run_matlab("q2_matlab_piece(0,0,'init');", timeout=180)
    if not has_token(out, "INIT_OK"):
        print("INIT failed", file=sys.stderr)
        return 1

    m = re.search(r"init N=(\d+)", out)
    if not m:
        log = OUT / "q2_matlab_serial.log"
        if log.exists():
            m = re.search(r"init N=(\d+)", log.read_text(encoding="utf-8", errors="replace"))
    n_all = int(m.group(1)) if m else 7537
    print(f"n_all={n_all}, chunk={CHUNK}", flush=True)

    # 2) coarse pieces
    for i0 in range(1, n_all + 1, CHUNK):
        i1 = min(n_all, i0 + CHUNK - 1)
        out = run_matlab(f"q2_matlab_piece({i0},{i1},'coarse');", timeout=300)
        if not has_token(out, "PIECE_OK"):
            print("retry coarse piece...", flush=True)
            time.sleep(1)
            out = run_matlab(f"q2_matlab_piece({i0},{i1},'coarse');", timeout=300)
            if not has_token(out, "PIECE_OK"):
                print(f"COARSE failed at [{i0},{i1}]", file=sys.stderr)
                return 2

    # 3) refine pieces（前 12 个候选）
    for i0 in range(1, 13, REFINE_CHUNK):
        i1 = min(12, i0 + REFINE_CHUNK - 1)
        out = run_matlab(f"q2_matlab_piece({i0},{i1},'refine');", timeout=420)
        if not has_token(out, "PIECE_OK"):
            print("retry refine...", flush=True)
            time.sleep(1)
            out = run_matlab(f"q2_matlab_piece({i0},{i1},'refine');", timeout=420)
            if not has_token(out, "PIECE_OK"):
                print(f"REFINE failed at [{i0},{i1}]", file=sys.stderr)
                return 3

    # 4) finalize
    out = run_matlab("q2_matlab_piece(0,0,'finalize');", timeout=240)
    if not has_token(out, "FINAL_OK"):
        print("FINALIZE failed", file=sys.stderr)
        return 4

    res = OUT / "q2_matlab_serial_result.txt"
    if res.exists():
        print("\n==== RESULT FILE ====")
        print(res.read_text(encoding="utf-8", errors="replace"))
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Helpers para executar subprocessos Python no mesmo ambiente do pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def run_python(args: Sequence[str], *, cwd: Path | None = None) -> None:
    cmd = [sys.executable, *args]
    workdir = str(cwd or ROOT)
    print(f"[INFO] Executando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=workdir)
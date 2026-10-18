"""Shared result type and helpers for the ILP / CP baselines.

Every baseline exposes `solve(givens, box_size, time_limit_s) -> BaselineResult`.
`t_build` is model-construction time; `t_solve` is elapsed time inside the
solver call. The requested solver limit applies to `t_solve`, not `t_build`.
The benchmark separately measures end-to-end worker wall time.

All baselines are pinned to one thread so the comparison with the (single
threaded) SAT solvers is like for like.
"""
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Tuple

Givens = List[Tuple[int, int, int]]  # (row, col, value) all 0-indexed

SOLVED = "solved"
INFEASIBLE = "infeasible"
TIMEOUT = "timeout"
LICENSE_LIMIT = "license_limit"  # free-edition size cap hit, not a solver failure
ERROR = "error"


@dataclass
class BaselineResult:
    status: str
    grid: Optional[List[List[int]]] = None   # n x n, values 1..n, only when solved
    t_build: float = 0.0  # exact observed model-build time; outside solver time limit
    t_solve: float = 0.0  # elapsed solver-call time; right-censored on TIMEOUT
    message: str = ""


def package_version(distribution: str) -> str:
    """Return installed package version, or an explicit unavailable marker."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def diagnostic(solver: str, version_text: str, api_status=None, original: str = "") -> str:
    """Keep solver identity, structured status and original diagnostic together."""
    pieces = [f"solver={solver}", f"version={version_text}"]
    if api_status is not None:
        pieces.append(f"api_status={api_status}")
    if original:
        pieces.append(f"original={original}")
    return "; ".join(pieces)

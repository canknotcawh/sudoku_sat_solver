"""Shared result type and helpers for the ILP / CP baselines.

Every baseline exposes `solve(givens, box_size, time_limit_s) -> BaselineResult`
and measures model-building time separately from solving time, matching how
the SAT pipeline reports t_encode and t_solve.

All baselines are pinned to one thread so the comparison with the (single
threaded) SAT solvers is like for like.
"""
from dataclasses import dataclass
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
    t_build: float = 0.0
    t_solve: float = 0.0
    message: str = ""


def looks_like_license_limit(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in (
        "size limit", "size-limited", "too large for size", "problem size", "1016", "community edition",
    ))

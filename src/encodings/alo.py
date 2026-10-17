"""At Least One: each square must has at least one digit"""
from typing import List

Clause = List[int]
Lits = List[int]


def alo(lits: Lits) -> List[Clause]:
    return [list(lits)]

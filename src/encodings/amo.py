"""At Most One: each row, column and region contains only one instance of each digit"""

import math
from typing import Callable, List

Clause = List[int]
Literals = List[int]
Lits = Literals

def amo_pairwise(lits: Lits, new_var: Callable[[], int] = None) -> List[Clause]:
    """O(n^2) clauses, no auxiliary variables. Simple and fine for small groups (n=9)."""
    clauses = []
    n = len(lits)
    for i in range(n):
        for j in range(i + 1, n):
            clauses.append([-lits[i], -lits[j]])
    return clauses

def amo_sequential(lits: Lits, new_var: Callable[[], int]) -> List[Clause]:
    """Sinz (2005) sequential counter encoding. O(n) clauses, O(n) aux vars."""
    n = len(lits)
    if n <= 1:
        return []
    clauses: List[Clause] = []
    s = [new_var() for _ in range(n - 1)]  # s[i] <=> at least one of lits[0..i] is true
    clauses.append([-lits[0], s[0]])
    for i in range(1, n - 1):
        clauses.append([-lits[i], s[i]])
        clauses.append([-s[i - 1], s[i]])
        clauses.append([-lits[i], -s[i - 1]])
    clauses.append([-lits[n - 1], -s[n - 2]])
    return clauses


def amo_commander(lits: Lits, new_var: Callable[[], int], group_size: int = 3) -> List[Clause]:
    """Commander encoding (Klieber & Kwon, 2007): recursive grouping, one commander
    literal per group, AMO enforced within each group and recursively across
    commanders. group_size is a tunable trade-off between clause count and depth."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer greater than or equal to 2")
    if len(lits) <= 1:
        return []
    if len(lits) <= group_size:
        return amo_pairwise(lits)

    clauses: List[Clause] = []
    commanders: Lits = []
    for i in range(0, len(lits), group_size):
        group = lits[i:i + group_size]
        if len(group) == 1:
            commanders.append(group[0])
            continue
        c = new_var()
        clauses.extend(amo_pairwise(group))     # AMO within the group
        clauses.append([-c] + list(group))       # commander true -> someone in group true
        for lit in group:
            clauses.append([-lit, c])            # group member true -> commander true
        commanders.append(c)
    clauses.extend(amo_commander(commanders, new_var, group_size))
    return clauses


def amo_binary(lits: Lits, new_var: Callable[[], int]) -> List[Clause]:
    """Binary/log encoding (Frisch & Peugniez, 2001). O(n log n) clauses,
    O(log n) aux bits — most compact for large groups, but harder to debug."""
    n = len(lits)
    if n <= 1:
        return []
    k = max(1, math.ceil(math.log2(n)))
    bits = [new_var() for _ in range(k)]
    clauses: List[Clause] = []
    for idx, lit in enumerate(lits):
        for b in range(k):
            bit_lit = bits[b] if (idx >> b) & 1 else -bits[b]
            clauses.append([-lit, bit_lit])
    return clauses


def amo_product(lits: Lits, new_var: Callable[[], int], threshold: int = 6) -> List[Clause]:
    """Product encoding (Chen, 2010).

    Lay the n literals out on a p x q grid, p = ceil(sqrt(n)), q = ceil(n / p),
    literal i sitting at (row i // q, col i % q). Add one auxiliary variable per
    row and per column, and tie each literal to both of its coordinates:

        x_i -> row_u      x_i -> col_v

    Two different literals occupy different cells, hence differ in their row or
    their column. If both were true, two rows (or two columns) would be true, so
    it is enough to forbid that with an AMO over the rows and an AMO over the
    columns. Those two AMOs are small (about sqrt(n) literals) and are encoded
    recursively; at or below `threshold` literals plain pairwise is cheaper
    (break-even against pairwise is at n = 7, and the extra variables make it
    not worth it before then).

    Cost: about 2n + 4*sqrt(n) clauses and 2*sqrt(n) auxiliary variables.
    """
    threshold = max(threshold, 2)  # below 2 the recursion would never shrink
    n = len(lits)
    if n <= threshold:
        return amo_pairwise(lits)

    p = math.ceil(math.sqrt(n))
    q = math.ceil(n / p)
    num_rows = math.ceil(n / q)  # rows actually used (never more than p)
    rows = [new_var() for _ in range(num_rows)]
    cols = [new_var() for _ in range(q)]

    clauses: List[Clause] = []
    for i, lit in enumerate(lits):
        u, v = divmod(i, q)
        clauses.append([-lit, rows[u]])
        clauses.append([-lit, cols[v]])

    clauses.extend(amo_product(rows, new_var, threshold))
    clauses.extend(amo_product(cols, new_var, threshold))
    return clauses


AMO_ENCODERS = {
    "pairwise": amo_pairwise,
    "sequential": amo_sequential,
    "commander": amo_commander,
    "binary": amo_binary,
    "product": amo_product,
}
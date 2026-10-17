"""Correctness tests for every AMO encoding.

For each encoding and group size n we check, with a SAT solver:
  * every assignment of the n literals with at most one true is satisfiable
  * every pair of literals true together is unsatisfiable
Together these say the CNF accepts exactly the "at most one" assignments,
because any assignment with two or more true literals contains a violated pair.

Run:  python -m tests.test_amo      (also collected by pytest)
"""
from functools import partial
from itertools import combinations

from pysat.solvers import Solver

from src.encodings.amo import AMO_ENCODERS, amo_product

SIZES = list(range(2, 31)) + [36, 50]


def build(encoder, n):
    lits = list(range(1, n + 1))
    counter = [n]

    def new_var():
        counter[0] += 1
        return counter[0]

    return lits, encoder(lits, new_var)


def check_encoder(name, encoder, sizes=SIZES):
    for n in sizes:
        lits, clauses = build(encoder, n)
        with Solver(name="cadical195", bootstrap_with=clauses) as s:
            assert s.solve(assumptions=[-l for l in lits]), (name, n, "none true")
            for i in lits:
                only_i = [i] + [-l for l in lits if l != i]
                assert s.solve(assumptions=only_i), (name, n, "only", i)
            for i, j in combinations(lits, 2):
                assert not s.solve(assumptions=[i, j]), (name, n, "pair", i, j)


def test_all_encoders():
    for name, encoder in AMO_ENCODERS.items():
        check_encoder(name, encoder)


def test_product_deep_recursion():
    # A low threshold forces several levels of recursion even for small n.
    for threshold in (2, 3):
        check_encoder(f"product(threshold={threshold})", partial(amo_product, threshold=threshold))


if __name__ == "__main__":
    for name, encoder in AMO_ENCODERS.items():
        check_encoder(name, encoder)
        print(f"PASS {name}")
    for threshold in (2, 3):
        check_encoder(f"product(threshold={threshold})", partial(amo_product, threshold=threshold))
        print(f"PASS product(threshold={threshold})")

from itertools import product

import pytest

from src.encodings.alo import alo
from src.encodings.amo import (
    AMO_ENCODERS,
    amo_commander,
    amo_product,
)


def _build(encoder, n, **kwargs):
    lits = list(range(1, n + 1))
    allocated = []

    def new_var():
        variable = n + 1 + len(allocated)
        allocated.append(variable)
        return variable

    if encoder is AMO_ENCODERS["pairwise"]:
        clauses = encoder(lits)
    else:
        clauses = encoder(lits, new_var, **kwargs)
    assert len(allocated) == len(set(allocated))
    assert all(variable > n for variable in allocated)
    return lits, clauses, allocated


def _satisfiable_with_inputs(clauses, input_lits, auxiliary_vars, input_values):
    base = dict(zip(input_lits, input_values))
    for auxiliary_values in product((False, True), repeat=len(auxiliary_vars)):
        assignment = {**base, **dict(zip(auxiliary_vars, auxiliary_values))}
        if all(
            any(assignment[abs(lit)] == (lit > 0) for lit in clause)
            for clause in clauses
        ):
            return True
    return False


def _assert_amo_truth_table(encoder, n, **kwargs):
    lits, clauses, auxiliary_vars = _build(encoder, n, **kwargs)
    for values in product((False, True), repeat=n):
        expected = sum(values) <= 1
        actual = _satisfiable_with_inputs(clauses, lits, auxiliary_vars, values)
        assert actual is expected, (encoder.__name__, n, values)


@pytest.mark.parametrize("name", list(AMO_ENCODERS))
def test_each_amo_matches_exhaustive_truth_table(name):
    encoder = AMO_ENCODERS[name]
    for n in range(8):
        _assert_amo_truth_table(encoder, n)


@pytest.mark.parametrize("group_size", [2, 4])
def test_commander_exhaustive_with_even_and_odd_group_remainders(group_size):
    for n in range(8):
        _assert_amo_truth_table(amo_commander, n, group_size=group_size)


def test_commander_rejects_group_size_that_cannot_shrink_recursion():
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        amo_commander([1, 2], lambda: 3, group_size=1)


@pytest.mark.parametrize("threshold,n", [(2, 5), (3, 7)])
def test_product_exhaustive_with_recursive_and_partial_rows(threshold, n):
    _assert_amo_truth_table(amo_product, n, threshold=threshold)


def test_alo_matches_exhaustive_truth_table_including_empty_group():
    for n in range(7):
        lits = list(range(1, n + 1))
        clauses = alo(lits)
        for values in product((False, True), repeat=n):
            assignment = dict(zip(lits, values))
            actual = all(
                any(assignment[abs(lit)] == (lit > 0) for lit in clause)
                for clause in clauses
            )
            assert actual is any(values), (n, values)

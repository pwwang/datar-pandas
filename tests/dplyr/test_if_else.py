# tests grabbed from:
# https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-if-else.R
# and
# https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-case-when.R
import pytest

import pandas
import numpy as np
from datar import f
from datar.base import NA, c
from datar.dplyr import (
    group_by,
    if_else,
    case_when,
    mutate,
    pull,
    case_match,
    group_vars,
)
from datar.data import mtcars
from datar.tibble import tibble
from datar_pandas.pandas import Series
from ..conftest import assert_iterable_equal


def test_scalar_true_false_are_vectorized():
    x = c(True, True, False, False)
    out = if_else(x, 1, 2)
    assert list(out) == [1, 1, 2, 2]

    # Series
    x = Series(c(True, True, False, False))
    out = if_else(x, 1, 2)
    assert isinstance(out, Series)
    assert list(out) == [1, 1, 2, 2]


def test_vector_true_false_ok():
    x = np.array([-1, 0, 1])
    out = if_else(x < 0, x, 0)
    assert list(out) == [-1, 0, 0]

    out = if_else(x > 0, x, 0)
    assert list(out) == [0, 0, 1]


def test_missing_values_are_missing():
    out = if_else(c(True, NA, False), -1, 1)
    # assert_iterable_equal(out, [-1, NA, 1])
    # NA as false
    assert_iterable_equal(out, [-1, 1, 1])

    out = if_else(c(True, NA, False), -1, 1, 0)
    assert_iterable_equal(out, [-1, 0, 1])


def test_if_else_errors():
    # ok, numbers are able to be converted to booleans
    out = if_else(range(1, 11), 1, 2)
    assert list(out) == [1.0] * 10

    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="size"):
        if_else(data < 2, [1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="size"):
        if_else(data < 2, [1, 2, 3], [1, 2])


# case_when ------------------
def test_matches_values_in_order():
    x = np.array([1, 2, 3])
    out = case_when(x <= 1, 1, x <= 2, 2, x <= 3, 3)
    assert list(out) == [1, 2, 3]


def test_unmatched_gets_missing_value():
    x = np.array([1, 2, 3])
    out = case_when(x <= 1, 1, x <= 2, 2)
    assert_iterable_equal(out, [1, 2, NA])


def test_missing_values_can_be_replaced():
    x = np.array([1, 2, 3, NA])
    out = case_when(x <= 1, 1, x <= 2, 2, pandas.isna(x), 0)
    assert_iterable_equal(out, [1, 2, NA, 0])


def test_na_conditions():
    out = case_when([True, False, NA], [1, 2, 3], True, 4)
    assert list(out) == [1, 4, 4]


def test_atomic_conditions():
    import warnings

    warnings.filterwarnings("error")
    out = case_when(True, [1, 2, 3], False, [4, 5, 6])
    assert list(out) == [1, 2, 3]

    out = case_when(NA, [1, 2, 3], True, [4, 5, 6])
    assert list(out) == [4, 5, 6]


def test_0len_conditions_and_values():
    out = case_when(True, [], False, [])
    assert list(out) == []

    out = case_when([], 1, [], 2)
    assert list(out) == []


def test_inside_mutate():
    out = (
        mtcars.iloc[:4,]
        >> mutate(out=case_when(f.cyl == 4, 1, f["am"] == 1, 2, True, 0))
        >> pull(to="list")
    )
    assert out == [2, 2, 1, 0]


def test_errors():
    x = np.array([NA] * 10)
    with pytest.raises(ValueError):
        # condition has to be the same length as data
        case_when(x, [True, False], [1, 2, 3], [False, True], [1, 2])
    with pytest.raises(TypeError):
        case_when()
    with pytest.raises(TypeError):
        case_when("a")
    # ok
    case_when([], 1)


# case_match ------------------
def test_case_match_lhs_can_match_multiple_values():
    assert_iterable_equal(case_match(1, [1, 2], "x"), ["x"])


def test_case_match_lhs_can_match_na():
    assert_iterable_equal(case_match(NA, NA, "x"), ["x"])


def test_case_match_rhs_recycling():
    x = np.array([1, 2, 3])
    assert_iterable_equal(case_match(x, [1, 3], x * 2), [2, NA, 6])


def test_case_match_requires_at_least_one_condition():
    with pytest.raises(ValueError):
        case_match(1)


def test_case_match_default_works():
    assert_iterable_equal(case_match(1, 3, 1, _default=2), [2])
    assert_iterable_equal(case_match([1, 2, 3, 4, 5], 6, 1, _default=2), [2] * 5)
    assert_iterable_equal(
        case_match([1, 2, 3, 4, 5], 6, [1, 2, 3, 4, 5], _default=[2, 3, 4, 5, 6]),
        [2, 3, 4, 5, 6],
    )


def test_case_match_dtypes():
    assert_iterable_equal(case_match(1, 1, 1.1, _dtypes=int), [1])


def test_case_match_on_grouped():
    gdf = tibble(g=[1, 2], x=[1, 2]) >> group_by(f.g)
    out = gdf >> mutate(y=case_match(f.x, 1, 2))
    assert_iterable_equal(group_vars(out), ["g"])
    assert_iterable_equal(out["y"].obj, [2, NA])

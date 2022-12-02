import pytest  # noqa

import numpy as np
from datar.base import (
    droplevels,
    factor,
    levels,
    as_factor,
    is_factor,
    nlevels,
    is_ordered,
    ordered,
    # is_categorical,
)
from datar_pandas.pandas import (
    Series,
    SeriesGroupBy,
    is_categorical_dtype,
    get_obj,
)
from ..conftest import (
    assert_,
    assert_equal,
    assert_not,
    assert_factor_equal,
    assert_iterable_equal,
)


def test_droplevels():
    fct = factor([1, 2, 3], levels=[1, 2, 3, 4])
    out = droplevels(fct)
    assert_iterable_equal(levels(out), [1, 2, 3])

    fct = Series(fct)
    out = droplevels(fct)
    assert_iterable_equal(fct, out)
    assert_iterable_equal(levels(out), [1, 2, 3])


def test_levels():
    assert_(levels(1) is None)
    assert_(levels(Series([1])) is None)


def test_factor():
    out = factor()
    assert_equal(len(out), 0)
    assert_equal(len(levels(out)), 0)
    assert_equal(len(factor(2)), 1)

    out = factor([1, 2, 3], exclude=None)
    assert_equal(len(out), 3)

    out = factor([1, 2, 3], exclude=1)
    assert_iterable_equal(out, [np.nan, 2, 3])
    assert_iterable_equal(levels(out), [2, 3])

    out = factor(out)
    assert_iterable_equal(out, [np.nan, 2, 3])
    assert_iterable_equal(levels(out), [2, 3])


def test_factor_sgb():
    x = Series([1, 2, 3]).groupby([1, 1, 3])
    out = factor(x)
    assert_(isinstance(out, SeriesGroupBy))
    assert_factor_equal(get_obj(out).values, factor([1, 2, 3]))


def test_as_factor():
    out = as_factor([1, 2, 3])
    assert_iterable_equal(out, [1, 2, 3])
    assert_iterable_equal(levels(out), [1, 2, 3])

    out2 = as_factor(out)
    assert_(out is out2)

    s = Series([1, 2, 3])
    out = as_factor(s)
    assert_(is_categorical_dtype(out))


def test_is_factor():
    out = as_factor([])
    assert_(is_factor(out))
    assert_not(is_factor([]))
    assert_(is_factor(Series(out)))


def test_nlevels():
    assert_equal(nlevels(1), 0)
    assert_equal(nlevels(factor([1, 2, 3])), 3)


def test_is_ordered():
    assert_not(is_ordered(1))
    assert_not(is_ordered(factor()))
    assert_(is_ordered(factor(ordered=True)))
    assert_(is_ordered(Series(factor(ordered=True))))
    assert_not(is_ordered(Series([1])))


def test_ordered():
    o = ordered([3, 1, 2])
    assert_iterable_equal(o, [3, 1, 2])

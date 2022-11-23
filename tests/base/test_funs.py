import pytest  # noqa

from datar.base import (
    cut,
    outer,
    diff,
    rank,
)
from datar.base import table, pi, paste0, rnorm, cumsum, seq
from datar_pandas.pandas import Interval, DataFrame, Series, assert_frame_equal
from ..conftest import assert_iterable_equal


def test_cut():
    z = rnorm(10000)
    tab = table(cut(z, breaks=range(-6, 7)))
    assert tab.shape == (1, 12)
    assert tab.columns.tolist() == [
        Interval(-6, -5, closed="right"),
        Interval(-5, -4, closed="right"),
        Interval(-4, -3, closed="right"),
        Interval(-3, -2, closed="right"),
        Interval(-2, -1, closed="right"),
        Interval(-1, 0, closed="right"),
        Interval(0, 1, closed="right"),
        Interval(1, 2, closed="right"),
        Interval(2, 3, closed="right"),
        Interval(3, 4, closed="right"),
        Interval(4, 5, closed="right"),
        Interval(5, 6, closed="right"),
    ]
    assert sum(tab.values.flatten()) == 10000

    z = cut([1] * 5, 4)
    assert_iterable_equal(
        z.to_numpy(), [Interval(0.9995, 1.0, closed="right")] * 5
    )
    assert_iterable_equal(
        z.categories.to_list(),
        [
            Interval(0.999, 0.9995, closed="right"),
            Interval(0.9995, 1.0, closed="right"),
            Interval(1.0, 1.0005, closed="right"),
            Interval(1.0005, 1.001, closed="right"),
        ],
    )

    z = rnorm(100)
    tab = table(cut(z, breaks=[pi / 3.0 * i for i in range(0, 4)]))
    assert str(tab.columns.tolist()[0]) == "(0.0, 1.05]"

    tab = table(
        cut(z, breaks=[pi / 3.0 * i for i in range(0, 4)], precision=3)
    )
    assert str(tab.columns.tolist()[0]) == "(0.0, 1.047]"

    aaa = [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]
    ct = cut(aaa, 3, precision=3, ordered_result=True)
    assert str(ct[0]) == "(0.994, 3.0]"


def test_diff():
    x = cumsum(cumsum(seq(1, 10)))
    # assert_iterable_equal(diff(x, lag=2), x[2:] - x[:-2])
    # assert_iterable_equal(diff(x, lag=2), seq(3, 10)**2)

    assert_iterable_equal(diff(diff(x)), diff(x, differences=2))

    assert_iterable_equal(diff(x, differences=40), [])

    x = Series([1, 2, 3, 4, 5]).groupby([1, 2, 2, 3, 3])
    out = diff(x)
    assert_iterable_equal(out.obj, [1, 1])
    assert out.grouper.ngroups == 3


def test_outer():
    out = outer([1, 2], [1, 2, 3])
    assert_frame_equal(out, DataFrame([[1, 2, 3], [2, 4, 6]]))

    out = outer(["1", "2"], ["1", "2", "3"], fun=paste0)
    assert_frame_equal(
        out, DataFrame([["11", "12", "13"], ["21", "22", "23"]])
    )


def test_rank():
    r = rank(Series([3, 1, 4, 15, 92]))
    assert_iterable_equal(r, [2, 1, 3, 4, 5])

    r = rank(Series([3, 1, 4, 15, 92]).groupby([1, 1, 2, 2, 3])).obj
    assert_iterable_equal(r, [2, 1, 1, 2, 1])

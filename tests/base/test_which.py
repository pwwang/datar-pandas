import pytest  # noqa

from datar.base import which, which_min, which_max
from datar_pandas.pandas import Series
from ..conftest import assert_iterable_equal, assert_equal


def test_which():
    assert_iterable_equal(which(Series([True, False, True])), [0, 2])
    out = which(Series([True, False, True]).groupby([1, 1, 2]))
    assert len(out) == 2
    assert_iterable_equal(out[1], [0])
    assert_iterable_equal(out[2], [0])


def test_which_min():
    assert_equal(which_min(Series([2, 1, 3])), 1)
    assert_iterable_equal(
        which_min(Series([2, 1, 3]).groupby([1, 1, 2])),
        [1, 0],
    )


def test_which_max():
    assert_equal(which_max(Series([2, 1, 3])), 2)
    assert_iterable_equal(
        which_max(Series([2, 1, 3]).groupby([1, 1, 2])),
        [0, 0],
    )

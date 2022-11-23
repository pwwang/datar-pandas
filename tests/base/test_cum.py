import pytest  # noqa

from datar.base import (
    cummax,
    cummin,
    cumprod,
    cumsum,
)
from ..conftest import (
    assert_iterable_equal,
    pd_data,
)

pd_data = pd_data()


def test_cumsum_pandas():
    assert_iterable_equal(cumsum(pd_data.series), [1, 3, 5, 8])
    assert_iterable_equal(cumsum(pd_data.sgb).obj, [1, 2, 4, 3])


def test_cumprod_pandas():
    assert_iterable_equal(cumprod(pd_data.series), [1, 2, 4, 12])
    assert_iterable_equal(cumprod(pd_data.sgb).obj, [1, 2, 4, 3])


def test_cummin_pandas():
    assert_iterable_equal(cummin(pd_data.series), [1, 1, 1, 1])
    assert_iterable_equal(cummin(pd_data.sgb).obj, [1, 2, 2, 3])


def test_cummax_pandas():
    assert_iterable_equal(cummax(pd_data.series), [1, 2, 2, 3])
    assert_iterable_equal(cummax(pd_data.sgb).obj, [1, 2, 2, 3])

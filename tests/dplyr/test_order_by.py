import pytest  # noqa

import numpy as np
from datar import f
from datar.base import cumsum, seq, c
from datar.dplyr import order_by, with_order, mutate
from datar.tibble import tibble

from ..conftest import assert_iterable_equal


def test_order_by():
    df = tibble(x=c[1:6])
    out = df >> mutate(y=order_by(c[5:], cumsum(f.x)))
    assert_iterable_equal(out.y, [15, 14, 12, 9, 5])

    with pytest.raises(ValueError):
        order_by(seq(5, 1), cumsum(seq(1, 5)))


def test_with_order():
    x = [1, 2, 3, 4, 5]

    out = with_order(seq(5, 1), cumsum, x)
    assert_iterable_equal(out, [15, 14, 12, 9, 5])

    out = with_order(seq(5, 1), list, x)
    assert_iterable_equal(out, [1, 2, 3, 4, 5])

    out = with_order(seq(5, 1), np.array, x)
    assert_iterable_equal(out, [1, 2, 3, 4, 5])

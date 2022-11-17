import pytest  # noqa

import numpy as np
from datar.base import (
    as_double,
    as_integer,
    as_numeric,
    is_double,
    is_integer,
)
from datar.base import factor
from datar.tibble import tibble
from datar_pandas.pandas import Series
from ..conftest import assert_, assert_iterable_equal


def test_as_double():
    assert_(is_double(as_double(Series([1]))))
    assert_(is_double(as_double(Series([1, 2]))))

    x = tibble(a=[1, 2, 3]).rowwise()
    out = as_double(x.a)
    assert_(is_double(out.obj))
    assert_(out.is_rowwise)


def test_as_integer():
    assert_(is_integer(as_integer(Series([1]))))
    fct = factor(list("abc"))
    assert_iterable_equal(as_integer(fct), [0, 1, 2])

    x = tibble(a=factor(["a", "b"])).group_by("a")
    out = as_integer(x.a)
    assert_iterable_equal(out.obj, [0, 1])


def test_as_numeric():
    assert_iterable_equal(as_numeric(Series(["1"])), [1])
    # assert_iterable_equal(as_numeric(Series(["1.1"]), _keep_na=False), [1.1])
    # assert_iterable_equal(
    #     as_numeric(Series(["1", np.nan]), _keep_na=True), [1, np.nan]
    # )
    assert_iterable_equal(as_numeric(Series(["1", "2"])), [1, 2])
    assert_iterable_equal(
        as_numeric(Series(["1", "2"]).groupby([1, 2])).obj,
        [1, 2],
    )

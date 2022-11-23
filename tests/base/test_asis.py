import pytest  # noqa

import numpy as np
from datar.base import (
    as_double,
    as_integer,
    as_numeric,
    as_character,
    as_complex,
    as_date,
    as_logical,
    as_null,
    as_factor,
    as_ordered,
    is_atomic,
    is_double,
    is_integer,
    is_character,
    is_complex,
    is_element,
    is_finite,
    is_infinite,
    is_false,
    is_true,
    is_numeric,
    is_null,
    is_na,
    is_ordered,
    as_pd_date,
    all_,
    any_,
    any_na,
    levels,
)
from datar.base import factor
from datar.tibble import tibble
from datar_pandas import pandas as pd
from datar_pandas.pandas import Series, Categorical
from ..conftest import assert_, assert_equal, assert_not, assert_iterable_equal


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
    out = is_integer(x.a)
    assert_iterable_equal(out, [False, False])

    x = tibble(a=factor(["a", "b"])).rowwise()
    out = as_integer(x.a)
    assert_iterable_equal(out.obj, [0, 1])
    assert out.is_rowwise


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


def test_as_character():
    ch = as_character(Series([1]))
    assert isinstance(ch, Series)
    assert_iterable_equal(ch, ["1"])

    # assert_iterable_equal(as_character(Series([1, np.nan])), ["1.0", np.nan])

    sgb = Series([1, 2, 3]).groupby([1, 1, 2])
    assert_iterable_equal(as_character(sgb).obj, ["1", "2", "3"])


def test_as_complex():
    ch = as_complex(Series([1]))
    assert isinstance(ch, Series)
    assert_iterable_equal(ch, [1 + 0j])

    sgb = Series([1, 2, 3]).groupby([1, 1, 2])
    assert_iterable_equal(as_complex(sgb).obj, [1 + 0j, 2 + 0j, 3 + 0j])


def test_as_logical():
    ch = as_logical(Series([1]))
    assert isinstance(ch, Series)
    assert ch.dtype.name == "bool"
    assert_iterable_equal(ch, [True])

    sgb = Series([2, 0]).groupby([1, 2])
    assert_iterable_equal(as_logical(sgb).obj, [True, False])


def test_as_null():
    ch = as_null(Series([1]))
    assert ch is None

    sgb = Series([1, 2, 3]).groupby([1, 1, 2])
    ch = as_null(sgb)
    assert ch is None


def test_as_date():
    x = pd.Series([1, 2, 3])
    assert_iterable_equal(
        as_date(x, origin="1970-01-01"),
        [
            pd.Timestamp("1970-01-02"),
            pd.Timestamp("1970-01-03"),
            pd.Timestamp("1970-01-04"),
        ],
    )

    x = pd.Series([1, 2, 3], dtype="datetime64[ns]")
    assert_iterable_equal(
        as_date(x),
        [
            pd.Timestamp("1970-01-01"),
            pd.Timestamp("1970-01-02"),
            pd.Timestamp("1970-01-03"),
        ],
    )

    x = pd.Series([1, 2, 3], dtype="datetime64[ns]").groupby([1, 1, 2])
    assert_iterable_equal(
        as_date(x).obj,
        [
            pd.Timestamp("1970-01-01"),
            pd.Timestamp("1970-01-02"),
            pd.Timestamp("1970-01-03"),
        ],
    )


def test_as_factor():
    out = as_factor([1, 2, 3])
    assert_iterable_equal(out, [1, 2, 3])
    assert_iterable_equal(levels(out), [1, 2, 3])

    out = as_factor(Series([1, 2, 3]))
    assert_iterable_equal(out, [1, 2, 3])
    assert_iterable_equal(levels(out), [1, 2, 3])

    out = as_factor(Series([1, 2, 3]).groupby([1, 1, 2]))
    assert_iterable_equal(out.obj, [1, 2, 3])
    assert_iterable_equal(levels(out.obj), [1, 2, 3])


def test_as_ordered():
    out = as_ordered([1, 2, 3])
    assert_iterable_equal(out, [1, 2, 3])
    assert_iterable_equal(levels(out), [1, 2, 3])
    assert out.ordered

    out = as_ordered(Categorical([1, 2, 3], ordered=False))
    assert_iterable_equal(out, [1, 2, 3])
    assert_iterable_equal(levels(out), [1, 2, 3])
    assert out.ordered

    out = as_ordered(Series([1, 2, 3]))
    assert_iterable_equal(out, [1, 2, 3])
    assert_iterable_equal(levels(out), [1, 2, 3])
    assert out.cat.ordered

    out = as_ordered(Series([1, 2, 3]).groupby([1, 1, 2]))
    assert_iterable_equal(out.obj, [1, 2, 3])
    assert_iterable_equal(levels(out.obj), [1, 2, 3])
    assert out.obj.cat.ordered


def test_as_pd_date():
    assert_equal(
        as_pd_date("Sep 16, 2021"), pd.Timestamp("2021-09-16 00:00:00")
    )


def test_is_atomic():
    assert_(is_atomic(1))
    assert_not(is_atomic([1]))
    assert_not(is_atomic(Series([1])))


def test_is_character():
    assert_not(is_character(Series([1])))
    assert_(is_character(Series([1, "a"])))
    assert_(is_character(Series(["111"])))
    assert_(is_character(Series(["111", "222"])))

    sgb = Series(["1", "2", "3"]).groupby([1, 1, 2])
    assert_iterable_equal(is_character(sgb), [True, True])


def test_is_complex():
    assert_not(is_complex(Series([1])))
    assert_(is_complex(Series([1 + 2j, 3 + 4j])))

    sgb = Series([1 + 2j, 3 + 4j, 5 + 6j]).groupby([1, 1, 2])
    assert_iterable_equal(is_complex(sgb), [True, True])


def test_is_element():
    df = tibble(x=[1, 2, 1, 2], y=[1, 1, 2, 2]).groupby("y")
    out = is_element(df.x, df.y)
    assert_iterable_equal(out, [True, False, False, True])

    out = is_element(1, df.x)
    assert_iterable_equal(out, [True, True])
    assert_iterable_equal(out.index, [1, 2])

    out = is_element(df.x, [2, 3])
    assert_iterable_equal(out.obj, [False, True, False, True])

    df = tibble(x=[1, 2, 1, 2], y=[1, 1, 2, 2]).rowwise()
    out = is_element(df.x, [2, 3])
    assert out.is_rowwise

    out = is_element(df.x.obj, [2, 3])
    assert_iterable_equal(out, [False, True, False, True])
    assert_iterable_equal(out.index, df.index)


def test_is_finite():
    assert_iterable_equal(is_finite(Series([1, 2, 3])), [True, True, True])
    assert_iterable_equal(
        is_finite(Series([1, 2, 3]).groupby([1, 1, 2])).obj, [True, True, True]
    )

    assert_iterable_equal(
        is_finite(Series([1, 2, 3, float("inf")])), [True, True, True, False]
    )
    assert_iterable_equal(
        is_finite(Series([1, 2, 3, float("inf")]).groupby([1, 1, 2, 2])).obj,
        [True, True, True, False],
    )


def test_is_infinite():
    assert_iterable_equal(is_infinite(Series([1, 2, 3])), [False, False, False])
    assert_iterable_equal(
        is_infinite(Series([1, 2, 3]).groupby([1, 1, 2])).obj,
        [False, False, False]
    )

    assert_iterable_equal(
        is_infinite(Series([1, 2, 3, float("inf")])),
        [False, False, False, True],
    )
    assert_iterable_equal(
        is_infinite(Series([1, 2, 3, float("inf")]).groupby([1, 1, 2, 2])).obj,
        [False, False, False, True],
    )


def test_is_false_is_true():
    assert_not(is_false(Series([1])))
    assert_not(is_true(Series([1])))
    assert_iterable_equal(is_false(Series([0]).groupby([1])), [False])
    assert_iterable_equal(is_true(Series([0]).groupby([1])), [False])
    assert_iterable_equal(
        is_false(Series([True, False]).groupby([1, 2])),
        [False, True],
    )
    assert_iterable_equal(is_true(Series([0]).groupby([1])), [False])


def test_all_any():
    assert_(all_(Series([True, True])))
    assert_(any_(Series([True, False])))
    assert_iterable_equal(
        all_(Series([True, True, False]).groupby([1, 1, 2])),
        [True, False],
    )
    assert_iterable_equal(
        any_(Series([True, True, False]).groupby([1, 2, 2])),
        [True, True],
    )


def test_is_numeric():
    assert_(is_numeric(Series([1])))
    assert_not(is_numeric(Series([1, "2"])))


def test_is_null():
    assert_not(is_null(Series([None])))
    assert_not(is_null(Series([1])))
    assert_not(is_null(Series([None, 1])))
    assert_iterable_equal(
        is_null(Series([None, 1]).groupby([1, 2])),
        [False, False],
    )


def test_is_na():
    assert_iterable_equal(is_na(Series([None])), [True])
    assert_iterable_equal(is_na(Series([1])), [False])
    assert_iterable_equal(is_na(Series([None, 1])), [True, False])
    assert_iterable_equal(
        is_na(Series([None, 1]).groupby([1, 2])).obj,
        [True, False],
    )


def test_is_ordered():
    assert_not(is_ordered(1))

    u = Series([1, 2, 3]).astype("category")
    o = Series([1, 2, 3]).astype("category").cat.as_ordered()
    assert_not(is_ordered(u))
    assert_(is_ordered(o))
    assert_iterable_equal(
        is_ordered(u.groupby([1, 1, 2])),
        [False, False],
    )
    assert_iterable_equal(
        is_ordered(o.groupby([1, 1, 2])),
        [True, True],
    )


def test_any_na():
    assert_(any_na(Series([1, np.nan])))
    assert_iterable_equal(
        any_na(Series([1, np.nan, 1]).groupby([1, 1, 2])),
        [True, False],
    )

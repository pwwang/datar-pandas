import pytest  # noqa: F401

import numpy as np
from datar.other import itemgetter, attrgetter, pd_str, pd_cat, pd_dt
from datar.tibble import tibble
from datar_pandas.pandas import Series, Categorical
from datar_pandas.collections import Collection

from .conftest import assert_iterable_equal


def test_itemgetter():
    arr = [1, 2, 3]
    out = itemgetter(arr, Series([2]))
    assert_iterable_equal(out, [3])

    arr = [1, 2, 3]
    out = itemgetter(arr, Series([1, 2]).groupby([1, 2]))
    assert_iterable_equal(out.index, [1, 2])
    assert_iterable_equal(out, [2, 3])

    arr = [1, 2, 3]
    out = itemgetter(arr, np.array([2]))
    assert_iterable_equal(out, [3])

    arr = [1, 2, 3]
    out = itemgetter(arr, Collection([2]))
    assert_iterable_equal(out, [3])

    df = tibble(x=[1, 2, 3], y=[2, 1, 0])
    out = itemgetter(df.x, df.y)
    assert_iterable_equal(out, [3, 2, 1])

    out = itemgetter(df.x, [2, 1, 0])
    assert_iterable_equal(out.index, [2, 1, 0])
    assert_iterable_equal(out, [3, 2, 1])

    df = tibble(x=[1, 2, 3], y=[1, 0, 0], g=[1, 1, 2]).group_by("g")
    out = itemgetter(df.x, df.y)
    assert_iterable_equal(out.index, [1, 1, 2])
    assert_iterable_equal(out, [2, 1, 3])

    out = itemgetter(df.x, [0])
    assert_iterable_equal(out.index, [1, 2])
    assert_iterable_equal(out, [1, 3])

    out = itemgetter(df.x, Collection([0]))
    assert_iterable_equal(out.index, [1, 2])
    assert_iterable_equal(out, [1, 3])


def test_attrgetter():
    s = Series(["aa", "bb", "cc"]).groupby([1, 1, 2], group_keys=True)
    out = attrgetter(s, "str").upper()
    assert_iterable_equal(out.obj.index, [1, 1, 2])
    assert_iterable_equal(out.obj, ["AA", "BB", "CC"])

    out = attrgetter(s, "str")[0]
    assert_iterable_equal(out.obj.index, [1, 1, 2])
    assert_iterable_equal(out.obj, ["a", "b", "c"])

    s = Series(Categorical(["aa", "bb", "cc"]))
    out = attrgetter(s, "cat").codes
    assert_iterable_equal(out, [0, 1, 2])

    s = Series(Categorical(["aa", "bb", "cc"])).groupby([1, 1, 2])
    out = attrgetter(s, "cat").codes
    assert_iterable_equal(out.obj, [0, 1, 2])


def test_pd_str():
    s = Series(["aa", "bb", "cc"])
    out = pd_str(s).upper()
    assert_iterable_equal(out, ["AA", "BB", "CC"])

    s = s.groupby([1, 1, 2], group_keys=True)
    out = pd_str(s).upper()
    assert_iterable_equal(out.obj, ["AA", "BB", "CC"])


def test_pd_cat():
    s = Series(Categorical(["aa", "bb", "cc"]))
    out = pd_cat(s).codes
    assert_iterable_equal(out, [0, 1, 2])

    s = Series(Categorical(["aa", "bb", "cc"])).groupby([1, 1, 2])
    out = pd_cat(s).codes
    assert_iterable_equal(out.obj, [0, 1, 2])


def test_pd_dt():
    s = Series(["2019-01-01", "2019-01-02", "2019-01-03"]).astype("datetime64")
    out = pd_dt(s).year
    assert_iterable_equal(out, [2019, 2019, 2019])

    s = s.groupby([1, 1, 2], group_keys=True)
    out = pd_dt(s).year
    assert_iterable_equal(out.obj, [2019, 2019, 2019])

    s = (
        Series(["2019-01-01", "2019-01-02", "2019-01-03"])
        .astype("datetime64")
        .groupby([1, 1, 2])
    )
    out = pd_dt(s).year
    assert_iterable_equal(out.obj, [2019, 2019, 2019])

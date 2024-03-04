import pytest  # noqa

import numpy as np
from datar import f
from datar.base import (
    length,
    lengths,
    sample,
    match,
    unique,
    order,
    rep,
    c,
    mean,
    rev,
    sort,
    seq_len,
    seq_along,
    seq,
)
from datar.dplyr import rowwise, mutate
from datar.tibble import tibble
from datar_pandas.pandas import Series, assert_frame_equal, get_obj
from datar_pandas.tibble import TibbleGrouped
from ..conftest import assert_iterable_equal, assert_equal, pd_data

pd_data = pd_data()


def test_length():
    assert_equal(length(pd_data.series), 4)
    assert_iterable_equal(length(pd_data.sgb), [1, 2, 1])


def test_lengths():
    assert_iterable_equal(lengths(pd_data.series), [1, 1, 1, 1])
    out = lengths(pd_data.sgb).values
    assert_equal(len(out), 3)
    assert_equal(out[0], 1)
    assert_iterable_equal(out[1], [1, 1])
    assert_equal(out[2], 1)


def test_sample():
    x = sample(Series(range(1, 13)))
    assert set(x) == set(range(1, 13))

    y = sample(Series(x), replace=True)
    assert len(y) <= 12

    z = sample(Series([0, 1]), 100, replace=True)
    assert set(z) == {0, 1}
    assert len(z) == 100

    w = sample(Series(list("abc")), 100, replace=True)
    assert set(w) == {"a", "b", "c"}
    assert len(z) == 100


def test_match():
    x = tibble(x=[1, 2, 1, 4], y=[2, 1, 3, 4], g=[1, 1, 2, 2])
    out = match(x.x, [1, 2])
    assert_iterable_equal(out, [0, 1, 0, -1])

    x = x.groupby('g')
    out = match(x.x, x.y)
    assert_iterable_equal(get_obj(out), [1, 0, -1, 1])

    out = match(x.x, [2, 4, 1, 0])
    assert_iterable_equal(get_obj(out), [2, 0, 2, 1])

    x = get_obj(x) >> rowwise()
    out = match(x.x, [2, 4, 1, 0])
    assert out.is_rowwise

    # GH #115
    df = tibble(x=[1, 1, 2, 2], y=["a", "b", "b", "b"])
    out = match(df.y, unique(df.y))
    assert_iterable_equal(out, [0, 1, 1, 1])

    gf = df.groupby("x")
    out = match(gf.y, unique(gf.y))
    assert_iterable_equal(get_obj(out), [0, 1, 0, 0])

    out = match(["a", "b"], df.y)
    assert_iterable_equal(out, [0, 1])

    with pytest.raises(ValueError):
        match(gf.y, df.y.groupby([1, 1, 1, 2]))

    # treat as normal series
    incompatible_y = unique(gf.y)
    incompatible_y.loc[3] = "c"
    out = match(gf.y, incompatible_y)
    assert_iterable_equal(get_obj(out), [0, 1, 1, 1])


def test_order():
    x = Series([5, 2, 3, 4])
    out = order(x)
    assert_iterable_equal(out, [1, 2, 3, 0])

    out = order(x, decreasing=True)
    assert_iterable_equal(out, [0, 3, 2, 1])

    x = Series([np.nan, 5, 2, 3, 4])
    out = order(x)
    assert_iterable_equal(out, [2, 3, 4, 1, 0])

    out = order(x, na_last=False)
    assert_iterable_equal(out, [0, 2, 3, 4, 1])

    x = Series([1, 2, 3, 4]).groupby([1, 1, 2, 2])
    out = order(x)
    assert_iterable_equal(get_obj(out), [0, 1, 0, 1])


def test_rep_sgb_param(caplog):
    df = tibble(
        x=[1, 1, 2, 2],
        times=[1, 2, 1, 2],
        length=[3, 4, 4, 3],
        each=[1, 1, 1, 1],
    ).group_by("x")
    out = rep([1, 2], df.times)
    assert_iterable_equal(get_obj(out), [1, 2, 2, 1, 2, 2])

    out = rep([1, 2], times=df.times, each=1, length=df.length)
    assert "first element" in caplog.text

    assert_iterable_equal(get_obj(out), [1, 2, 2, 1, 2, 2, 1])
    assert_iterable_equal(out._grouper.size(), [3, 4])

    df2 = tibble(x=[1, 2], each=[1, 1]).group_by("x")
    out = rep(df2.x, each=df2.each)
    assert_iterable_equal(get_obj(out), [1, 2])
    out = rep(df2.x, times=df2.each, length=df2.each, each=df2.each)
    assert_iterable_equal(get_obj(out), [1, 2])
    out = rep(3, each=df2.each)
    assert_iterable_equal(get_obj(out), [3, 3])

    out = rep(get_obj(df2.x), 2)
    assert_iterable_equal(out, [1, 2, 1, 2])


def test_rep_df():
    df = tibble(x=[0, 1, 2])
    with pytest.raises(ValueError):
        rep(df, each=2)

    out = rep(df, times=2, length=5)
    assert_frame_equal(out, tibble(x=[0, 1, 2, 0, 1]))


def test_rep_grouped_df():
    df = tibble(x=[0, 1, 2], g=[1, 1, 2]).group_by("g")
    out = rep(df, 2, length=5)
    assert isinstance(out, TibbleGrouped)
    assert_iterable_equal(get_obj(out.x), [0, 1, 2, 0, 1])
    assert out._datar["grouped"]._grouper.ngroups == 2


def test_c():
    x = Series([1, 2, 3, 4]).groupby([1, 1, 2, 2])
    out = c(7, [8, 9], x)
    assert_iterable_equal(get_obj(out), [7, 8, 9, 1, 2, 7, 8, 9, 3, 4])

    df = tibble(x=[1, 2, 3, 4], y=rep([1, 2], each=2)) >> rowwise()
    out = df >> mutate(z=mean(c(f.x, f.y)))
    assert_iterable_equal(get_obj(out.z), [1.0, 1.5, 2.5, 3.0])


def test_rev():
    x = Series([1, 2, 3])
    out = rev(x)
    assert_iterable_equal(out, [3, 2, 1])
    assert_iterable_equal(out.index, [0, 1, 2])

    x = Series([1, 2, 3]).groupby([1, 1, 2])
    out = rev(x)
    assert_iterable_equal(get_obj(out), [2, 1, 3])


def test_sort():
    out = sort(Series([1, 2, 3]))
    assert_iterable_equal(out, [1, 2, 3])
    out = sort(Series([1, 2, 3]), decreasing=True)
    assert_iterable_equal(out, [3, 2, 1])
    # out = sort([NA, 1, 2, 3])
    # assert_iterable_equal(out, [1, 2, 3])
    out = sort(Series([np.nan, 1, 2, 3]), na_last=True)
    assert_iterable_equal(out, Series([1, 2, 3, np.nan]))
    out = sort(Series([np.nan, 1, 2, 3]), na_last=False)
    assert_iterable_equal(out, [np.nan, 1, 2, 3])

    out = sort(Series([np.nan, 1, 2, 3]).groupby([1, 1, 2, 2]), na_last=True)
    assert_iterable_equal(get_obj(out), Series([1, np.nan, 2, 3]))


def test_seq_len():
    x = Series([1, 2, 3, 4])
    out = seq_len(x)
    assert_iterable_equal(out, [1])

    x = x.groupby([1, 1, 2, 2])
    out = seq_len(x)
    assert len(out) == 2
    assert_iterable_equal(out.index, [1, 2])
    assert_iterable_equal(out[1], [1])
    assert_iterable_equal(out[2], [1, 2, 3])


def test_seq_along():
    x = Series([1, 2, 3, 4])
    out = seq_along(x)
    assert_iterable_equal(out, [1, 2, 3, 4])

    x = x.groupby([1, 1, 2, 2])
    out = seq_along(x)
    assert len(out) == 2
    assert_iterable_equal(out.index, [1, 2])
    assert_iterable_equal(out[1], [1, 2])
    assert_iterable_equal(out[2], [1, 2])


def test_seq():
    # seq works like seq_along with Series or SeriesGroupBy
    x = Series([1, 2, 3, 4])
    out = seq(x)
    assert_iterable_equal(out, [1, 2, 3, 4])

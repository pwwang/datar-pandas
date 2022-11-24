import pytest
from datar.base import (
    colnames,
    rownames,
    set_colnames,
    set_rownames,
    dim,
    nrow,
    ncol,
    diag,
    t,
    unique,
    duplicated,
    max_col,
    complete_cases,
    head,
    tail,
)
from datar.base import c, NA
from datar.tibble import tibble
from datar_pandas.pandas import DataFrame

from ..conftest import assert_iterable_equal, assert_equal


def test_rowcolnames():
    df = DataFrame(dict(x=[1, 2, 3]))
    assert_equal(colnames(df), ["x"])
    assert_equal(rownames(df).tolist(), [0, 1, 2])
    df = DataFrame([1, 2, 3], index=["a", "b", "c"])
    assert_equal(colnames(df), [0])
    assert_equal(rownames(df).tolist(), ["a", "b", "c"])

    df = set_colnames(df, ["y"])
    assert_iterable_equal(df.columns, ["y"])

    df = set_colnames(df, ["y"], nested=False)
    assert_iterable_equal(df.columns, ["y"])

    assert_iterable_equal(colnames(df, nested=False), ["y"])

    df = set_rownames(df, ["a", "b", "c"])
    assert_iterable_equal(df.index, ["a", "b", "c"])

    df = tibble(a=tibble(x=1, y=1), b=tibble(u=2, v=3), z=2)
    df = df >> set_colnames(["c", "d", "w"], nested=True)
    assert_iterable_equal(df.columns, ["c$x", "c$y", "d$u", "d$v", "w"])


def test_diag():
    out = dim(3 >> diag())
    assert out == (3, 3)
    out = dim(10 >> diag(3, 4))
    assert out == (3, 4)
    x = c(1j, 2j) >> diag()
    assert x.iloc[0, 0] == 0 + 1j
    assert x.iloc[0, 1] == 0 + 0j
    assert x.iloc[1, 0] == 0 + 0j
    assert x.iloc[1, 1] == 0 + 2j
    x = True >> diag(3)
    assert sum(x.values.flatten()) == 3
    x = c(2, 1) >> diag(4)
    assert_iterable_equal(x >> diag(), [2, 1, 2, 1])

    with pytest.raises(ValueError):
        x >> diag(3, 3)

    x = 1 >> diag(4)
    assert_iterable_equal(x >> diag(3) >> diag(), [3, 3, 3, 3])


def test_ncol():
    df = tibble(x=tibble(a=1, b=2))
    out = ncol(df)
    assert out == 1
    out = ncol(df, nested=False)
    assert out == 2


def test_nrow():
    df = tibble(x=tibble(a=1, b=2))
    out = nrow(df)
    assert out == 1


def test_unique():
    df = tibble(x=[1, 1, 1, 1], y=[1, 1, 2, 2]).group_by("y")
    out = unique(df.x)
    assert_iterable_equal(out, [1, 1])


def test_t():
    df = tibble(x=1, y=2)
    out = t(df)
    assert out.shape == (2, 1)
    assert_iterable_equal(out.index, ["x", "y"])


def test_duplicated():
    df = tibble(x=[1, 1, 2, 2])
    assert_iterable_equal(duplicated(df), [False, True, False, True])


def test_max_col():
    df = tibble(a=[1, 7, 4], b=[8, 5, 3], c=[6, 2, 9], d=[8, 7, 9])
    assert_iterable_equal(max_col(df[["a", "b", "c"]], "random"), [1, 0, 2])
    out = max_col(df, "random")
    assert out[0] in [1, 3]
    assert out[1] in [0, 3]
    assert out[2] in [2, 3]
    assert_iterable_equal(max_col(df, "first"), [1, 0, 2])
    assert_iterable_equal(max_col(df, "last"), [3, 3, 3])


def test_complete_cases():
    df = tibble(
        a=[NA, 1, 2],
        b=[4, NA, 6],
        c=[7, 8, 9],
    )
    out = complete_cases(df)
    assert_iterable_equal(out, [False, False, True])


def test_head_tail():
    df = tibble(x=range(20))
    z = df >> head()
    assert z.shape[0] == 6
    z = df >> head(3)
    assert z.shape[0] == 3

    z = df >> tail()
    assert z.shape[0] == 6
    z = df >> tail(3)
    assert z.shape[0] == 3

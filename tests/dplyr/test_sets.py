# tests grabbed from:
# https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-sets.R

import pytest
import numpy
from datar import f
from datar.base import (
    nrow,
    union,
    intersect,
    setdiff,
    factor,
    letters,
    c,
    NA,
    is_factor,
    levels,
    seq,
    is_na,
    rep,
    rev,
    seq_len,
    rnorm,
    length,
    setequal,
)
from datar.dplyr import (
    group_by,
    group_vars,
    group_rows,
    bind_rows,
    distinct,
    union_all,
    filter,
)
from datar.tibble import tibble
from datar_pandas.pandas import assert_frame_equal

from ..conftest import assert_equal, assert_


def test_set_uses_coercion_rules():
    df1 = tibble(x=[1, 2], y=[1, 1])
    df2 = tibble(x=[1, 2], y=[1, 2])

    assert_equal(nrow(union(df1, df2)), 3)
    assert_equal(nrow(intersect(df1, df2)), 1)
    assert_equal(nrow(setdiff(df1, df2)), 1)

    df1 = tibble(x=factor(letters[:10]))
    df2 = tibble(x=letters[5:15])
    res = intersect(df1, df2)
    assert res.equals(tibble(x=letters[5:10]))

    res = intersect(df2, df1)
    assert res.equals(tibble(x=letters[5:10]))

    res = union(df1, df2)
    assert res.equals(tibble(x=letters[:15]))

    res = union(df2, df1)
    assert res.equals(tibble(x=c(letters[5:15], letters[:5])))

    res = setdiff(df1, df2)
    assert res.equals(tibble(x=letters[:5]))

    res = setdiff(df2, df1)
    assert res.equals(tibble(x=letters[10:15]))


def test_setdiff_handles_factors_with_na():
    # test_that("setdiff handles factors with NA (#1526)", {
    df1 = tibble(x=factor(c(NA, "a")))
    df2 = tibble(x=factor("a"))

    res = setdiff(df1, df2)
    assert is_factor(res.x)
    assert levels(res.x) == ["a"]
    assert_equal(is_na(res.x[0]), True)


def test_intersect_does_not_unnecessarily_coerce():
    # test_that("intersect does not unnecessarily coerce (#1722)", {
    df = tibble(a=1)
    res = intersect(df, df)
    assert numpy.issubdtype(res.a.dtype, numpy.integer)


def test_set_operations_reconstruct_grouping_metadata():
    # test_that("set operations reconstruct grouping metadata (#3587)", {
    df1 = tibble(x=seq(1, 4), g=rep([1, 2], each=2)) >> group_by(f.g)
    df2 = tibble(x=seq(3, 6), g=rep([2, 3], each=2))

    out = setdiff(df1, df2)
    exp = filter(df1, f.x < 3)
    assert out.equals(exp)

    out = intersect(df1, df2)
    exp = filter(df1, f.x >= 3).reset_index(drop=True)
    assert_frame_equal(out, exp)

    out = union(df1, df2)
    exp = tibble(x=seq(1, 6), g=rep([1, 2, 3], each=2)) >> group_by(f.g)
    assert out.equals(exp)
    assert_equal(group_vars(out), group_vars(exp))

    out = setdiff(df1, df2) >> group_rows()
    assert out == [[0, 1]]

    out = intersect(df1, df2) >> group_rows()
    assert out == [[0, 1]]

    out = union(df1, df2) >> group_rows()
    assert out == [[0, 1], [2, 3], [4, 5]]


def test_set_operations_keep_the_ordering_of_the_data():
    # test_that("set operations keep the ordering of the data (#3839)", {
    rev_df = lambda df: df.iloc[rev(seq_len(nrow(df))) - 1, :]

    df1 = tibble(x=seq(1, 4), g=rep([1, 2], each=2))
    df2 = tibble(x=seq(3, 6), g=rep([2, 3], each=2))

    out = setdiff(df1, df2)
    exp = filter(df1, f.x < 3)
    assert out.equals(exp)

    out = setdiff(rev_df(df1), df2)
    exp = filter(rev_df(df1), f.x < 3).reset_index(drop=True)
    assert out.equals(exp)

    out = intersect(df1, df2)
    exp = filter(df1, f.x >= 3).reset_index(drop=True)
    assert out.equals(exp)

    out = intersect(rev_df(df1), df2)
    exp = filter(rev_df(df1), f.x >= 3).reset_index(drop=True)
    assert out.equals(exp)

    out = union(df1, df2)
    exp = tibble(x=seq(1, 6), g=rep([1, 2, 3], each=2))
    assert out.equals(exp)

    out = union(rev_df(df1), df2)
    exp = tibble(x=c(seq(4, 1), [5, 6]), g=rep([2, 1, 3], each=2))
    # assert out.equals(exp)
    assert_frame_equal(out, exp)

    out = union(df1, rev_df(df2))
    exp = tibble(x=c(seq(1, 4), [6, 5]), g=rep([1, 2, 3], each=2))
    assert out.equals(exp)


def test_set_operations_remove_duplicates():
    df1 = tibble(x=seq(1, 4), g=rep([1, 2], each=2)) >> bind_rows(f)
    df2 = tibble(x=seq(3, 6), g=rep([2, 3], each=2))

    out = setdiff(df1, df2)
    exp = filter(df1, f.x < 3) >> distinct()
    assert out.equals(exp)

    out = intersect(df1, df2).reset_index(drop=True)
    exp = filter(df1, f.x >= 3) >> distinct()
    assert out.equals(exp.reset_index(drop=True))

    out = union(df1, df2)
    exp = tibble(x=seq(1, 6), g=rep([1, 2, 3], each=2))
    assert out.equals(exp)

    out = union_all(df1, df2)
    exp = tibble(
        x=c(seq(1, 4), seq(1, 4), seq(3, 6)),
        g=c(rep([1, 2], each=2), rep([1, 2], each=2), rep([2, 3], each=2)),
    )
    assert out.equals(exp)

    out = union_all(df1 >> group_by(f.g), df2)
    assert out.equals(exp)
    assert_equal(group_vars(out), ["g"])


def test_set_equality():
    df1 = tibble(x=seq(1, 4), g=rep([1, 2], each=2)) >> group_by(f.g)
    df2 = tibble(x=seq(3, 6), g=rep([2, 3], each=2))

    assert_(setequal(df1, df1))
    assert_(setequal(df2, df2))
    assert_(not setequal(df1, df2))
    assert_(not setequal(df2, df1))


# Errors ------------------------------------------------------------------


def test_errors():
    alfa = tibble(
        land=c("Sverige", "Norway", "Danmark", "Island", "GB"),
        data=rnorm(length(f.land)),
    )
    beta = tibble(
        land=c("Norge", "Danmark", "Island", "Storbritannien"),
        data2=rnorm(length(f.land)),
    )

    with pytest.raises(ValueError, match="not compatible"):
        intersect(alfa, beta)
    with pytest.raises(ValueError, match="not compatible"):
        union(alfa, beta)
    with pytest.raises(ValueError, match="not compatible"):
        setdiff(alfa, beta)

    with pytest.raises(ValueError, match="not compatible"):
        intersect(tibble(x=1), tibble(x=1, y=2))


def test_intersect_union_setdiff_keep_y_groups():
    x = tibble(x=[1, 2, 3])
    y = x.group_by('x')

    out = intersect(x, y)
    assert out.group_vars == ['x']
    assert out.shape[0] == 3

    out = union(x, y)
    assert out.group_vars == ['x']
    assert out.shape[0] == 3

    out = union_all(x, y)
    assert out.group_vars == ['x']
    assert out.shape[0] == 6

    out = setdiff(x, y)
    assert out.group_vars == ['x']
    assert out.shape[0] == 0

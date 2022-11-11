# tests grabbed from:
# https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-slice.r
import pytest
from datar import f
from datar.data import mtcars
from datar.tibble import tibble, as_tibble
from datar.base import nrow, c, NA, rep, seq, dim, colnames
from datar.dplyr import (
    slice,
    slice_sample,
    slice_min,
    slice_head,
    slice_max,
    slice_tail,
    group_by,
    ungroup,
    rowwise,
    group_keys,
    mutate,
    select,
    arrange,
    row_number,
    group_rows,
    filter,
    n,
)
from datar_pandas.tibble import TibbleRowwise
from datar_pandas.api.dplyr.slice_ import _n_from_prop
from datar_pandas.pandas import Categorical, Series, assert_frame_equal

from ..conftest import assert_iterable_equal, assert_equal


def test_empty_slice_returns_input():
    df = tibble(x=[1, 2, 3])
    sf = slice(df)
    assert sf.equals(df)


def test_slice_handles_numeric_input():
    g = mtcars >> arrange(f.cyl) >> group_by(f.cyl)
    res = g >> slice(0)
    assert_equal(nrow(res), 3)
    exp = g >> filter(row_number() == 1)
    assert_frame_equal(res, exp)

    res1 = mtcars >> slice(0) >> as_tibble()
    res2 = mtcars >> filter(row_number() == 1)
    assert_frame_equal(res1, res2)


def test_slice_silently_ignores_out_of_range_values():
    res1 = slice(mtcars, c(2, 100))
    res2 = slice(mtcars, 2)
    assert_frame_equal(res1, res2)

    g = group_by(mtcars, f.cyl)
    res1 = slice(g, c(2, 100))
    res2 = slice(g, 2)
    assert_frame_equal(res1, res2)


def test_slice_works_with_negative_indices():
    res = slice(mtcars, ~c[:2])
    exp = mtcars.tail(-2)
    assert_frame_equal(res, exp)


def test_slice_works_with_grouped_data():
    g = mtcars >> arrange(f.cyl) >> group_by(f.cyl)

    res = slice(g, c[:2])
    exp = filter(g, row_number() < 3)
    assert_frame_equal(res, exp)

    res = slice(g, ~c[:2])
    exp = filter(g, row_number() >= 3)
    assert_frame_equal(res, exp)

    g = group_by(tibble(x=c(1, 1, 2, 2, 2)), f.x)
    # out = group_keys(slice(g, 3, _preserve=True))
    # assert out.x.tolist() == [1, 2]
    out = group_keys(slice(g, 2, _preserve=False))
    assert out.x.tolist() == [2]

    gf = tibble(x=c[1:4]) >> group_by(
        g=Categorical([1, 1, 2], categories=[1, 2, 3]),
        _drop=False,
    )
    with pytest.raises(TypeError):
        gf >> slice("a")
    with pytest.raises(ValueError):
        gf >> slice(~c[:2], 1)

    out = gf >> slice(0)
    assert out.shape[0] == 2

    out = gf >> slice(
        Series([1, 0, 0]).groupby(gf._datar["grouped"].grouper.result_index)
    )
    assert_iterable_equal(out.x.obj, [2, 3])


def test_slice_gives_correct_rows():
    a = tibble(value=[f"row{i}" for i in range(1, 11)])
    out = slice(a, c(0, 1, 2))
    assert out.value.tolist() == ["row1", "row2", "row3"]

    out = slice(a, c(3, 5, 8))
    assert out.value.tolist() == ["row4", "row6", "row9"]

    a = tibble(
        value=[f"row{i}" for i in range(1, 11)], group=rep([1, 2], each=5)
    ) >> group_by(f.group)

    out = slice(a, c[:3])
    assert out.value.obj.tolist() == [f"row{i}" for i in [1, 2, 3, 6, 7, 8]]

    out = slice(a, c(1, 3))
    assert out.value.obj.tolist() == [f"row{i}" for i in [2, 4, 7, 9]]


def test_slice_handles_na():
    df = tibble(x=[1, 2, 3])
    out = slice(df, NA)
    assert_equal(nrow(out), 0)
    out = slice(df, c(1, NA))
    assert_equal(nrow(out), 1)
    out = df >> slice(c(~c(1), NA)) >> nrow()
    assert out == 2

    df = tibble(x=[1, 2, 3, 4], g=rep([1, 2], 2)) >> group_by(f.g)
    out = slice(df, c(1, NA))
    assert_equal(nrow(out), 2)
    out = df >> slice(c(~c(1), NA)) >> nrow()
    assert out == 2


def test_slice_handles_logical_NA():
    df = tibble(x=[1, 2, 3])
    out = slice(df, NA)
    assert_equal(nrow(out), 0)


def test_slice_handles_empty_df():
    df = tibble(x=[])
    res = df >> slice(c[:3])
    assert_equal(nrow(res), 0)
    assert_equal(colnames(res), ["x"])


def test_slice_works_fine_if_n_gt_nrow():
    by_slice = mtcars >> arrange(f.cyl) >> group_by(f.cyl)
    slice_res = by_slice >> slice(7)
    filter_res = by_slice >> filter(row_number() == 8)
    assert slice_res.equals(filter_res)


def test_slice_strips_grouped_indices():
    res = mtcars >> group_by(f.cyl) >> slice(1) >> mutate(mpgplus=f.mpg + 1)
    assert_equal(nrow(res), 3)
    out = group_rows(res)
    assert out == [[0], [1], [2]]


def test_slice_works_with_0col_dfs():
    out = tibble(a=[1, 2, 3]) >> select(~f.a) >> slice(1) >> nrow()
    assert out == 1


def test_slice_correctly_computes_positive_indices_from_negative_indices():
    x = tibble(y=range(1, 11))
    # negative in dplyr meaning exclusive
    out = slice(x, ~c[9:30]).equals(tibble(y=range(1, 10)))
    assert out


def test_slice_accepts_star_args():
    out1 = slice(mtcars, 1, 2)
    out2 = slice(mtcars, [1, 2])
    assert out1.equals(out2)

    out3 = slice(mtcars, 0, n() - 1)
    out4 = slice(mtcars, c(0, nrow(mtcars) - 1))
    assert out3.equals(out4)

    g = mtcars >> group_by(f.cyl)
    out5 = slice(g, 0, n() - 1)
    out6 = slice(g, c(0, n() - 1))
    assert out5.equals(out6)


def test_slice_does_not_evaluate_the_expression_in_empty_groups():
    res = mtcars >> group_by(f.cyl) >> filter(f.cyl == 6) >> slice(c[:2])
    assert_equal(nrow(res), 2)

    # sample_n is Superseded in favor of slice_sample
    # res = mtcars >> \
    #     group_by(f.cyl) >> \
    #     filter(f.cyl==6) >> \
    #     sample_n(size=3)
    # assert nrow(res) == 3


def test_slice_handles_df_columns():
    df = tibble(
        x=[1, 2], y=tibble(a=[1, 2], b=[3, 4]), z=tibble(A=[1, 2], B=[3, 4])
    )
    out = slice(df, 0)
    assert out.equals(df.iloc[[0], :])

    gdf = group_by(df, f.x)
    out = slice(gdf, 0).equals(gdf)
    assert out
    # TODO: group_by a stacked df is not supported yet
    gdf = group_by(df, f["y$a"], f["y$b"])
    out = slice(gdf, 0).equals(gdf)
    assert out
    gdf = group_by(df, f["z$A"], f["z$B"])
    out = slice(gdf, 0).equals(gdf)
    assert out


# # Slice variants ----------------------------------------------------------


def test_functions_silently_truncate_results():
    df = tibble(x=range(1, 6))
    out = df >> slice_head(n=6) >> nrow()
    assert out == 5
    out = df >> slice_tail(n=6) >> nrow()
    assert out == 5
    out = df >> slice_sample(n=6) >> nrow()
    assert out == 5
    out = df >> slice_min(f.x, n=6) >> nrow()
    assert out == 5
    out = df >> slice_max(f.x, n=6) >> nrow()
    assert out == 5


def test_proportion_computed_correctly():
    df = tibble(x=range(1, 11))

    out = df >> slice_head(prop=0.11) >> nrow()
    assert out == 1
    out = df >> slice_tail(prop=0.11) >> nrow()
    assert out == 1
    out = df >> slice_sample(prop=0.11) >> nrow()
    assert out == 1
    out = df >> slice_min(f.x, prop=0.11) >> nrow()
    assert out == 1
    out = df >> slice_max(f.x, prop=0.11) >> nrow()
    assert out == 1
    out = df >> slice_max(f.x, prop=0.11, with_ties=False) >> nrow()
    assert out == 1
    out = df >> slice_min(f.x, prop=0.11, with_ties=False) >> nrow()
    assert out == 1


def test_min_and_max_return_ties_by_default():
    df = tibble(x=c(1, 1, 1, 2, 2))

    out = df >> slice_min(f.x) >> nrow()
    assert out == 3
    out = df >> slice_max(f.x) >> nrow()
    assert out == 2

    out = df >> slice_min(f.x, with_ties=False) >> nrow()
    assert out == 1
    out = df >> slice_max(f.x, with_ties=False) >> nrow()
    assert out == 1


def test_min_and_max_reorder_results():
    df = tibble(id=range(1, 5), x=c(2, 3, 1, 2))
    out = df >> slice_min(f.x, n=2)
    assert out.id.tolist() == [3, 1, 4]
    out = df >> slice_min(f.x, n=2, with_ties=False)
    assert out.id.tolist() == [3, 1]
    out = df >> slice_max(f.x, n=2)
    assert out.id.tolist() == [2, 1, 4]
    out = df >> slice_max(f.x, n=2, with_ties=False)
    assert out.id.tolist() == [2, 1]


def test_min_and_max_ignore_nas():
    df = tibble(id=range(1, 5), x=c(2, NA, 1, 2), y=[NA] * 4)
    out = df >> slice_min(f.x, n=2)
    assert out.id.tolist() == [3, 1, 4]
    out = df >> slice_min(f.y, n=2) >> nrow()
    assert out == 0
    out = df >> slice_max(f.x, n=2)
    assert out.id.tolist() == [1, 4]
    out = df >> slice_max(f.y, n=2) >> nrow()
    assert out == 0


def test_arguments_to_sample_are_passed_along():
    df = tibble(x=range(1, 101), wt=c(1, rep(0, 99)))
    out = df >> slice_sample(n=1, weight_by=f.wt)
    assert out.x.tolist() == [1]

    out = df >> slice_sample(n=2, weight_by=f.wt, replace=True)
    assert out.x.tolist() == [1, 1]


def test_slice_any_checks_for_empty_args_kwargs():
    df = tibble(x=range(1, 11))
    # python recognize n=5
    # with pytest.raises(ValueError):
    #     slice_head(df, 5)
    # with pytest.raises(ValueError):
    #     slice_tail(df, 5)
    with pytest.raises(TypeError):
        df >> slice_min(n=5)
    with pytest.raises(TypeError):
        df >> slice_max(n=5)
    # with pytest.raises(ValueError):
    #     slice_sample(df, 5)


def test_slice_any_checks_for_constant_n_and_prop():
    df = tibble(x=range(1, 11))
    with pytest.raises(TypeError):
        slice_head(df, n=f.x)  # ok with n()
    with pytest.raises(TypeError):
        slice_head(df, prop=f.x)

    with pytest.raises(TypeError):
        slice_tail(df, n=f.x)
    with pytest.raises(TypeError):
        slice_tail(df, prop=f.x)

    with pytest.raises(TypeError):
        slice_min(df, f.x, n=f.x)
    with pytest.raises(TypeError):
        slice_min(df, f.x, prop=f.x)

    with pytest.raises(TypeError):
        slice_max(df, f.x, n=f.x)
    with pytest.raises(TypeError):
        slice_max(df, f.x, prop=f.x)

    with pytest.raises(TypeError):
        slice_sample(df, n=f.x)
    with pytest.raises(TypeError):
        slice_sample(df, prop=f.x)


def test_slice_sample_dose_not_error_on_0rows():
    df = tibble(dummy=[], weight=[])
    res = slice_sample(df, prop=0.5, weight_by=f.weight)
    assert_equal(nrow(res), 0)


# # Errors ------------------------------------------------------------------
def test_rename_errors_with_invalid_grouped_df():
    df = tibble(x=[1, 2, 3])

    # Incompatible type
    with pytest.raises(TypeError):
        slice(df, object())
    with pytest.raises(TypeError):
        slice(df, {"a": 1})

    # Mix of positive and negative integers
    with pytest.raises(ValueError):
        mtcars >> slice(c(~c(1), 2))
    with pytest.raises(ValueError):
        mtcars >> slice(c(c[2:4], ~c(1)))

    # n and prop are carefully validated
    # with pytest.raises(ValueError):
    #     _n_from_prop(10, n=1, prop=1)
    with pytest.raises(TypeError):
        _n_from_prop(10, n="a")
    with pytest.raises(TypeError):
        _n_from_prop(10, prop="a")
    with pytest.raises(ValueError):
        _n_from_prop(10, n=-1)
    with pytest.raises(ValueError):
        _n_from_prop(10, prop=-1)
    with pytest.raises(TypeError):
        _n_from_prop(10, n=n())
    with pytest.raises(TypeError):
        _n_from_prop(10, prop=n())


# tests for datar
def test_mixed_rows():
    df = tibble(x=range(5))

    # order kept
    # 0   1   2   3   4
    #        -3      -1
    #             3
    out = slice(df, c(-c(3, 1), 3))
    assert out.x.tolist() == [2, 4, 3]

    # 0   1   2   3   4
    #            -2  -1
    #             3
    out = slice(df, c(-c[1:3], 3))
    assert out.x.tolist() == [4, 3, 3]

    # 0   1   2   3   4
    # 0       2
    #                -1
    out = slice(df, c(~c(0, 2), ~c(-1)))
    assert out.x.tolist() == [1, 3]

    out = df >> slice(c(~c[3:], ~c(1)))
    assert out.x.tolist() == [0, 2]


def test_slice_sample_n_defaults_to_1():
    df = tibble(g=rep([1, 2], each=3), x=seq(1, 6))
    out = df >> slice_sample(n=None)
    assert_equal(dim(out), (1, 2))


def test_slicex_on_grouped_data():
    gf = tibble(g=rep([1, 2], each=3), x=seq(1, 6)) >> group_by(f.g)

    out = gf >> slice_min(f.x)
    assert out.equals(tibble(g=[1, 2], x=[1, 4]))
    out = gf >> slice_max(f.x)
    assert out.equals(tibble(g=[1, 2], x=[3, 6]))
    out = gf >> slice_sample()
    assert_equal(dim(out), (2, 2))


def test_n_from_prop():
    assert _n_from_prop(1, prop=0.5) == 0
    assert _n_from_prop(2, prop=0.5) == 1
    assert _n_from_prop(4, prop=0.5) == 2


# slice_head/tail on grouped data


def test_slice_head_tail_on_grouped_data():
    df = tibble(g=[1, 1, 1, 2, 2, 2], x=[1, 2, 3, 4, 5, 6]) >> group_by(f.g)
    out = slice_head(df, 1) >> ungroup()
    assert_frame_equal(out, tibble(g=[1, 2], x=[1, 4]))
    out = slice_tail(df, 1) >> ungroup()
    assert_frame_equal(out, tibble(g=[1, 2], x=[3, 6]))


def test_slice_family_on_rowwise_df():
    df = tibble(x=c[1:6]) >> rowwise()
    out = df >> slice_head(prop=0.1)
    assert out.shape[0] == 0

    out = df >> slice([0, 1, 2])
    assert isinstance(out, TibbleRowwise)
    assert_equal(nrow(out), 5)

    out = df >> slice_head(n=3)
    assert isinstance(out, TibbleRowwise)
    assert_equal(nrow(out), 5)

    out = df >> slice_tail(n=3)
    assert isinstance(out, TibbleRowwise)
    assert_equal(nrow(out), 5)

    out = df >> slice_min(f.x, n=3)
    assert isinstance(out, TibbleRowwise)
    assert_equal(nrow(out), 5)

    out = df >> slice_max(f.x, n=3)
    assert isinstance(out, TibbleRowwise)
    assert_equal(nrow(out), 5)

    out = df >> slice_sample(n=3)
    assert isinstance(out, TibbleRowwise)
    assert_equal(nrow(out), 5)


def test_preserve_prop_not_support(caplog):
    df = tibble(x=c[:5]) >> group_by(f.x)
    df >> slice(f.x == 2, _preserve=True)
    assert "_preserve" in caplog.text

    with pytest.raises(ValueError):
        df >> slice_min(f.x, prop=0.5)

    with pytest.raises(ValueError):
        df >> slice_max(f.x, prop=0.5)

    with pytest.raises(ValueError):
        df >> slice_sample(f.x, prop=0.5)


def test_wrong_indices():
    df = tibble(x=c[:3])
    with pytest.raises(TypeError):
        df >> slice("a")

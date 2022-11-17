import warnings
import pytest

import numpy as np
from datar import f
from datar.base import NA, c
from datar_pandas.pandas import (
    DataFrame,
    Series,
    SeriesGroupBy,
    assert_frame_equal,
    assert_series_equal,
)
from datar_pandas.tibble import TibbleGrouped

# from datar.datar import drop_index
from datar.tibble import tibble, tribble
from datar.base import (
    sum,
    mean,
    median,
    var,
    pmax,
    pmin,
    prod,
    abs,
    ceiling,
    col_means,
    col_medians,
    col_sds,
    col_sums,
    cov,
    exp,
    floor,
    log,
    log10,
    log1p,
    log2,
    max,
    min,
    row_means,
    row_medians,
    round,
    row_sums,
    row_sds,
    scale,
    sign,
    signif,
    sqrt,
    trunc,
    weighted_mean,
    quantile,
)
from datar.base import pi
from ..conftest import pd_data, assert_equal, assert_iterable_equal

pd_data = pd_data()


@pytest.mark.parametrize(
    "fun, x, expected, args, kwargs",
    [
        (sum, pd_data.series, 8, (), {}),
        (sum, pd_data.sgb, [1, 4, 3], (), {}),
        (mean, pd_data.series, 2, (), {}),
        (mean, pd_data.sgb, [1, 2, 3], (), {}),
        (median, pd_data.series, 2, (), {}),
        (median, pd_data.sgb, [1, 2, 3], (), {}),  # 5
        (var, pd_data.series, 0.6666666666, (), {}),
        (var, pd_data.sgb, [NA, 0, NA], (), {}),
        (exp, pd_data.series, [2.718, 7.389, 7.389, 20.086], (), {}),
        (exp, pd_data.sgb, [2.718, 7.389, 7.389, 20.086], (), {}),
        (prod, pd_data.series, 12, (), {}),  # 10
        (prod, pd_data.sgb, [1, 4, 3], (), {}),
        (min, pd_data.series, 1, (), {}),
        (min, pd_data.sgb, [1, 2, 3], (), {}),
        (max, pd_data.series, 3, (), {}),
        (max, pd_data.sgb, [1, 2, 3], (), {}),
        (round, pd_data.series, [1, 2, 2, 3], (), {}),
        (round, Series(pd_data.farray), [1, 2, 2, 3], (), {}),
        (round, pd_data.sgb, [1, 2, 2, 3], (), {}),
        (trunc, pd_data.series, [1, 2, 2, 3], (), {}),
        (trunc, Series(pd_data.farray), [1, 2, 2, 3], (), {}),  # 20
        (trunc, pd_data.sgb, [1, 2, 2, 3], (), {}),
        (sqrt, pd_data.series, [1, 1.414, 1.414, 1.732], (), {}),
        (sqrt, pd_data.sgb, [1, 1.414, 1.414, 1.732], (), {}),
        (abs, pd_data.series, [1, 2, 2, 3], (), {}),
        (abs, Series(pd_data.negarray), [1, 2, 2, 3], (), {}),
        (abs, pd_data.sgb, [1, 2, 2, 3], (), {}),
        (sign, pd_data.series, [1, 1, 1, 1], (), {}),
        (sign, Series(pd_data.negarray), [-1, -1, -1, -1], (), {}),
        (sign, pd_data.sgb, [1, 1, 1, 1], (), {}),
        (ceiling, pd_data.series, [1, 2, 2, 3], (), {}),  # 30
        (ceiling, Series(pd_data.farray), [2, 3, 3, 4], (), {}),
        (ceiling, pd_data.sgb, [1, 2, 2, 3], (), {}),
        (floor, pd_data.series, [1, 2, 2, 3], (), {}),
        (floor, Series(pd_data.farray), [1, 2, 2, 3], (), {}),
        (floor, pd_data.sgb, [1, 2, 2, 3], (), {}),
    ],
)
def test_arithm_func(fun, x, expected, args, kwargs):
    out = fun(x, *args, **kwargs)
    if isinstance(out, SeriesGroupBy):
        out = out.obj
    if isinstance(expected, list):
        assert_iterable_equal(out, expected, approx=1e-3)
    elif isinstance(expected, Series):
        assert_series_equal(out, expected)
    else:
        assert_equal(out, expected, approx=1e-3)


def test_na_rm_error():

    with pytest.raises(TypeError):
        with warnings.catch_warnings():
            # Dropping invalid columns in SeriesGroupBy.agg is deprecated.
            warnings.simplefilter("ignore", FutureWarning)
            sum(pd_data.sgb, na_rm=False)


def test_pmin():

    assert_iterable_equal(pmin(pd_data.series, pd_data.series), [1, 2, 2, 3])
    assert_iterable_equal(pmin(pd_data.series, 2), [1, 2, 2, 2])
    assert_iterable_equal(pmin(2, pd_data.series), [1, 2, 2, 2])
    assert_iterable_equal(pmin(pd_data.sgb, pd_data.sgb).obj, [1, 2, 2, 3])
    assert_iterable_equal(pmin(pd_data.tr.x, pd_data.tr.x).obj, [1, 2, 2, 3])


def test_pmax():

    assert_iterable_equal(pmax(pd_data.series, pd_data.series), [1, 2, 2, 3])
    assert_iterable_equal(pmax(pd_data.series, 2), [2, 2, 2, 3])
    assert_iterable_equal(pmax(2, pd_data.series), [2, 2, 2, 3])
    assert_iterable_equal(pmax(pd_data.sgb, pd_data.sgb).obj, [1, 2, 2, 3])


def test_cov():
    df = tibble(x=c[1:4], y=c[4:7])
    out = df >> cov()
    assert_frame_equal(
        out.reset_index(drop=True), tibble(x=[1.0, 1.0], y=[1.0, 1.0])
    )

    out = [1, 2, 3] >> cov([4, 5, 6])
    assert out == 1.0

    with pytest.raises(ValueError):
        cov(df, 1)

    gf = tibble(x=[1, 1, 1, 2, 2, 2], y=[1, 2, 3, 4, 5, 6]).group_by("x")
    with pytest.raises(ValueError):
        cov(gf, 1)

    out = cov(gf)
    assert_iterable_equal(out.y, [1, 1])
    assert_iterable_equal(out.index, [1, 2])

    out = cov(gf.y, [3, 3, 3])
    assert_iterable_equal(out, [0, 0])

    with pytest.raises(ValueError):
        cov(gf.y)


def test_col_row_verbs():
    df = tribble(f.x, f.y, f.z, 1, NA, 6, 2, 4, 9, 3, 6, 15)
    assert_iterable_equal(row_medians(df), [NA, 4, 6])
    assert_iterable_equal(row_medians(df, na_rm=True), [3.5, 4, 6])
    assert_iterable_equal(col_medians(df), [2, NA, 9])
    assert_iterable_equal(col_medians(df, na_rm=True), [2, 5, 9])

    assert_iterable_equal(row_means(df), [NA, 5, 8])
    assert_iterable_equal(row_means(df, na_rm=True), [3.5, 5, 8])
    assert_iterable_equal(col_means(df), [2, NA, 10])
    assert_iterable_equal(col_means(df, na_rm=True), [2, 5, 10])

    assert_iterable_equal(row_sums(df), [NA, 15, 24])
    assert_iterable_equal(row_sums(df, na_rm=True), [7, 15, 24])
    assert_iterable_equal(col_sums(df), [6, NA, 30])
    assert_iterable_equal(col_sums(df, na_rm=True), [6, 10, 30])

    assert_iterable_equal(
        row_sds(df), [NA, 3.605551275463989, 6.244997998398398], approx=True
    )
    assert_iterable_equal(
        row_sds(df, na_rm=True),
        [3.5355339059327378, 3.605551275463989, 6.244997998398398],
        approx=True,
    )
    assert_iterable_equal(
        col_sds(df), [1.0, NA, 4.58257569495584], approx=True
    )
    assert_iterable_equal(
        col_sds(df, na_rm=True),
        [1.0, 1.4142135623730951, 4.58257569495584],
        approx=True,
    )

    # grouped
    df = tibble(x=[1, 1, 2, 2], y=[3, 4, 3, 4]).group_by("x")
    assert_iterable_equal(col_sums(df).y, [7, 7])
    assert_iterable_equal(col_means(df).y, [3.5, 3.5])
    assert_iterable_equal(col_medians(df).y, [3.5, 3.5])
    assert_iterable_equal(col_sds(df).y, [0.7071, 0.7071], approx=1e-3)

    with pytest.raises(ValueError):
        col_sums(df, na_rm=True)


def test_scale():

    out = Series([1, 2, 3, 4, 3, 2]).groupby([1, 1, 1, 2, 2, 2]) >> scale()
    assert_iterable_equal(out.obj, [-1.0, 0, 1, 1, 0, -1])

    df = tibble(x=[1, 2, 3], y=[4, 5, 6])
    assert_frame_equal(scale(df, False, False), df)
    assert_iterable_equal(scale(df.x, [1]), [0.0, 0.6325, 1.2649], approx=1e-3)
    with pytest.raises(ValueError):
        scale(df, center=[1])
    with pytest.raises(ValueError):
        scale(df, scale_=[1])

    df = tibble(x=["a", "b"])
    with pytest.raises(ValueError):  # must be all numeric
        scale(df)


def test_signif():
    x2 = Series(pi * 100.0 ** np.array([-1, 0, 1, 2, 3]))
    out = signif(x2, 3)
    assert_iterable_equal(
        out, [3.14e-02, 3.14e00, 3.14e02, 3.14e04, 3.14e06], approx=1e-2
    )


def test_log():
    df = DataFrame(
        {
            "x": [exp(1), exp(2), 4, 2, 10, np.e - 1],
            "base": [np.e, np.e, 4, 2, 10, np.e],
        }
    )
    assert_iterable_equal(
        log(df.x, df.base),
        [1, 2, 1, 1, 1, 0.5413],
        approx=1e-3,
    )
    assert_iterable_equal(
        log2(df.x),
        [1.4427, 2.8854, 2, 1, 3.3219, 0.7810],
        approx=1e-3,
    )
    assert_iterable_equal(
        log10(df.x),
        [0.4343, 0.8686, 0.6021, 0.3010, 1.0, 0.2351],
        approx=1e-3,
    )
    assert_iterable_equal(
        log1p(df.x),
        [1.3133, 2.1269, 1.6094, 1.0986, 2.3979, 1.0],
        approx=1e-3,
    )

    gf = TibbleGrouped.from_groupby(df.groupby("base", sort=False))
    assert_iterable_equal(
        log(gf.x, gf.base).obj,
        [1, 2, 0.5413, 1, 1, 1],
        approx=1e-3,
    )
    assert_iterable_equal(
        log2(gf.x).obj,
        [1.4427, 2.8854, 2, 1, 3.3219, 0.7810],
        approx=1e-3,
    )
    assert_iterable_equal(
        log10(gf.x).obj,
        [0.4343, 0.8686, 0.6021, 0.3010, 1.0, 0.2351],
        approx=1e-3,
    )
    assert_iterable_equal(
        log1p(gf.x).obj,
        [1.3133, 2.1269, 1.6094, 1.0986, 2.3979, 1.0],
        approx=1e-3,
    )

    rf = gf.rowwise()
    assert_iterable_equal(
        log(rf.x, rf.base).obj,
        [1, 2, 1, 1, 1, 0.5413],
        approx=1e-3,
    )


def test_weighted_mean():
    df = tibble(x=[1, 2, 3], w=[1, 2, 3], w2=[-1, 0, 1])
    assert_equal(weighted_mean(df.x, df.w), 2.3333333, approx=1e-3)
    assert_equal(weighted_mean(df.x, None), 2.0)
    assert_equal(weighted_mean(df.x, df.w2), NA)

    df = tibble(x=[1, 2, 3], w=[1, 2, 3], g=[1, 1, 2]).group_by("g")
    assert_iterable_equal(
        weighted_mean(df.x, df.w), [1.6667, 3.0], approx=1e-2
    )


def test_quantile():
    df = tibble(x=[1, 2, 3], g=[1, 1, 2])
    assert_equal(quantile(df.x, 0.5), 2.0)
    assert_iterable_equal(quantile(df.x, [0.5, 1]), [2, 3])

    gf = df.group_by("g")
    assert_iterable_equal(quantile(gf.x, 0.5), [1.5, 3.0])

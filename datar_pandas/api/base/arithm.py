from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import datar_numpy.api.arithm  # noqa: F401
from datar_numpy.utils import make_array
from datar.apis.base import (
    ceiling,
    cov,
    floor,
    mean,
    median,
    pmax,
    pmin,
    sqrt,
    var,
    scale,
    min_,
    max_,
    round_,
    sum_,
    abs_,
    prod,
    sign,
    signif,
    trunc,
    exp,
    log,
    log2,
    log10,
    log1p,
    sd,
    weighted_mean,
    quantile,
    proportions,
    col_sums,
    row_sums,
    col_sds,
    row_sds,
    col_means,
    row_means,
    col_medians,
    row_medians,
)

from ...typing import Data, Int, Number, Bool
from ...common import is_null
from ...pandas import (
    DataFrame,
    Series,
    GroupBy,
    SeriesGroupBy,
    NDFrame,
    is_numeric_dtype,
)
from ...tibble import Tibble, TibbleGrouped
from ...factory import func_bootstrap

func_bootstrap(ceiling, func=np.ceil, kind="transform")
func_bootstrap(floor, func=np.floor, kind="transform")
func_bootstrap(mean, func=np.mean, kind="agg")
func_bootstrap(median, func=np.median, kind="agg")
func_bootstrap(sqrt, func=np.sqrt, kind="transform")
func_bootstrap(var, func="var", kind="agg")
func_bootstrap(min_, func=np.min, kind="agg")
func_bootstrap(max_, func=np.max, kind="agg")
func_bootstrap(round_, func=np.round, kind="transform")
func_bootstrap(sum_, func=np.sum, kind="agg")
func_bootstrap(abs_, func=np.abs, kind="transform")
func_bootstrap(prod, func=np.prod, kind="agg")
func_bootstrap(sign, func=np.sign, kind="transform")
func_bootstrap(trunc, func=np.trunc, kind="transform")
func_bootstrap(exp, func=np.exp, kind="transform")
func_bootstrap(log2, func=np.log2, kind="transform")
func_bootstrap(log10, func=np.log10, kind="transform")
func_bootstrap(log1p, func=np.log1p, kind="transform")
func_bootstrap(sd, func=np.std, kind="agg")
func_bootstrap(proportions, func=lambda x: x / x.sum(), kind="transform")
func_bootstrap(signif, func=signif.dispatch(object, backend="numpy"))
func_bootstrap(
    log,
    func=log.dispatch(object, backend="numpy"),
    post="transform",
)
func_bootstrap(
    weighted_mean,
    func=weighted_mean.dispatch(object, backend="numpy"),
    exclude="na_rm",
)
func_bootstrap(
    quantile,
    exclude={"probs", "na_rm", "names", "type_", "digits"},
    func=quantile.dispatch(object, backend="numpy"),
    # kind="agg",
)


def _check_all_numeric(x: DataFrame, fun_name: str) -> None:
    if x.apply(is_numeric_dtype).all():
        return

    raise ValueError(f"In {fun_name}(...): input must be all numeric.")


def _na_rm_check(__data: NDFrame, na_rm: Bool, *args: Any, **kwargs: Any):
    """Warn about na_rm False on GroupBy objects"""
    if isinstance(__data, (GroupBy, TibbleGrouped)) and na_rm is not None:
        raise ValueError("`na_rm` is not supported on grouped objects.")
    return __data, args, {**kwargs, "skipna": na_rm}


@cov.register(DataFrame, backend="pandas")
def _cov_df(x, y=None, na_rm: bool = False, ddof: int = 1):
    """Covariance of DataFrame"""
    if y is not None:
        raise ValueError(
            "In `cov(...)`: No `y` is allowed when `x` is a data frame."
        )
    return x.cov(ddof=ddof)


@cov.register(TibbleGrouped, backend="pandas")
def _cov_tibble_grouped(
    x: TibbleGrouped,
    y: Data[Number] = None,
    ddof: Int = 1,
) -> Tibble:
    """Covariance of TibbleGrouped"""
    if y is not None:
        raise ValueError(
            "In `cov(...)`: No `y` is allowed when `x` is a data frame."
        )

    with warnings.catch_warnings():
        # size-1 group will warning about ddof
        warnings.simplefilter("ignore", RuntimeWarning)
        return x._datar["grouped"].cov(ddof=ddof).droplevel(-1)


@cov.register(SeriesGroupBy, backend="pandas")
def _cov_seriesgroupby(
    x: SeriesGroupBy,
    y: Data[Number] = None,
    ddof: Int = 1,
) -> Series:
    """Covariance of SeriesGrouped"""
    if y is None:
        raise ValueError(
            "In `cov(...)`: `y` is required when `x` is a SeriesGroupBy."
        )

    df = TibbleGrouped.from_args(x=x, cov=y)
    with warnings.catch_warnings():
        # size-1 group will warning about ddof
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            df._datar["grouped"].cov(ddof=ddof).droplevel(-1)["cov"].iloc[::2]
        )


@scale.register(DataFrame, backend="pandas")
def _scale_df(
    x: DataFrame,
    center: Bool | Data[Number] = True,
    scale_: Bool | Data[Number] = True,
) -> Series:
    _check_all_numeric(x, "scale")
    center_true = center is True

    # center
    ncols = x.shape[1]

    if center_true:
        center = x.mean(numeric_only=True)

    elif center is not False:
        center = make_array(center)
        if len(center) != ncols:
            raise ValueError(
                f"length of `center` ({len(center)}) must equal "
                f"the number of columns of `x` ({ncols})"
            )

    if center is not False:
        x = x.subtract(center)

    # scale
    if scale_ is True:

        def _rms(col: Series) -> Series:
            nonnas = col[~is_null(col)] ** 2
            return np.sqrt(nonnas.sum() / (len(nonnas) - 1))

        scale_ = x.std(numeric_only=True) if center_true else x.agg(_rms)

    elif scale_ is not False:
        scale_ = make_array(scale_)
        if len(scale_) != ncols:
            raise ValueError(
                f"length of `scale_` ({len(center)}) must equal "
                f"the number of columns of `x` ({ncols})"
            )

    if scale_ is not False:
        x = x.div(scale_)

    if center is False and scale_ is False:
        x = x.copy()

    return x


@scale.register(Series, backend="pandas")
def _scale_series(
    x: Series,
    center: Bool | Data[Number] = True,
    scale_: Bool | Data[Number] = True,
) -> Series:
    """Scaling on series"""
    out = _scale_df(x.to_frame(), center, scale_)
    out = out.iloc[:, 0]
    return out


@scale.register(SeriesGroupBy, backend="pandas")
def _scale_seriesgroupby(
    x: SeriesGroupBy,
    center: Bool | Data[Number] | Series = True,
    scale_: Bool | Data[Number] | Series = True,
) -> Series:
    """Scaling on series"""
    return x.transform(
        scale.dispatch(Series, backend="pandas"),
        center=center,
        scale_=scale_,
    ).groupby(
        x.grouper,
        sort=x.sort,
        dropna=x.dropna,
        observed=x.observed,
    )


@pmin.register((Series, DataFrame), backend="pandas")
def _pmin_ndframe(
    *x: NDFrame,
    na_rm: Bool = False,
) -> Series:
    """Get the min value rowwisely"""
    return Tibble.from_args(*x, _name_repair="minimal").min(
        axis=1,
        skipna=na_rm,
    )


@pmin.register((SeriesGroupBy, TibbleGrouped), backend="pandas")
def _pmin_grouped(
    *x: SeriesGroupBy | TibbleGrouped,
    na_rm: Bool = False,
) -> SeriesGroupBy:
    """Get the min value rowwisely"""
    gf = TibbleGrouped.from_args(*x, _name_repair="minimal")
    out = gf.min(axis=1, skipna=na_rm)
    g = gf._datar["grouped"]
    return out.groupby(
        g.grouper,
        sort=g.sort,
        observed=g.observed,
        dropna=g.dropna,
    )


@pmax.register((Series, DataFrame), backend="pandas")
def _pmax_ndframe(
    *x: NDFrame,
    na_rm: Bool = False,
) -> Series:
    """Get the max value rowwisely"""
    return Tibble.from_args(*x, _name_repair="minimal").max(
        axis=1,
        skipna=na_rm,
    )


@pmax.register((SeriesGroupBy, TibbleGrouped), backend="pandas")
def _pmax_grouped(
    *x: SeriesGroupBy | TibbleGrouped,
    na_rm: Bool = False,
) -> SeriesGroupBy:
    """Get the max value rowwisely"""
    gf = TibbleGrouped.from_args(*x, _name_repair="minimal")
    out = gf.max(axis=1, skipna=na_rm)
    g = gf._datar["grouped"]
    return out.groupby(
        g.grouper,
        sort=g.sort,
        observed=g.observed,
        dropna=g.dropna,
    )


@col_sums.register(DataFrame, backend="pandas")
def _col_sums_df(x: DataFrame, na_rm: Bool = None) -> Series:
    _check_all_numeric(x, "col_sums")
    return x.sum(skipna=bool(na_rm), numeric_only=True)


@col_sums.register(TibbleGrouped, backend="pandas")
def _col_sums_tibble_grouped(x: TibbleGrouped, na_rm: Bool = None) -> Series:
    _check_all_numeric(x, "col_sums")
    _na_rm_check(x, na_rm)
    x = x._datar["grouped"]
    return x.sum(numeric_only=True)


@row_sums.register(DataFrame, backend="pandas")
def _row_sums_df(x: DataFrame, na_rm: Bool = None) -> Series:
    _check_all_numeric(x, "row_sums")
    return x.sum(skipna=bool(na_rm), numeric_only=True, axis=1)


@col_means.register(DataFrame, backend="pandas")
def _col_means_df(x: DataFrame, na_rm: Bool = None) -> Series:
    _check_all_numeric(x, "col_means")
    return x.mean(skipna=bool(na_rm), numeric_only=True)


@col_means.register(TibbleGrouped, backend="pandas")
def _col_means_tibble_grouped(x: TibbleGrouped, na_rm: Bool = None) -> Series:
    _check_all_numeric(x, "col_means")
    _na_rm_check(x, na_rm)
    x = x._datar["grouped"]
    return x.mean(numeric_only=True)


@row_means.register(DataFrame, backend="pandas")
def _row_means_df(x: DataFrame, na_rm: Bool = None) -> Series:
    _check_all_numeric(x, "row_means")
    return x.mean(skipna=bool(na_rm), numeric_only=True, axis=1)


@col_sds.register(DataFrame, backend="pandas")
def _col_sds_df(x: DataFrame, na_rm: Bool = None, ddof: Int = 1) -> Series:
    _check_all_numeric(x, "col_sds")
    return x.std(skipna=bool(na_rm), numeric_only=True, ddof=ddof)


@col_sds.register(TibbleGrouped, backend="pandas")
def _col_sds_tibble_grouped(
    x: TibbleGrouped,
    na_rm: Bool = None,
    ddof: Int = 1,
) -> Series:
    _check_all_numeric(x, "col_sds")
    _na_rm_check(x, na_rm)
    x = x._datar["grouped"]
    return x.std(ddof=ddof)


@row_sds.register(DataFrame, backend="pandas")
def _row_sds_df(x: DataFrame, na_rm: Bool = None, ddof: Int = 1) -> Series:
    _check_all_numeric(x, "row_sds")
    return x.std(skipna=bool(na_rm), numeric_only=True, ddof=ddof, axis=1)


@col_medians.register(DataFrame, backend="pandas")
def _col_medians_df(x: DataFrame, na_rm: Bool = None) -> Series:
    _check_all_numeric(x, "col_medians")
    return x.median(skipna=bool(na_rm), numeric_only=True)


@col_medians.register(TibbleGrouped, backend="pandas")
def _col_medians_tibble_grouped(
    x: TibbleGrouped, na_rm: Bool = None
) -> Series:
    _check_all_numeric(x, "col_medians")
    _na_rm_check(x, na_rm)
    x = x._datar["grouped"]
    return x.median(numeric_only=True)


@row_medians.register(DataFrame, backend="pandas")
def _row_medians_df(x: DataFrame, na_rm: Bool = None) -> Series:
    _check_all_numeric(x, "row_medians")
    return x.median(skipna=bool(na_rm), numeric_only=True, axis=1)

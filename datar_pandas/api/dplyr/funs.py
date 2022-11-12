"""Functions from R-dplyr"""
import numpy as np

from datar.apis.dplyr import (
    between,
    cummean,
    cumall,
    cumany,
    coalesce,
    na_if,
    near,
    nth,
    first,
    last,
)

from ... import pandas as pd
from ...pandas import DataFrame, Series, SeriesGroupBy
from ...common import is_scalar
from ...factory import func_bootstrap
from ...tibble import TibbleGrouped


@between.register(object, backend="pandas")
def _between_obj(x, left, right, inclusive: str = "both"):
    return Series(x).between(left, right, inclusive=inclusive)


@func_bootstrap(between, kind="transform")
def _between(x, left, right, inclusive: str = "both"):
    return x.between(left, right, inclusive)


# faster
between.register(SeriesGroupBy, func="between", backend="pandas")


@cummean.register(object, backend="pandas")
def _cummean_obj(x):
    return Series(x, dtype=float).expanding().mean()


@func_bootstrap(cummean, kind="transform")
def _cummean(x: Series):
    return x.expanding().mean()


@cumall.register(object, backend="pandas")
def _cumall_obj(x, na_as: bool = False):
    return Series(x, dtype=object).fillna(na_as).cumprod().astype(bool)


@func_bootstrap(cumall, kind="transform")
def _cumall(x, na_as: bool = False):
    return x.fillna(na_as).cumprod().astype(bool)


@cumany.register(object, backend="pandas")
def _cumany_obj(x, na_as: bool = False):
    return Series(x, dtype=object).fillna(na_as).cumsum().astype(bool)


@func_bootstrap(cumany, kind="transform")
def _cumany(x, na_as: bool = False):
    return x.fillna(na_as).cumsum().astype(bool)


@func_bootstrap(coalesce, post="transform")
def _coalesce(x, *replace):
    if isinstance(x, DataFrame):
        for repl in replace:
            if isinstance(repl, DataFrame):
                x = x.combine_first(repl)
            else:
                x = x.combine_first(pd.concat([repl] * x.columns.size, axis=1))
    else:
        for repl in replace:
            x = x.combine_first(repl)

    return x


def _na_if_post(__out, x, y, __args_raw=None):
    if not isinstance(__out, TibbleGrouped):
        return __out
    rawx = __args_raw["x"]
    out = __out.groupby(
        rawx.grouper,
        sort=rawx.sort,
        dropna=rawx.dropna,
        observed=rawx.observed,
    )
    if getattr(rawx, "is_rowwise", False):
        out.is_rowwise = True
    return out


@func_bootstrap(na_if, post=_na_if_post)
def _na_if(x, y, __args_raw=None):
    if is_scalar(x) and is_scalar(y):  # rowwise
        return np.nan if x == y else x

    rawx = __args_raw["x"] if __args_raw else x
    lenx = 1 if is_scalar(rawx) else len(rawx)
    if lenx < y.size:
        raise ValueError(
            f"`y` must be length {lenx} (same as `x`), not {y.size}."
        )
    x[(x == y).values] = np.nan
    return x


def _near_post(__out, a, b, __args_raw=None):
    if not isinstance(__out, TibbleGrouped):
        return __out

    rawx = __args_raw["a"]
    out = __out.groupby(
        rawx.grouper,
        sort=rawx.sort,
        dropna=rawx.dropna,
        observed=rawx.observed,
    )
    if getattr(rawx, "is_rowwise", False):
        out.is_rowwise = True
    return out


@func_bootstrap(near, post=_near_post)
def _near(a, b, __args_raw=None):
    return np.isclose(a, b)


def _nth_(x, n, order_by=np.nan, default=np.nan, __args_raw=None):
    if not isinstance(n, int):
        raise TypeError("`nth` expects `n` to be an integer")

    order_by_null = pd.isnull(__args_raw["order_by"])
    if is_scalar(order_by_null):
        order_by_null = np.array([order_by_null], dtype=bool)

    if not order_by_null.all():
        x = x.iloc[order_by.argsort().values]

    try:
        return x.iloc[n]
    except (ValueError, IndexError, TypeError):
        return default


@func_bootstrap(nth, exclude={"n", "default"})
def _nth(x, n, order_by=np.nan, default=np.nan, __args_raw=None):
    return _nth_(
        x, n, order_by=order_by, default=default, __args_raw=__args_raw
    )


@func_bootstrap(first, exclude="default")
def _first(
    x,
    order_by=np.nan,
    default=np.nan,
    __args_raw=None,
):
    """Get the first element of x"""
    return _nth_(
        x, 0, order_by=order_by, default=default, __args_raw=__args_raw
    )


@func_bootstrap(last, exclude="default")
def _last(
    x,
    order_by=np.nan,
    default=np.nan,
    __args_raw=None,
):
    """Get the last element of x"""
    return _nth_(
        x, -1, order_by=order_by, default=default, __args_raw=__args_raw
    )

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
from ...pandas import Series, SeriesGroupBy
from ...common import is_scalar
from ...factory import func_bootstrap
from ...tibble import TibbleGrouped, TibbleRowwise


@func_bootstrap(between, kind="transform")
def _between(x, left, right, inclusive="both"):
    return x.between(left, right, inclusive)


# faster
between.register(SeriesGroupBy, "between")


@func_bootstrap(cummean, kind="transform")
def _cummean(x: Series):
    return x.cumsum() / (np.arange(x.size) + 1.0)


@func_bootstrap(cumall, kind="transform")
def _cumall(x, na_as: bool = False):
    return x.fillna(na_as).cumprod().astype(bool)


@func_bootstrap(cumany, kind="transform")
def _cumany(x, na_as: bool = False):
    return x.fillna(na_as).cumsum().astype(bool)


@func_bootstrap(coalesce, data_args={"x", "replace"})
def _coalesce(x, *replace):
    for repl in replace:
        x = x.combine_first(repl)

    return x


@func_bootstrap(na_if, data_args={"x", "y"})
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


@na_if.register((TibbleGrouped, TibbleRowwise), func="default", post="decor")
def _na_if_post(__out, x, y, __args_raw=None):
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


func_bootstrap(
    near,
    func=lambda a, b, __args_raw=None: np.isclose(a, b),
    data_args={"a", "b"},
    name="near",
    qualname="near",
    module="datar.dplyr",
    doc="""Compare numbers with tolerance

    Args:
        a: and
        b: Numbers to compare
        rtol: The relative tolerance parameter
        atol: The absolute tolerance parameter
        equal_nan: Whether to compare NaN's as equal.
            If True, NA's in `a` will be
            considered equal to NA's in `b` in the output array.

    Returns:
        A bool array indicating element-wise equvalence between a and b
    """,
    post=_near_post,
)


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


@func_bootstrap(nth, data_args={"x", "order_by"})
def _nth(x, n, order_by=np.nan, default=np.nan, __args_raw=None):
    return _nth_(
        x, n, order_by=order_by, default=default, __args_raw=__args_raw
    )


@func_bootstrap(first, data_args={"x", "order_by"})
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


@func_bootstrap(last, data_args={"x", "order_by"})
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

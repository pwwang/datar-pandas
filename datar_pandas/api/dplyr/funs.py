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
from datar_numpy.utils import make_array

from ... import pandas as pd
from ...pandas import DataFrame, Series, SeriesGroupBy
from ...utils import as_series
from ...common import is_scalar
from ...factory import func_bootstrap
from ...tibble import Tibble


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


@coalesce.register(object, backend="pandas")
def _coalesce_obj(x, *replace):
    df = Tibble.from_args(x, *replace)
    x = df.iloc[:, 0]
    for col in df.columns[1:]:
        x = x.combine_first(df[col])
    return x


@func_bootstrap(coalesce, post="transform")
def _coalesce(x, *replace):
    if isinstance(x, DataFrame):
        for repl in replace:
            if isinstance(repl, DataFrame):
                x = x.combine_first(repl)
            else:  # pragma: no cover
                x = x.combine_first(pd.concat([repl] * x.columns.size, axis=1))
    else:
        for repl in replace:
            x = x.combine_first(repl)

    return x


@na_if.register(object, backend="pandas")
def _na_if_obj(x, y):
    if is_scalar(x) and is_scalar(y):
        return np.nan if x == y else x

    x = make_array(x)
    y = make_array(y)
    if x.size < y.size:
        raise ValueError(
            f"`y` must be length {x.size} (same as `x`), not {y.size}."
        )

    try:
        eqs = np.equal(x, y)
    except np.core._exceptions.UFuncTypeError:
        eqs = np.equal(x.astype(object), y.astype(object))

    if eqs.any():
        if np.issubdtype(x.dtype, np.int_):
            x = x.astype(float)
        elif not np.issubdtype(x.dtype, np.float_):
            x = x.astype(object)
        x[eqs] = np.nan

    return x


@func_bootstrap(na_if, post="transform")
def _na_if(x, y):
    x[(x == y).values] = np.nan
    return x


@near.register(object, backend="pandas")
def _near_obj(a, b):
    return np.isclose(a, b)


@func_bootstrap(near, post="transform")
def _near(a, b):
    return np.isclose(a, b)


def _nth_(x, n, order_by=np.nan, default=np.nan, __args_raw=None):
    if not isinstance(n, int):
        raise TypeError("`nth` expects `n` to be an integer")

    if __args_raw is None:
        order_by_null = pd.isnull(order_by)
    else:
        order_by_null = pd.isnull(__args_raw["order_by"])

    if is_scalar(order_by_null):
        order_by_null = np.array([order_by_null], dtype=bool)

    if not order_by_null.all():
        x = x.iloc[as_series(order_by).argsort().values]

    try:
        return x.iloc[n]
    except (ValueError, IndexError, TypeError):
        return default


@nth.register(object, backend="pandas")
def _nth_obj(x, n, order_by=np.nan, default=np.nan, __args_raw=None):
    return _nth_(
        as_series(x),
        n,
        order_by=order_by,
        default=default,
        __args_raw=__args_raw,
    )


@func_bootstrap(nth, exclude={"n", "default"})
def _nth(x, n, order_by=np.nan, default=np.nan, __args_raw=None):
    return _nth_(
        x,
        n,
        order_by=order_by,
        default=default,
        __args_raw=__args_raw,
    )


@first.register(object, backend="pandas")
def _first_obj(x, order_by=np.nan, default=np.nan, __args_raw=None):
    return _nth_(
        as_series(x),
        0,
        order_by=order_by,
        default=default,
        __args_raw=__args_raw,
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
        x,
        0,
        order_by=order_by,
        default=default,
        __args_raw=__args_raw,
    )


@last.register(object, backend="pandas")
def _last_obj(x, order_by=np.nan, default=np.nan, __args_raw=None):
    return _nth_(
        as_series(x),
        -1,
        order_by=order_by,
        default=default,
        __args_raw=__args_raw,
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
        x,
        -1,
        order_by=order_by,
        default=default,
        __args_raw=__args_raw,
    )

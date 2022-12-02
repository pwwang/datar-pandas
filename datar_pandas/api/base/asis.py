import builtins

import numpy as np
# load numpy implementation of datar
import datar_numpy.api.asis  # noqa: F401
# make sure all_ and any_ loaded first
import datar_numpy.api.sets  # noqa: F401
from datar.apis.base import (
    is_atomic,
    is_character,
    is_complex,
    is_double,
    is_integer,
    is_element,
    is_finite,
    is_false,
    is_infinite,
    is_logical,
    is_na,
    is_null,
    is_numeric,
    is_true,
    as_character,
    as_complex,
    as_date,
    as_double,
    as_integer,
    as_logical,
    as_numeric,
    is_factor,
    is_ordered,
    as_factor,
    as_ordered,
    all_,
    any_,
)

from ... import pandas as pd
from ...pandas import (
    Categorical,
    PandasObject,
    Series,
    SeriesGroupBy,
    is_string_dtype,
    is_complex_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_bool_dtype,
    is_numeric_dtype,
    is_categorical_dtype,
    get_obj,
)
from ...common import is_scalar
from ...utils import as_series
from ...factory import func_bootstrap, func_factory
from ...tibble import Tibble

is_character.register(Series, backend="pandas")(is_string_dtype)
is_complex.register(Series, backend="pandas")(is_complex_dtype)
is_double.register(Series, backend="pandas")(is_float_dtype)
is_integer.register(Series, backend="pandas")(is_integer_dtype)
is_logical.register(Series, backend="pandas")(is_bool_dtype)
is_numeric.register(Series, backend="pandas")(is_numeric_dtype)
is_factor.register(object, backend="pandas")(is_categorical_dtype)

is_character.register(SeriesGroupBy, backend="pandas")(
    lambda x: x.agg(is_string_dtype)
)
is_complex.register(SeriesGroupBy, backend="pandas")(
    lambda x: x.agg(is_complex_dtype)
)
is_double.register(SeriesGroupBy, backend="pandas")(
    lambda x: x.agg(is_float_dtype)
)
is_integer.register(SeriesGroupBy, backend="pandas")(
    lambda x: x.agg(is_integer_dtype)
)
is_logical.register(SeriesGroupBy, backend="pandas")(
    lambda x: x.agg(is_bool_dtype)
)
is_numeric.register(SeriesGroupBy, backend="pandas")(
    lambda x: x.agg(is_numeric_dtype)
)
is_factor.register(SeriesGroupBy, backend="pandas")(
    lambda x: x.agg(is_categorical_dtype)
)


@is_ordered.register(object, backend="pandas")
def _is_ordered(x):
    return False


@is_ordered.register(Categorical, backend="pandas")
def _is_ordered_cat(x):
    return x.ordered


func_bootstrap(is_atomic, func=is_scalar, kind="agg")
func_bootstrap(is_finite, func=np.isfinite, kind="transform")
func_bootstrap(is_infinite, func=np.isinf, kind="transform")
func_bootstrap(is_na, func=pd.isna, kind="transform")
func_bootstrap(is_null, func=lambda x: False, kind="agg")
func_bootstrap(
    is_ordered,
    func=lambda x: is_categorical_dtype(x) and x.cat.ordered,
    kind="agg",
)


func_bootstrap(as_character, func=lambda x: x.astype(str), kind="transform")
func_bootstrap(as_complex, func=lambda x: x.astype(complex), kind="transform")
func_bootstrap(as_double, func=lambda x: x.astype(float), kind="transform")
func_bootstrap(as_integer, func=lambda x: x.astype(int), kind="transform")
func_bootstrap(as_logical, func=lambda x: x.astype(bool), kind="transform")
func_bootstrap(
    as_numeric,
    func=as_numeric.dispatch(object, backend="numpy"),
    kind="transform",
)
# func_bootstrap(as_factor, func=lambda x: Categorical(x), kind="transform")
# func_bootstrap(
#     as_ordered,
#     func=lambda x: Categorical(x, ordered=True),
#     kind="transform",
# )


@as_factor.register(object, backend="pandas")
def _as_factor_obj(x):
    return Categorical(x)


@as_factor.register(Categorical, backend="pandas")
def _as_factor_cat(x):
    return x


@as_factor.register(Series, backend="pandas")
def _as_factor_series(x):
    return x.astype("category")


@as_factor.register(SeriesGroupBy, backend="pandas")
def _as_factor_series_groupby(x):
    return get_obj(x).astype("category").groupby(
        x.grouper,
        sort=x.sort,
        observed=x.observed,
        dropna=x.dropna,
    )


@as_ordered.register(object, backend="pandas")
def _as_ordered_obj(x):
    return Categorical(x, ordered=True)


@as_ordered.register(Categorical, backend="pandas")
def _as_ordered_cat(x: Categorical):
    return x if x.ordered else x.as_ordered()


@as_ordered.register(Series, backend="pandas")
def _as_ordered_series(x):
    return x.astype("category").cat.as_ordered()


@as_ordered.register(SeriesGroupBy, backend="pandas")
def _as_ordered_series_groupby(x):
    return get_obj(x).astype("category").cat.as_ordered().groupby(
        x.grouper,
        sort=x.sort,
        observed=x.observed,
        dropna=x.dropna,
    )


@as_integer.register(Categorical, backend="pandas")
def _as_integer_cat(x):
    return x.codes


@as_integer.register(Series, backend="pandas")
def _as_integer_ser(x):
    if is_categorical_dtype(x):
        return x.cat.codes
    return x.astype(int)


@as_integer.register(SeriesGroupBy, backend="pandas")
def _as_integer_sgb(x: SeriesGroupBy):
    out = as_integer(get_obj(x), __ast_fallback="normal", __backend="pandas")
    out = as_series(out)
    out.index = get_obj(x).index
    out = out.groupby(
        x.grouper,
        sort=x.sort,
        observed=x.observed,
        dropna=x.dropna,
    )
    if getattr(x, "is_rowwise", False):
        out.is_rowwise = True
    return out


@is_element.register(PandasObject, backend="pandas")
def _is_element(x, y):
    if isinstance(x, SeriesGroupBy) and isinstance(y, SeriesGroupBy):
        df = Tibble.from_args(x=x, y=y)
        return (
            df._datar["grouped"]
            .apply(lambda g: np.isin(g.x, g.y))
            .explode()
            .astype(bool)
        )

    if isinstance(x, SeriesGroupBy):
        out = x.transform(np.isin, test_elements=y).groupby(
            x.grouper,
            observed=x.observed,
            sort=x.sort,
            dropna=x.dropna,
        )
        if getattr(x, "is_rowwise", False):
            out.is_rowwise = True
        return out

    if isinstance(y, SeriesGroupBy):
        return y.apply(lambda e: np.isin(x, e)).explode().astype(bool)

    if isinstance(x, Series):
        return x.isin(y)

    return np.isin(x, y)  # pragma: no cover


@func_bootstrap(as_date, kind="transform")
def _as_date(
    x,
    format=None,
    try_formats=None,
    optional=False,
    tz=0,
    origin=None,
):
    out = pd.to_datetime(
        as_date.dispatch(np.ndarray, backend="numpy")(
            x.values,
            format=format,
            try_formats=try_formats,
            optional=optional,
            origin=origin,
            tz=tz,
        )
    )
    out.index = x.index
    return out


as_pd_date = func_factory(
    kind="transform",
    name="as_pd_date",
    doc="""Alias of pandas.to_datetime(), but registered as a function
    so that it can be used in verbs.

    See https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

    Args:
        x: The argument to be converted to datetime
        *args: and
        **kwargs: Other arguments passing to `pandas.to_datetime()`

    Returns:
        Converted datetime
    """,
    func=pd.to_datetime,
)


@as_pd_date.register(object, backend="pandas")
def _as_pd_data_obj(x):
    return pd.to_datetime(x)


func_bootstrap(
    is_true,
    func=lambda x: x.size == 1 and x.values[0].item() is True,
    kind="agg",
)

func_bootstrap(
    is_false,
    func=lambda x: x.size == 1 and x.values[0].item() is False,
    kind="agg",
)

func_bootstrap(all_, func=builtins.all, kind="agg")
func_bootstrap(any_, func=builtins.any, kind="agg")

import builtins
import numpy as np
from datar.apis.base import (
    # is_atomic,
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
    as_null,
    as_numeric,
    is_factor,
    is_ordered,
    as_factor,
    as_ordered,
    all_,
    any_,
)
from datar.apis.tibble import tibble

from ... import pandas as pd
from ...pandas import (
    Categorical,
    Series,
    SeriesGroupBy,
    is_string_dtype,
    is_complex_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_bool_dtype,
    is_numeric_dtype,
    is_categorical_dtype,
)
from ...contexts import Context
from ...factory import func_bootstrap, func_factory

is_character.register(Series, context=Context.EVAL)(is_string_dtype)
is_complex.register(Series, context=Context.EVAL)(is_complex_dtype)
is_double.register(Series, context=Context.EVAL)(is_float_dtype)
is_integer.register(Series, context=Context.EVAL)(is_integer_dtype)
is_logical.register(Series, context=Context.EVAL)(is_bool_dtype)
is_numeric.register(Series, context=Context.EVAL)(is_numeric_dtype)
is_factor.register(object, context=Context.EVAL)(is_categorical_dtype)

is_character.register(SeriesGroupBy, context=Context.EVAL)(
    lambda x: x.agg(is_string_dtype)
)
is_complex.register(SeriesGroupBy, context=Context.EVAL)(
    lambda x: x.agg(is_complex_dtype)
)
is_double.register(SeriesGroupBy, context=Context.EVAL)(
    lambda x: x.agg(is_float_dtype)
)
is_integer.register(SeriesGroupBy, context=Context.EVAL)(
    lambda x: x.agg(is_integer_dtype)
)
is_logical.register(SeriesGroupBy, context=Context.EVAL)(
    lambda x: x.agg(is_bool_dtype)
)
is_numeric.register(SeriesGroupBy, context=Context.EVAL)(
    lambda x: x.agg(is_numeric_dtype)
)
is_factor.register(SeriesGroupBy, context=Context.EVAL)(
    lambda x: x.agg(is_categorical_dtype)
)

func_bootstrap(
    is_finite,
    func=np.isfinite,
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    is_infinite,
    func=np.isinf,
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    is_na,
    func=pd.isna,
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    is_null,
    func=pd.isnull,
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    is_ordered,
    func=lambda x: is_categorical_dtype(x) and x.cat.ordered,
    context=Context.EVAL,
    kind="transform",
)

func_bootstrap(
    as_character,
    func=lambda x: x.astype(str),
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    as_complex,
    func=lambda x: x.astype(complex),
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    as_double,
    func=lambda x: x.astype(float),
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    as_integer,
    func=lambda x: x.astype(int),
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    as_logical,
    func=lambda x: x.astype(bool),
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    as_null,
    func=lambda x: None,
    context=Context.EVAL,
    kind="agg",
)
func_bootstrap(
    as_numeric,
    func=as_numeric.dispatch(object),
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    as_factor,
    func=lambda x: Categorical(x),
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    as_ordered,
    func=lambda x: Categorical(x, ordered=True),
    context=Context.EVAL,
    kind="transform",
)


@is_element.register(object, context=Context.EVAL)
def _is_element(x, y):
    if isinstance(x, SeriesGroupBy) and isinstance(y, SeriesGroupBy):
        df = tibble(x=x, y=y)
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

    return np.isin(x, y)


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
        as_date.dispatch(np.ndarray)(
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


func_bootstrap(is_true, func=lambda x: False, context=Context.EVAL, kind="agg")
func_bootstrap(is_false, func=lambda x: False, context=Context.EVAL, kind="agg")

func_bootstrap(all_, func=builtins.all, context=Context.EVAL, kind="agg")
func_bootstrap(any_, func=builtins.any, context=Context.EVAL, kind="agg")

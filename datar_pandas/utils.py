import textwrap
from typing import Any, Hashable, Mapping
from functools import singledispatch

from .common import is_null
from .pandas import DataFrame, Series, SeriesGroupBy

# Specify a "no default" value so that None can be used as a default value
NO_DEFAULT = object()


@singledispatch
def name_of(value: Any) -> str:
    out = str(value)
    out = textwrap.shorten(out, 16, break_on_hyphens=False, placeholder="...")
    return out


name_of.register(Series, lambda x: x.name)
name_of.register(SeriesGroupBy, lambda x: x.obj.name)
name_of.register(DataFrame, lambda x: None)


def apply_dtypes(df: DataFrame, dtypes) -> None:
    """Apply dtypes to data frame"""
    if dtypes is None or dtypes is False:
        return

    if dtypes is True:
        inferred = df.convert_dtypes()
        for col in df:
            df[col] = inferred[col]
        return

    if not isinstance(dtypes, dict):
        dtypes = dict(zip(df.columns, [dtypes] * df.shape[1]))

    for column, dtype in dtypes.items():
        if column in df:
            df[column] = df[column].astype(dtype)
        else:
            for col in df:
                if col.startswith(f"{column}$"):
                    df[col] = df[col].astype(dtype)


def dict_get(d: Mapping[str, Any], key: Hashable, default: Any = NO_DEFAULT):
    """Get value from dict in case nan is in the key"""
    try:
        return d[key]
    except KeyError:
        if is_null(key):
            for k, v in d.items():
                if is_null(k):
                    return v

        if default is NO_DEFAULT:
            raise
        return default

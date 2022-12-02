from __future__ import annotations

import textwrap
import warnings
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Mapping
from functools import singledispatch

import numpy as np
from pipda import Expression, evaluate_expr

from .common import is_null, unique
from .collections import Collection
from .pandas import DataFrame, Series, SeriesGroupBy, get_obj

if TYPE_CHECKING:
    from pipda import ContextType
    from datar_numpy.utils import Version

# Specify a "no default" value so that None can be used as a default value
NO_DEFAULT = object()
# Just for internal and testing uses
NA_character_ = "<NA>"
# Sentinels
NA_integer_ = np.random.randint(np.iinfo(np.int32).max)
NA_real_ = np.nan
NA_compex_ = complex(NA_real_, NA_real_)

DEFAULT_COLUMN_PREFIX = "_VAR_"


class ExpressionWrapper:
    """A wrapper around an expression to bypass evaluation"""

    __slots__ = ("expr",)

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def __str__(self) -> str:
        return str(self.expr)

    def _pipda_eval(self, data: Any, context: ContextType = None) -> Any:
        return evaluate_expr(self.expr, data, context)


@singledispatch
def name_of(value: Any) -> str:
    out = str(value)
    out = textwrap.shorten(out, 16, break_on_hyphens=False, placeholder="...")
    return out


name_of.register(Series, lambda x: x.name)
name_of.register(SeriesGroupBy, lambda x: get_obj(x).name)
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


def vars_select(
    all_columns: Iterable[str],
    *columns: int | str,
    raise_nonexists: bool = True,
):
    # TODO: support selecting data-frame columns
    """Select columns

    Args:
        all_columns: The column pool to select
        *columns: arguments to select from the pool
        raise_nonexist: Whether raise exception when column not exists
            in the pool

    Returns:
        The selected indexes for columns

    Raises:
        KeyError: When the column does not exist in the pool
            and raise_nonexists is True.
    """
    # if not isinstance(all_columns, Index):
    #     all_columns = Index(all_columns, dtype=object)

    columns = [
        column.name
        if isinstance(column, Series)
        else get_obj(column).name
        if isinstance(column, SeriesGroupBy)
        else column
        for column in columns
    ]
    for col in columns:
        if not isinstance(col, str):
            continue
        colidx = all_columns.get_indexer_for([col])
        if colidx.size > 1:
            raise ValueError(
                "Names must be unique. Name "
                f'"{col}" found at locations {list(colidx)}.'
            )

    selected = Collection(*columns, pool=list(all_columns))
    if raise_nonexists and selected.unmatched and selected.unmatched != {None}:
        raise KeyError(f"Columns `{selected.unmatched}` do not exist.")

    return unique(selected).astype(int)


def as_series(x: Any) -> Series:
    """Convert to a series.

    To avoid Seris([]) to raise a warning about dtype

    Args:
        x: The input to convert

    Returns:
        The converted series
    """
    if isinstance(x, Series):
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return Series(x)


def pandas_version() -> Version:
    """Get pandas version"""
    import pandas as pd
    from datar_numpy.utils import Version

    return Version(*map(int, pd.__version__.split(".")[:3]))

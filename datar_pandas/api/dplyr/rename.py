"""Rename columns

https://github.com/tidyverse/dplyr/blob/master/R/rename.R
"""

from typing import Any, Callable, Sequence

from datar.apis.dplyr import group_vars, rename, rename_with

from ...pandas import DataFrame
from ...contexts import Context
from ...utils import vars_select
from ...tibble import TibbleGrouped
from .select import _eval_select


@rename.register(DataFrame, context=Context.SELECT, backend="pandas")
def _rename(_data: DataFrame, **kwargs: str) -> DataFrame:
    gvars = group_vars(
        _data,
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
    )
    all_columns = _data.columns
    selected, new_names = _eval_select(
        all_columns,
        _group_vars=gvars,
        _missing_gvars_inform=False,
        **kwargs,
    )
    rename_map = {} if new_names is None else new_names

    out = _data.copy()
    # new_names: old -> new
    # cannot do with duplicates
    # out.rename(columns=new_names, inplace=True)
    out.columns = [
        rename_map.get(col, col) if i in selected else col
        for i, col in enumerate(all_columns)
    ]

    if isinstance(out, TibbleGrouped):
        out._datar["group_vars"] = [rename_map.get(name, name) for name in gvars]
        out.regroup()

    return out


@rename_with.register(DataFrame, context=Context.SELECT, backend="pandas")
def _rename_with(
    _data: DataFrame,
    _fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> DataFrame:
    if not args:
        cols = _data.columns.tolist()
    else:
        cols = args[0]
        args = args[1:]

    if isinstance(cols, Sequence) and not isinstance(cols, str):
        selected = vars_select(_data.columns, *cols)
    else:
        selected = vars_select(_data.columns, cols)

    cols = _data.columns[selected]
    new_columns = {_fn(col, *args, **kwargs): col for col in cols}  # type: ignore
    return rename(
        _data,
        **new_columns,
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
    )

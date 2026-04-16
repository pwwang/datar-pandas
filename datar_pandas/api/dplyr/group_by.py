"""Group by verbs and functions
See source https://github.com/tidyverse/dplyr/blob/master/R/group-by.r
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union, cast

from datar.core.names import NameNonUniqueError
from datar.apis.dplyr import (
    mutate,
    group_by,
    ungroup,
    rowwise,
    group_by_drop_default,
)

from ...pandas import DataFrame, GroupBy, get_obj, Grouper
from ...tibble import Tibble, TibbleGrouped, TibbleRowwise
from ...contexts import Context
from ...utils import vars_select
from ...common import setdiff, union
from ..tibble.tibble import as_tibble
from .group_data import group_vars


@group_by.register(DataFrame, context=Context.PENDING, backend="pandas")
def _group_by(
    _data: DataFrame,
    *args: Any,
    _add: bool = False,  # not working, since _data is not grouped
    _drop: Optional[bool] = None,
    _sort: bool = False,
    _dropna: bool = False,
    **kwargs: Any,
) -> Union[Tibble, TibbleGrouped]:

    if _drop is None:
        _drop = group_by_drop_default(_data)

    if any([isinstance(arg, Grouper) for arg in args]):
        if len(args) + len(kwargs) > 1:
            raise ValueError(
                "When using `pandas.Grouper`, it must be the only argument "
                "to `group_by()`."
            )

        return Tibble(_data).group_by([args[0]], drop=_drop, sort=_sort, dropna=_dropna)

    _data = mutate(
        _data,
        *args,
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
        **kwargs,
    )
    _data.reset_index(drop=True, inplace=True)
    new_cols = _data._datar["mutated_cols"]
    if len(new_cols) == 0:
        return Tibble(_data, copy=False)

    return Tibble(_data, copy=False).group_by(
        new_cols,
        drop=_drop,
        sort=_sort,
        dropna=_dropna,
    )


@group_by.register(TibbleGrouped, context=Context.PENDING, backend="pandas")
def _group_by_grouped(
    _data: TibbleGrouped,
    *args: Any,
    _add: bool = False,
    _drop: Optional[bool] = None,
    _sort: bool = False,
    _dropna: bool = False,
    **kwargs: Any,
) -> TibbleGrouped:
    if any([isinstance(arg, Grouper) for arg in args]) or any(
        [isinstance(v, Grouper) for v in kwargs.values()]
    ):
        raise ValueError("Can't use `pandas.Grouper` when the data is already grouped.")

    if _drop is None:
        _drop = group_by_drop_default(_data)

    _data = mutate(
        _data,
        *args,
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
        **kwargs,
    )
    new_cols = _data._datar["mutated_cols"]
    gvars = (
        union(
            group_vars(
                _data,
                __ast_fallback="normal",  # type: ignore
                __backend="pandas",  # type: ignore
            ),
            new_cols,
        )
        if _add
        else new_cols
    )

    return group_by(
        Tibble(_data, copy=False),
        *gvars,
        _drop=_drop,
        _sort=_sort,  # type: ignore
        _dropna=_dropna,  # type: ignore
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
    )


@rowwise.register(DataFrame, context=Context.SELECT, backend="pandas")
def _rowwise(
    _data: DataFrame,
    *cols: Union[str, int],
) -> TibbleRowwise:
    if not _data.columns.is_unique:
        raise NameNonUniqueError("Cann't rowwise a data frame with duplicated names.")
    idxes = vars_select(_data.columns, *cols)
    gvars = _data.columns[idxes]
    return as_tibble(
        _data.reset_index(drop=True),
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
    ).rowwise(gvars)


@rowwise.register(TibbleGrouped, context=Context.SELECT, backend="pandas")
def _rowwise_grouped(
    _data: TibbleGrouped,
    *cols: Union[str, int],
) -> TibbleRowwise:
    # grouped dataframe's columns are unique already
    if cols:
        raise ValueError(
            "Can't re-group when creating rowwise data. "
            "Either first `ungroup()` or call `rowwise()` without arguments."
        )

    cols = tuple(cast(Tuple[Union[str, int], ...], _data.group_vars))
    return rowwise(
        _data._datar["grouped"].obj,
        *cols,
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
    )


@rowwise.register(TibbleRowwise, context=Context.SELECT, backend="pandas")
def _rowwise_rowwise(
    _data: TibbleRowwise,
    *cols: Union[str, int],
) -> TibbleRowwise:
    idxes = vars_select(_data.columns, *cols)
    gvars = _data.columns[idxes]
    return _data.rowwise(gvars)


@ungroup.register(object, context=Context.SELECT, backend="pandas")
def _ungroup(
    x: Any,
    *cols: Union[str, int],
) -> DataFrame:
    if cols:
        raise ValueError("`*cols` is not empty.")
    return x


@ungroup.register(TibbleGrouped, context=Context.SELECT, backend="pandas")
def _ungroup_grouped(
    x: TibbleGrouped,
    *cols: Union[str, int],
) -> Union[Tibble, TibbleGrouped]:
    obj = x._datar["grouped"].obj
    if not cols:
        return Tibble(obj)

    old_groups = group_vars(
        x,
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
    )
    to_remove = vars_select(obj.columns, *cols)
    new_groups = setdiff(old_groups, obj.columns[to_remove])

    return group_by(
        obj,
        *new_groups,
        __ast_fallback="normal",  # type: ignore
        __backend="pandas",  # type: ignore
    )


@ungroup.register(TibbleRowwise, context=Context.SELECT, backend="pandas")
def _ungroup_rowwise(
    x: TibbleRowwise,
    *cols: Union[str, int],
) -> DataFrame:
    if cols:
        raise ValueError("`*cols` is not empty.")
    return Tibble(x)


@ungroup.register(GroupBy, context=Context.SELECT, backend="pandas")
def _ungroup_groupby(
    x: GroupBy,
    *cols: Union[str, int],
) -> DataFrame:
    if cols:
        raise ValueError("`*cols` is not empty.")
    return get_obj(x)


@group_by_drop_default.register(DataFrame, backend="pandas")
def _group_by_drop_default(_tbl: DataFrame) -> bool:
    """Get the groupby _drop attribute of dataframe"""
    grouped = getattr(_tbl, "_datar", {}).get("grouped", None)
    if not grouped:
        return True
    return grouped.observed

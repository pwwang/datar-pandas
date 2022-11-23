"""Iterate over groups

https://github.com/tidyverse/dplyr/blob/master/R/group_split.R
https://github.com/tidyverse/dplyr/blob/master/R/group_map.R
https://github.com/tidyverse/dplyr/blob/master/R/group_trim.R
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from pipda import register_verb
from datar.core.utils import logger
from datar.apis.dplyr import (
    group_keys,
    group_rows,
    group_vars,
    group_by,
    ungroup,
    select,
    mutate,
    across,
    where,
    group_map,
    group_modify,
    group_walk,
    group_trim,
    with_groups,
    group_split,
)

from ... import pandas as pd
from ...typing import Data, Int, Str
from ...pandas import DataFrame
from ...contexts import Context
from ...tibble import TibbleGrouped, TibbleRowwise, reconstruct_tibble
from ...common import setdiff, intersect
from ..base.asis import is_factor
from ..base.factor import droplevels


def _nargs(fun):
    """Get the number of arguments of a function"""
    return len(inspect.signature(fun).parameters)


@group_map.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_map(
    _data: DataFrame,
    _f: Callable,
    *args: Any,
    _keep: bool = False,
    **kwargs: Any,
):
    keys = (
        group_keys(_data, __ast_fallback="normal", __backend="pandas")
        if _nargs(_f) > 1
        else None
    )
    for i, chunk in enumerate(
        group_split(
            _data,
            _keep=_keep,
            __ast_fallback="normal",
            __backend="pandas",
        )
    ):
        if keys is None:
            yield _f(chunk)
        else:
            yield _f(chunk, keys.iloc[[i], :], *args, **kwargs)


def _group_map_list(
    _data: DataFrame,
    _f: Callable,
    *args: Any,
    _keep: bool = False,
    **kwargs: Any,
):
    """List version of group_map"""
    return list(
        group_map(
            _data,
            _f,
            *args,
            **kwargs,
            _keep=_keep,
            __ast_fallback="normal",
            __backend="pandas",
        )
    )


group_map.list = register_verb(
    DataFrame,
    func=_group_map_list,
    context=Context.PENDING,
)


@group_modify.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_modify(
    _data: DataFrame,
    _f: Callable,
    *args: Any,
    _keep: bool = False,
    **kwargs: Any,
):
    return _f(_data, *args, **kwargs)


@group_modify.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _group_modify_grouped(
    _data: DataFrame,
    _f: Callable,
    *args: Any,
    _keep: bool = False,
    **kwargs: Any,
) -> DataFrame:
    gvars = group_vars(_data, __ast_fallback="normal", __backend="pandas")
    func = (lambda df, keys: _f(df)) if _nargs(_f) == 1 else _f

    def fun(df, keys):
        res = func(df, keys, *args, **kwargs)
        if not isinstance(res, DataFrame):
            raise ValueError("The result of `_f` should be a data frame.")
        bad = intersect(res.columns, gvars)
        if len(bad) > 0:
            raise ValueError(
                "The returned data frame cannot contain the original grouping "
                f"variables: {bad}."
            )

        return pd.concat(
            (
                keys.iloc[[0] * res.shape[0], :].reset_index(drop=True),
                res.reset_index(drop=True),
            ),
            axis=1,
        )

    chunks = group_map(
        _data,
        fun,
        _keep=_keep,
        __ast_fallback="normal",
        __backend="pandas",
    )
    out = pd.concat(chunks, axis=0)

    return reconstruct_tibble(out, _data)


@group_walk.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_walk(
    _data: DataFrame,
    _f: Callable,
    *args: Any,
    _keep: bool = False,
    **kwargs: Any,
) -> None:
    """Walk along data in each groups, but don't return anything"""
    # list to trigger generator
    list(
        group_map(
            _data,
            _f,
            *args,
            **kwargs,
            _keep=_keep,
            __ast_fallback="normal",
            __backend="pandas",
        )
    )


@group_trim.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_trim(_data: DataFrame, _drop: bool = None) -> DataFrame:
    return _data


@group_trim.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _group_trim_grouped(
    _data: TibbleGrouped,
    _drop: bool = None,
) -> TibbleGrouped:
    ungrouped = ungroup(_data, __ast_fallback="normal", __backend="pandas")

    fgroups = select(
        ungrouped,
        where(is_factor),
        __ast_fallback="normal",
        __backend="pandas",
    )
    dropped = mutate(
        ungrouped,
        across(
            fgroups.columns.tolist(),
            droplevels,
        ),
        __ast_fallback="normal",
        __backend="pandas",
    )

    return reconstruct_tibble(dropped, _data, drop=_drop)


@with_groups.register(DataFrame, context=Context.PENDING, backend="pandas")
def _with_groups(
    _data: DataFrame,
    _groups: Data[Int | Str],
    _func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _groups is None:
        grouped = ungroup(_data, __ast_fallback="normal", __backend="pandas")
    else:
        # all_columns = _data.columns
        # _groups = evaluate_expr(_groups, _data, Context.SELECT)
        # _groups = all_columns[vars_select(all_columns, _groups)]
        grouped = group_by(
            _data,
            _groups,
            __ast_fallback="normal",
            __backend="pandas",
        )

    return _func(grouped, *args, **kwargs)


@group_split.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_split(
    _data: DataFrame,
    *args: Any,
    _keep: bool = True,
    **kwargs: Any,
):
    data = group_by(
        _data,
        *args,
        **kwargs,
        __ast_fallback="normal",
        __backend="pandas",
    )
    yield from group_split_impl(data, _keep=_keep)


@group_split.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _group_split_grouped(
    _data: DataFrame,
    *args: Any,
    _keep: bool = True,
    **kwargs: Any,
):
    # data = group_by(_data, *args, **kwargs, _add=True)
    if args or kwargs:
        logger.warning(
            "`*args` and `**kwargs` are ignored in "
            "`group_split(<TibbleGrouped>)`, please use "
            "`group_by(..., _add=True) >> group_split()`."
        )
    return group_split_impl(_data, _keep=_keep)


@group_split.register(TibbleRowwise, context=Context.EVAL, backend="pandas")
def _group_split_rowwise(
    _data: DataFrame,
    *args: Any,
    _keep: bool = True,
    **kwargs: Any,
):
    if args or kwargs:
        logger.warning(
            "`*args` and `**kwargs` is ignored in "
            "`group_split(<TibbleRowwise>)`."
        )
    if _keep is not None:
        logger.warning(
            "`_keep` is ignored in " "`group_split(<TibbleRowwise>)`."
        )

    return group_split_impl(_data, _keep=True)


def _group_split_list(
    _data: DataFrame,
    *args: Any,
    _keep: bool = True,
    **kwargs: Any,
):
    """List version of group_split"""
    return list(
        group_split(
            _data,
            *args,
            _keep=_keep,
            __ast_fallback="normal",
            __backend="pandas",
            **kwargs,
        )
    )


group_split.list = register_verb(
    DataFrame,
    func=_group_split_list,
    context=Context.PENDING,
)


def group_split_impl(data, _keep):
    """Implement splitting data frame by groups"""
    out = ungroup(data, __ast_fallback="normal", __backend="pandas")
    indices = group_rows(data, __ast_fallback="normal", __backend="pandas")

    if not _keep:
        remove = group_vars(data, __ast_fallback="normal", __backend="pandas")
        _keep = out.columns
        _keep = setdiff(_keep, remove)
        out = out[_keep]

    for rows in indices:
        yield out.iloc[rows, :].reset_index(drop=True)

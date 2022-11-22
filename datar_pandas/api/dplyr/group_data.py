from __future__ import annotations

from typing import Sequence, List

from datar.apis.dplyr import (
    group_data,
    group_keys,
    group_rows,
    group_indices,
    group_vars,
    group_size,
    n_groups,
)

from ...utils import dict_get
from ...contexts import Context
from ...tibble import Tibble, TibbleGrouped, TibbleRowwise
from ...pandas import DataFrame, GroupBy


@group_data.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_data(_data: DataFrame) -> Tibble:
    return Tibble(
        {
            "_rows": group_rows(
                _data,
                __ast_fallback="normal",
                __backend="pandas",
            ),
        }
    )


@group_data.register(
    (TibbleGrouped, GroupBy),
    context=Context.EVAL,
    backend="pandas",
)
def _group_data_grouped(_data: TibbleGrouped | GroupBy) -> Tibble:
    gpdata = group_keys(_data, __ast_fallback="normal", __backend="pandas")
    gpdata["_rows"] = group_rows(
        _data,
        __ast_fallback="normal",
        __backend="pandas",
    )
    return gpdata


@group_keys.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_keys(_data: DataFrame) -> Tibble:
    return Tibble(index=[0])


@group_keys.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _group_keys_grouped(_data: TibbleGrouped) -> Tibble:
    grouper = _data._datar["grouped"].grouper
    return Tibble(grouper.result_index.to_frame(index=False), copy=False)


@group_keys.register(TibbleRowwise, context=Context.EVAL, backend="pandas")
def _group_keys_rowwise(_data: TibbleRowwise) -> Tibble:
    return Tibble(_data.loc[:, _data.group_vars])


@group_rows.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_rows(_data: DataFrame) -> List[List[int]]:
    rows = list(range(_data.shape[0]))
    return [rows]


@group_rows.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _group_rows_grouped(_data: TibbleGrouped) -> List[List[int]]:
    return group_rows(
        _data._datar["grouped"],
        __ast_fallback="normal",
        __backend="pandas",
    )


@group_rows.register(GroupBy, context=Context.EVAL, backend="pandas")
def _group_rows_groupby(_data: GroupBy) -> List[List[int]]:
    grouper = _data.grouper
    return [
        list(dict_get(grouper.indices, group_key))
        for group_key in grouper.result_index
    ]


@group_indices.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_indices(_data: DataFrame) -> List[int]:
    return [0] * _data.shape[0]


@group_indices.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _group_indices_gruoped(_data: TibbleGrouped) -> List[int]:
    ret = {}
    for row in group_data(
        _data,
        __ast_fallback="normal",
        __backend="pandas",
    ).itertuples():
        for index in row[-1]:
            ret[index] = row.Index
    return [ret[key] for key in sorted(ret)]


@group_vars.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_vars(_data: DataFrame) -> Sequence[str]:
    return getattr(_data, "group_vars", [])


@group_size.register(DataFrame, context=Context.EVAL, backend="pandas")
def _group_size(_data: DataFrame) -> Sequence[int]:
    """Gives the size of each group"""
    return [_data.shape[0]]


@group_size.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _group_size_grouped(_data: TibbleGrouped) -> Sequence[int]:
    return list(
        map(
            len,
            group_rows(_data, __ast_fallback="normal", __backend="pandas"),
        )
    )


@n_groups.register(DataFrame, context=Context.EVAL, backend="pandas")
def _n_groups(_data: DataFrame) -> int:
    """Gives the total number of groups."""
    return 1


@n_groups.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _n_groups_grouped(_data: TibbleGrouped) -> int:
    return _data._datar["grouped"].ngroups


@n_groups.register(TibbleRowwise, context=Context.EVAL, backend="pandas")
def _n_groups_rowwise(_data: TibbleRowwise) -> int:
    return _data.shape[0]

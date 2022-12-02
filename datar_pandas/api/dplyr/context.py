"""Context dependent expressions

See souce https://github.com/tidyverse/dplyr/blob/master/R/context.R
"""
import numpy as np
from datar.apis.dplyr import (
    group_data,
    group_keys,
    n,
    cur_data_all,
    cur_data,
    cur_group,
    cur_group_id,
    cur_group_rows,
    cur_column,
)

from ...typing import Data, Int
from ...common import setdiff
from ...contexts import Context
from ...pandas import DataFrame, Series, get_obj
from ...tibble import Tibble, TibbleGrouped
from ...middlewares import CurColumn
from ...utils import dict_get


# n used directly in count
# @register_func(DataFrame, verb_arg_only=True)
@n.register(DataFrame, context=Context.EVAL, backend="pandas")
def _n(_data: DataFrame) -> int:
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    return _data.shape[0]


@n.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _n_grouped(_data: TibbleGrouped) -> Data[Int]:
    _data = _data._datar.get("summarise_source", _data)
    grouped = _data._datar["grouped"]

    out = grouped.grouper.size().to_frame().reset_index()
    out = out.groupby(
        grouped.grouper.names,
        sort=grouped.sort,
        observed=grouped.observed,
        dropna=grouped.dropna,
    )[0]

    return out


@cur_data_all.register(DataFrame, context=Context.EVAL, backend="pandas")
def _cur_data_all(_data: DataFrame) -> Series:
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    return Series([_data.copy()], dtype=object)


@cur_data_all.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _cur_data_all_grouped(_data: TibbleGrouped) -> Series:
    _data = _data._datar.get("summarise_source", _data)
    grouped = _data._datar["grouped"]
    return Series(
        [
            get_obj(grouped).loc[dict_get(grouped.grouper.groups, key), :]
            for key in grouped.grouper.result_index
        ],
        name="cur_data_all",
        dtype=object,
        index=grouped.grouper.result_index,
    )


@cur_data.register(DataFrame, context=Context.EVAL, backend="pandas")
def _cur_data(_data: DataFrame) -> Series:
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    cols = setdiff(_data.columns, _data.group_vars or [])
    return Series([_data[cols]], dtype=object)


@cur_data.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _cur_data_grouped(_data: TibbleGrouped) -> Series:
    _data = _data._datar.get("summarise_source", _data)
    cols = setdiff(_data.columns, _data.group_vars or [])
    return (
        _data._datar["grouped"].apply(lambda g: Series([g[cols]])).iloc[:, 0]
    )


@cur_group.register(DataFrame, context=Context.EVAL, backend="pandas")
def _cur_group(_data: DataFrame) -> Tibble:
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    return Tibble(index=[0])


@cur_group.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _cur_group_grouped(_data: TibbleGrouped) -> Series:
    _data = _data._datar.get("summarise_source", _data)
    out = group_keys(_data, __ast_fallback="normal", __backend="pandas")
    # split each row as a df
    out = out.apply(lambda row: row.to_frame().T, axis=1)
    out.index = _data._datar["grouped"].grouper.result_index
    return out


@cur_group_id.register(DataFrame, context=Context.EVAL, backend="pandas")
def _cur_group_id(_data: DataFrame) -> int:
    return 0


@cur_group_id.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _cur_group_id_grouped(_data: TibbleGrouped) -> Series:
    _data = _data._datar.get("summarise_source", _data)
    grouper = _data._datar["grouped"].grouper
    return Series(np.arange(grouper.ngroups), index=grouper.result_index)


@cur_group_rows.register(DataFrame, context=Context.EVAL, backend="pandas")
def _cur_group_rows(_data: DataFrame) -> np.ndarray:
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    gdata = group_data(_data, __ast_fallback="normal", __backend="pandas")
    if isinstance(_data, TibbleGrouped):
        return gdata.set_index(_data.group_vars)["_rows"]

    return gdata["_rows"]


@cur_column.register(backend="pandas")
def _cur_column() -> CurColumn:
    return CurColumn()

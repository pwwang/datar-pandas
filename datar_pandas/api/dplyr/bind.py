"""Bind multiple data frames by row and column

See https://github.com/tidyverse/dplyr/blob/master/R/bind.r
"""
from __future__ import annotations
from typing import Any, Callable

from datar.core.utils import logger
from datar.core.names import repair_names
from datar.apis.dplyr import bind_rows, bind_cols

from ... import pandas as pd
from ...pandas import DataFrame, Categorical, union_categoricals
from ...common import is_factor, is_scalar, is_null
from ...tibble import Tibble, TibbleGrouped, reconstruct_tibble


def _construct_tibble(data):
    if not isinstance(data, dict):
        return Tibble(data, copy=False)

    data = data.copy()
    for key, val in data.items():
        data[key] = [val] if is_scalar(val) else val

    return Tibble(data, copy=False)


@bind_rows.register((DataFrame, list, dict, type(None)), backend="pandas")
def _bind_rows(
    *datas: DataFrame | list | dict | None,
    _id: str = None,
    _copy: bool = True,
    **kwargs: Any,
) -> DataFrame:
    if _id is not None and not isinstance(_id, str):
        raise ValueError("`_id` must be a scalar string.")

    if not datas:
        _data = None
    else:
        _data, datas = datas[0], datas[1:]

    key_data = {}
    if isinstance(_data, list):
        _data = [d for d in _data if d is not None]
        for i, dat in enumerate(_data):
            key_data[i] = _construct_tibble(dat)
    elif _data is not None:
        key_data[0] = _construct_tibble(_data)

    for i, dat in enumerate(datas):
        if isinstance(dat, list):
            for df in dat:
                key_data[len(key_data)] = _construct_tibble(df)
        elif dat is not None:
            key_data[len(key_data)] = _construct_tibble(dat)

    for key, val in kwargs.items():
        if val is not None:
            key_data[key] = _construct_tibble(val)

    if not key_data:
        return Tibble()

    # handle categorical data
    for col in list(key_data.values())[0].columns:
        all_series = [
            dat[col]
            for dat in key_data.values()
            if col in dat and not dat[col].isna().all()
        ]
        all_categorical = [
            is_factor(ser) or is_null(ser).all()
            for ser in all_series
        ]
        if all(all_categorical):
            union_cat = union_categoricals(all_series)
            for data in key_data.values():
                if col not in data:  # in case it is 0-column df
                    continue
                data[col] = Categorical(
                    data[col],
                    categories=union_cat.categories,
                    ordered=is_factor(data[col])
                    and data[col].cat.ordered,
                )
        elif any(all_categorical):
            logger.warning("Factor information lost during rows binding.")

    if _id is not None:
        return (
            pd.concat(
                key_data.values(),
                keys=key_data.keys(),
                names=[_id, None],
                copy=_copy,
            )
            .reset_index(level=0)
            .reset_index(drop=True)
        )

    to_concat = [
        kdata
        for kdata in
        key_data.values()
        if kdata.shape[0] > 0
    ]
    if not to_concat:
        return key_data[0].loc[[], :]

    return pd.concat(to_concat, copy=_copy).reset_index(drop=True)


@bind_rows.register(TibbleGrouped, backend="pandas")
def _bind_rows_grouped(
    *datas: Any,
    _id: str = None,
    **kwargs: Any,
) -> TibbleGrouped:
    grouped = [data for data in datas if isinstance(data, TibbleGrouped)]
    grouped = grouped[0]
    out = bind_rows.dispatch(DataFrame, backend="pandas")(
        *datas,
        _id=_id,
        **kwargs,
    )
    return reconstruct_tibble(out, grouped)


@bind_cols.register((DataFrame, dict, type(None)), backend="pandas")
def _bind_cols(
    *datas: DataFrame | dict | None,
    _name_repair: str | Callable = "unique",
    _copy=True,
) -> DataFrame:
    ds = [
        Tibble.from_args(**d)
        if isinstance(d, dict)
        else d
        for d in datas
        if d is not None
    ]

    if not ds:
        return Tibble()

    ret = pd.concat(ds, axis=1, copy=_copy)
    ret.columns = repair_names(ret.columns.tolist(), repair=_name_repair)
    return ret

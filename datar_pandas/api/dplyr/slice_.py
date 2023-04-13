"""Subset rows using their positions

https://github.com/tidyverse/dplyr/blob/master/R/slice.R
"""
from __future__ import annotations

import builtins
from math import ceil, floor
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Union

import numpy as np
from pipda import Expression
from datar.core.utils import logger
from datar.apis.dplyr import (
    slice_,
    slice_head,
    slice_tail,
    slice_sample,
    slice_min,
    slice_max,
)

from ... import pandas as pd
from ...pandas import DataFrame, SeriesGroupBy, get_obj
from ...common import is_integer, is_scalar
from ...collections import Collection
from ...broadcast import _ungroup
from ...contexts import Context
from ...utils import dict_get
from ...tibble import Tibble, TibbleGrouped, TibbleRowwise
from ..base.seq import c_

if TYPE_CHECKING:
    from ...pandas import Index


@slice_.register(DataFrame, context=Context.SELECT, backend="pandas")
def _slice(
    _data: DataFrame,
    *rows: Union[int, str],
    _preserve: bool = False,
) -> Tibble:
    # if _preserve:
    #     logger.warning("`slice()` doesn't support `_preserve` argument yet.")

    if not rows:
        return _data.copy()

    rows = _sanitize_rows(rows, _data.shape[0])
    return _data.take(rows)


@slice_.register(TibbleGrouped, context=Context.SELECT, backend="pandas")
def _slice_grouped(
    _data: TibbleGrouped,
    *rows: Any,
    _preserve: bool = False,
) -> TibbleGrouped:
    """Slice on grouped dataframe"""
    if _preserve:
        logger.warning("`slice()` doesn't support `_preserve` argument yet.")

    grouped = _data._datar["grouped"]
    indices = _sanitize_rows(
        rows,
        grouped.grouper.indices,
        grouped.grouper.result_index,
    )

    return _data.take(indices)


@slice_head.register(DataFrame, context=Context.EVAL, backend="pandas")
def _slice_head(
    _data: DataFrame,
    n: int = None,
    prop: float = None,
) -> Tibble:
    n = _n_from_prop(_data.shape[0], n, prop)
    return slice_(
        _data,
        builtins.slice(None, n),
        __ast_fallback="normal",
        __backend="pandas",
    )


@slice_head.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _slice_head_grouped(
    _data: DataFrame,
    n: int = None,
    prop: float = None,
) -> TibbleGrouped:
    """Slice on grouped dataframe"""
    grouped = _data._datar["grouped"]
    # Calculate n's of each group
    ns = grouped.grouper.size().transform(lambda x: _n_from_prop(x, n, prop))
    # Get indices of each group
    # A better way?
    indices = np.concatenate(
        [
            grouped.grouper.indices[key][: ns[key]]
            for key in grouped.grouper.result_index
        ]
    )

    return _data.take(indices)


@slice_head.register(TibbleRowwise, context=Context.EVAL, backend="pandas")
def _slice_head_rowwise(
    _data: TibbleRowwise,
    n: int = None,
    prop: float = None,
) -> TibbleRowwise:
    """Slice on grouped dataframe"""
    n = _n_from_prop(1, n, prop)

    if n >= 1:
        return _data.copy()

    return _data.take([])


@slice_tail.register(DataFrame, context=Context.EVAL, backend="pandas")
def _slice_tail(
    _data: DataFrame,
    n: int = 1,
    prop: float = None,
) -> Tibble:
    n = _n_from_prop(_data.shape[0], n, prop)
    return slice_(
        _data,
        builtins.slice(
            _data.shape[0] if n == 0 else -n,
            None,
        ),
        __ast_fallback="normal",
        __backend="pandas",
    )


@slice_tail.register(TibbleGrouped, context=Context.EVAL, backend="pandas")
def _slice_tail_grouped(
    _data: DataFrame,
    n: int = None,
    prop: float = None,
) -> TibbleGrouped:
    grouped = _data._datar["grouped"]
    # Calculate n's of each group
    ns = grouped.grouper.size().transform(lambda x: _n_from_prop(x, n, prop))
    # Get indices of each group
    # A better way?
    indices = np.concatenate(
        [
            grouped.grouper.indices[key][-ns[key]:]
            for key in grouped.grouper.result_index
        ]
    )

    return _data.take(indices)


@slice_tail.register(TibbleRowwise, context=Context.PENDING, backend="pandas")
def _slice_tail_rowwise(
    _data: TibbleRowwise,
    n: int = None,
    prop: float = None,
) -> TibbleRowwise:
    """Slice on grouped dataframe"""
    return slice_head(
        _data,
        n=n,
        prop=prop,
        __ast_fallback="normal",
        __backend="pandas",
    )


@slice_min.register(DataFrame, context=Context.EVAL, backend="pandas")
def _slice_min(
    _data: DataFrame,
    order_by: Expression,
    n: int = 1,
    prop: float = None,
    with_ties: Union[bool, str] = True,
) -> Tibble:
    if isinstance(_data, TibbleGrouped) and prop is not None:
        raise ValueError(
            "`slice_min()` doesn't support `prop` for grouped data yet."
        )

    if isinstance(_data, TibbleRowwise):
        n = _n_from_prop(1, n, prop)
    else:
        n = _n_from_prop(_data.shape[0], n, prop)

    sliced = order_by.nsmallest(n, keep="all" if with_ties else "first")
    sliced = sliced[~pd.isnull(sliced)]
    return _data.reindex(sliced.index.get_level_values(-1))


@slice_max.register(DataFrame, context=Context.EVAL, backend="pandas")
def _slice_max(
    _data: DataFrame,
    order_by: Iterable[Any],
    n: int = 1,
    prop: float = None,
    with_ties: Union[bool, str] = True,
) -> DataFrame:
    if isinstance(_data, TibbleGrouped) and prop is not None:
        raise ValueError(
            "`slice_max()` doesn't support `prop` for grouped data yet."
        )

    if isinstance(_data, TibbleRowwise):
        n = _n_from_prop(1, n, prop)
    else:
        n = _n_from_prop(_data.shape[0], n, prop)

    sliced = order_by.nlargest(n, keep="all" if with_ties else "first")
    sliced = sliced[~pd.isnull(sliced)]
    return _data.reindex(sliced.index.get_level_values(-1))


@slice_sample.register(DataFrame, context=Context.EVAL, backend="pandas")
def _slice_sample(
    _data: DataFrame,
    n: int = 1,
    prop: float = None,
    weight_by: Iterable[Union[int, float]] = None,
    replace: bool = False,
    random_state: Any = None,
) -> DataFrame:
    if (
        prop is not None
        and isinstance(_data, TibbleGrouped)
        and not isinstance(_data, TibbleRowwise)
    ):
        raise ValueError(
            "`slice_sample()` doesn't support `prop` for grouped data yet."
        )

    if isinstance(_data, TibbleRowwise):
        n = _n_from_prop(1, n, prop)
    else:
        n = _n_from_prop(_data.shape[0], n, prop)

    if n == 0:
        # otherwise _data.sample raises error when weight_by is empty as well
        return _data.take([])

    return _data.sample(
        n=n,
        replace=replace,
        weights=_ungroup(weight_by),
        random_state=random_state,
    )


def _n_from_prop(
    total: int,
    n: int | float = None,
    prop: float = None,
) -> int:
    """Get n from a proportion"""
    if n is None and prop is None:
        return 1
    if n is not None and not isinstance(n, (int, float)):
        raise TypeError(f"Expect `n` a number, got {type(n)}.")
    if prop is not None and not isinstance(prop, (int, float)):
        raise TypeError(f"Expect `prop` a number, got {type(n)}.")
    # if (n is not None and n < 0) or (prop is not None and prop < 0):
    #     raise ValueError("`n` and `prop` should not be negative.")
    if prop is not None:
        if prop < 0:
            return max(ceil((1.0 + prop) * total), 0)
        return floor(float(total) * min(prop, 1.0))

    if n < 0:
        return max(ceil(total + n), 0)
    return min(floor(n), total)


def _sanitize_rows(
    rows: Iterable,
    indices: Union[int, Mapping] = None,
    result_index: "Index" = None,
) -> np.ndarray:
    """Sanitize rows passed to slice"""

    if is_scalar(indices) and is_integer(indices):
        rows = Collection(*rows, pool=indices)
        if rows.error:
            raise rows.error from None
        return np.array(rows, dtype=int)

    out = []
    if any(isinstance(row, SeriesGroupBy) for row in rows):
        rows = c_(*rows)
        for key in result_index:
            idx = dict_get(indices, key)
            if idx.size == 0:
                continue

            gidx = dict_get(rows.grouper.indices, key)
            out.extend(idx.take(get_obj(rows).take(gidx)))
    else:
        for key in result_index:
            idx = dict_get(indices, key)
            if idx.size == 0:
                continue
            grows = Collection(*rows, pool=idx.size)
            if grows.error:
                raise grows.error from None
            out.extend(idx.take(grows))

    return np.array(out, dtype=int)

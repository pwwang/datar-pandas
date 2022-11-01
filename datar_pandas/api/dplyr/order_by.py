"""Helper functions for ordering window function output

https://github.com/tidyverse/dplyr/blob/master/R/order-by.R
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
from functools import singledispatch

import numpy as np
from pipda import FunctionCall
from datar.apis.base import order as order_fun
from datar.apis.dplyr import order_by, with_order

from ..other import itemgetter
from ...factory import func_bootstrap
from ...pandas import Series, PandasObject


@order_by.impl
def _order_by(order: Sequence, call: FunctionCall):
    order = order_fun(order)
    if not isinstance(call, FunctionCall) or len(call._pipda_args) < 1:
        raise ValueError(
            "In `order_by()`: `call` must be a registered "
            f"function call with data, not `{type(call).__name__}`. \n"
            "            This function should be called as an argument "
            "of a verb. If you want to call it regularly, try `with_order()`"
        )

    x = itemgetter(call._pipda_args[0], order)
    call._pipda_args = (x, *call._pipda_args[1:])
    return itemgetter(call, order)


@singledispatch
def _with_order(seq, order):
    return [seq[i] for i in order]


@_with_order.register(np.ndarray)
def _(seq, order):
    return seq.take(order)


@_with_order.register(Series)
def _(seq, order):
    out = seq.take(order)
    out.index = seq.index
    return out


def _with_order_post(__out, order, func, x, *args, __args_raw=None, **kwargs):
    """Keep the raw values if input is not Series-alike"""
    if (
        not isinstance(__args_raw["x"], PandasObject)
        and isinstance(__out, Series)
    ):
        return __out.values

    return __out


@func_bootstrap(with_order, data_args={'order', 'x'}, post=_with_order_post)
def _with_order(
    order: Any,
    func: Callable,
    x: Any,
    *args: Any,
    __args_raw: Mapping[str, Any] = None,
    **kwargs: Any,
) -> Sequence:
    order = order.data if isinstance(order, PandasData) else order
    order = order_fun(order, __ast_fallback="normal")

    x = _with_order(x, order)
    out = func(x, *args, **kwargs)
    out = _with_order(out, order)

    return out

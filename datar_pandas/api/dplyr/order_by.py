"""Helper functions for ordering window function output

https://github.com/tidyverse/dplyr/blob/master/R/order-by.R
"""
from __future__ import annotations

from typing import Any, Callable
from functools import singledispatch

import numpy as np
from pipda import FunctionCall
from datar.apis.dplyr import order_by, with_order

from ...factory import func_bootstrap
from ...pandas import Series
from ..base.seq import order as order_fun
from ..other import itemgetter


@order_by.register(backend="pandas")
def _order_by(order, call: FunctionCall):
    order = order_fun(order, __ast_fallback="normal")
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


@with_order.register(object, backend="pandas")
def _with_order_obj(
    order: Any,
    func: Callable,
    x: Any,
    *args: Any,
    **kwargs: Any,
):
    order = order_fun(order, __ast_fallback="normal")

    x = _with_order(x, order)
    out = func(x, *args, **kwargs)
    out = _with_order(out, order)

    return out


@func_bootstrap(with_order, exclude={"func", "args", "kwargs"})
def _with_order_bootstrap(
    order: Any,
    func: Callable,
    x: Any,
    *args: Any,
    **kwargs: Any,
):
    order = order_fun(order, __ast_fallback="normal")

    x = _with_order(x, order)
    out = func(x, *args, **kwargs)
    out = _with_order(out, order)

    return out

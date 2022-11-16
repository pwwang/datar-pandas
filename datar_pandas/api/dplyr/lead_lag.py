"""Compute lagged or leading values

https://github.com/tidyverse/dplyr/blob/master/R/lead-lag.R
"""
import numpy as np
from datar.apis.dplyr import with_order, lead, lag

from ...pandas import Series
from ...common import is_scalar
from ...utils import as_series
from ...factory import func_bootstrap


def _shift(x, n, default=None, order_by=None):
    if not isinstance(n, int):
        raise ValueError("`lead-lag` expect an integer for `n`.")

    if not is_scalar(default) and len(default) > 1:
        raise ValueError("`lead-lag` Expect scalar or length-1 `default`.")

    if not is_scalar(default):
        default = default[0]

    if order_by is not None:
        # newx = newx.reset_index(drop=True)
        out = with_order(order_by, Series.shift, x, n, fill_value=default)
    else:
        out = x.shift(n, fill_value=default)

    return out


@lead.register(object, backend="pandas")
def _lead_obj(x, n=1, default=np.nan, order_by=None):
    return _shift(as_series(x), n=-n, default=default, order_by=order_by)


@func_bootstrap(lead, kind="transform")
def _lead(x, n=1, default=np.nan, order_by=None):
    """Find next values in a vector

    Args:
        series: Vector of values
        n: Positive integer of length 1, giving the number of positions to
            lead or lag by
        default: Value used for non-existent rows.
        order_by: Override the default ordering to use another vector or column

    Returns:
        Lead or lag values with default values filled to series.
    """
    return _shift(x, n=-n, default=default, order_by=order_by)


@lag.register(object, backend="pandas")
def _lag_obj(x, n=1, default=np.nan, order_by=None):
    return _shift(as_series(x), n=n, default=default, order_by=order_by)


@func_bootstrap(lag, kind="transform")
def _lag(x, n=1, default=np.nan, order_by=None):
    """Find previous values in a vector

    See lead()
    """
    return _shift(x, n=n, default=default, order_by=order_by)

"""Windowed rank functions.

See https://github.com/tidyverse/dplyr/blob/master/R/rank.R
"""
from datar.apis.dplyr import (
    row_number_,
    ntile_,
    min_rank_,
    dense_rank_,
    percent_rank_,
    cume_dist_,
)

from ._rank import (
    _row_number,
    _ntile,
    _rank,
    _percent_rank,
    _cume_dist,
)


@row_number_.register(object, backend="pandas")
def _row_number_registered(x):
    return _row_number(x)


@ntile_.register(object, backend="pandas")
def _ntile_registered(x, *, n=None):
    return _ntile(x, n)


@min_rank_.register(object, backend="pandas")
def _min_rank_registered(x, *, na_last="keep"):
    return _rank(x, na_last=na_last, method="min")


@dense_rank_.register(object, backend="pandas")
def _dense_rank_registered(x, *, na_last="keep"):
    return _rank(x, na_last=na_last, method="dense")


@percent_rank_.register(object, backend="pandas")
def _percent_rank_registered(x, *, na_last="keep"):
    return _percent_rank(x, na_last)


@cume_dist_.register(object, backend="pandas")
def _cume_dist_registered(x, *, na_last="keep"):
    return _cume_dist(x, na_last)

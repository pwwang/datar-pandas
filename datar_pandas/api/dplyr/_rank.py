from types import GeneratorType
from functools import singledispatch
from typing import cast

import numpy as np

from ... import pandas as pd
from ...utils import get_grouper
from ...pandas import (
    Categorical,
    DataFrame,
    Series,
    GroupBy,
    SeriesGroupBy,
    NDFrame,
    get_obj,
)
from ...common import is_scalar
from ...tibble import TibbleGrouped


@singledispatch
def _rank(
    data,
    na_last,
    method,
    percent=False,
):
    """Rank the data"""
    is_series = isinstance(data, Series)

    if not is_series:
        data = Series(data)

    out = data.rank(
        method=method,
        pct=percent,
        na_option=("keep" if na_last == "keep" else "top" if not na_last else "bottom"),
    )

    return out if is_series else out.values


@_rank.register(GroupBy)
def _(
    data: GroupBy,
    na_last,
    method,
    percent=False,
):
    out = data.rank(
        method=method,
        pct=percent,
        na_option=("keep" if na_last == "keep" else "top" if not na_last else "bottom"),
    )
    return out


@singledispatch
def _row_number(x):
    return _rank(x, na_last="keep", method="first")


@_row_number.register(SeriesGroupBy)
def _(x):
    out = x.transform(_row_number)
    return out.groupby(
        get_grouper(x),
        observed=x.observed,
        sort=x.sort,
        dropna=x.dropna,
    )


@_row_number.register(TibbleGrouped)
def _(x):
    grouped = x._datar["grouped"]
    return _row_number(
        Series(np.arange(x.shape[0]), index=x.index).groupby(
            get_grouper(grouped),
            observed=grouped.observed,
            sort=grouped.sort,
            dropna=grouped.dropna,
        )
    )


@_row_number.register(NDFrame)
def _(x):
    if x.ndim > 1:
        if x.shape[1] == 0:
            return np.array([], dtype=int)
        x = Series(np.arange(x.shape[0]), index=x.index)
    return _rank(x, na_last="keep", method="first")


@singledispatch
def _ntile(x, n):
    if is_scalar(x):
        x = [x]

    return _ntile(np.array(x), n)


def _ntile_array_values(x, n):
    """Compute ntile labels with R-like large-group-first behavior."""
    if x.size == 0:
        return np.array([], dtype=float)

    na_mask = pd.isnull(x)
    if na_mask.all():
        return np.array([np.nan] * x.size)

    non_na_count = (~na_mask).sum()
    n = min(n, non_na_count)
    ranked = Series(x).rank(method="min", na_option="keep").values

    out = np.array([np.nan] * x.size)
    out[~na_mask] = np.floor((ranked[~na_mask] - 1) * n / non_na_count) + 1
    return out


@_ntile.register(GeneratorType)
def _ntile_generator(x, n):
    return _ntile(np.array(list(x)), n)


@_ntile.register(Series)
def _ntile_series(x, n):
    return Series(_ntile(x.values, n), index=x.index, name=x.name)


@_ntile.register(DataFrame)
def _ntile_df(x, n):
    x = _row_number(x)
    return _ntile(x, n)


@_ntile.register(TibbleGrouped)
def _ntile_tibblegrouped(x, n):
    grouped = x._datar["grouped"]
    if x.shape[1] == 0:  # pragma: no cover
        x = _row_number(grouped)
    else:
        # Not using x.iloc[:, 0] to preserve the grouping structure
        x = x[x.columns[0]]
    return _ntile(x, n)


@_ntile.register(np.ndarray)
def _ntile_ndarray(x, n):
    out = _ntile_array_values(x, n)
    if out.size == 0:
        return Categorical([])

    labels = out[~pd.isnull(out)].astype(int)
    categories = np.arange(labels.max()) + 1 if labels.size else []
    return Categorical(
        labels if labels.size == out.size else out, categories=categories
    )


@_ntile.register(GroupBy)
def _ntile_groupby(x, n):
    return x.transform(lambda grup: _ntile_array_values(grup.values, n))


@singledispatch
def _percent_rank(x, na_last="keep"):
    if len(x) == 0:
        dtype = getattr(x, "dtype", None)
        return np.array(x, dtype=dtype)

    return np.asarray(_percent_rank(Series(x), na_last))


@_percent_rank.register(NDFrame)
def _(x, na_last="keep"):
    ranking = Series(_rank(x, na_last, "min", True), index=x.index)
    minrank = ranking.min()
    maxrank = ranking.max()

    ret = (ranking - minrank) / (maxrank - minrank)
    ret[pd.isnull(ranking)] = np.nan
    return ret


@_percent_rank.register(GroupBy)
def _(x, na_last="keep"):
    ranking = cast(Series, _rank(x, na_last, "min", True)).groupby(
        get_grouper(x),
        observed=x.observed,
        sort=x.sort,
        dropna=x.dropna,
    )
    maxs = ranking.transform("max")
    mins = ranking.transform("min")
    ret = ranking.transform(lambda r: (r - mins) / (maxs - mins))
    ret[get_obj(ranking).isna()] = np.nan
    return ret


@singledispatch
def _cume_dist(x, na_last="keep"):
    if is_scalar(x):
        x = [x]

    if len(x) == 0:
        dtype = getattr(x, "dtype", None)
        return np.array(x, dtype=dtype)

    return np.asarray(_cume_dist(Series(x), na_last))


@_cume_dist.register(NDFrame)
def _(x, na_last="keep"):
    ranking = Series(_rank(x, na_last, "min"), index=x.index)
    total = (~pd.isnull(ranking)).sum()
    ret = ranking.transform(lambda r: ranking.le(r).sum() / total)
    ret[pd.isnull(ranking)] = np.nan
    return ret


@_cume_dist.register(GroupBy)
def _(x, na_last="keep"):
    return x.transform(_cume_dist.dispatch(Series), na_last=na_last)

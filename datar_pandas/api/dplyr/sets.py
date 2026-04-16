"""Set operations

https://github.com/tidyverse/dplyr/blob/master/R/sets.r
"""

from typing import Any, cast

import numpy as np
from datar.apis.dplyr import (
    ungroup,
    bind_rows,
    intersect,
    union,
    setdiff,
    union_all,
    setequal,
    symdiff,
)

from ... import pandas as pd
from ...utils import meta_kwargs
from ...broadcast import get_grouper, _grouper_compatible
from ...pandas import DataFrame, SeriesGroupBy
from ...common import (
    setdiff as _setdiff,
    union as _union,
    intersect as _intersect,
)
from ...tibble import TibbleGrouped, reconstruct_tibble

META_KWARGS = cast(Any, meta_kwargs)


def _check_xy(x, y):
    """Check the dimension and columns of x and y for set operations"""
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "not compatible:\n"
            f"- different number of columns: {x.shape[1]} vs {y.shape[1]}"
        )

    in_y_not_x = _setdiff(y.columns, x.columns)
    in_x_not_y = _setdiff(x.columns, y.columns)
    if in_y_not_x.size > 0 or in_x_not_y.size > 0:
        msg = ["not compatible:"]
        if in_y_not_x:
            msg.append(f"- Cols in `y` but not `x`: {in_y_not_x}.")
        if in_x_not_y:
            msg.append(f"- Cols in `x` but not `y`: {in_x_not_y}.")
        raise ValueError("\n".join(msg))


@intersect.register(DataFrame, backend="pandas")
def _intersect_df(x: DataFrame, y: DataFrame) -> DataFrame:
    """Intersect of two dataframes

    Args:
        _data, data2, *datas: Dataframes to perform operations
        on: The columns to the dataframes to perform operations on

    Returns:
        The dataframe of intersect of input dataframes
    """
    _check_xy(x, y)
    from .distinct import distinct

    out = distinct(
        pd.merge(x, ungroup(y, **META_KWARGS), how="inner"),
        **META_KWARGS,
    )
    # In pandas 3, merging str (StringDtype) with category can produce object
    # Restore x's column dtypes when the merge produced object dtype
    for col in x.columns:
        if (
            col in out.columns
            and out[col].dtype == object
            and x[col].dtype != object
            and not isinstance(x[col].dtype, pd.CategoricalDtype)
        ):
            try:
                out[col] = out[col].astype(x[col].dtype)
            except (TypeError, ValueError):  # pragma: no cover
                pass
    if isinstance(y, TibbleGrouped):
        return reconstruct_tibble(out, y)
    return out


@intersect.register(TibbleGrouped, backend="pandas")
def _intersect_grouped(x, y):
    newx = ungroup(x, **META_KWARGS)
    newy = ungroup(y, **META_KWARGS)
    out = intersect.dispatch(DataFrame)(newx, newy)
    return reconstruct_tibble(out, x)


@intersect.register(SeriesGroupBy, backend="pandas")
def _intersect_sg(x, y) -> SeriesGroupBy:
    """Intersect of two SeriesGroupBy objects"""
    if isinstance(x, SeriesGroupBy) and isinstance(y, SeriesGroupBy):
        xgrouper = get_grouper(x)
        ygrouper = get_grouper(y)
        if not _grouper_compatible(xgrouper, ygrouper, broadcastable=False):
            raise ValueError("Groupby objects are not compatible for intersection.")

        # intersect each group
        y_groups = y.groups
        y_obj = y.obj
        x = (
            x.apply(lambda s: intersect(s, y_obj.iloc[y_groups[s.name]]))
            .explode(ignore_index=False)
            .convert_dtypes()
        )
        return x.groupby(x.index)

    if isinstance(x, SeriesGroupBy):
        x = (
            x.apply(lambda s: intersect(s, y))
            .explode(ignore_index=False)
            .convert_dtypes()
        )
        return x.groupby(x.index)

    y = y.apply(lambda s: intersect(x, s)).explode(ignore_index=False).convert_dtypes()
    return y.groupby(y.index)


@union.register(DataFrame, backend="pandas")
def _union_df(x, y):
    """Union of two dataframes, ignoring grouping structure and indxes

    Args:
        x: and
        y: Dataframes to perform operations

    Returns:
        The dataframe of union of input dataframes
    """
    _check_xy(x, y)
    from .distinct import distinct

    out = distinct(
        pd.merge(x, ungroup(y, **META_KWARGS), how="outer"),
        **META_KWARGS,
    )
    out.reset_index(drop=True, inplace=True)
    if isinstance(y, TibbleGrouped):
        return reconstruct_tibble(out, y)
    return out


@union.register(TibbleGrouped, backend="pandas")
def _union_grouped(x, y):
    out = union.dispatch(DataFrame)(
        ungroup(x, **META_KWARGS),
        ungroup(y, **META_KWARGS),
    )
    return reconstruct_tibble(out, x)


@setdiff.register(DataFrame, backend="pandas")
def _setdiff_df(x, y):
    """Set diff of two dataframes

    Args:
        _data, *datas: Dataframes to perform operations
        on: The columns to the dataframes to perform operations on

    Returns:
        The dataframe of setdiff of input dataframes
    """
    _check_xy(x, y)
    indicator = "__datar_setdiff__"
    out = pd.merge(
        x,
        ungroup(y, **META_KWARGS),
        how="left",
        indicator=indicator,
    )

    from .distinct import distinct

    out = distinct(
        out[out[indicator] == "left_only"]
        .drop(columns=[indicator])
        .reset_index(drop=True),
        **META_KWARGS,
    )
    # In pandas 3, merging str with category can produce object dtype
    for col in x.columns:
        if (
            col in out.columns
            and out[col].dtype == object
            and x[col].dtype != object
            and not isinstance(x[col].dtype, pd.CategoricalDtype)
        ):
            try:
                out[col] = out[col].astype(x[col].dtype)
            except (TypeError, ValueError):  # pragma: no cover
                pass
    if isinstance(y, TibbleGrouped):
        return reconstruct_tibble(out, y)
    return out


@setdiff.register(TibbleGrouped, backend="pandas")
def _setdiff_grouped(x, y):
    out = setdiff.dispatch(DataFrame)(
        ungroup(x, **META_KWARGS),
        ungroup(y, **META_KWARGS),
    )
    return reconstruct_tibble(out, x)


@union_all.register(object, backend="pandas")
def _union_all_obj(x, y):
    return np.concatenate([x, y])


@union_all.register(DataFrame, backend="pandas")
def _union_all(x, y):
    """Union of all rows of two dataframes

    Args:
        _data, *datas: Dataframes to perform operations
        on: The columns to the dataframes to perform operations on

    Returns:
        The dataframe of union of all rows of input dataframes
    """
    _check_xy(x, y)
    out = bind_rows(x, ungroup(y, **META_KWARGS), **META_KWARGS)
    if isinstance(y, TibbleGrouped):
        return reconstruct_tibble(out, y)
    return out


@union_all.register(TibbleGrouped, backend="pandas")
def _union_all_grouped(x, y):
    out = union_all.dispatch(DataFrame)(
        ungroup(x, **META_KWARGS),
        ungroup(y, **META_KWARGS),
    )
    return reconstruct_tibble(out, x)


@setequal.register(DataFrame, backend="pandas")
def _set_equal_df(x, y, equal_na=True):
    """Check if two dataframes equal, grouping structures are ignored.

    Args:
        x: The first dataframe
        y: The second dataframe
        equal_na: To be compatible with non-dataframe version. Takes no effect
            for dataframe.

    Returns:
        True if they equal else False
    """
    x = ungroup(x, **META_KWARGS)
    y = ungroup(y, **META_KWARGS)
    _check_xy(x, y)

    x = x.sort_values(by=x.columns.to_list()).reset_index(drop=True)
    y = y.sort_values(by=y.columns.to_list()).reset_index(drop=True)
    return x.equals(y)


@symdiff.register(object, backend="pandas")
def _symdiff(x, y):
    """Symmetric difference of two vectors"""
    return _setdiff(_union(x, y), _intersect(x, y))


@symdiff.register(DataFrame, backend="pandas")
def _symdiff_df(x, y):
    """Symmetric difference of two dataframes"""
    _x = ungroup(x, **META_KWARGS)
    _y = ungroup(y, **META_KWARGS)
    _check_xy(_x, _y)

    out = setdiff(
        union(_x, _y, **META_KWARGS),
        intersect(_x, _y, **META_KWARGS),
        **META_KWARGS,
    )
    return reconstruct_tibble(out, x)

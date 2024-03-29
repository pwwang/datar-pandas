"""Set operations

https://github.com/tidyverse/dplyr/blob/master/R/sets.r
"""
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
from ...pandas import DataFrame
from ...common import (
    setdiff as _setdiff,
    union as _union,
    intersect as _intersect,
)
from ...tibble import TibbleGrouped, reconstruct_tibble


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
        pd.merge(x, ungroup(y, **meta_kwargs), how="inner"),
        **meta_kwargs,
    )
    if isinstance(y, TibbleGrouped):
        return reconstruct_tibble(out, y)
    return out


@intersect.register(TibbleGrouped, backend="pandas")
def _intersect_grouped(x, y):
    newx = ungroup(x, **meta_kwargs)
    newy = ungroup(y, **meta_kwargs)
    out = intersect.dispatch(DataFrame)(newx, newy)
    return reconstruct_tibble(out, x)


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
        pd.merge(x, ungroup(y, **meta_kwargs), how="outer"),
        **meta_kwargs,
    )
    out.reset_index(drop=True, inplace=True)
    if isinstance(y, TibbleGrouped):
        return reconstruct_tibble(out, y)
    return out


@union.register(TibbleGrouped, backend="pandas")
def _union_grouped(x, y):
    out = union.dispatch(DataFrame)(
        ungroup(x, **meta_kwargs),
        ungroup(y, **meta_kwargs),
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
        ungroup(y, **meta_kwargs),
        how="left",
        indicator=indicator,
    )

    from .distinct import distinct

    out = distinct(
        out[out[indicator] == "left_only"]
        .drop(columns=[indicator])
        .reset_index(drop=True),
        **meta_kwargs,
    )
    if isinstance(y, TibbleGrouped):
        return reconstruct_tibble(out, y)
    return out


@setdiff.register(TibbleGrouped, backend="pandas")
def _setdiff_grouped(x, y):
    out = setdiff.dispatch(DataFrame)(
        ungroup(x, **meta_kwargs),
        ungroup(y, **meta_kwargs),
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
    out = bind_rows(x, ungroup(y, **meta_kwargs), **meta_kwargs)
    if isinstance(y, TibbleGrouped):
        return reconstruct_tibble(out, y)
    return out


@union_all.register(TibbleGrouped, backend="pandas")
def _union_all_grouped(x, y):
    out = union_all.dispatch(DataFrame)(
        ungroup(x, **meta_kwargs),
        ungroup(y, **meta_kwargs),
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
    x = ungroup(x, **meta_kwargs)
    y = ungroup(y, **meta_kwargs)
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
    _x = ungroup(x, **meta_kwargs)
    _y = ungroup(y, **meta_kwargs)
    _check_xy(_x, _y)

    out = setdiff(
        union(_x, _y, **meta_kwargs),
        intersect(_x, _y, **meta_kwargs),
        **meta_kwargs,
    )
    return reconstruct_tibble(out, x)

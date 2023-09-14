"""Provide functions to manipulate multiple rows

https://github.com/tidyverse/dplyr/blob/master/R/rows.R
"""
from typing import Tuple
import numpy as np
from datar.core.utils import logger, arg_match
from datar.apis.dplyr import (
    bind_rows,
    coalesce,
    rows_insert,
    rows_append,
    rows_update,
    rows_patch,
    rows_upsert,
    rows_delete,
)

from ... import pandas as pd
from ...pandas import DataFrame
from ...common import is_scalar, setdiff
from ...contexts import Context

_meta_args = {"__ast_fallback": "normal", "__backend": "pandas"}


@rows_insert.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_insert(
    x,
    y,
    by=None,
    conflict="error",
    **kwargs,
):
    if kwargs:  # pragma: no cover
        raise ValueError("Unsupported arguments: %s" % kwargs.keys())

    conflict = arg_match(conflict, "conflict", ["error", "ignore"])

    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    idx_x, idx_y = _rows_match(x[key], y[key])
    if idx_x.size > 0 and conflict == "error":
        raise ValueError("Attempting to insert duplicate rows.")

    idx_y = np.isin(y.index, idx_y, invert=True)
    return bind_rows(x, y.loc[idx_y, :], **_meta_args)


@rows_append.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_append(x, y, **kwargs):
    if kwargs:  # pragma: no cover
        raise ValueError("Unsupported arguments: %s" % kwargs.keys())

    _rows_check_key_df(x, y.columns, df_name="x")
    return bind_rows(x, y, **_meta_args)


@rows_update.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_update(x, y, by=None, unmatched="error", **kwargs):
    if kwargs:  # pragma: no cover
        raise ValueError("Unsupported arguments: %s" % kwargs.keys())

    unmatched = arg_match(unmatched, "unmatched", ["error", "ignore"])

    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    idx_x, idx_y = _rows_match(x[key], y[key])

    if y.index.difference(idx_y).size > 0 and unmatched == "error":
        raise ValueError("Attempting to update missing rows.")

    if np.unique(idx_x).size < idx_x.size:
        raise ValueError("`y` key values must be unique.")

    x = x.copy()
    x.loc[idx_x, y.columns] = y.loc[idx_y, :].values
    return x


@rows_patch.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_patch(x, y, by=None, unmatched="error", **kwargs):
    if kwargs:  # pragma: no cover
        raise ValueError("Unsupported arguments: %s" % kwargs.keys())

    unmatched = arg_match(unmatched, "unmatched", ["error", "ignore"])

    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    idx_x, idx_y = _rows_match(x[key], y[key])

    if idx_x.size == 0:
        raise ValueError("Attempting to patch missing rows.")

    if y.index.difference(idx_y).size > 0 and unmatched == "error":
        raise ValueError("`y` must contain keys that already exist in `x`.")

    if np.unique(idx_x).size < idx_x.size:
        raise ValueError("`y` key values must be unique.")

    z = x.copy()

    other_cols = y.columns.difference(key)
    z.loc[idx_x, other_cols] = coalesce(
        x.loc[idx_x, other_cols],
        y.loc[idx_y, other_cols].set_index(idx_x),
    )

    for col in other_cols:
        z[col] = z[col].astype(x[col].dtype)

    return z


@rows_upsert.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_upsert(x, y, by=None):
    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    idx_x, idx_y = _rows_match(x[key], y[key])

    if np.unique(idx_x).size < idx_x.size:
        raise ValueError("`y` key values must be unique.")

    x = x.copy()
    x.loc[idx_x, y.columns] = y.loc[idx_y, :].values
    return bind_rows(x, y.loc[~y.index.isin(idx_y)], **_meta_args)


@rows_delete.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_delete(
    x,
    y,
    by=None,
    unmatched="error",
    **kwargs,
):
    if kwargs:  # pragma: no cover
        raise ValueError("Unsupported arguments: %s" % kwargs.keys())

    unmatched = arg_match(unmatched, "unmatched", ["error", "ignore"])

    key = _rows_check_key(by, x, y, allow_y_extra=True)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    extra_cols = y.columns.difference(key)
    if len(extra_cols) > 0:
        logger.info("Ignoring extra columns: %s", extra_cols)

    idx_x, idx_y = _rows_match(x[key], y[key])

    if y.index.difference(idx_y).size > 0 and unmatched == "error":
        raise ValueError("Attempting to delete missing rows.")

    x = x.copy()
    return x.loc[~x.index.isin(idx_x), :]


# helpers -----------------------------------------------------------------


def _rows_check_key(by, x, y, allow_y_extra=False):
    """Check the key and return the valid key"""
    if by is None:
        by = y.columns[0]
        logger.info("Matching, by=%r", by)

    if is_scalar(by):
        by = [by]  # type: ignore

    for by_elem in by:
        if not isinstance(by_elem, str):
            raise ValueError("`by` must be a string or a list of strings.")

    bad = setdiff(y.columns, x.columns)
    if len(bad) > 0 and not allow_y_extra:
        raise ValueError("All columns in `y` must exist in `x`.")

    return by


def _rows_check_key_df(df, by, df_name) -> None:
    """Check key with the data frame"""
    y_miss = setdiff(by, df.columns)
    if len(y_miss) > 0:
        raise ValueError(f"All `by` columns must exist in `{df_name}`.")

    # if any(df.duplicated(by)):
    #     raise ValueError(f"`{df_name}` key values are not unique.")


def _rows_match(
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mimic vctrs::vec_match"""
    x_id_col = "__x_id__"
    y_id_col = "__y_id__"
    xin = x.index.name
    yin = y.index.name
    x.index.name = x_id_col
    y.index.name = y_id_col
    xi = x.reset_index()
    yi = y.reset_index()
    x.index.name = xin
    y.index.name = yin
    merge_col = "__merge__"
    df = xi.merge(yi, how="left", indicator=merge_col)
    df = df[df[merge_col] == "both"]
    return df[x_id_col].values.astype(int), df[y_id_col].values.astype(int)

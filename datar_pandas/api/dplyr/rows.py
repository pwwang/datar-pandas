"""Provide functions to manipulate multiple rows

https://github.com/tidyverse/dplyr/blob/master/R/rows.R
"""
import numpy as np
from datar.core.utils import logger
from datar.apis.dplyr import (
    bind_rows,
    left_join,
    coalesce,
    rows_insert,
    rows_update,
    rows_patch,
    rows_upsert,
    rows_delete,
)

from ... import pandas as pd
from ...pandas import DataFrame
from ...common import is_scalar, setdiff
from ...contexts import Context
from ..tibble.verbs import rownames_to_column

_meta_args = {"__ast_fallback": "normal", "__backend": "pandas"}


@rows_insert.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_insert(x, y, by=None, copy=True):
    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    idx = _rows_match(y[key], x[key])
    bad = ~pd.isnull(idx)
    if any(bad):
        raise ValueError("Attempting to insert duplicate rows.")

    return bind_rows(x, y, _copy=copy, **_meta_args)


@rows_update.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_update(x, y, by=None, copy=True):
    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    idx = _rows_match(y[key], x[key])
    bad = pd.isnull(idx)
    if any(bad):
        raise ValueError("Attempting to update missing rows.")

    idx = idx.astype(int)

    if copy:
        x = x.copy()

    # Join at the beginning? NaNs will be produced and dtypes will be changed
    # in y
    # Try it in pandas2
    y_joined = left_join(x.loc[idx, key], y, by=key, **_meta_args).set_index(
        idx
    )

    x.loc[idx, y.columns] = y_joined
    return x


@rows_patch.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_patch(x, y, by=None, copy=True):
    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    idx = _rows_match(y[key], x[key])
    bad = pd.isnull(idx)
    if any(bad):
        raise ValueError("Attempting to patch missing rows.")

    new_data = []
    for col in y.columns:
        new_data.append(coalesce(x.loc[idx, col].values, y[col]))

    if copy:
        x = x.copy()
    x.loc[idx, y.columns] = np.array(new_data).T
    return x


@rows_upsert.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_upsert(x, y, by=None, copy=True):
    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    idx = _rows_match(y[key], x[key])
    new = pd.isnull(idx)
    # idx of x
    idx_existing = idx[~new]

    x.loc[idx_existing, y.columns] = y.loc[~new].values
    return bind_rows(x, y.loc[new], _copy=copy, **_meta_args)


@rows_delete.register(DataFrame, context=Context.EVAL, backend="pandas")
def _rows_delete(
    x,
    y,
    by=None,
    copy=True,
):
    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, df_name="x")
    _rows_check_key_df(y, key, df_name="y")

    extra_cols = setdiff(y.columns, key)
    if len(extra_cols) > 0:
        logger.info("Ignoring extra columns: %s", extra_cols)

    idx = _rows_match(y[key], x[key])
    bad = pd.isnull(idx)

    if any(bad):
        raise ValueError("Attempting to delete missing rows.")

    if copy:
        x = x.copy()

    return x.loc[~x.index.isin(idx), :]


# helpers -----------------------------------------------------------------


def _rows_check_key(by, x, y):
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
    if len(bad) > 0:
        raise ValueError("All columns in `y` must exist in `x`.")

    return by


def _rows_check_key_df(df, by, df_name) -> None:
    """Check key with the data frame"""
    y_miss = setdiff(by, df.columns)
    if len(y_miss) > 0:
        raise ValueError(f"All `by` columns must exist in `{df_name}`.")

    # if any(df.duplicated(by)):
    #     raise ValueError(f"`{df_name}` key values are not unique.")


def _rows_match(x: pd.DataFrame, y: pd.DataFrame, for_: str = "x"):
    """Mimic vctrs::vec_match"""
    id_col = "__id__"
    y_with_id = rownames_to_column(y, var=id_col, **_meta_args)

    return left_join(x, y_with_id, **_meta_args)[id_col].values

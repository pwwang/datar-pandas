"""Function from R-base that can be used as verbs"""
import numpy as np
from datar.core.utils import arg_match
from datar.apis.base import (
    colnames,
    rownames,
    set_colnames,
    set_rownames,
    dim,
    nrow,
    ncol,
    diag,
    t,
    unique,
    duplicated,
    max_col,
    complete_cases,
    head,
    tail,
)

from ... import pandas as pd
from ...pandas import DataFrame, Series, Index, SeriesGroupBy, get_obj
from ...common import is_scalar, unique as _unique


@colnames.register(DataFrame, backend="pandas")
def _colnames(df, nested=True):
    has_nest = any("$" in str(col) for col in df.columns)
    if not has_nest or not nested:
        return df.columns.values

    # x, y, y -> x, y
    names = Index([col.split("$", 1)[0] for col in df.columns])
    names = _unique(names)
    return names if nested else df.columns.values


@set_colnames.register(DataFrame, backend="pandas")
def _set_colnames(df, names, nested=True):
    df = df.copy()
    if not nested:
        df.columns = names
        return df

    # x, y$a, y$b with names m, n -> m, n$a, n$b
    old_names = colnames(
        df,
        nested=True,
        __ast_fallback="normal",
        __backend="pandas",
    )
    mapping = dict(zip(old_names, names))
    names = []
    for col in df.columns:
        strcol = str(col)
        if "$" in strcol:
            prefix, suffix = strcol.split("$", 1)
            new_prefix = mapping[prefix]
            names.append(f"{new_prefix}${suffix}")
        else:
            names.append(mapping[col])
    df.columns = names
    return df


@rownames.register(DataFrame, backend="pandas")
def _rownames(df):
    return df.index.values


@set_rownames.register(DataFrame, backend="pandas")
def _set_rownames(df, names):
    df = df.copy()
    df.index = names
    return df


@dim.register(DataFrame, backend="pandas")
def _dim(x, nested=True):
    return (
        nrow(x, __ast_fallback="normal", __backend="pandas"),
        ncol(x, nested, __ast_fallback="normal", __backend="pandas"),
    )


@nrow.register(DataFrame, backend="pandas")
def _nrow(_data) -> int:
    return _data.shape[0]


@ncol.register(DataFrame, backend="pandas")
def _ncol(_data, nested=True):
    if not nested:
        return _data.shape[1]

    return len(
        colnames(
            _data,
            nested=nested,
            __ast_fallback=True,
            __backend="pandas",
        )
    )


@diag.register(object, backend="pandas")
def _diag(x=1, nrow=None, ncol=None):
    if nrow is None and isinstance(x, int):
        nrow = x
        x = 1
    if ncol is None:
        ncol = nrow
    if is_scalar(x):
        nmax = max(nrow, ncol)
        x = [x] * nmax
    elif nrow is not None:
        nmax = max(nrow, ncol)
        nmax = nmax // len(x)
        x = x * nmax

    x = np.array(x)
    ret = DataFrame(np.diag(x), dtype=x.dtype)
    return ret.iloc[:nrow, :ncol]


@diag.register(DataFrame, backend="pandas")
def _diag_df(
    x,
    nrow=None,
    ncol=None,
):
    if nrow is not None and ncol is not None:
        raise ValueError("Extra arguments received for diag.")

    x = x.copy()
    if nrow is not None:
        np.fill_diagonal(x.values, nrow)
        return x
    return np.diag(x)


@t.register(DataFrame, backend="pandas")
def _t(_data, copy=False):
    return _data.transpose(copy=copy)


@unique.register(SeriesGroupBy, backend="pandas")
def _unique_sgb(x):
    return x.apply(pd.unique).explode().astype(get_obj(x).dtype)


@duplicated.register(DataFrame, backend="pandas")
def _duplicated(
    x,
    incomparables=None,
    from_last=False,
):
    keep = "first" if not from_last else "last"
    return x.duplicated(keep=keep).values


@max_col.register(DataFrame, backend="pandas")
def _max_col(df, ties_method="random"):
    ties_method = arg_match(
        ties_method, "ties_method", ["random", "first", "last"]
    )

    def which_max_with_ties(ser: Series):
        """Find index with max if ties happen"""
        indices = np.flatnonzero(ser == max(ser))
        if len(indices) == 1 or ties_method == "first":
            return indices[0]
        if ties_method == "random":
            return np.random.choice(indices)
        return indices[-1]

    return df.apply(which_max_with_ties, axis=1).values


@complete_cases.register(DataFrame, backend="pandas")
def _complete_cases(_data):
    return _data.apply(lambda row: row.notna().all(), axis=1).values


# actually from R::utils
@head.register((DataFrame, Series), backend="pandas")
def _head(_data, n=6):
    return _data.head(n)


@tail.register((DataFrame, Series), backend="pandas")
def _tail(_data, n=6):
    return _data.tail(n).reset_index(drop=True)

"""Provides forcats verbs to manipulate factor level orders"""
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from datar.core.utils import logger
from datar.apis.forcats import (
    lvls_reorder,
    fct_relevel,
    fct_inorder,
    fct_infreq,
    fct_inseq,
    last2,
    first2,
    fct_reorder,
    fct_reorder2,
    fct_shuffle,
    fct_rev,
    fct_shift,
)

from ... import pandas as pd
from ...pandas import Categorical, DataFrame, Series, SeriesGroupBy, get_obj
from ...common import is_scalar, intersect, setdiff
from ...collections import Collection
from ...contexts import Context
from ..base.arithm import median
from ..base.asis import as_integer
from ..base.factor import levels, nlevels
from ..base.seq import seq_len, sample, rev, match, append, order
from ..base.table import table
from ..base.verbs import duplicated
from .utils import check_factor, ForcatsRegType
from .lvls import lvls_seq


@fct_relevel.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_relevel(
    _f,
    *lvls: Any,
    after: int = None,
) -> Categorical:
    """Reorder factor levels by hand

    Args:
        _f: A factor (categoriccal), or a string vector
        *lvls: Either a function (then `len(lvls)` should equal to `1`) or
            the new levels.
            A function will be called with the current levels as input, and the
            return value (which must be a character vector) will be used to
            relevel the factor.
            Any levels not mentioned will be left in their existing order,
            by default after the explicitly mentioned levels.
        after: Where should the new values be placed?

    Returns:
        The factor with levels replaced
    """

    _f = check_factor(_f)
    old_levels = levels(_f, __ast_fallback="normal", __backend="pandas")
    if len(lvls) == 1 and callable(lvls[0]):
        first_levels = lvls[0](old_levels)
    else:
        first_levels = Collection(lvls)

    unknown = setdiff(first_levels, old_levels)

    if len(unknown) > 0:
        logger.warning("[fct_relevel] Unknown levels in `_f`: %s", unknown)
        first_levels = intersect(first_levels, old_levels)

    new_levels = append(
        setdiff(old_levels, first_levels).astype(old_levels.dtype),
        first_levels,
        after=after,
        __ast_fallback="normal",
        __backend="numpy",
    )

    return lvls_reorder(
        _f,
        match(
            new_levels,
            old_levels,
            __ast_fallback="normal",
            __backend="numpy",
        ),
        __ast_fallback="normal",
        __backend="pandas",
    )


@fct_inorder.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_inorder(_f, ordered: bool = None) -> Categorical:
    """Reorder factor levels by first appearance

    Args:
        _f: A factor
        ordered: A logical which determines the "ordered" status of the
            output factor.

    Returns:
        The factor with levels reordered
    """
    is_sgb = isinstance(_f, SeriesGroupBy)
    _f1 = get_obj(_f) if is_sgb else _f

    _f1 = check_factor(_f1)
    dups = duplicated(_f1, __ast_fallback="normal", __backend="numpy")
    idx = as_integer(_f1, __ast_fallback="normal", __backend="pandas")[~dups]
    idx = idx[~pd.isnull(_f1[~dups])]
    out = lvls_reorder(
        _f1,
        idx,
        ordered=ordered,
        __ast_fallback="normal",
        __backend="pandas",
    )

    if not is_sgb:
        return out

    return Series(out, get_obj(_f).index).groupby(
        _f.grouper,
        observed=_f.observed,
        sort=_f.sort,
        dropna=_f.dropna,
    )


@fct_infreq.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_infreq(_f, ordered: bool = None) -> Categorical:
    """Reorder factor levels by frequency

    Args:
        _f: A factor
        ordered: A logical which determines the "ordered" status of the
            output factor.

    Returns:
        The factor with levels reordered
    """
    _f = check_factor(_f)
    return lvls_reorder(
        _f,
        order(
            table(
                _f, __ast_fallback="normal", __backend="pandas"
            ).values.flatten(),
            decreasing=True,
            __ast_fallback="normal",
            __backend="numpy",
        ),
        ordered=ordered,
        __ast_fallback="normal",
        __backend="pandas",
    )


@fct_inseq.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_inseq(_f, ordered: bool = None) -> Categorical:
    """Reorder factor levels by numeric order

    Args:
        _f: A factor
        ordered: A logical which determines the "ordered" status of the
            output factor.

    Returns:
        The factor with levels reordered
    """
    _f = check_factor(_f)
    levs = levels(_f, __ast_fallback="normal", __backend="pandas")
    num_levels = []
    for lev in levs:
        try:
            numlev = as_integer(
                lev, __ast_fallback="normal", __backend="numpy"
            )
        except (ValueError, TypeError):
            numlev = np.nan
        num_levels.append(numlev)

    if all(pd.isnull(num_levels)):
        raise ValueError(
            "At least one existing level must be coercible to numeric."
        )

    return lvls_reorder(
        _f,
        order(
            num_levels,
            na_last=True,
            __ast_fallback="normal",
            __backend="numpy",
        ),
        ordered=ordered,
        __ast_fallback="normal",
        __backend="pandas",
    )


@last2.register(object, backend="pandas")
def _last2(_x: Iterable, _y: Sequence) -> Any:
    """Find the last element of `_y` ordered by `_x`

    Args:
        _x: The vector used to order `_y`
        _y: The vector to get the last element of

    Returns:
        Last element of `_y` ordered by `_x`
    """
    return list(
        _y[
            order(
                _x,
                na_last=False,
                __ast_fallback="normal",
                __backend="numpy",
            )
        ]
    )[-1]


@first2.register(object, backend="pandas")
def _first2(_x: Sequence, _y: Sequence) -> Any:
    """Find the first element of `_y` ordered by `_x`

    Args:
        _x: The vector used to order `_y`
        _y: The vector to get the first element of

    Returns:
        First element of `_y` ordered by `_x`
    """
    return _y[order(_x, __ast_fallback="normal", __backend="numpy")][0]


@fct_reorder.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_reorder(
    _f,
    _x: Sequence,
    *args: Any,
    _fun: Callable = median,
    _desc: bool = False,
    **kwargs: Any,
) -> Categorical:
    """Reorder factor levels by sorting along another variable

    Args:
        _f: A factor
        _x: The levels of `f` are reordered so that the values
            of `_fun(_x)` are in ascending order.
        _fun: The summary function, have to be passed by keyword
        *args, **kwargs: Other arguments for `_fun`.
        _desc: Order in descending order?

    Returns:
        The factor with levels reordered
    """
    _f = check_factor(_f)
    if is_scalar(_x):
        _x = [_x]

    if len(_f) != len(_x):
        raise ValueError("Unmatched length between `_x` and `_f`.")

    summary = DataFrame({"f": _f, "x": _x}).groupby(
        "f", observed=False, sort=False, dropna=False
    )
    args = args[1:]
    if getattr(_fun, "_pipda_functype", None) in ("pipeable", "verb"):
        # simulate tapply
        # TODO: test
        summary = summary.agg(  # pragma: no cover
            lambda col: _fun(col, *args, **kwargs, __ast_fallback="normal")
        )
    else:
        summary = summary.agg(lambda col: _fun(col, *args, **kwargs))

    if not is_scalar(summary.iloc[0, 0]):
        raise ValueError("`fun` must return a single value per group.")

    return lvls_reorder(
        _f,
        order(
            summary.iloc[:, 0],
            decreasing=_desc,
            __ast_fallback="normal",
            __backend="numpy",
        ),
        __ast_fallback="normal",
        __backend="pandas",
    )


@fct_reorder2.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_reorder2(
    _f,
    _x: Sequence,
    _y: Sequence,
    *args: Any,
    _fun: Callable = last2,
    _desc: bool = True,
    **kwargs: Any,
) -> Categorical:
    """Reorder factor levels by sorting along another variable

    Args:
        _f: A factor
        _x: and
        _y: The levels of `f` are reordered so that the values
            of `_fun(_x, _y)` are in ascending order.
        _fun: The summary function, have to be passed by keyword
        *args, **kwargs: Other arguments for `_fun`.
        _desc: Order in descending order?

    Returns:
        The factor with levels reordered
    """
    _f = check_factor(_f)
    if is_scalar(_x):
        _x = [_x]
    if is_scalar(_y):
        _y = [_y]
    if len(_f) != len(_x) or len(_f) != len(_y):
        raise ValueError("Unmatched length between `_x` and `_f`.")

    summary = DataFrame({"f": _f, "x": _x, "y": _y}).groupby(
        "f", observed=False, sort=False, dropna=False
    )
    args = args[1:]

    if (
        getattr(_fun, "_pipda_functype", None) in ("pipeable", "verb")
    ):  # pragma: no cover
        kwargs["__ast_fallback"] = "normal"

    summary = summary.apply(
        lambda row: _fun(
            row.x.reset_index(drop=True),
            row.y.reset_index(drop=True),
            *args,
            **kwargs,
        )
    )

    if not isinstance(summary, Series) or not is_scalar(summary.values[0]):
        raise ValueError("`fun` must return a single value per group.")

    return lvls_reorder(
        _f,
        order(
            summary,
            decreasing=_desc,
            __ast_fallback="normal",
            __backend="numpy",
        ),
        __ast_fallback="normal",
        __backend="pandas",
    )


@fct_shuffle.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_shuffle(_f) -> Categorical:
    """Randomly permute factor levels

    Args:
        _f: A factor

    Returns:
        The factor with levels randomly permutated
    """
    _f = check_factor(_f)

    return lvls_reorder(
        _f,
        sample(lvls_seq(_f), __ast_fallback="normal", __backend="numpy"),
        __ast_fallback="normal",
        __backend="pandas",
    )


@fct_rev.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_rev(_f) -> Categorical:
    """Reverse order of factor levels

    Args:
        _f: A factor

    Returns:
        The factor with levels reversely ordered
    """
    _f = check_factor(_f)

    return lvls_reorder(
        _f,
        rev(lvls_seq(_f), __ast_fallback="normal", __backend="numpy"),
        __ast_fallback="normal",
        __backend="pandas",
    )


@fct_shift.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_shift(_f, n: int = 1) -> Categorical:
    """Shift factor levels to left or right, wrapping around at end

    Args:
        f: A factor.
        n: Positive values shift to the left; negative values shift to
            the right.

    Returns:
        The factor with levels shifted
    """
    nlvls = nlevels(_f, __ast_fallback="normal", __backend="pandas")
    lvl_order = (
        seq_len(nlvls, __ast_fallback="normal", __backend="numpy") + n - 1
    ) % nlvls

    return lvls_reorder(
        _f, lvl_order, __ast_fallback="normal", __backend="pandas"
    )

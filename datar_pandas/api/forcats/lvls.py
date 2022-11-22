"""Lower-level APIs to manipulate the factors"""
from typing import Iterable, List

import numpy as np
from datar.apis.forcats import (
    lvls_reorder,
    lvls_revalue,
    lvls_union,
    lvls_expand,
)

from ...pandas import Categorical
from ...contexts import Context
from ...common import is_integer, setdiff, union, unique
from ..base.asis import as_character, as_integer, is_ordered
from ..base.factor import levels, nlevels, factor
from ..base.seq import match, seq_along
from ..dplyr.recode import recode_factor
from ..dplyr.sets import setequal
from .utils import check_factor, ForcatsRegType


def lvls_seq(_f):
    """Get the index sequence of a factor levels"""
    return (
        seq_along(
            levels(_f, __ast_fallback="normal", __backend="pandas"),
            __ast_fallback="normal",
            __backend="numpy",
        )
        - 1
    )


def refactor(_f, new_levels: Iterable, ordered: bool = None) -> Categorical:
    """Refactor using new levels"""
    if ordered is None:
        ordered = is_ordered(_f, __ast_fallback="normal", __backend="pandas")

    new_f = factor(
        _f,
        levels=new_levels,
        exclude=np.nan,
        ordered=ordered,
        __ast_fallback="normal",
        __backend="pandas",
    )
    # keep attributes?
    return new_f


@lvls_reorder.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _lvls_reorder(
    _f,
    idx,
    ordered: bool = None,
) -> Categorical:
    """Leaves values of a factor as they are, but changes the order by
    given indices

    Args:
        f: A factor (or character vector).
        idx: A integer index, with one integer for each existing level.
        new_levels: A character vector of new levels.
        ordered: A logical which determines the "ordered" status of the
          output factor. `None` preserves the existing status of the factor.

    Returns:
        The factor with levels reordered
    """
    _f = check_factor(_f)
    if not is_integer(idx):
        raise ValueError("`idx` must be integers")

    len_idx = len(idx)
    seq_lvls = lvls_seq(_f)
    if not setequal(
        idx,
        seq_lvls,
        __ast_fallback="normal",
        __backend="numpy",
    ) or len_idx != nlevels(_f, __ast_fallback="normal", __backend="pandas"):
        raise ValueError(
            "`idx` must contain one integer for each level of `f`"
        )

    return refactor(
        _f,
        levels(_f, __ast_fallback="normal", __backend="pandas")[idx],
        ordered=ordered,
    )


@lvls_revalue.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _lvls_revalue(
    _f,
    new_levels: Iterable,
) -> Categorical:
    """changes the values of existing levels; there must
    be one new level for each old level

    Args:
        _f: A factor
        new_levels: A character vector of new levels.

    Returns:
        The factor with the new levels
    """
    _f = check_factor(_f)

    if len(new_levels) != nlevels(
        _f,
        __ast_fallback="normal",
        __backend="pandas",
    ):
        raise ValueError(
            "`new_levels` must be the same length as `levels(f)`: expected ",
            f"{nlevels(_f, __ast_fallback='normal', __backend='pandas')} "
            f"new levels, got {len(new_levels)}.",
        )

    u_levels = unique(new_levels)
    if len(new_levels) > len(u_levels):
        # has duplicates
        index = match(
            new_levels,
            u_levels,
            __ast_fallback="normal",
            __backend="numpy",
        )
        out = factor(
            as_character(
                index[
                    as_integer(_f, __ast_fallback="normal", __backend="pandas")
                ]
            ),
            __ast_fallback="normal",
            __backend="pandas",
        )
        return recode_factor(
            out,
            dict(
                zip(
                    levels(out, __ast_fallback="normal", __backend="pandas"),
                    u_levels,
                )
            ),
            __ast_fallback="normal",
            __backend="pandas",
        ).values

    recodings = dict(
        zip(
            levels(_f, __ast_fallback="normal", __backend="pandas"), new_levels
        )
    )
    return recode_factor(
        _f, recodings, __ast_fallback="normal", __backend="pandas"
    ).values


@lvls_expand.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _lvls_expand(
    _f,
    new_levels: Iterable,
) -> Categorical:
    """Expands the set of levels; the new levels must
    include the old levels.

    Args:
        _f: A factor
        new_levels: The new levels. Must include the old ones

    Returns:
        The factor with the new levels
    """
    _f = check_factor(_f)
    levs = levels(_f, __ast_fallback="normal", __backend="pandas")

    missing = setdiff(levs, new_levels)
    if len(missing) > 0:
        raise ValueError(
            "Must include all existing levels. Missing: {missing}"
        )

    return refactor(_f, new_levels=new_levels)


@lvls_union.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _lvls_union(
    fs,
) -> List:
    """Find all levels in a list of factors

    Args:
        fs: A list of factors

    Returns:
        A list of all levels
    """
    out = []
    for fct in fs:
        fct = check_factor(fct)
        levs = levels(fct, __ast_fallback="normal", __backend="pandas")
        out = union(out, levs)
    return out

"""Provides functions to add or remove levels"""
from typing import Any, Iterable, List

from datar.apis.forcats import (
    lvls_expand,
    lvls_union,
    fct_expand,
    fct_explicit_na,
    fct_drop,
    fct_unify,
)

from ... import pandas as pd
from ...pandas import Categorical
from ...common import is_scalar, union, intersect, setdiff
from ...contexts import Context
from ..base.factor import levels
from ..base.table import table
from .lvls import refactor
from .utils import check_factor, ForcatsRegType


@fct_expand.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_expand(_f, *additional_levels: Any) -> Categorical:
    """Add additional levels to a factor

    Args:
        _f: A factor
        *additional_levels: Additional levels to add to the factor.
            Levels that already exist will be silently ignored.

    Returns:
        The factor with levels expanded
    """
    _f = check_factor(_f)
    levs = levels(_f)
    addlevs = []
    for alev in additional_levels:
        if is_scalar(alev):
            addlevs.append(alev)
        else:
            addlevs.extend(alev)
    new_levels = union(levs, addlevs)
    return lvls_expand(
        _f,
        new_levels,
        __ast_fallback="normal",
        __backend="pandas",
    )


@fct_explicit_na.register(
    ForcatsRegType,
    context=Context.EVAL,
    backend="pandas",
)
def _fct_explicit_na(_f, na_level: Any = "(Missing)") -> Categorical:
    """Make missing values explicit

    This gives missing values an explicit factor level, ensuring that they
    appear in summaries and on plots.

    Args:
        _f: A factor
        na_level: Level to use for missing values.
            This is what NAs will be changed to.

    Returns:
        The factor with explict na_levels
    """
    _f = check_factor(_f)
    # levs = levels(_f, __calling_env=CallingEnvs.REGULAR)
    is_missing = pd.isnull(_f)
    # is_missing_level = is_null(levs)

    if any(is_missing):
        _f = fct_expand(_f, na_level)
        _f[is_missing] = na_level
        return _f

    # NAs cannot be a level in pandas.Categorical
    # if any(is_missing_level):
    #     levs[is_missing_level] = na_level
    #     return lvls_revalue(_f, levs)

    return _f


@fct_drop.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_drop(_f, only: Any = None) -> Categorical:
    """Drop unused levels

    Args:
        _f: A factor
        only: A character vector restricting the set of levels to be dropped.
            If supplied, only levels that have no entries and appear in
            this vector will be removed.

    Returns:
        The factor with unused levels dropped
    """
    _f = check_factor(_f)

    levs = levels(_f)
    count = table(_f).iloc[0, :]

    to_drop = levs[count == 0]
    if only is not None and is_scalar(only):
        only = [only]

    if only is not None:
        to_drop = intersect(to_drop, only)

    return refactor(_f, new_levels=setdiff(levs, to_drop))


@fct_unify.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_unify(
    fs,
    levels: Iterable = None,
) -> List[Categorical]:
    """Unify the levels in a list of factors

    Args:
        fs: A list of factors
        levels: Set of levels to apply to every factor. Default to union
            of all factor levels

    Returns:
        A list of factors with the levels expanded
    """
    if levels is None:
        levels = lvls_union(fs)

    out = []
    for fct in fs:
        fct = check_factor(fct)
        out.append(
            lvls_expand(
                fct,
                new_levels=levels,
                __ast_fallback="normal",
                __backend="pandas",
            )
        )
    return out

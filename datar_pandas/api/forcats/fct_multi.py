"""Provides functions for multiple factors"""
import itertools

from datar.apis.forcats import fct_c, fct_cross, lvls_union

from ...pandas import Categorical
from ...common import intersect, is_null
from ..base import expand_grid
from ..base.factor import factor, levels
from ..base.string import paste
from .utils import check_factor


@fct_c.register(object, backend="pandas")
def _fct_c(*fs) -> Categorical:
    """Concatenate factors, combining levels

    This is a useful way of patching together factors from multiple sources
    that really should have the same levels but don't.

    Args:
        fs: The factors to be concatenated

    Returns:
        The concatenated factor
    """
    if not fs:
        return factor(__ast_fallback="normal", __backend="pandas")

    levs = lvls_union(fs, __ast_fallback="normal", __backend="pandas")
    allvals = list(itertools.chain(*fs))
    return factor(
        allvals,
        levels=levs,
        __ast_fallback="normal",
        __backend="pandas",
    )


@fct_cross.register(object, backend="pandas")
def _fct_cross(
    *fs,
    sep: str = ":",
    keep_empty: bool = False,
) -> Categorical:
    """Combine levels from two or more factors to create a new factor

    Computes a factor whose levels are all the combinations of
    the levels of the input factors.

    Args:
        *fs: Factors to combine
        sep: A string to separate levels
        keep_empty: If True, keep combinations with no observations as levels

    Returns:
        The new factor
    """
    if not fs or (len(fs) == 1 and len(check_factor(fs[0])) == 0):
        return factor(__ast_fallback="normal", __backend="pandas")

    fs = [check_factor(fct) for fct in fs]
    newf = paste(*fs, sep=sep, __ast_fallback="normal", __backend="numpy")

    old_levels = (
        levels(fct, __ast_fallback="normal", __backend="pandas")
        for fct in fs
    )
    grid = expand_grid(*old_levels, __ast_fallback="normal", __backend="pandas")
    new_levels = paste(
        *(grid[col] for col in grid),
        sep=sep,
        __ast_fallback="normal",
        __backend="numpy",
    )

    if not keep_empty:
        new_levels = intersect(new_levels, newf[~is_null(newf)])

    return factor(
        newf,
        levels=new_levels,
        __ast_fallback="normal",
        __backend="pandas",
    )

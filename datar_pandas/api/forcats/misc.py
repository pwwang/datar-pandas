"""Provides other helper functions for factors"""
from typing import Any, Iterable

import numpy as np
from datar import f
from datar.apis.forcats import fct_inorder, fct_count, fct_match, fct_unique

from ... import pandas as pd
from ...common import is_scalar, setdiff
from ...pandas import Categorical, DataFrame
from ...contexts import Context
from ..base.arithm import proportions
from ..base.asis import is_ordered
from ..base.factor import levels, nlevels, factor
from ..base.table import tabulate
from ..dplyr.mutate import mutate
from ..dplyr.arrange import arrange
from ..dplyr.desc import desc
from .utils import check_factor, ForcatsRegType


@fct_count.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_count(_f, sort: bool = False, prop=False) -> Categorical:
    """Count entries in a factor

    Args:
        _f: A factor
        sort: If True, sort the result so that the most common values float to
            the top
        prop: If True, compute the fraction of marginal table.

    Returns:
        A data frame with columns `f`, `n` and `p`, if prop is True
    """
    f2 = check_factor(_f)
    n_na = sum(pd.isnull(f2))

    df = DataFrame(
        {
            "f": fct_inorder(
                levels(f2, __ast_fallback="normal", __backend="pandas"),
                __ast_fallback="normal",
                __backend="pandas",
            ),
            "n": tabulate(
                f2,
                nlevels(f2, __ast_fallback="normal", __backend="pandas"),
                __ast_fallback="normal",
                __backend="pandas",
            ),
        }
    )

    if n_na > 0:
        df.loc[df.shape[0], :] = {"f": np.nan, "n": n_na}

    if sort:
        df = arrange(
            df,
            desc(f.n, __ast_fallback="normal", __backend="pandas"),
            __ast_fallback="normal",
            __backend="pandas",
        )
    if prop:
        df = mutate(
            df,
            p=proportions(f.n, __ast_fallback="normal", __backend="pandas"),
            __ast_fallback="normal",
            __backend="pandas",
        )

    return df


@fct_match.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_match(_f, lvls: Any) -> Iterable[bool]:
    """Test for presence of levels in a factor

    Do any of `lvls` occur in `_f`?

    Args:
        _f: A factor
        lvls: A vector specifying levels to look for.

    Returns:
        A logical factor
    """
    _f = check_factor(_f)

    if is_scalar(lvls):
        lvls = [lvls]

    bad_lvls = setdiff(
        lvls,
        levels(_f, __ast_fallback="normal", __backend="pandas"),
    )
    if len(bad_lvls) > 0:
        bad_lvls = np.array(bad_lvls)[~pd.isnull(bad_lvls)]
    if len(bad_lvls) > 0:
        raise ValueError(f"Levels not present in factor: {bad_lvls}.")

    return np.isin(_f, lvls)


@fct_unique.register(ForcatsRegType, context=Context.EVAL, backend="pandas")
def _fct_unique(_f) -> Categorical:
    """Unique values of a factor

    Args:
        _f: A factor

    Returns:
        The factor with the unique values in `_f`
    """
    lvls = levels(_f, __ast_fallback="normal", __backend="pandas")
    is_ord = is_ordered(_f, __ast_fallback="normal", __backend="pandas")
    return factor(
        lvls,
        levels=lvls,
        ordered=is_ord,
        __ast_fallback="normal",
        __backend="pandas",
    )

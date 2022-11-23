"""Count observations by group

See souce code https://github.com/tidyverse/dplyr/blob/master/R/count-tally.R
"""
from typing import Any
from datar import options_context
from datar.core.defaults import f
from datar.core.utils import logger
from datar.apis.dplyr import (
    n,
    group_by,
    ungroup,
    group_by_drop_default,
    group_vars,
    mutate,
    summarise,
    arrange,
    desc,
    count,
    add_count,
    tally,
    add_tally,
)

from ...typing import Data, Number
from ...pandas import DataFrame
from ...contexts import Context
from ...tibble import reconstruct_tibble


@count.register(DataFrame, context=Context.PENDING, backend="pandas")
def _count(
    x: DataFrame,
    *args: Any,
    wt: Data[Number] = None,
    sort: bool = False,
    name: str = None,
    _drop: bool = None,
    **kwargs: Any,
) -> DataFrame:
    if _drop is None:
        _drop = group_by_drop_default(x)

    if args or kwargs:
        out = group_by(
            x,
            *args,
            **kwargs,
            _add=True,
            _drop=_drop,
            __ast_fallback="normal",
            __backend="pandas",
        )
    else:
        out = x

    out = tally(
        out,
        wt=wt,
        sort=sort,
        name=name,
        __ast_fallback="normal",
        __backend="pandas",
    )

    return reconstruct_tibble(out, x)


@tally.register(DataFrame, context=Context.PENDING, backend="pandas")
def _tally(
    x: DataFrame,
    wt: Data[Number] = None,
    sort: bool = False,
    name: str = None,
):
    name = _check_name(
        name,
        group_vars(x, __ast_fallback="normal", __backend="pandas"),
    )
    # thread-safety?
    with options_context(dplyr_summarise_inform=False):
        out = summarise(
            x,
            __ast_fallback="normal",
            __backend="pandas",
            **{name: n() if wt is None else wt.sum()},
        )

    if sort:
        out = arrange(
            ungroup(out, __ast_fallback="normal", __backend="pandas"),
            # desc(f[name], __calling_env=CallingEnvs.PIPING)
            # FunctionCall(desc, (f[name], ), {}),
            desc(f[name]),
            __ast_fallback="normal",
            __backend="pandas",
        )
        out.reset_index(drop=True, inplace=True)
        return reconstruct_tibble(out, x)

    return out


@add_count.register(DataFrame, context=Context.PENDING, backend="pandas")
def _add_count(
    x: DataFrame,
    *args: Any,
    wt: Data[Number] = None,
    sort: bool = False,
    name: str = "n",
    **kwargs: Any,
):
    if args or kwargs:
        out = group_by(
            x,
            *args,
            **kwargs,
            _add=True,
            __ast_fallback="normal",
            __backend="pandas",
        )
    else:
        out = x

    out = add_tally(
        out,
        wt=wt,
        sort=sort,
        name=name,
        __ast_fallback="normal",
        __backend="pandas",
    )
    return out


@add_tally.register(DataFrame, context=Context.PENDING, backend="pandas")
def _add_tally(
    x: DataFrame,
    wt: Data[Number] = None,
    sort: bool = False,
    name: str = "n",
) -> DataFrame:
    name = _check_name(name, x.columns)

    out = mutate(
        x,
        **{name: n() if wt is None else wt.sum()},
        __ast_fallback="normal",
        __backend="pandas",
    )

    if sort:
        sort_ed = arrange(
            ungroup(out, __ast_fallback="normal", __backend="pandas"),
            # desc(f[name], __calling_env=CallingEnvs.PIPING)
            # FunctionCall(desc, (f[name], ), {}),
            desc(f[name]),
            __ast_fallback="normal",
            __backend="pandas",
        )
        sort_ed.reset_index(drop=True, inplace=True)
        return reconstruct_tibble(sort_ed, x)

    return out


# Helpers -----------------------------------------------------------------

def _check_name(name, invars):
    """Check if count is valid"""
    if name is None:
        name = _n_name(invars)

        if name != "n":
            logger.warning(
                "Storing counts in `%s`, as `n` already present in input. "
                'Use `name="new_name" to pick a new name.`',
                name,
            )
    elif not isinstance(name, str):
        raise ValueError("`name` must be a single string.")

    return name


def _n_name(invars):
    """Make sure that name does not exist in invars"""
    name = "n"
    while name in invars:
        name = "n" + name
    return name

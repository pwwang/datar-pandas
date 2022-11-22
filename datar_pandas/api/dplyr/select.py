"""Subset columns using their names and types

See source https://github.com/tidyverse/dplyr/blob/master/R/select.R
"""
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union

from datar.core.utils import logger
from datar.apis.dplyr import group_vars, select

from ...pandas import DataFrame, Index

from ...contexts import Context
from ...tibble import Tibble, TibbleGrouped
from ...utils import vars_select
from ...collections import Inverted
from ...common import setdiff, union, intersect


@select.register(DataFrame, context=Context.SELECT, backend="pandas")
def _select(
    _data: DataFrame,
    *args: Union[str, Iterable, Inverted],
    **kwargs: Mapping[str, str],
) -> Tibble:
    all_columns = _data.columns
    gvars = group_vars(_data, __ast_fallback="normal", __backend="pandas")
    selected_idx, new_names = _eval_select(
        all_columns,
        *args,
        **kwargs,
        _group_vars=gvars,
    )
    out = _data.copy()
    # nested dfs?
    if new_names:
        out.rename(columns=new_names, inplace=True)
        if (
            isinstance(out, TibbleGrouped)
            and len(intersect(gvars, list(new_names))) > 0
        ):
            out._datar["group_vars"] = [
                new_names.get(gvar, gvar) for gvar in gvars
            ]

    return out.iloc[:, selected_idx]


def _eval_select(
    _all_columns: Index,
    *args: Any,
    _group_vars: Sequence[str],
    _missing_gvars_inform: bool = True,
    **kwargs: Any,
) -> Tuple[Sequence[int], Mapping[str, str]]:
    """Evaluate selections to get locations

    Returns:
        A tuple of (selected columns, dict of old-to-new renaming columns)
    """
    selected_idx = vars_select(
        _all_columns,
        *args,
        *kwargs.values(),
    )

    if _missing_gvars_inform:
        missing = setdiff(_group_vars, _all_columns[selected_idx])
        if len(missing) > 0:
            logger.info("Adding missing grouping variables: %s", missing)

    selected_idx = union(
        _all_columns.get_indexer_for(_group_vars),
        selected_idx,
    )

    if not kwargs:
        return selected_idx, None

    rename_idx = vars_select(_all_columns, *kwargs.values())
    new_names = dict(zip(_all_columns[rename_idx], kwargs))
    return selected_idx, new_names

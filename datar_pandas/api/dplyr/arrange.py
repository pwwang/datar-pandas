"""Arrange rows by column values

See source https://github.com/tidyverse/dplyr/blob/master/R/arrange.R
"""
from typing import Any

from datar.apis.dplyr import mutate, arrange
from datar.core.names import NameNonUniqueError

from ...pandas import DataFrame

from ...contexts import Context
from ...common import union
from ...tibble import TibbleGrouped


@arrange.register(DataFrame, context=Context.PENDING, backend="pandas")
def _arrange(
    _data: DataFrame,
    *args: Any,
    _by_group: bool = False,
    **kwargs: Any,
) -> DataFrame:
    if not args and not kwargs and not _by_group:
        return _data.copy()

    if not _data.columns.is_unique:
        raise NameNonUniqueError(
            "Cannot arrange a data frame with duplicate names."
        )

    gvars = getattr(_data, "group_vars", [])

    sorting_df = mutate(
        _data,
        *args,
        __ast_fallback="normal",
        __backend="pandas",
        **kwargs,
    )
    if _by_group:
        sorting_cols = union(gvars, sorting_df._datar["mutated_cols"])
    else:
        sorting_cols = sorting_df._datar["mutated_cols"]

    sorting_df = DataFrame(sorting_df, copy=False).sort_values(
        list(sorting_cols), na_position="last"
    )
    out = _data.reindex(sorting_df.index)
    if isinstance(_data, TibbleGrouped):
        out.reset_index(drop=True, inplace=True)

    return out

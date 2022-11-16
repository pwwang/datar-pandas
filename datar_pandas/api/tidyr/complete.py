"""Complete a data frame with missing combinations of data

https://github.com/tidyverse/tidyr/blob/HEAD/R/complete.R
"""
from typing import Iterable, Mapping, Any

from datar.apis.dplyr import full_join, ungroup
from datar.apis.tidyr import replace_na, expand, complete

from ...pandas import DataFrame

from ...contexts import Context
from ...tibble import reconstruct_tibble


@complete.register(DataFrame, context=Context.PENDING, backend="pandas")
def _complete(
    data: DataFrame,
    *args: Iterable[Any],
    fill: Mapping[str, Any] = None,
    **kwargs: Iterable[Any],
) -> DataFrame:
    """Turns implicit missing values into explicit missing values.

    Args:
        data: A data frame
        *args: columns to expand. Columns can be atomic lists.
            - To find all unique combinations of x, y and z, including
              those not present in the data, supply each variable as a
              separate argument: `expand(df, x, y, z)`.
            - To find only the combinations that occur in the data, use
              nesting: `expand(df, nesting(x, y, z))`.
            - You can combine the two forms. For example,
              `expand(df, nesting(school_id, student_id), date)` would
              produce a row for each present school-student combination
              for all possible dates.

    Returns:
        Data frame with missing values completed
    """
    full = expand(
        ungroup(data, __ast_fallback="normal"),
        *args,
        **kwargs,
        __ast_fallback="normal",
    )
    if full.shape[0] == 0:
        return data.copy()

    full = full_join(
        full,
        data,
        by=full.columns.tolist(),
        __ast_fallback="normal",
    )
    full = replace_na(full, fill, __ast_fallback="normal")

    return reconstruct_tibble(data, full)
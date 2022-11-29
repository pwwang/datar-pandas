"""Unite multiple columns into one by pasting strings together"""
from typing import Union

from datar.apis.tidyr import unite

from ... import pandas as pd
from ...pandas import DataFrame, Series
from ...common import setdiff
from ...contexts import Context
from ...utils import vars_select
from ...tibble import reconstruct_tibble
from ..dplyr.group_by import ungroup


@unite.register(DataFrame, context=Context.SELECT, backend="pandas")
def _unite(
    data: DataFrame,
    col: str,
    *columns: Union[str, int],
    sep: str = "_",
    remove: bool = True,
    na_rm: bool = True,
) -> DataFrame:
    """Unite multiple columns into one by pasting strings together

    Args:
        data: A data frame.
        col: The name of the new column, as a string or symbol.
        *columns: Columns to unite
        sep: Separator to use between values.
        remove: If True, remove input columns from output data frame.
        na_rm: If True, missing values will be remove prior to uniting
            each value.

    Returns:
        The dataframe with selected columns united
    """
    all_columns = data.columns
    if not columns:
        unite_idx = range(data.shape[1])
        columns = all_columns
    else:
        unite_idx = vars_select(data, columns)
        columns = all_columns[unite_idx]

    out = ungroup(data, __ast_fallback="normal", __backend="pandas").copy()

    united = Series(out[columns].values.tolist(), index=out.index)
    if sep is not None:
        if na_rm:
            united = united.transform(
                lambda lst: [elem for elem in lst if pd.notnull(elem)]
            )

        united = united.transform(lambda x: sep.join(str(elem) for elem in x))

    # get indexes to relocate
    insert_at = int(min(unite_idx))
    out.insert(insert_at, col, united, allow_duplicates=True)

    if remove:
        to_remove = [i if i < insert_at else i + 1 for i in unite_idx]
        out = out.iloc[:, setdiff(range(out.shape[1]), to_remove)]

    return reconstruct_tibble(out, data)

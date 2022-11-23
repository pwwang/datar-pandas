"""Fill in missing values with previous or next value

https://github.com/tidyverse/tidyr/blob/HEAD/R/fill.R
"""

from typing import Union
from datar.apis.tidyr import fill

from ...pandas import DataFrame
from ...contexts import Context
from ...utils import vars_select
from ...tibble import TibbleGrouped, reconstruct_tibble


@fill.register(DataFrame, context=Context.SELECT, backend="pandas")
def _fill(
    _data: DataFrame,
    *columns: Union[str, int],
    _direction: str = "down",
) -> DataFrame:
    """Fills missing values in selected columns using the next or
    previous entry.

    See https://tidyr.tidyverse.org/reference/fill.html

    Args:
        _data: A dataframe
        *columns: Columns to fill
        _direction: Direction in which to fill missing values.
            Currently either "down" (the default), "up",
            "downup" (i.e. first down and then up) or
            "updown" (first up and then down).

    Returns:
        The dataframe with NAs being replaced.
    """
    data = _data.copy()
    if not columns:
        data = data.fillna(
            method="ffill" if _direction.startswith("down") else "bfill",
        )
        if _direction in ("updown", "downup"):
            data = data.fillna(
                method="ffill" if _direction.endswith("down") else "bfill",
            )
    else:
        colidx = vars_select(data.columns, *columns)
        data[data.columns[colidx]] = fill(
            data.iloc[:, colidx],
            _direction=_direction,
            __ast_fallback="normal",
            __backend="pandas",
        )
    return data


@fill.register(TibbleGrouped, context=Context.SELECT, backend="pandas")
def _fill_grouped(
    _data: TibbleGrouped,
    *columns: str,
    _direction: str = "down",
) -> TibbleGrouped:
    # TibbleGrouped
    out = _data._datar["grouped"].apply(
        fill,
        *columns,
        _direction=_direction,
        __ast_fallback="normal",
        __backend="pandas",
        # drop the index, pandas 1.4 and <1.4 act differently
    ).sort_index().reset_index(drop=True)
    return reconstruct_tibble(out, _data)

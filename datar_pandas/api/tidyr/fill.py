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
        if _direction.startswith("down"):
            data = data.ffill()
        else:
            data = data.bfill()

        if _direction in ("updown", "downup"):
            if _direction.endswith("down"):
                data = data.ffill()
            else:
                data = data.bfill()
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
    # Use transform instead of apply to avoid include_groups issue in pandas 3
    data = _data.copy()
    if not columns:  # pragma: no cover
        colidx = list(range(len(_data.columns)))
    else:
        colidx = vars_select(_data.columns, *columns)
    cols = list(_data.columns[colidx])
    grouped = _data._datar["grouped"]
    group_vars = list(_data.group_vars)

    for col in cols:
        sgb = grouped[col]
        if _direction.startswith("down"):
            data[col] = sgb.transform(lambda g: g.ffill())
        else:  # pragma: no cover
            data[col] = sgb.transform(lambda g: g.bfill())

        if len(_direction) > 4:  # pragma: no cover
            # Second pass with re-grouped data
            sgb2 = data[col].groupby(data[group_vars] if group_vars else data.index)
            if _direction.endswith("down"):
                data[col] = sgb2.transform(lambda g: g.ffill())
            else:
                data[col] = sgb2.transform(lambda g: g.bfill())

    return reconstruct_tibble(data, _data)

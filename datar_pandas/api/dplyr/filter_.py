"""Subset rows using column values

See source https://github.com/tidyverse/dplyr/blob/master/R/filter.R
"""
import operator

import numpy as np
from datar.core.utils import logger
from datar.apis.dplyr import ungroup, filter_

from ...typing import Data, Bool
from ...pandas import DataFrame, Series
from ...contexts import Context
from ...broadcast import broadcast_to
from ...tibble import Tibble, TibbleGrouped, reconstruct_tibble
from ...operators import _binop


@filter_.register(DataFrame, context=Context.EVAL, backend="pandas")
def _filter(
    _data: DataFrame,
    *conditions: Data[Bool],
    _preserve: bool = False,
) -> Tibble:
    if _preserve:
        logger.warning("`filter()` doesn't support `_preserve` argument yet.")

    if _data.shape[0] == 0 or not conditions:
        return _data.copy()

    condition = np.array(True)
    for cond in conditions:
        condition = _binop(operator.and_, condition, cond)

    grouper = None
    if isinstance(_data, TibbleGrouped):
        grouper = _data._datar["grouped"].grouper

    condition = broadcast_to(condition, _data.index, grouper)
    if isinstance(condition, np.bool_):
        condition = bool(condition)

    if condition is True:
        return _data.copy()
    if condition is False:
        return _data.take([])

    if isinstance(condition, Series):
        condition = condition.values

    out = ungroup(_data, __ast_fallback="normal", __backend="pandas")[condition]
    if isinstance(_data, TibbleGrouped):
        out.reset_index(drop=True, inplace=True)

    return reconstruct_tibble(out, _data)

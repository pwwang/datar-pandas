"""Replace NAs with specified values"""
from functools import singledispatch
from typing import Any, Iterable

import numpy as np
from datar.apis.tidyr import replace_na

from ... import pandas as pd
from ...pandas import DataFrame, Series
from ...common import is_scalar
from ...contexts import Context


@singledispatch
def _replace_na(data: Iterable[Any], replace: Any) -> Iterable[Any]:
    """Replace NA for any iterables"""
    return np.array(
        [
            replace
            if is_scalar(elem) and pd.isnull(elem)
            else elem
            for elem in data
        ]
    )


@_replace_na.register(np.ndarray)
@_replace_na.register(Series)
def _replace_na_array_or_series(data, replace):
    """Replace NA for np.ndarray or Series"""
    ret = data.copy()
    ret[pd.isnull(ret)] = replace
    return ret


@_replace_na.register(DataFrame)
def _replace_na_df(data: DataFrame, replace: Any) -> DataFrame:
    """Replace NA for np.ndarray or DataFrame"""
    # TODO: allow replace to be a list as an entire value to replace
    return data.fillna(replace)


@replace_na.register(
    (DataFrame, Series, np.ndarray, list, tuple, set),
    context=Context.EVAL,
    backend="pandas",
)
def _replace_na_registered(
    data: Iterable[Any],
    data_or_replace: Any = None,
    replace: Any = None,
) -> Any:
    """Replace NA with a value

    This function can be also used not as a verb. As a function called as
    an argument in a verb, data is passed implicitly. Then one could
    pass data_or_replace as the data to replace.

    Args:
        data: The data piped in
        data_or_replace: When called as argument of a verb, this is the
            data to replace. Otherwise this is the replacement.
        replace: The value to replace with
            Can only be a scalar or dict for data frame.
            So replace NA with a list is not supported yet.

    Returns:
        Corresponding data with NAs replaced
    """
    if data_or_replace is None and replace is None:
        return data.copy()

    if replace is None:
        # no replace, then data_or_replace should be replace
        replace = data_or_replace
    else:
        # replace specified, determine data
        # If data_or_replace is specified, it's data
        data = data if data_or_replace is None else data_or_replace

    return _replace_na(data, replace)

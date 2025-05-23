"""Vectorised if and multiple if-else

https://github.com/tidyverse/dplyr/blob/master/R/if_else.R
https://github.com/tidyverse/dplyr/blob/master/R/case_when.R
"""
import numpy as np
from datar.apis.dplyr import if_else, case_when, case_match

from ... import pandas as pd
from ...utils import meta_kwargs, get_grouper
from ...pandas import Series, SeriesGroupBy, get_obj, option_context
from ...tibble import Tibble, reconstruct_tibble
from ..dplyr.group_by import ungroup


@if_else.register(object, backend="pandas")
def _if_else(condition, true, false, missing=None):
    if missing is None:
        missing = np.nan
        na_conds = False
    else:
        na_conds = pd.isnull(condition)

    newcond = condition
    if isinstance(condition, Series):
        newcond = condition.fillna(False)
    elif isinstance(condition, np.ndarray):
        newcond = np.nan_to_num(condition, copy=True)
    else:
        newcond = np.nan_to_num(condition, copy=True, nan=0.0)

    newcond = newcond.astype(bool)

    out = case_when(
        na_conds,  # 0
        missing,  # 1
        ~newcond,  # 2
        false,  # 3
        newcond,  # 4
        true,  # 5
        True,  # 6
        missing,  # 7
        **meta_kwargs,
    )

    if isinstance(condition, Series):
        out.index = condition.index
        out.name = condition.name

    return out


@if_else.register(SeriesGroupBy, backend="pandas")
def _if_else_sgb(condition, true, false, missing=None):
    if missing is None:
        missing = np.nan
    df = Tibble.from_args(
        condition,
        true,
        false,
        missing,
        _name_repair="minimal",
    )
    # use obj so df.x won't get a SeriesGroupBy
    grouped = df._datar["grouped"]
    # print(get_obj(grouped))
    out = if_else(
        get_obj(grouped).iloc[:, 0],
        get_obj(grouped).iloc[:, 1],
        get_obj(grouped).iloc[:, 2],
        get_obj(grouped).iloc[:, 3],
    )
    return out.groupby(
        get_grouper(condition),
        observed=condition.observed,
        sort=condition.sort,
        dropna=condition.dropna,
    )


@case_when.register(object, backend="pandas")
def _case_when(when, case, *when_cases):
    if len(when_cases) % 2 != 0:
        raise ValueError("Case-value not paired.")

    when_cases = (when, case, *when_cases)

    is_series = any(
        isinstance(wc, (Series, SeriesGroupBy)) for wc in when_cases
    )
    df = Tibble.from_args(*when_cases, _name_repair="minimal")
    ungrouped = ungroup(df, **meta_kwargs)

    value = Series(np.nan, index=ungrouped.index)
    with option_context("future.no_silent_downcasting", True):
        for i in range(ungrouped.shape[1] - 1, 0, -2):
            condition = ungrouped.iloc[:, i - 1].fillna(False).values.astype(bool)
            value[condition] = ungrouped.iloc[:, i][condition]

    value = value.to_frame(name="when_case_result")
    value = reconstruct_tibble(value, df)
    value = value["when_case_result"]
    return value if is_series else value.values


@case_match.register(object, backend="pandas")
def _case_match(_x, *args, _default=None, _dtypes=None):
    if len(args) % 2 != 0 or len(args) == 0:
        raise ValueError("condition-value not paired.")

    x = ungroup(_x, **meta_kwargs) if isinstance(_x, SeriesGroupBy) else _x
    cases = (
        arg
        if i % 2 == 1
        else np.in1d(x, arg).astype(bool) | pd.isnull(x)
        for i, arg in enumerate(args)
    )
    cases = (*cases, True, [None] if _default is None else _default)
    out = case_when(*cases, **meta_kwargs)
    if _dtypes is not None:
        out = out.astype(_dtypes)

    if not isinstance(_x, SeriesGroupBy):
        return out

    return out.groupby(
        get_grouper(_x),
        observed=_x.observed,
        sort=_x.sort,
        dropna=_x.dropna,
    )

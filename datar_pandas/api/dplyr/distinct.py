"""Subset distinct/unique rows

See source https://github.com/tidyverse/dplyr/blob/master/R/distinct.R
"""
from typing import Any
from pipda.reference import Reference
from datar.apis.dplyr import mutate, distinct, n_distinct

from ...pandas import DataFrame, Series, PandasObject
from ...contexts import Context
from ...tibble import Tibble, TibbleGrouped, reconstruct_tibble
from ...common import union, setdiff, intersect, unique
from ...factory import func_bootstrap


@distinct.register(DataFrame, context=Context.PENDING, backend="pandas")
def _distinct(
    _data: DataFrame,
    *args: Any,
    _keep_all: bool = False,
    **kwargs: Any,
) -> Tibble:
    if not args and not kwargs:
        out = _data.drop_duplicates()
    else:
        if (
            not kwargs
            # optimize:
            # iris >> distinct(f.Species, f.Sepal_Length)
            # We don't need to do mutation
            and all(
                isinstance(expr, Reference)
                and expr._pipda_level == 1
                and expr._pipda_ref in _data.columns
                for expr in args
            )
        ):
            subset = [expr._pipda_ref for expr in args]
            ucols = getattr(_data, "group_vars", [])
            ucols.extend(subset)
            ucols = unique(ucols)
            uniq = DataFrame(_data).drop_duplicates(subset=subset)[ucols]
        else:
            # keep_none_prefers_new_order
            uniq = DataFrame(
                mutate(
                    _data,
                    *args,
                    **kwargs,
                    _keep="none",
                    __ast_fallback="normal",
                    __backend="pandas",
                )
            ).drop_duplicates()

        if not _keep_all:
            # keep original order
            out = uniq[
                union(
                    intersect(_data.columns, uniq.columns),
                    setdiff(uniq.columns, _data.columns),
                )
            ]
        else:
            out = _data.loc[uniq.index, :].copy()
            out[uniq.columns.tolist()] = uniq

    if isinstance(out, TibbleGrouped):
        out = out.reset_index(drop=True)

    return reconstruct_tibble(Tibble(out, copy=False), _data)


@func_bootstrap(n_distinct, kind="agg")
def _n_distinct_bootstrap(x: PandasObject, na_rm: bool = True):
    """Get the length of distinct elements"""
    return x.nunique(dropna=na_rm)


@n_distinct.register(object, context=Context.EVAL, backend="pandas")
def _n_distinct(x: Any, na_rm: bool = True):
    return Series(x).nunique(dropna=na_rm)

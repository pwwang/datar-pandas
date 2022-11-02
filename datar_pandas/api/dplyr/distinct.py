"""Subset distinct/unique rows

See source https://github.com/tidyverse/dplyr/blob/master/R/distinct.R
"""
from typing import Any
from pipda.reference import Reference
from datar.apis.dplyr import mutate, distinct, n_distinct

from ...pandas import DataFrame, Series, GroupBy, PandasObject
from ...utils import PandasData
from ...contexts import Context
from ...tibble import Tibble, TibbleGrouped, reconstruct_tibble
from ...common import union, setdiff, intersect, unique
from ...factory import func_bootstrap


@distinct.register(DataFrame, context=Context.PENDING)
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
            ucols = unique(ucols, __ast_fallback="normal")
            uniq = _data.drop_duplicates(subset=subset)[ucols]
        else:
            # keep_none_prefers_new_order
            uniq = (
                mutate(
                    _data,
                    *args,
                    **kwargs,
                    _keep="none",
                    __ast_fallback="normal",
                )
            ).drop_duplicates()

        if not _keep_all:
            # keep original order
            out = uniq[
                union(
                    intersect(
                        _data.columns,
                        uniq.columns,
                        __ast_fallback="normal",
                    ),
                    setdiff(
                        uniq.columns,
                        _data.columns,
                        __ast_fallback="normal",
                    ),
                    __ast_fallback="normal",
                )
            ]
        else:
            out = _data.loc[uniq.index, :].copy()
            out[uniq.columns.tolist()] = uniq

    return reconstruct_tibble(_data, Tibble(out, copy=False))


@n_distinct.register((object, PandasData), context=Context.EVAL)
def _n_distinct(x: Any, na_rm: bool = True):
    x = x.data if isinstance(x, PandasData) else x
    return Series(x).nunique(dropna=na_rm)


@func_bootstrap(n_distinct, kind="agg")
def _n_distinct_bootstrap(x: PandasObject, na_rm: bool = True):
    """Get the length of distinct elements"""
    return x.nunique(dropna=na_rm)


@n_distinct.register(TibbleGrouped, context=Context.EVAL)
def _n_distinct_grouped(x: Any, na_rm: bool = True):
    return x._datar["grouped"].agg("nunique", dropna=na_rm)


@n_distinct.register(GroupBy, context=Context.EVAL)
def _n_distinct_groupby(x: Any, na_rm: bool = True):
    return x.agg("nunique", dropna=na_rm)

"""Mutating joins"""
from datar.apis.dplyr import (
    filter_,
    inner_join,
    left_join,
    right_join,
    full_join,
    semi_join,
    anti_join,
    nest_join,
)

from ... import pandas as pd
from ...typing import Data, Str
from ...pandas import Categorical, DataFrame, SeriesGroupBy, get_obj
from ...common import is_factor, is_scalar, intersect, setdiff, union
from ...contexts import Context
from ...tibble import reconstruct_tibble


def _join(
    x,
    y,
    how,
    by=None,
    copy=False,
    suffix=("_x", "_y"),
    # na_matches = "", # TODO: how?
    keep=False,
):
    """General join"""
    # make sure df.x returns a Series not SeriesGroupBy for TibbleGrouped
    newx = DataFrame(x, copy=False)
    y = DataFrame(y, copy=False)

    if by is not None and not by:
        ret = pd.merge(newx, y, how="cross", copy=copy, suffixes=suffix)

    elif isinstance(by, dict):
        left_on = list(by)
        right_on = list(by.values())
        ret = pd.merge(
            newx,
            y,
            left_on=left_on,
            right_on=right_on,
            how=how,
            copy=copy,
            suffixes=suffix,
        )
        if not keep:
            to_drop = setdiff(right_on, left_on)
            ret.drop(columns=to_drop, inplace=True)

    elif keep:
        if by is None:
            by = intersect(newx.columns, y.columns)
        # on=... doesn't keep both by columns in left and right
        left_on = [f"{col}{suffix[0]}" for col in by]
        right_on = [f"{col}{suffix[1]}" for col in by]
        newx = newx.rename(columns=dict(zip(by, left_on)))
        y = y.rename(columns=dict(zip(by, right_on)))
        ret = pd.merge(
            newx,
            y,
            left_on=left_on,
            right_on=right_on,
            how=how,
            copy=copy,
            suffixes=suffix,
        )

    else:
        if by is None:
            by = intersect(newx.columns, y.columns)

        by = [by] if is_scalar(by) else list(by)
        ret = pd.merge(newx, y, on=by, how=how, copy=copy, suffixes=suffix)
        for col in by:
            # try recovering factor columns
            xcol = x[col]
            ycol = y[col]
            if isinstance(xcol, SeriesGroupBy):
                xcol = get_obj(xcol)
            if isinstance(ycol, SeriesGroupBy):  # pragma: no cover
                ycol = get_obj(ycol)
            if is_factor(xcol) and is_factor(ycol):
                ret[col] = Categorical(
                    ret[col],
                    categories=union(
                        xcol.cat.categories,
                        ycol.cat.categories,
                    ),
                )

    return reconstruct_tibble(ret, x)


@inner_join.register(
    DataFrame,
    context=Context.EVAL,
    kw_context={"by": Context.SELECT},
    backend="pandas",
)
def _inner_join(
    x: DataFrame,
    y: DataFrame,
    *,
    by: Data[Str] = None,
    copy: bool = False,
    suffix: Data[Str] = ("_x", "_y"),
    keep: bool = False,
) -> DataFrame:
    return _join(
        x,
        y,
        how="inner",
        by=by,
        copy=copy,
        suffix=suffix,
        keep=keep,
    )


@left_join.register(
    DataFrame,
    context=Context.EVAL,
    kw_context={"by": Context.SELECT},
    backend="pandas",
)
def _left_join(
    x: DataFrame,
    y: DataFrame,
    *,
    by: Data[Str] = None,
    copy: bool = False,
    suffix: Data[Str] = ("_x", "_y"),
    keep: bool = False,
) -> DataFrame:
    return _join(
        x,
        y,
        how="left",
        by=by,
        copy=copy,
        suffix=suffix,
        keep=keep,
    )


@right_join.register(
    DataFrame,
    context=Context.EVAL,
    kw_context={"by": Context.SELECT},
    backend="pandas",
)
def _right_join(
    x: DataFrame,
    y: DataFrame,
    *,
    by: Data[Str] = None,
    copy: bool = False,
    suffix: Data[Str] = ("_x", "_y"),
    keep: bool = False,
) -> DataFrame:
    return _join(
        x,
        y,
        how="right",
        by=by,
        copy=copy,
        suffix=suffix,
        keep=keep,
    )


@full_join.register(
    DataFrame,
    context=Context.EVAL,
    kw_context={"by": Context.SELECT},
    backend="pandas",
)
def _full_join(
    x: DataFrame,
    y: DataFrame,
    *,
    by: Data[Str] = None,
    copy: bool = False,
    suffix: Data[Str] = ("_x", "_y"),
    keep: bool = False,
) -> DataFrame:
    return _join(
        x,
        y,
        how="outer",
        by=by,
        copy=copy,
        suffix=suffix,
        keep=keep,
    )


@semi_join.register(
    DataFrame,
    context=Context.EVAL,
    kw_context={"by": Context.SELECT},
    backend="pandas",
)
def _semi_join(
    x: DataFrame,
    y: DataFrame,
    *,
    by: Data[Str] = None,
    copy: bool = False,
) -> DataFrame:
    on = _merge_on(by)
    right_on = on.get("right_on", on.get("on", y.columns))

    ret = pd.merge(
        DataFrame(x, copy=False),
        # fix #71: semi_join returns duplicated rows
        DataFrame(y, copy=False).drop_duplicates(right_on),
        how="left",
        copy=copy,
        suffixes=["", "_y"],
        indicator="__merge__",
        **on,
    )
    ret = ret.loc[ret["__merge__"] == "both", x.columns]
    return reconstruct_tibble(ret, x)


@anti_join.register(
    DataFrame,
    context=Context.EVAL,
    kw_context={"by": Context.SELECT},
    backend="pandas",
)
def anti_join(
    x: DataFrame,
    y: DataFrame,
    *,
    by: Data[Str] = None,
    copy: bool = False,
) -> DataFrame:
    ret = pd.merge(
        DataFrame(x, copy=False),
        DataFrame(y, copy=False),
        how="left",
        copy=copy,
        suffixes=["", "_y"],
        indicator=True,
        **_merge_on(by),
    )
    ret = ret.loc[ret._merge != "both", x.columns]
    return reconstruct_tibble(ret, x)


@nest_join.register(
    DataFrame,
    context=Context.EVAL,
    kw_context={"by": Context.SELECT},
    backend="pandas",
)
def _nest_join(
    x: DataFrame,
    y: DataFrame,
    *,
    by: Data[Str] = None,
    copy: bool = False,
    keep: bool = False,
    name: str = None,
) -> DataFrame:
    on = by
    newx = DataFrame(x, copy=False)
    y = DataFrame(y, copy=False)
    if isinstance(by, (list, tuple, set)):
        on = dict(zip(by, by))
    elif by is None:
        common_cols = intersect(newx.columns, y.columns)
        on = dict(zip(common_cols, common_cols))
    elif not isinstance(by, dict):
        on = {by: by}

    if copy:
        newx = newx.copy()

    def get_nested_df(row):
        row = getattr(row, "obj", row)
        condition = None
        for key in on:
            if condition is None:
                condition = y[on[key]] == row[key]
            else:
                condition = condition & (y[on[key]] == row[key])
        df = filter_(y, condition, __ast_fallback="normal", __backend="pandas")
        if not keep:
            df = df[setdiff(df.columns, list(on.values()))]

        return df

    y_matched = newx.apply(get_nested_df, axis=1)
    y_name = name or "_y_joined"
    if y_name:
        y_matched = y_matched.to_frame(name=y_name)

    out = pd.concat([newx, y_matched], axis=1)
    return reconstruct_tibble(out, x)


def _merge_on(by):
    """Calculate argument on for pandas.merge()"""
    if by is None:
        return {}
    if isinstance(by, dict):
        return {"left_on": list(by), "right_on": list(by.values())}
    return {"on": by}

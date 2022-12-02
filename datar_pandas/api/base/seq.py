import numpy as np
from datar_numpy.utils import make_array
from datar.apis.base import (  # noqa: F401
    append,
    c_,
    length,
    lengths,
    match,
    order,
    rep,
    rev,
    sample,
    seq,
    seq_along,
    seq_len,
    sort,
)
from datar_numpy.api import seq as _  # noqa: F401

from ... import pandas as pd
from ...broadcast import _grouper_compatible
from ...collections import Collection
from ...common import is_integer, is_scalar
from ...factory import func_bootstrap
from ...pandas import (
    DataFrame,
    PandasObject,
    Series,
    SeriesGroupBy,
    NDFrame,
    get_obj,
)
from ...tibble import Tibble, TibbleGrouped, reconstruct_tibble

func_bootstrap(length, func=lambda x: x.shape[0], kind="agg")
func_bootstrap(
    lengths,
    func=lambda x: x.agg(lambda y: 1 if is_scalar(y) else len(y)),
    kind="agg",
)
func_bootstrap(
    sample,
    func=sample.dispatch(object, backend="numpy"),
    exclude={"size", "replace", "prob"},
)


@match.register(PandasObject, backend="pandas")
def _match(x, table, nomatch=-1):
    def match_dummy(xx, tab):
        sorter = np.argsort(tab)
        if isinstance(sorter, Series):
            sorter = sorter.values
        searched = np.searchsorted(tab, xx, sorter=sorter).ravel()
        out = sorter.take(searched, mode="clip")
        out[~np.isin(xx, tab)] = nomatch
        return out

    if isinstance(x, SeriesGroupBy):
        # length of each group may differ
        # table could be, for example, unique elements of each group in x
        x1 = x.agg(tuple)
        x1 = x1.groupby(
            x1.index,
            observed=x.observed,
            sort=x.sort,
            dropna=x.dropna,
        )
        df = get_obj(x1).to_frame()
        if isinstance(table, SeriesGroupBy):
            t1 = table.agg(tuple)
            t1 = t1.groupby(
                t1.index,
                observed=table.observed,
                sort=table.sort,
                dropna=table.dropna,
            )
            if not _grouper_compatible(x1.grouper, t1.grouper):
                raise ValueError("Grouping of x and table are not compatible")
            df["table"] = get_obj(t1)
        elif isinstance(table, Series):
            t1 = table.groupby(
                table.index,
                observed=True,
                sort=False,
                dropna=False,
            ).agg(tuple)
            t1 = t1.groupby(
                t1.index,
                observed=x1.observed,
                sort=x1.sort,
                dropna=x1.dropna,
            )
            if not _grouper_compatible(x1.grouper, t1.grouper):
                df["table"] = [make_array(table)] * df.shape[0]
            else:
                df["table"] = get_obj(t1)
        else:
            df["table"] = [make_array(table)] * df.shape[0]

        out = (
            df
            # not working for pandas 1.3.0
            # .agg(lambda row: match_dummy(*row), axis=1)
            .apply(lambda row: match_dummy(*row), axis=1)
            .explode()
            .astype(int)
        ).groupby(
            x.grouper,
            observed=x.observed,
            sort=x.sort,
            dropna=x.dropna,
        )
        if getattr(x, "is_rowwise", False):
            out.is_rowwise = True
        return out

    if isinstance(x, Series):
        return Series(match_dummy(x, table), index=x.index)

    return match_dummy(x, table)


def _order_post(out, x, decreasing=False, na_last=True):
    if not isinstance(x, SeriesGroupBy):
        return out

    return (
        out.explode()
        .astype(int)
        .groupby(
            x.grouper,
            observed=x.observed,
            sort=x.sort,
            dropna=x.dropna,
        )
    )


@func_bootstrap(order, kind="transform", post=_order_post)
def _order(x: Series, decreasing=False, na_last=True):
    if not na_last or decreasing:
        na = -np.inf
    else:
        na = np.inf

    out = np.argsort(x.fillna(na))
    if decreasing:
        out = out[::-1]
        out.index = x.index
    return out


@rep.register(SeriesGroupBy, backend="pandas")
def _rep_sgb(x, times=1, length=None, each=1):
    data = {}
    times_sgb = isinstance(times, SeriesGroupBy)
    length_sgb = isinstance(length, SeriesGroupBy)
    each_sgb = isinstance(each, SeriesGroupBy)
    if times_sgb:
        data["times"] = times
    if length_sgb:
        data["length"] = length
    if each_sgb:
        data["each"] = each
    # in case x is not grouped
    data["x"] = x

    df = Tibble.from_args(**data)
    out = (
        df._datar["grouped"]
        .apply(
            lambda subdf: rep.dispatch(object, backend="numpy")(
                subdf["x"],
                times=subdf["times"] if times_sgb else times,
                length=subdf["length"] if length_sgb else length,
                each=subdf["each"] if each_sgb else each,
            )
        )
        .explode()
        .astype(df["x"].obj.dtype)
    )
    grouping = out.index
    return out.reset_index(drop=True).groupby(
        grouping,
        observed=df._datar["grouped"].observed,
        sort=df._datar["grouped"].sort,
        dropna=df._datar["grouped"].dropna,
    )


@rep.register(DataFrame, backend="pandas")
def _rep_df(x, times=1, length=None, each=1):
    if not is_integer(each) or each != 1:
        raise ValueError("`each` has to be 1 to replicate a data frame.")

    out = pd.concat([x] * times, ignore_index=True)
    if length is not None:
        out = out.iloc[:length, :]

    return out


@rep.register(TibbleGrouped, backend="pandas")
def _rep_grouped(x, times=1, length=None, each=1):
    out = rep.dispatch(DataFrame)(x, times, length, each)
    return reconstruct_tibble(out, x)


@c_.register(SeriesGroupBy, backend="pandas")
def _c_sgb(*args):
    values = []
    for elem in args:
        if isinstance(elem, SeriesGroupBy):
            values.append(elem.agg(list))
        elif is_scalar(elem):
            values.append(elem)
        else:
            values.extend(elem)

    df = Tibble.from_args(*values)
    # pandas 1.3.0 expand list into columns after aggregation
    # pandas 1.3.2 has this fixed
    # https://github.com/pandas-dev/pandas/issues/42727
    out = df.agg(
        lambda row: Collection(*row),
        axis=1,
    )
    if isinstance(out, DataFrame):  # pragma: no cover
        # pandas < 1.3.2
        out = Series(out.values.tolist(), index=out.index, dtype=object)

    out = out.explode().convert_dtypes()
    # TODO: check observed, sort and dropna?
    out = out.reset_index(drop=True).groupby(out.index)
    return out


# Define different function so that it has higher priority
@c_.register(NDFrame, backend="pandas")
def _c_ndframe(*args):
    return Collection(*args)


@c_.register(object, backend="pandas", favored=True)
def _c(*args):
    if any(isinstance(arg, SeriesGroupBy) for arg in args):
        return c_.dispatch(SeriesGroupBy, backend="pandas")(*args)

    return c_.dispatch(NDFrame, backend="pandas")(*args)


@func_bootstrap(rev, kind="transform")
def _rev(x, __args_raw=None):
    out = x[::-1]
    out.index = x.index
    return out


@func_bootstrap(sort, kind="transform")
def _sort(
    x,
    decreasing=False,
    na_last=True,
):
    idx = order(x.values, decreasing=decreasing, na_last=na_last)
    out = x.iloc[idx]
    out.index = x.index
    return out


func_bootstrap(seq_len, func=seq_len.dispatch(np.ndarray, backend="numpy"))
func_bootstrap(seq_along, func=seq_along.dispatch(object, backend="numpy"))

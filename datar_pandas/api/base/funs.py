"""Some functions from R-base

If a function uses DataFrame/TibbleGrouped as first argument, it may be
registered by `register_verb` and should be placed in `./verbs.py`
"""
# numpy impl of diff
import datar_numpy.api.sets  # noqa: F401
from datar.apis.base import (
    cut,
    diff,
    outer,
    rank,
)

from ... import pandas as pd
from ...factory import func_bootstrap
from ...tibble import Tibble


@func_bootstrap(cut)
def _cut(
    x,
    breaks,
    labels=None,
    include_lowest=False,
    right=True,
    precision=2,
    ordered_result=False,
):
    if labels is None:
        ordered_result = True

    return pd.cut(
        x,
        breaks,
        labels=labels,
        include_lowest=include_lowest,
        right=right,
        precision=precision,
        ordered=ordered_result,
    )


def _diff_sgb_post(out, x, lag=1, differences=1):
    """Post process diff on SeriesGroupBy object"""
    non_na_out = out[out.transform(len) > 0]
    non_na_out = non_na_out.explode()
    grouping = pd.Categorical(non_na_out.index, categories=out.index.unique())
    return (
        non_na_out.explode()
        .reset_index(drop=True)
        .groupby(grouping, observed=False, sort=x.sort, dropna=x.dropna)
    )


func_bootstrap(
    diff,
    func=diff.dispatch(object, backend="numpy"),
    post=_diff_sgb_post,
)


func_bootstrap(
    outer,
    func=outer.dispatch(object, backend="numpy"),
    post=lambda out, *args, **kwargs: Tibble(out),
)


func_bootstrap(
    rank,
    func=rank.dispatch(object, backend="numpy"),
    kind="transform",
)

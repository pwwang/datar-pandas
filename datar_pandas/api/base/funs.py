"""Some functions from R-base

If a function uses DataFrame/TibbleGrouped as first argument, it may be
registered by `register_verb` and should be placed in `./verbs.py`
"""
from datar.apis.base import (
    cut,
    diff,
    outer,
    rank,
)
# numpy impl of diff
from datar_numpy.api import sets as _  # noqa: F401
# numpy impl of rank
from datar_numpy.api import seq as _  # noqa: F401, F811

from ... import pandas as pd
from ...factory import func_bootstrap
from ...tibble import Tibble


@func_bootstrap(cut)
@cut.register(object, backend="pandas")
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
    exclude={"lag", "differences"},
)


func_bootstrap(
    rank,
    func=rank.dispatch(object, backend="numpy"),
    kind="transform",
)


@outer.register(object, backend="pandas", favored=True)
def _outer(x, y, fun="*"):
    out = outer(x, y, fun=fun, __backend="numpy")
    return Tibble(out)

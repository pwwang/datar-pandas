from datar.apis.base import (
    rbinom,
    rcauchy,
    rchisq,
    rexp,
    rnorm,
    rpois,
    runif,
)


from ...factory import func_bootstrap
from ...contexts import Context
from ...pandas import SeriesGroupBy


def _sgb_post(__out, n, __args_raw=None, **kwargs):
    """Post process SeriesGroupBy object"""
    n = __args_raw["n"]
    if not isinstance(n, SeriesGroupBy):
        return __out

    return __out.groupby(
        __out.index.get_level_values(0),
        observed=n.observed,
        sort=n.sort,
        dropna=n.dropna,
    ).agg(list)


func_bootstrap(
    rbinom,
    func=rbinom.dispatch(object),
    context=Context.EVAL,
    post=_sgb_post,
    data_args={"n", "size", "prob"},
)

func_bootstrap(
    rcauchy,
    func=rcauchy.dispatch(object),
    context=Context.EVAL,
    post=_sgb_post,
    data_args={"n", "location", "scale"},
)

func_bootstrap(
    rchisq,
    func=rchisq.dispatch(object),
    context=Context.EVAL,
    post=_sgb_post,
    data_args={"n", "df"},
)

func_bootstrap(
    rexp,
    func=rexp.dispatch(object),
    context=Context.EVAL,
    post=_sgb_post,
    data_args={"n", "rate"},
)

func_bootstrap(
    rnorm,
    func=rnorm.dispatch(object),
    context=Context.EVAL,
    post=_sgb_post,
    data_args={"n", "mean", "sd"},
)

func_bootstrap(
    rpois,
    func=rpois.dispatch(object),
    context=Context.EVAL,
    post=_sgb_post,
    data_args={"n", "lambda"},
)

func_bootstrap(
    runif,
    func=runif.dispatch(object),
    context=Context.EVAL,
    post=_sgb_post,
    data_args={"n", "min", "max"},
)

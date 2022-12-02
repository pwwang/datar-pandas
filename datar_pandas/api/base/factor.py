"""Factors, implemented with pandas' Categorical

The huge difference:
R's factors support NAs in levels but Categorical cannot have NAs in categories.
"""
from datar.apis.base import (
    factor,
    ordered,
    levels,
    nlevels,
    droplevels,
)

from ...common import is_scalar
from ...factory import func_bootstrap
from ...pandas import (
    Categorical,
    Series,
    SeriesGroupBy,
    is_categorical_dtype,
    get_obj,
)


@func_bootstrap(droplevels, kind="transform")
def _droplevels_bootsstrap(x):
    return x.cat.remove_unused_categories()


@droplevels.register(Categorical, backend="pandas")
def _droplevels_cat(x):
    return x.remove_unused_categories()


@func_bootstrap(levels)
def _levels(x):
    if not is_categorical_dtype(x):
        return None

    return x.cat.categories.values.copy()


@levels.register(object, backend="pandas")
def _levels_object(x):
    return None


@levels.register(Categorical, backend="pandas")
def _levels_cat(x):
    return x.categories.values.copy()


@func_bootstrap(nlevels, kind="agg")
def _nlevels_bootstrap(x) -> int:
    lvls = levels(x, __ast_fallback="normal", __backend="pandas")
    return 0 if lvls is None else len(lvls)


@nlevels.register(object, backend="pandas")
def _nlevels_object(x) -> int:
    return 0


@nlevels.register(Categorical, backend="pandas")
def _nlevels_cat(x):
    return x.categories.size


@factor.register(object, backend="pandas")
def _factor(
    x=None,
    *,
    levels=None,
    labels=None,
    exclude=None,
    ordered=False,
    nmax=None,
):
    if isinstance(x, SeriesGroupBy):
        out = factor(
            get_obj(x),
            levels=levels,
            exclude=exclude,
            ordered=ordered,
            __ast_fallback="normal",
        )
        return Series(out, index=get_obj(x).index).groupby(
            x.grouper,
            observed=x.observed,
            sort=x.sort,
            dropna=x.dropna,
        )

    if x is None:
        x = []

    # pandas v1.3.0
    # FutureWarning: Allowing scalars in the Categorical constructor
    # is deprecated and will raise in a future version.
    if is_scalar(x):
        x = [x]

    if is_categorical_dtype(x):
        x = x.to_numpy()

    ret = Categorical(x, categories=levels, ordered=ordered)
    if exclude in [False, None]:
        return ret

    if is_scalar(exclude):
        exclude = [exclude]

    return ret.remove_categories(exclude)


@ordered.register(object, backend="pandas")
def _ordered(x, levels=None):
    return factor(x, levels=levels).as_ordered()

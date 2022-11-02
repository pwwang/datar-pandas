"""Factors, implemented with pandas' Categorical

The huge difference:
R's factors support NAs in levels but Categorical cannot have NAs in categories.
"""
import numpy as np
from datar.apis.base import (
    factor,
    ordered,
    levels,
    nlevels,
    droplevels,
)

from ...common import is_scalar
from ...contexts import Context
from ...utils import PandasData
from ...factory import func_bootstrap
from ...pandas import Categorical, Series, SeriesGroupBy, is_categorical_dtype


@func_bootstrap(droplevels, kind="transform")
def _droplevels_bootsstrap(x):
    return x.cat.remove_unused_categories()


@droplevels.register(Categorical, context=Context.EVAL)
def _droplevels_cat(x):
    return x.remove_unused_categories()


@func_bootstrap(levels, kind="transform")
def _levels(x):
    if not is_categorical_dtype(x):
        return None

    return x.cat.categories.values.copy()


@levels.register(Categorical, context=Context.EVAL)
def _levels_cat(x):
    return x.categories


@func_bootstrap(nlevels, kind="transform")
def _nlevels_bootstrap(x) -> int:
    lvls = levels(x)
    return 0 if lvls is None else len(lvls)


@nlevels.register(Categorical, context=Context.EVAL)
def _nlevels_cat(x):
    return nlevels.dispatch(Series)(x)


@factor.register((object, PandasData), context=Context.EVAL)
def _factor(x, levels=None, exclude=np.nan, ordered=False):
    x = x.data if isinstance(x, PandasData) else x
    if isinstance(x, SeriesGroupBy):
        out = factor(
            x.obj,
            levels=levels,
            exclude=exclude,
            ordered=ordered,
            __ast_fallback="normal",
        )
        return Series(out, index=x.obj.index).groupby(
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


@ordered.register((object, PandasData), context=Context.EVAL)
def _ordered(x, levels=None):
    x = x.data if isinstance(x, PandasData) else x
    return factor(x, levels=levels).as_ordered()

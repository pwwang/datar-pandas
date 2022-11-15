"""Provides desc"""
import numpy as np
from datar.apis.dplyr import desc
from datar_numpy.utils import make_array

from ...pandas import Categorical, Series
from ...tibble import SeriesCategorical
from ...factory import func_bootstrap


@func_bootstrap(desc, kind="transform")
def _desc(x):
    try:
        out = -x
    except (ValueError, TypeError):
        cat = Categorical(x.values)
        out = desc.dispatch(SeriesCategorical)(
            Series(cat, index=x.index)
        )
    out.name = None
    return out


@desc.register(object, backend="pandas")
def _desc_obj(x):
    try:
        out = -make_array(x)
    except (ValueError, TypeError):
        cat = Categorical(x)
        out = desc.dispatch(Categorical)(cat)

    return out


@desc.register(Categorical, backend="pandas")
def _desc_cat(x):
    code = x.codes.astype(float)
    code[code == -1.0] = np.nan
    return -code


@desc.register(SeriesCategorical, backend="pandas")
def _desc_scat(x):
    cat = x.values
    code = cat.codes.astype(float)
    code[code == -1.0] = np.nan
    return Series(-code, index=x.index)

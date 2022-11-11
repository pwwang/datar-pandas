"""Provides desc"""
import numpy as np
from datar.apis.dplyr import desc

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


@desc.register(SeriesCategorical)
def _desc_cat(x):
    cat = x.values
    code = cat.codes.astype(float)
    code[code == -1.0] = np.nan
    return Series(-code, index=x.index)

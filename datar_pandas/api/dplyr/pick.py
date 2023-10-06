"""Apply a function (or functions) across multiple columns

See source https://github.com/tidyverse/dplyr/blob/master/R/across.R
"""
from ...utils import meta_kwargs
from ...pandas import DataFrame
from ...contexts import Context
from ...common import intersect
from datar.apis.base import c
from datar.apis.dplyr import pick, group_vars


@pick.register(DataFrame, backend="pandas", context=Context.SELECT)
def _pick(_data, *args):
    if not args:
        raise ValueError("must pick at least one column")

    gvars = group_vars(_data, **meta_kwargs)
    cols = c(*args, **meta_kwargs)

    if not gvars:
        return _data[c(*args, **meta_kwargs)]

    if len(intersect(gvars, cols)) > 0:
        raise ValueError("cannot pick grouping columns")

    return _data[cols]

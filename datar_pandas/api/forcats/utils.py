"""Utilities for forcats"""
import numpy as np

from ...common import is_scalar, is_factor
from ...pandas import Categorical, Series, Index, SeriesGroupBy


ForcatsRegType = (
    Series,
    SeriesGroupBy,
    Categorical,
    Index,
    list,
    tuple,
    np.ndarray,
)


def check_factor(_f) -> Categorical:
    """Make sure the input become a factor"""
    if not is_factor(_f):
        if is_scalar(_f):
            _f = [_f]
        return Categorical(_f)

    return _f

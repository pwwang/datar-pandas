from typing import Any

from datar.apis.base import (
    is_factor as _is_factor,
    is_logical as _is_logical,
    is_integer as _is_integer,
    intersect as _intersect,
    setdiff as _setdiff,
    union as _union,
    # unique as _unique,
)
from datar_numpy.utils import is_scalar  # noqa: F401
from datar_numpy.api import asis as _  # noqa: F401
from datar_numpy.api import sets as _  # noqa: F401, F811

from . import pandas as pd
from .pandas import unique  # noqa: F401
from .typing import Data, Bool

np_meta_kwargs = {"__backend": "numpy", "__ast_fallback": "normal"}


def is_null(x: Any) -> Data[Bool]:
    return pd.isnull(x)


def is_factor(x: Any) -> bool:
    from .utils import meta_kwargs
    return _is_factor(x, **meta_kwargs)


def is_integer(x: Any) -> bool:
    return _is_integer(x, **np_meta_kwargs)


def is_logical(x: Any) -> bool:
    return _is_logical(x, **np_meta_kwargs)


def intersect(x: Any, y: Any) -> Any:
    return _intersect(x, y, **np_meta_kwargs)


def setdiff(x: Any, y: Any) -> Any:
    return _setdiff(x, y, **np_meta_kwargs)


def union(x: Any, y: Any) -> Any:
    return _union(x, y, **np_meta_kwargs)

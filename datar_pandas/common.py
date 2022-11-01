from typing import Any

from datar.base import (
    is_atomic,
    is_factor as _is_factor,
    is_logical as _is_logical,
    is_integer as _is_integer,
    intersect as _intersect,
    setdiff as _setdiff,
    union as _union,
    unique as _unique,
)

from . import pandas as pd
from .typing import Data, Bool


def is_scalar(x: Any) -> bool:
    """Check if x is a scalar"""
    return is_atomic(x, __ast_fallback="normal")


def is_null(x: Any) -> Data[Bool]:
    return pd.isnull(x)


def is_factor(x: Any) -> bool:
    return _is_factor(x, __ast_fallback="normal")


def is_integer(x: Any) -> bool:
    return _is_integer(x, __ast_fallback="normal")


def is_logical(x: Any) -> bool:
    return _is_logical(x, __ast_fallback="normal")


def intersect(x: Any, y: Any) -> Any:
    return _intersect(x, y, __ast_fallback="normal")


def setdiff(x: Any, y: Any) -> Any:
    return _setdiff(x, y, __ast_fallback="normal")


def union(x: Any, y: Any) -> Any:
    return _union(x, y, __ast_fallback="normal")


def unique(x: Any) -> Any:
    return _unique(x, __ast_fallback="normal")

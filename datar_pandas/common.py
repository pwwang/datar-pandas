from typing import Any

from datar.base import is_atomic, is_integer, intersect, setdiff

from . import pandas as pd
from .typing import Data, Bool


def is_scalar(x: Any) -> bool:
    """Check if x is a scalar"""
    return is_atomic(x, __ast_fallback="normal")


def is_null(x: Any) -> Data[Bool]:
    return pd.isnull(x)


def is_integer(x: Any) -> bool:
    return is_integer(x, __ast_fallback="normal")


def intersect(x: Any, y: Any) -> Any:
    return intersect(x, y, __ast_fallback="normal")


def setdiff(x: Any, y: Any) -> Any:
    return setdiff(x, y, __ast_fallback="normal")

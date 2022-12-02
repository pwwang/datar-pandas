from typing import List, Tuple, TypeVar, Union
from numbers import Number as PyNumber

import numpy as np
from pipda import Expression

from .pandas import Series

T = TypeVar("T")

Number = Union[PyNumber, np.number, Expression]
Int = Union[int, np.integer, Expression]
Float = Union[float, np.floating, Expression]
Bool = Union[bool, np.bool_, Expression]
Str = Union[str, np.str_, Expression]

Data = Union[T, np.ndarray, Tuple[T, ...], List[T], Series, Expression]

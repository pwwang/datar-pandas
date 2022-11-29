import inspect
import numpy as np

from datar.apis.base import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    cospi,
    sin,
    sinh,
    sinpi,
    tan,
    tanh,
    tanpi,
    atan2,
)
from datar_numpy.api import trig as _  # noqa: F401

from ...factory import func_bootstrap
from ...pandas import DataFrame, SeriesGroupBy
from ...tibble import Tibble, TibbleRowwise

func_bootstrap(acos, func=np.arccos, kind="transform")
func_bootstrap(acosh, func=np.arccosh, kind="transform")
func_bootstrap(asin, func=np.arcsin, kind="transform")
func_bootstrap(asinh, func=np.arcsinh, kind="transform")
func_bootstrap(atan, func=np.arctan, kind="transform")
func_bootstrap(atanh, func=np.arctanh, kind="transform")
func_bootstrap(cos, func=np.cos, kind="transform")
func_bootstrap(cosh, func=np.cosh, kind="transform")
func_bootstrap(sin, func=np.sin, kind="transform")
func_bootstrap(sinh, func=np.sinh, kind="transform")
func_bootstrap(tan, func=np.tan, kind="transform")
func_bootstrap(tanh, func=np.tanh, kind="transform")
func_bootstrap(cospi, func=lambda x: np.cos(x * np.pi), kind="transform")
func_bootstrap(tanpi, func=lambda x: np.tan(x * np.pi), kind="transform")
func_bootstrap(sinpi, func=lambda x: np.sin(x * np.pi), kind="transform")


def _atan2_post(__out, y, x):
    if isinstance(__out, DataFrame):
        __out = __out.iloc[:, 0]

    sgb = None
    if isinstance(x, SeriesGroupBy):
        sgb = x
    elif isinstance(y, SeriesGroupBy):
        sgb = y

    if sgb is None:
        return __out

    out = __out.groupby(
        sgb.grouper,
        sort=sgb.sort,
        dropna=sgb.dropna,
        observed=sgb.observed,
    )

    if getattr(sgb, "is_rowwise", False):
        out.is_rowwise = True
    return out


_atan2 = func_bootstrap(
    atan2,
    func=np.arctan2,
    post=_atan2_post,
    signature=inspect.signature(lambda y, x: None),
)


@_atan2.apply_df.register(TibbleRowwise)
def _atan2_rowwise(data, bound, exclude, func=None):
    df = Tibble(data, copy=False)
    return np.arctan2(df.y, df.x)

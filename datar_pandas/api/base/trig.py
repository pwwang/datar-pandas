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

from ...factory import func_bootstrap
from ...contexts import Context
from ...pandas import DataFrame, SeriesGroupBy
from ...tibble import Tibble, TibbleRowwise

func_bootstrap(acos, func=np.arccos, context=Context.EVAL, kind="transform")
func_bootstrap(acosh, func=np.arccosh, context=Context.EVAL, kind="transform")
func_bootstrap(asin, func=np.arcsin, context=Context.EVAL, kind="transform")
func_bootstrap(asinh, func=np.arcsinh, context=Context.EVAL, kind="transform")
func_bootstrap(atan, func=np.arctan, context=Context.EVAL, kind="transform")
func_bootstrap(atanh, func=np.arctanh, context=Context.EVAL, kind="transform")
func_bootstrap(cos, func=np.cos, context=Context.EVAL, kind="transform")
func_bootstrap(cosh, func=np.cosh, context=Context.EVAL, kind="transform")
func_bootstrap(sin, func=np.sin, context=Context.EVAL, kind="transform")
func_bootstrap(sinh, func=np.sinh, context=Context.EVAL, kind="transform")
func_bootstrap(tan, func=np.tan, context=Context.EVAL, kind="transform")
func_bootstrap(tanh, func=np.tanh, context=Context.EVAL, kind="transform")
func_bootstrap(cospi, func=lambda x: np.cos(x * np.pi), context=Context.EVAL, kind="transform")
func_bootstrap(tanpi, func=lambda x: np.tan(x * np.pi), context=Context.EVAL, kind="transform")
func_bootstrap(sinpi, func=lambda x: np.sin(x * np.pi), context=Context.EVAL, kind="transform")

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
    data_args={"y", "x"},
    signature=inspect.signature(lambda y, x: None),
)

@_atan2.kind_apply.register(TibbleRowwise)
def _atan2_rowwise(data, bound, data_args, func=None):
    df = Tibble(data, copy=False)
    return np.arctan2(df.y, df.x)

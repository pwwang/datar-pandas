"""Cumulative functions"""
import numpy as np
from datar.apis.base import cumsum, cumprod, cummax, cummin

from ...contexts import Context
from ...factory import func_bootstrap

func_bootstrap(
    cumsum,
    func=np.cumsum,
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    cumprod,
    func=np.cumprod,
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    cummax,
    func=np.maximum.accumulate,
    context=Context.EVAL,
    kind="transform",
)
func_bootstrap(
    cummin,
    func=np.minimum.accumulate,
    context=Context.EVAL,
    kind="transform",
)

"""Cumulative functions"""
import numpy as np
from datar.apis.base import cumsum, cumprod, cummax, cummin
from datar_numpy.api import cum as _  # noqa: F401

from ...factory import func_bootstrap

func_bootstrap(cumsum, func=np.cumsum, kind="transform")
func_bootstrap(cumprod, func=np.cumprod, kind="transform")
func_bootstrap(cummax, func=np.maximum.accumulate, kind="transform")
func_bootstrap(cummin, func=np.minimum.accumulate, kind="transform")

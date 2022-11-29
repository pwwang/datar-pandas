"""Functions related to complex numbers"""
import numpy as np
from datar.apis.base import (
    re_,
    im,
    mod,
    arg,
    conj,
)
from datar_numpy.api import complex as _  # noqa: F401

from ...factory import func_bootstrap

func_bootstrap(re_, func=np.real, kind="transform")
func_bootstrap(im, func=np.imag, kind="transform")
func_bootstrap(mod, func=np.abs, kind="transform")
func_bootstrap(arg, func=np.angle, kind="transform")
func_bootstrap(conj, func=np.conj, kind="transform")

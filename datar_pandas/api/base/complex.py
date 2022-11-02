"""Functions related to complex numbers"""
import numpy as np
from datar.apis.base import (
    re_,
    im,
    mod,
    arg,
    conj,
)

from ...contexts import Context
from ...factory import func_bootstrap

func_bootstrap(re_, func=np.real, context=Context.EVAL, kind="transform")
func_bootstrap(im, func=np.imag, context=Context.EVAL, kind="transform")
func_bootstrap(mod, func=np.abs, context=Context.EVAL, kind="transform")
func_bootstrap(arg, func=np.angle, context=Context.EVAL, kind="transform")
func_bootstrap(conj, func=np.conj, context=Context.EVAL, kind="transform")

import numpy as np
from datar.apis.base import (
    which,
    which_max,
    which_min,
)
from datar_numpy.api import which as _  # noqa: F401

from ...factory import func_bootstrap

func_bootstrap(which, func=np.flatnonzero)
func_bootstrap(which_max, func=np.argmax, kind="agg")
func_bootstrap(which_min, func=np.argmin, kind="agg")

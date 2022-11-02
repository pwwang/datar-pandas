import numpy as np
from datar.apis.base import (
    which,
    which_max,
    which_min,
)

from ...factory import func_bootstrap
from ...contexts import Context

func_bootstrap(
    which,
    context=Context.EVAL,
    func=np.flatnonzero,
    kind="transform",
)
func_bootstrap(
    which_max,
    context=Context.EVAL,
    func=np.argmax,
    kind="agg",
)
func_bootstrap(
    which_min,
    context=Context.EVAL,
    func=np.argmin,
    kind="agg",
)

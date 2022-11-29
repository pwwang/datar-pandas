from datar.apis.base import (
    bessel_i,
    bessel_j,
    bessel_k,
    bessel_y,
)
from datar_numpy.api import bessel as _  # noqa: F401

from ...factory import func_bootstrap


func_bootstrap(
    bessel_i,
    func=bessel_i.dispatch(object, backend="numpy"),
    kind="transform",
)

func_bootstrap(
    bessel_j,
    func=bessel_j.dispatch(object, backend="numpy"),
    kind="transform",
)

func_bootstrap(
    bessel_k,
    func=bessel_k.dispatch(object, backend="numpy"),
    kind="transform",
)

func_bootstrap(
    bessel_y,
    func=bessel_y.dispatch(object, backend="numpy"),
    kind="transform",
)

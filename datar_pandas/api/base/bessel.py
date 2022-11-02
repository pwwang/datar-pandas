from datar.apis.base import (
    bessel_i,
    bessel_j,
    bessel_k,
    bessel_y,
)

from ...factory import func_bootstrap
from ...contexts import Context


func_bootstrap(
    bessel_i,
    func=bessel_i.dispatch(object),
    context=Context.EVAL,
    kind="transform",
)

func_bootstrap(
    bessel_j,
    func=bessel_j.dispatch(object),
    context=Context.EVAL,
    kind="transform",
)

func_bootstrap(
    bessel_k,
    func=bessel_k.dispatch(object),
    context=Context.EVAL,
    kind="transform",
)

func_bootstrap(
    bessel_y,
    func=bessel_y.dispatch(object),
    context=Context.EVAL,
    kind="transform",
)

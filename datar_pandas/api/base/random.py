from datar.apis.base import (
    rbinom,
    rcauchy,
    rchisq,
    rexp,
    rnorm,
    rpois,
    runif,
)
from datar_numpy.api import random as _  # noqa: F401

from ...factory import func_bootstrap


func_bootstrap(
    rbinom,
    func=rbinom.dispatch(object, backend="numpy"),
)

func_bootstrap(
    rcauchy,
    func=rcauchy.dispatch(object, backend="numpy"),
)

func_bootstrap(
    rchisq,
    func=rchisq.dispatch(object, backend="numpy"),
)

func_bootstrap(
    rexp,
    func=rexp.dispatch(object, backend="numpy"),
)

func_bootstrap(
    rnorm,
    func=rnorm.dispatch(object, backend="numpy"),
)

func_bootstrap(
    rpois,
    func=rpois.dispatch(object, backend="numpy"),
)

func_bootstrap(
    runif,
    func=runif.dispatch(object, backend="numpy"),
)

from datar.apis.base import (
    beta,
    lbeta,
    gamma,
    lgamma,
    psigamma,
    digamma,
    trigamma,
    choose,
    lchoose,
    factorial,
    lfactorial,
)
from datar_numpy.api import special as _  # noqa: F401

from ...factory import func_bootstrap

func_bootstrap(beta, func=beta.dispatch(object, backend="numpy"))

func_bootstrap(lbeta, func=lbeta.dispatch(object, backend="numpy"))

func_bootstrap(gamma, func=gamma.dispatch(object, backend="numpy"))

func_bootstrap(lgamma, func=lgamma.dispatch(object, backend="numpy"))

func_bootstrap(psigamma, func=psigamma.dispatch(object, backend="numpy"))

func_bootstrap(digamma, func=digamma.dispatch(object, backend="numpy"))

func_bootstrap(trigamma, func=trigamma.dispatch(object, backend="numpy"))

func_bootstrap(choose, func=choose.dispatch(object, backend="numpy"))

func_bootstrap(lchoose, func=lchoose.dispatch(object, backend="numpy"))

func_bootstrap(factorial, func=factorial.dispatch(object, backend="numpy"))

func_bootstrap(lfactorial, func=lfactorial.dispatch(object, backend="numpy"))

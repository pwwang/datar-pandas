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

from ...factory import func_bootstrap
from ...contexts import Context


func_bootstrap(
    beta,
    func=beta.dispatch(object),
    context=Context.EVAL,
    data_args={"a", "b"},
)

func_bootstrap(
    lbeta,
    func=lbeta.dispatch(object),
    context=Context.EVAL,
    data_args={"a", "b"},
)

func_bootstrap(
    gamma,
    func=gamma.dispatch(object),
    context=Context.EVAL,
)

func_bootstrap(
    lgamma,
    func=lgamma.dispatch(object),
    context=Context.EVAL,
)

func_bootstrap(
    psigamma,
    func=psigamma.dispatch(object),
    context=Context.EVAL,
    data_args={"x", "deriv"},
)

func_bootstrap(
    digamma,
    func=digamma.dispatch(object),
    context=Context.EVAL,
)

func_bootstrap(
    trigamma,
    func=trigamma.dispatch(object),
    context=Context.EVAL,
)

func_bootstrap(
    choose,
    func=choose.dispatch(object),
    context=Context.EVAL,
    data_args={"n", "k"},
)

func_bootstrap(
    lchoose,
    func=lchoose.dispatch(object),
    context=Context.EVAL,
    data_args={"n", "k"},
)

func_bootstrap(
    factorial,
    func=factorial.dispatch(object),
    context=Context.EVAL,
)

func_bootstrap(
    lfactorial,
    func=lfactorial.dispatch(object),
    context=Context.EVAL,
)

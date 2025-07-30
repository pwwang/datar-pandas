import sys
import pytest  # noqa

from datar.base import (
    beta,
    choose,
    digamma,
    factorial,
    gamma,
    lbeta,
    lchoose,
    lfactorial,
    lgamma,
    psigamma,
    trigamma,
)
from datar.base import Inf, NA
from datar_pandas.pandas import Series
from ..conftest import assert_iterable_equal


@pytest.mark.parametrize(
    "a,b,log,exp",
    [
        (Series([1]), 2, False, [0.5]),
        (Series([1]), 2, True, [-0.6931472]),
        (1, Series([1, 2]), False, [1.0, 0.5]),
        (1, Series([1, 2]), True, [0.0, -0.6931472]),
        ([1, 2], Series([2]), False, [0.5, 0.1666666666667]),
        ([1, 2], Series([2]), True, [-0.6931472, -1.7917595]),
    ],
)
def test_beta(a, b, log, exp):
    fun = lbeta if log else beta
    out = fun(a, b)
    assert_iterable_equal(out, exp, approx=True)


@pytest.mark.parametrize(
    "x,fun,exp",
    [
        (Series([0]), gamma, [Inf]),  # 0
        (Series([0]), lgamma, [Inf]),  # 1
        (Series([0]), digamma, [-Inf]),  # 2
        (Series([0]), trigamma, [Inf]),  # 3
        (Series([0]), factorial, [1]),  # 4
        (Series([0]), lfactorial, [0]),  # 5
        (Series([1]), digamma, [-0.5772157]),  # 6
        (Series([1]), trigamma, [1.644934]),  # 7
        (Series([-1]), lgamma, [Inf]),  # 8
        (Series([-1]), digamma, [NA]),  # 9
        (Series([-1]), gamma, [Inf if sys.version_info < (3, 11) else NA]),  # 10
        (Series([-1]), trigamma, [Inf]),
        (Series([-1]), factorial, [0]),
        (Series([-1, 1]), factorial, [0, 1]),
        (Series([0, 1, 2]), gamma, [Inf, 1, 1]),
        (Series([0, 1, 2]), lgamma, [Inf, 0, 0]),
    ],
)
def test_gamma(x, fun, exp):
    out = fun(x)
    assert_iterable_equal(out, exp, approx=True)


@pytest.mark.parametrize(
    "n,k,log,exp",
    [
        (5, Series([2]), False, [10]),
        (5, Series([2]), True, [2.302585]),
        (Series([4, 5]), 2, False, [6, 10]),
        (Series([4, 5]), 2, True, [1.791759469, 2.302585]),
    ],
)
def test_choose(n, k, log, exp):
    fun = lchoose if log else choose
    out = fun(n, k)
    assert_iterable_equal(exp, out, approx=True)


@pytest.mark.parametrize(
    "x,deriv,exp",
    [
        (Series([1]), 0, [-0.5772157]),
        (Series([2]), 1, [0.6449341]),
        (Series([2]), -1, [NA]),
        (Series([1]), (0, 1), [-0.5772157, 1.6449341]),
        (-2, Series([1]), [Inf]),
    ],
)
def test_psigamma(x, deriv, exp):
    out = psigamma(x, deriv)
    assert_iterable_equal(exp, out, approx=True)

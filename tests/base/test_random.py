import pytest  # noqa

# import numpy as np
from datar.base import rnorm, rpois, runif, rcauchy, rchisq, rexp, rbinom
from datar.tibble import tibble

from ..conftest import assert_iterable_equal


def test_rnorm():
    df = tibble(x=[1, 2, 3], y=[1, 1, 2])
    assert_iterable_equal(rnorm(df.x, df.y, 0), [1, 1, 2])

    gf = df.group_by("y")
    out = rnorm(gf.x, 1, 0)
    assert_iterable_equal(out.values[0], [1, 1])
    assert_iterable_equal(out.values[1], [1, 1, 1])

    rf = df.rowwise()
    out = rnorm(rf.x, rf.y, 0)
    assert len(out) == 3


def test_rbinorm():
    df = tibble(x=[1, 2, 3], y=[1, 1, 2])
    assert_iterable_equal(rbinom(df.x, df.y, 1), [1, 1, 2])

    gf = df.group_by("y")
    out = rbinom(gf.x, 1, 1)
    assert_iterable_equal(out.values[0], [1, 1])
    assert_iterable_equal(out.values[1], [1, 1, 1])

    rf = df.rowwise()
    out = rbinom(rf.x, rf.y, 1)
    assert len(out) == 3


def test_runif():
    df = tibble(x=[1, 2, 3], y=[1, 1, 2])
    assert len(runif(df.x, df.y, 0)) == 3

    gf = df.group_by("y")
    out = runif(gf.x, 1, 0)
    assert len(out) == 2

    rf = df.rowwise()
    out = runif(rf.x, rf.y, 0)
    assert len(out) == 3


def test_rpois():
    df = tibble(x=[1, 2, 3], y=[1, 1, 2])
    assert len(rpois(df.x, df.y)) == 3

    gf = df.group_by("y")
    out = rpois(gf.x, 1)
    assert len(out) == 2

    rf = df.rowwise()
    out = rpois(rf.x, rf.y)
    assert len(out) == 3


def test_rcauchy():
    df = tibble(x=[1, 2, 3], y=[1, 1, 2])
    assert len(rcauchy(df.x, df.y)) == 3

    gf = df.group_by("y")
    out = rcauchy(gf.x)
    assert len(out) == 2

    rf = df.rowwise()
    out = rcauchy(rf.x)
    assert len(out) == 3


def test_rchisq():
    df = tibble(x=[1, 2, 3], y=[1, 1, 2])
    assert len(rchisq(df.x, df.y)) == 3

    gf = df.group_by("y")
    out = rchisq(gf.x, 1)
    assert len(out) == 2

    rf = df.rowwise()
    out = rchisq(rf.x, rf.y)
    assert len(out) == 3


def test_rexp():
    df = tibble(x=[1, 2, 3], y=[1, 1, 2])
    assert len(rexp(df.x, df.y)) == 3

    gf = df.group_by("y")
    out = rexp(gf.x, 1)
    assert len(out) == 2

    rf = df.rowwise()
    out = rexp(rf.x, rf.y)
    assert len(out) == 3

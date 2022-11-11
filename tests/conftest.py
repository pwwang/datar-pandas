
import pytest


def pytest_sessionstart(session):
    from datar import options

    options(import_names_conflict="silent", backends=["numpy", "pandas"])
    # set_seed(8888)


SENTINEL = 85258525.85258525


def assert_iterable_equal(x, y, na=SENTINEL, approx=False):
    import pandas as pd

    x = [na if pd.isnull(elt) else elt for elt in x]
    y = [na if pd.isnull(elt) else elt for elt in y]
    if approx is True:
        x = pytest.approx(x)
    elif approx:
        x = pytest.approx(x, rel=approx)
    assert x == y


def assert_factor_equal(x, y, na=8525.8525, approx=False):
    xlevs = x.categories
    ylevs = y.categories
    assert_iterable_equal(x, y, na=na, approx=approx)
    assert_iterable_equal(xlevs, ylevs, na=na, approx=approx)


def assert_(x):
    assert x


# pytest modifies node for assert
def assert_equal(x, y, approx=False):
    if approx is True:
        x = pytest.approx(x)
    elif approx:
        x = pytest.approx(x, rel=approx)
    assert x == y


def is_installed(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

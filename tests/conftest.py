
import pytest
import numpy as np
from collections import namedtuple


def pytest_addoption(parser):
    parser.addoption("--modin", action="store_true")


def pytest_sessionstart(session):
    from datar import options
    from datar.core import plugin  # noqa: F401

    modin = session.config.getoption("modin")

    options(
        use_modin=modin,
        import_names_conflict="silent",
        backends=["numpy", "pandas"],
    )
    # set_seed(8888)


SENTINEL = 85258525.85258525


def _isna(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return False
    try:
        return np.isnan(x)
    except (ValueError, TypeError):
        return False


def assert_iterable_equal(x, y, na=SENTINEL, approx=False):
    import pandas as pd

    x = [na if pd.isnull(elt) else elt for elt in x]
    y = [na if pd.isnull(elt) else elt for elt in y]
    if approx is True:
        x = pytest.approx(x)
    elif approx:
        x = pytest.approx(x, rel=approx)
    assert x == y, f"{x} != {y}"


def assert_factor_equal(x, y, na=8525.8525, approx=False):
    xlevs = x.categories
    ylevs = y.categories
    assert_iterable_equal(x, y, na=na, approx=approx)
    assert_iterable_equal(xlevs, ylevs, na=na, approx=approx)


def assert_(x):
    assert x, f"{x} is not True"


def assert_not(x):
    assert not x, f"{x} is not False"


# pytest modifies node for assert
def assert_equal(x, y, approx=False):
    if _isna(x) and _isna(y):
        return
    if approx is True:
        x = pytest.approx(x)
    elif approx:
        x = pytest.approx(x, rel=approx)
    assert x == y, f"{x} != {y}"


def is_installed(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def pd_data():
    from datar_pandas.pandas import DataFrame, Series
    from datar_pandas.tibble import Tibble, TibbleGrouped
    out = namedtuple(
        "pd_data",
        "scalar list tuple array series sgb df gf tibble tg tr",
    )
    out.scalar = 1
    out.float = 1.2
    out.neg = -1
    out.list = [1, 2, 2, 3]
    out.tuple = (1, 2, 2, 3)
    out.array = np.array([1, 2, 2, 3])
    out.naarray = np.array([1, 2, np.nan, 3])
    out.farray = np.array([1.2, 2.2, 2.2, 3.2])
    out.negarray = np.array([-1, -2, -2, -3])
    out.series = Series([1, 2, 2, 3])
    out.sgb = Series([1, 2, 2, 3], name="x").groupby(
        Series([1, 2, 2, 3], name="x")
    )
    out.df = DataFrame({"x": [1, 2, 2, 3]})
    out.gf = out.df.groupby([1, 2, 2, 3])
    out.tibble = Tibble(out.df, copy=True)
    out.tg = TibbleGrouped.from_groupby(out.gf)
    out.tr = out.tibble.rowwise()
    return out

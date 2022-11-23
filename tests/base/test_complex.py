import pytest  # noqa: F401
from datar.base import (
    re,
    im,
    mod,
    arg,
    conj,
    is_complex,
)

from ..conftest import assert_iterable_equal, assert_


def test_complex_pandas():
    import pandas as pd

    x = pd.Series([1 + 2j, 3 + 4j])
    assert_iterable_equal(re(x), [1, 3])
    assert_iterable_equal(im(x), [2, 4])
    assert_iterable_equal(mod(x), [2.236068, 5], approx=1e-5)
    assert_iterable_equal(arg(x), [1.107149, 0.927295], approx=1e-5)
    assert_iterable_equal(conj(x), [1 - 2j, 3 - 4j])
    assert_(is_complex(x))

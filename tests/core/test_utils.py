import pytest

import numpy as np
from datar.base import is_integer
from datar.tibble import tibble
from datar_pandas.utils import apply_dtypes, dict_get
from ..conftest import assert_


def test_apply_dtypes():
    df = tibble(x=[1.0, 2.0])
    apply_dtypes(df, True)
    assert_(is_integer(df.x))


def test_dict_get():
    d = {'a': 1, 'b': 2, np.nan: 3}
    assert dict_get(d, 'a') == 1
    assert dict_get(d, 'b') == 2
    assert dict_get(d, float("nan")) == 3
    assert dict_get(d, 'c', None) is None
    with pytest.raises(KeyError):
        dict_get(d, 'c')

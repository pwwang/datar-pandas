import pytest
from simplug import ResultError
from datar_pandas.pandas import DataFrame


def test_no_such():
    with pytest.raises(ResultError):
        from datar.data import nosuch  # noqa: F401


def test_data():
    from datar.data import iris
    assert isinstance(iris, DataFrame)

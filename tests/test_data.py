import pytest
from datar_pandas.pandas import DataFrame


def test_no_such():
    with pytest.raises(ImportError):
        from datar.data import nosuch  # noqa: F401


def test_data():
    from datar.data import iris
    assert isinstance(iris, DataFrame)

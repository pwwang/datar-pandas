import pytest  # noqa

from datar import f
from datar.tibble import tibble
from datar.dplyr import mutate
from datar_pandas.factory import func_factory


def test_args_frame():
    @func_factory(kind="apply")
    def mn(x, __args_frame=None):
        return __args_frame.x.mean()

    out = tibble(x=[1, 2, 3]) >> mutate(y=mn(f.x))
    assert out.y.tolist() == [2, 2, 2]


def test_pre():
    @func_factory(kind="apply", pre=lambda x: ((x + 1, ), {}))
    def mn(x):
        return x.mean()

    out = tibble(x=[1, 2, 3]) >> mutate(y=mn(f.x))
    assert out.y.tolist() == [3, 3, 3]

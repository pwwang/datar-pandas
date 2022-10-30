from pandas import Series
from datar.apis.base import mean

@mean.register(object)
def _(x):
    return Series(x).mean()


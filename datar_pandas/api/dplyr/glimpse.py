from typing import Callable, Optional

from datar.apis.dplyr import glimpse

from ...pandas import DataFrame
from ...contexts import Context
from ...middlewares import glimpse_formatter, Glimpse


@glimpse.register(DataFrame, context=Context.EVAL, backend="pandas")
def _glimpse(
    x: DataFrame,
    width: Optional[int] = None,
    formatter: Callable = glimpse_formatter,
) -> Glimpse:
    return Glimpse(x, width=width, formatter=formatter)

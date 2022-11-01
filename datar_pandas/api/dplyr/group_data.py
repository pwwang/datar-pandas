from typing import Sequence

from datar.apis.dplyr import (
    group_vars,

)

from ...pandas import DataFrame
from ...contexts import Context


@group_vars.register(DataFrame, context=Context.EVAL)
def group_vars(_data: DataFrame) -> Sequence[str]:
    """Gives names of grouping variables as character vector"""
    return getattr(_data, "group_vars", [])

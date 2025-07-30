"""Reframe a data frame."""

from typing import Any

from datar.core.names import NameNonUniqueError
from datar.apis.dplyr import reframe, group_vars

from ...common import is_scalar
from ...pandas import DataFrame
from ...contexts import Context
from ...tibble import Tibble, TibbleGrouped
from .summarise import _summarise_build


@reframe.register(DataFrame, context=Context.PENDING, backend="pandas")
def _reframe(
    _data: DataFrame,
    *args: Any,
    **kwargs: Any,
) -> Tibble:
    if not _data.columns.is_unique:
        raise NameNonUniqueError("Can't transform a data frame with duplicate names.")

    out, _ = _summarise_build(_data, *args, **kwargs)

    if isinstance(_data, TibbleGrouped):
        gvars = group_vars(_data, __ast_fallback="normal", __backend="pandas")
        non_gvars = out.columns.difference(gvars).to_list()
        # remove the empty rows
        # faster way?
        mask = out[non_gvars].agg(
            lambda row: all(is_scalar(x) or len(x) > 0 for x in row),
            axis=1,
        )
        out = out[mask].reset_index(drop=True)
        out = out.explode(non_gvars, ignore_index=True)
        out = out.convert_dtypes()

    return out

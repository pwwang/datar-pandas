"""Uncount a data frame"""
from typing import Any, Iterable

from datar.apis.tidyr import uncount

from ...pandas import DataFrame, Series, is_number
from ...common import is_scalar
from ...utils import get_grouper
from ...broadcast import broadcast_to
from ...contexts import Context
from ...tibble import reconstruct_tibble
from ..dplyr.group_by import ungroup


@uncount.register(DataFrame, context=Context.EVAL, backend="pandas")
def _uncount(
    data: DataFrame,
    weights,
    _remove: bool = True,
    _id: str = None,
) -> DataFrame:
    """Duplicating rows according to a weighting variable

    Args:
        data: A data frame
        weights: A vector of weights. Evaluated in the context of data
        _remove: If TRUE, and weights is the name of a column in data,
            then this column is removed.
        _id: Supply a string to create a new variable which gives a
            unique identifier for each created row (0-based).

    Returns:
        dataframe with rows repeated.
    """
    grouped = getattr(data, "_datar", {}).get("grouped", None)
    undata = ungroup(data, __ast_fallback="normal", __backend="pandas").copy()
    weights = broadcast_to(
        weights,
        data.index,
        None if grouped is None else get_grouper(grouped),
    )
    if is_scalar(weights):
        weights = Series(weights, index=data.index)

    _check_weights(weights)

    if not undata.index.is_unique:
        raise ValueError("Cannot uncount a frame with duplicated index.")

    if weights.name in undata and _remove:
        del undata[weights.name]

    out = undata.reindex(undata.index.repeat(weights.values))
    if _id:
        out.index.name = _id
        out = out.reset_index()
    else:
        out = out.reset_index(drop=True)

    return reconstruct_tibble(out, data)


def _check_weights(weights: Iterable[Any]) -> None:
    """Check if uncounting weights are valid"""
    for weight in weights:
        if not is_number(weight):
            raise ValueError("`weights` must evaluate to numerics.")
        if weight < 0:
            raise ValueError("All elements in `weights` must be >= 0.")

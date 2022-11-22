"""Relocate columns"""
from typing import Any, Union

from datar.apis.dplyr import group_vars, relocate

from ...pandas import DataFrame
from ...contexts import Context
from ...tibble import Tibble, TibbleGrouped
from ...common import setdiff, union, intersect
from ..tibble.tibble import as_tibble
from .select import _eval_select


@relocate.register(DataFrame, context=Context.SELECT, backend="pandas")
def _relocate(
    _data: DataFrame,
    *args: Any,
    _before: Union[int, str] = None,
    _after: Union[int, str] = None,
    **kwargs: Any,
) -> Tibble:
    gvars = group_vars(_data, __ast_fallback="normal", __backend="pandas")
    _data = as_tibble(_data.copy(), __ast_fallback="normal", __backend="pandas")

    all_columns = _data.columns
    to_move, new_names = _eval_select(
        all_columns,
        *args,
        **kwargs,
        _group_vars=gvars,
        _missing_gvars_inform=False,
    )

    to_move = list(to_move)
    if _before is not None and _after is not None:
        raise ValueError("Must supply only one of `_before` and `_after`.")

    # length = len(all_columns)
    if _before is not None:
        where = min(
            _eval_select(
                all_columns,
                _before,
                _group_vars=[],
                _missing_gvars_inform=False,
            )[0]
        )
        if where not in to_move:
            to_move.append(where)

    elif _after is not None:
        where = max(
            _eval_select(
                all_columns,
                _after,
                _group_vars=[],
                _missing_gvars_inform=False,
            )[0]
        )
        if where not in to_move:
            to_move.insert(0, where)
    else:
        where = 0
        if where not in to_move:
            to_move.append(where)

    lhs = setdiff(range(where), to_move)
    rhs = setdiff(range(where + 1, len(all_columns)), to_move)
    pos = union(lhs, union(to_move, rhs))

    out = _data.iloc[:, pos]
    # out = out.copy()
    if new_names:
        out.rename(columns=new_names, inplace=True)
        if (
            isinstance(out, TibbleGrouped)
            and len(intersect(gvars, list(new_names))) > 0
        ):
            out._datar["group_vars"] = [
                new_names.get(gvar, gvar) for gvar in gvars
            ]

    return out

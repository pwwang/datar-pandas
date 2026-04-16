"""Apply a function (or functions) across multiple columns

See source https://github.com/tidyverse/dplyr/blob/master/R/across.R
"""

from typing import Any, Optional, Sequence, cast

from pipda import evaluate_expr
from datar.apis.dplyr import across, c_across, if_all, if_any

from ...pandas import DataFrame
from ...tibble import reconstruct_tibble
from ...utils import vars_select
from ...middlewares import Across, IfAll, IfAny
from ...contexts import Context
from ...collections import Collection
from .tidyselect import everything


@across.register(DataFrame, backend="pandas", context=Context.PENDING)
def _across(
    _data: DataFrame,
    *args: Any,
    _names: Optional[str] = None,
    _fn_context: Context = Context.EVAL,
    **kwargs: Any,
) -> DataFrame:
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)

    if not args:
        args = (None, None)
    elif len(args) == 1:
        args = (args[0], None)
    _cols, _fns, *rest = args
    other_args = tuple(rest)
    _cols = evaluate_expr(_cols, _data, cast(Any, Context.SELECT))

    return Across(
        _data,
        _cols,
        _fns,
        _names,
        other_args,
        kwargs,
    ).evaluate(_fn_context)


@c_across.register(DataFrame, backend="pandas", context=Context.SELECT)
def _c_across(
    _data: DataFrame,
    _cols: Optional[Sequence[str]] = None,
) -> DataFrame:
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)

    if not _cols and not isinstance(_cols, Collection):
        _cols = _data >> everything()

    selected = vars_select(_data.columns, cast(Any, _cols))
    return reconstruct_tibble(_data.iloc[:, cast(Any, selected)], _data)


@if_any.register(DataFrame, backend="pandas", context=Context.SELECT)
def _if_any(
    _data: DataFrame,
    *args: Any,
    _names: Optional[Sequence[str]] = None,
    _context: Optional[Context] = None,
    **kwargs: Any,
) -> DataFrame:
    if not args:
        args = (None, None)
    elif len(args) == 1:
        args = (args[0], None)
    _cols, _fns, *rest = args
    other_args = tuple(rest)
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)

    return IfAny(
        _data,
        _cols,
        _fns,
        _names,
        other_args,
        kwargs,
    ).evaluate(_context)


@if_all.register(DataFrame, backend="pandas", context=Context.SELECT)
def _if_all(
    _data: DataFrame,
    # _cols: Iterable[str] = None,
    # _fns: Union[Mapping[str, Callable]] = None,
    *args: Any,
    _names: Optional[Sequence[str]] = None,
    _context: Optional[Context] = None,
    **kwargs: Any,
) -> DataFrame:
    if not args:
        args = (None, None)
    elif len(args) == 1:
        args = (args[0], None)
    _cols, _fns, *rest = args
    other_args = tuple(rest)
    _data = getattr(_data, "_datar", {}).get("summerise_source", _data)

    return IfAll(
        _data,
        _cols,
        _fns,
        _names,
        other_args,
        kwargs,
    ).evaluate(_context)

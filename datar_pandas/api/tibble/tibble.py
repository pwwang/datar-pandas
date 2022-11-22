from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np
from pipda import Expression, ReferenceAttr, ReferenceItem, evaluate_expr
from datar.apis.tibble import (
    tibble,
    tibble_,
    tibble_row,
    tribble,
    as_tibble,
)

from ...common import is_scalar
from ...contexts import Context
from ...utils import ExpressionWrapper, name_of
from ...pandas import DataFrame, DataFrameGroupBy, PandasObject
from ...tibble import Tibble, TibbleGrouped

if TYPE_CHECKING:
    from pandas._typing import Dtype


@tibble.register(backend="pandas")
def _tibble(
    *args,
    _name_repair: str | Callable = "check_unique",
    _rows: int = None,
    _dtypes: Dtype | Mapping[str, Dtype] = None,
    _drop_index: bool = False,
    _index=None,
    **kwargs,
) -> Tibble:
    """Constructs a data frame, and make sure the references to external data
    and self work

    >>> df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> df >> mutate(c=f.a, d=f.c)
    """
    # Scan the args, kwargs see if we have any reference to self (f.c above)
    eval_data = {}
    evaled_args = []
    evaled_kws = {}
    for val in args:
        key = name_of(val)
        if isinstance(val, Expression):
            try:
                # Check if any refers to self (args, kwargs)
                evaluate_expr(val, eval_data, Context.EVAL)
            except (KeyError, NotImplementedError):
                # If not, it refers to external data
                evaled_args.append(val)
            else:
                # Otherwise, wrap it so that it gets bypass the evaluation
                evaled_args.append(ExpressionWrapper(val))
        else:
            evaled_args.append(val)

        if isinstance(val, DataFrame):
            for col in val.columns:
                eval_data[col] = 1
        else:
            eval_data[key] = 1

    for key, val in kwargs.items():
        if isinstance(val, Expression):
            try:
                evaluate_expr(val, eval_data, Context.EVAL)
            except (KeyError, NotImplementedError):
                evaled_kws[key] = val
            else:
                evaled_kws[key] = ExpressionWrapper(val)
        else:
            evaled_kws[key] = val

        eval_data[key] = 1

    return tibble_(
        *evaled_args,
        _name_repair=_name_repair,
        _rows=_rows,
        _dtypes=_dtypes,
        _drop_index=_drop_index,
        _index=_index,
        __ast_fallback="normal",
        __backend="pandas",
        **evaled_kws,
    )


@tibble_.register(
    (
        (object, np.ndarray, *PandasObject)
        if isinstance(PandasObject, tuple)
        else (object, np.ndarray, PandasObject)
    ),
    backend="pandas",
)
def _tibble_(
    *args,
    _name_repair: str | Callable = "check_unique",
    _rows: int = None,
    _dtypes: Dtype | Mapping[str, Dtype] = None,
    _drop_index: bool = False,
    _index=None,
    **kwargs,
) -> Tibble:
    out = Tibble.from_args(
        *args,
        **kwargs,
        _name_repair=_name_repair,
        _rows=_rows,
        _dtypes=_dtypes,
    )
    if _drop_index:
        return out.reset_index(drop=True)

    if _index is not None:
        out.index = [_index] if is_scalar(_index) else _index

    return out


@tribble.register(backend="pandas")
def _tribble(
    *dummies: Any,
    _name_repair: str | Callable = "minimal",
    _dtypes: Dtype | Mapping[str, Dtype] = None,
) -> Tibble:
    columns = []
    data = []
    for i, dummy in enumerate(dummies):
        # columns
        if (
            isinstance(dummy, (ReferenceAttr, ReferenceItem))
            and dummy._pipda_level == 1
        ):
            columns.append(dummy._pipda_ref)

        elif not columns:
            raise ValueError(
                "Must specify at least one column using the `f.<name>` syntax."
            )

        else:
            ncols = len(columns)
            if not data:
                data = [[] for _ in range(ncols)]

            data[i % ncols].append(dummy)

    # only columns provided
    if not data:
        data = [[] for _ in range(len(columns))]

    if len(data[-1]) != len(data[0]):
        raise ValueError(
            "Data must be rectangular. "
            f"{sum(len(dat) for dat in data)} cells is not an integer "
            f"multiple of {len(columns)} columns."
        )

    return Tibble.from_pairs(
        columns,
        data,
        _name_repair=_name_repair,
        _dtypes=_dtypes,
    )


@tibble_row.register(backend="pandas")
def _tibble_row(
    *args: Any,
    _name_repair: str | Callable = "check_unique",
    _dtypes: Dtype | Mapping[str, Dtype] = None,
    **kwargs: Any,
) -> Tibble:
    """Constructs a data frame that is guaranteed to occupy one row.
    Scalar values will be wrapped with `[]`
    Args:
        *args: and
        **kwargs: A set of name-value pairs.
        _name_repair: treatment of problematic column names:
            - "minimal": No name repair or checks, beyond basic existence,
            - "unique": Make sure names are unique and not empty,
            - "check_unique": (default value), no name repair,
                but check they are unique,
            - "universal": Make the names unique and syntactic
            - a function: apply custom name repair
    Returns:
        A constructed dataframe
    """
    if not args and not kwargs:
        df = Tibble(index=range(1))  # still one row
    else:
        df = tibble(*args, **kwargs, _name_repair=_name_repair, _dtypes=_dtypes)

    if df.shape[0] > 1:
        raise ValueError("All arguments must be size one, use `[]` to wrap.")

    return df


@as_tibble.register((dict, DataFrame), context=Context.EVAL, backend="pandas")
def _as_tibble_df(df: DataFrame | dict) -> Tibble:
    return Tibble(df)


@as_tibble.register(DataFrameGroupBy, context=Context.EVAL, backend="pandas")
def _as_tibble_dfg(df: DataFrameGroupBy) -> TibbleGrouped:
    """Convert a pandas DataFrameGroupBy object to TibbleGrouped object"""
    return TibbleGrouped.from_groupby(df)


@as_tibble.register(Tibble, context=Context.EVAL, backend="pandas")
def _as_tibble_tbl(df: Tibble) -> Tibble:
    """Convert a pandas DataFrame object to Tibble object"""
    return df

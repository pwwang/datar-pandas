"""Extract a single colum

https://github.com/tidyverse/dplyr/blob/master/R/pull.R
"""
from datar.core.utils import arg_match
from datar.apis.dplyr import pull

from ...pandas import DataFrame, Series
from ...common import is_scalar
# from ...tibble import Tibble, TibbleGrouped
from ...contexts import Context
from ..tibble.tibble import as_tibble


@pull.register(
    DataFrame,
    context=Context.SELECT,
    kw_context={"name": Context.EVAL},
    backend="pandas",
)
def _pull(_data, var=-1, *, name=None, to=None):
    # make sure pull(df, 'x') pulls a dataframe for columns
    # x$a, x$b in df

    to = arg_match(
        to, "to", ["list", "array", "frame", "series", "dict", None]
    )
    if name is not None and is_scalar(name):
        name = [name]

    _data = as_tibble(_data, __ast_fallback="normal", __backend="pandas")
    if isinstance(var, int):
        var = _data.columns[var]
        var = var.split("$", 1)[0]

    pulled = _data[var]
    pulled = getattr(pulled, "obj", pulled)
    # if var in _data.columns and isinstance(pulled, DataFrame):
    #     pulled = pulled.iloc[:, 0]

    if to is None:
        if name is not None and len(name) == len(pulled):
            to = "dict"
        else:
            to = "frame" if isinstance(pulled, DataFrame) else "series"

    if to == "dict":
        if name is None or len(name) != len(pulled):
            raise ValueError(
                "No `name` provided or length mismatches with the values."
            )
        return dict(zip(name, pulled))
    if to == "list":
        return pulled.values.tolist()
    if to == "array":
        return pulled.values
    if to == "frame":
        value = pulled if isinstance(pulled, DataFrame) else pulled.to_frame()
        if name and len(name) != value.shape[1]:
            raise ValueError(
                f"Expect {value.shape[1]} names but got {len(name)}."
            )
        if name:
            value.columns = name
        return value
    # if to == 'series':
    if isinstance(pulled, DataFrame) and pulled.shape[1] == 1:
        pulled = pulled.iloc[:, -1]
    if isinstance(pulled, Series):
        if name:
            pulled.name = name[0]
        return pulled
    # df
    if name and len(name) != pulled.shape[1]:
        raise ValueError(
            f"Expect {pulled.shape[1]} names but got {len(name)}."
        )

    out = pulled.to_dict("series")
    if not name:
        return out

    for newname, oldname in zip(name, out):
        out[newname] = out.pop(oldname)
    return out


# @pull.register(
#     TibbleGrouped,
#     context=Context.PENDING,
#     backend="pandas",
# )
# def _pull_grouped(_data, var=-1, name=None, to=None):
#     """Pull a column from a grouped data frame"""
#     return pull(
#         Tibble(_data, copy=False),
#         var=var,
#         name=name,
#         to=to,
#         __ast_fallback="normal",
#     )

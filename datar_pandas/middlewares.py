"""Middlewares for datar"""
import html
import textwrap
from abc import ABC, abstractmethod
from functools import singledispatch
from shutil import get_terminal_size
from typing import Any, Mapping, Tuple

from pipda import Context, evaluate_expr

from .common import is_scalar
from .broadcast import add_to_tibble
from .pandas import DataFrame, SeriesGroupBy
from .utils import vars_select
from .tibble import Tibble, TibbleGrouped, TibbleRowwise
from .api.dplyr.tidyselect import everything


class Across:
    """Across object"""

    def __init__(
        self,
        data,
        cols=None,
        fns=None,
        names=None,
        args=None,
        kwargs=None,
    ):
        cols = (
            data >> everything()
            if cols is None
            else cols
        )
        if is_scalar(cols):
            cols = [cols]

        cols = data.columns[vars_select(data.columns, cols)]

        fns_list = []
        if callable(fns):
            fns_list.append({"fn": fns})
        elif isinstance(fns, (list, tuple)):
            fns_list.extend(
                {"fn": fn, "_fn": i, "_fn1": i + 1, "_fn0": i}
                for i, fn in enumerate(fns)
            )
        elif isinstance(fns, dict):
            fns_list.extend(
                {"fn": value, "_fn": key} for key, value in fns.items()
            )
        elif fns is not None:
            raise ValueError(
                "Argument `_fns` of across must be None, a function, "
                "a formula, or a dict of functions."
            )

        self.data = data
        self.cols = cols
        self.fns = fns_list
        self.names = names
        self.args = args or ()
        self.kwargs = kwargs or {}

    def evaluate(self, context=None):
        """Evaluate object with context"""
        if isinstance(context, Context):  # pragma: no cover
            context = context.value

        if not self.fns:
            self.fns = [{"fn": lambda x: x}]

        ret = None
        # Instead of df.apply(), we can recycle groupby values and more
        for column in self.cols:
            for fn_info in self.fns:
                render_data = fn_info.copy()
                render_data["_col"] = column
                fn = render_data.pop("fn")
                name_format = self.names
                if not name_format:
                    name_format = (
                        "{_col}_{_fn}" if "_fn" in render_data else "{_col}"
                    )

                name = name_format.format(**render_data)
                args = CurColumn.replace_args(self.args, column)
                kwargs = CurColumn.replace_kwargs(self.kwargs, column)

                if (
                    getattr(fn, "_pipda_functype", None) == "verb"
                    and fn.dependent
                ):
                    value = fn(  # pragma: no cover
                        self.data,
                        self.data[column],
                        *args,
                        __ast_fallback="normal",
                        **kwargs,
                    )
                else:
                    value = fn(
                        self.data[column],
                        *evaluate_expr(args, self.data, context),
                        **evaluate_expr(kwargs, self.data, context),
                    )

                ret = add_to_tibble(ret, name, value, broadcast_tbl=True)

        return Tibble() if ret is None else ret


class IfCross(Across, ABC):
    """Base class for IfAny and IfAll"""

    @staticmethod
    @abstractmethod
    def aggregate(values):
        """How to aggregation by rows"""

    def evaluate(
        self,
        context=None,
    ):
        """Evaluate the object with context"""
        # Fill NA first and then do and/or
        # Since NA | True -> False for pandas
        return (
            super()
            .evaluate(context)
            .apply(self.__class__.aggregate, axis=1)
            .astype(bool)
        )


class IfAny(IfCross):
    """For calls from dplyr's if_any"""

    @staticmethod
    def aggregate(values):
        """How to aggregation by rows"""
        return values.fillna(False).astype(bool).any()


class IfAll(IfCross):
    """For calls from dplyr's if_all"""

    @staticmethod
    def aggregate(values):
        """How to aggregation by rows"""
        return values.fillna(False).astype(bool).all()


@singledispatch
def glimpse_formatter(x):
    """Formatter passed to glimpse to format a single element of a dataframe."""
    return str(x)


@glimpse_formatter.register(DataFrame)
def _dataframe_formatter(x):
    """Format a dataframe element."""
    return f"<DF {x.shape[0]}x{x.shape[1]}>"


@glimpse_formatter.register(str)
def _str_formatter(x):
    """Format a string"""
    return repr(x)


class Glimpse:
    """Glimpse class

    Args:
        x: The data to be glimpseed
        width: The width of the output
        formatter: The formatter to use to format data elements
    """
    def __init__(self, x, width, formatter) -> None:
        self.x = x
        self.width = width or get_terminal_size((100, 20)).columns
        self.formatter = formatter
        self.colwidths = (0, 0)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        self._calculate_output_widths()
        return "\n".join(
            (
                "\n".join(self._general()),
                "\n".join(self._variables()),
            )
        )

    def _repr_html_(self):
        out = []
        for gen in self._general():
            out.append(f"<div><i>{gen}</i></div>")
        out.append("<table>")
        out.extend(self._variables(fmt="html"))
        out.append("</table>")
        return "\n".join(out)

    def _general(self):
        if isinstance(self.x, TibbleGrouped):
            groups = ", ".join((str(name) for name in self.x.group_vars))
            group_title = (
                "Rowwise" if isinstance(self.x, TibbleRowwise) else "Groups"
            )
            return (
                f"Rows: {self.x.shape[0]}",
                f"Columns: {self.x.shape[1]}",
                f"{group_title}: {groups} "
                f"[{self.x._datar['grouped'].grouper.ngroups}]",
            )

        return (
            f"Rows: {self.x.shape[0]}",
            f"Columns: {self.x.shape[1]}",
        )

    def _calculate_output_widths(self):
        colname_width = max(len(str(colname)) for colname in self.x.columns)
        dtype_width = max(len(str(dtype)) for dtype in self.x.dtypes) + 2
        self.colwidths = (colname_width, dtype_width)

    def _variables(self, fmt="str"):
        for col in self.x:
            yield self._format_variable(
                col,
                self.x[col].dtype,
                self.x[col].obj.values
                if isinstance(self.x[col], SeriesGroupBy)
                else self.x[col].values,
                fmt=fmt,
            )

    def _format_variable(self, col, dtype, data, fmt="str"):
        if fmt == "str":
            return self._format_variable_str(col, dtype, data)

        return self._format_variable_html(col, dtype, data)

    def _format_data(self, data):
        """Format the data for the glimpse view

        Formatting 10 elements in a batch in case of a long dataframe.
        Since we don't need to format all the data, but only the first a few
        till the line (terminal width or provided width) overflows.
        """
        out = ""
        placeholder = "…"
        i = 0
        chunk_size = 10
        while not out.endswith(placeholder) and i < data.size:
            if out:
                out += ", "
            out += ", ".join(
                self.formatter(d) for d in data[i:i + chunk_size]
            )
            i += chunk_size
            out = textwrap.shorten(
                out,
                break_long_words=True,
                break_on_hyphens=True,
                width=self.width - 4 - sum(self.colwidths),
                placeholder=placeholder,
            )
        return out

    def _format_variable_str(self, col, dtype, data):
        name_col = col.ljust(self.colwidths[0])
        dtype_col = f'<{dtype}>'.ljust(self.colwidths[1])
        data_col = self._format_data(data)
        return f". {name_col} {dtype_col} {data_col}"

    def _format_variable_html(self, col, dtype, data):
        name_col = f". <b>{col}</b>"
        dtype_col = f"<i>&lt;{dtype}&gt;</i>"
        data_col = html.escape(self._format_data(data))
        return (
            f"<tr><th style=\"text-align: left\">{name_col}</th>"
            f"<td style=\"text-align: left\">{dtype_col}</td>"
            f"<td style=\"text-align: left\">{data_col}</td></tr>"
        )


class CurColumn:
    """Current column in across"""

    @classmethod
    def replace_args(cls, args: Tuple[Any], column: str) -> Tuple[Any, ...]:
        """Replace self with the real column in args"""
        return tuple(column if isinstance(arg, cls) else arg for arg in args)

    @classmethod
    def replace_kwargs(
        cls, kwargs: Mapping[str, Any], column: str
    ) -> Mapping[str, Any]:
        """Replace self with the real column in kwargs"""
        return {
            key: column if isinstance(val, cls) else val
            for key, val in kwargs.items()
        }

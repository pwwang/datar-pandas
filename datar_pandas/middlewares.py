"""Middlewares for datar"""
from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple

from pipda import Context, Verb, evaluate_expr

from .common import is_scalar
from .broadcast import add_to_tibble
from .pandas import DataFrame
from .utils import vars_select
from .tibble import Tibble
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
        if isinstance(context, Context):
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

                if isinstance(fn, Verb):
                    if fn.registered(DataFrame):
                        value = fn(
                            self.data,
                            self.data[column],
                            *args,
                            __ast_fallback="normal",
                            **kwargs,
                        )
                    else:
                        value = fn(
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

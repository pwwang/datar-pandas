"""Provides specific contexts for datar"""
from enum import Enum

from pipda.context import (
    ContextBase,
    ContextEval as ContextEvalPipda,
    ContextPending,
    ContextSelect,
)

from .pandas import DataFrame
from .tibble import Tibble


class ContextAutoEvalError(Exception):
    """Raised when auto evaluation fails"""


class ContextEval(ContextEvalPipda):
    """Evaluation context"""

    def _save_used_ref(self, parent, ref, level) -> None:
        """Increments the counters for used references"""
        if (
            not isinstance(parent, Tibble)
            or not isinstance(ref, str)
            or level != 1
            or "used_refs" not in parent._datar
        ):
            return

        parent._datar["used_refs"].add(ref)

    def getitem(self, parent, ref, level):
        """Interpret f[ref]"""
        self._save_used_ref(parent, ref, level)
        return super().getitem(parent, ref, level)

    def getattr(self, parent, ref, level):
        """Evaluate f.a"""

        self._save_used_ref(parent, ref, level)
        if isinstance(parent, DataFrame):
            return parent[ref]

        return super().getattr(parent, ref, level)

    @property
    def ref(self) -> ContextBase:
        """Defines how `item` in `f[item]` is evaluated.

        This function should return a `ContextBase` object."""
        return Context.SELECT


class ContextAutoEval(ContextEvalPipda):
    """Auto-evaluation of expressions inside a function so that
    `data >> mutate(x=f.col, y=f.x * 2)` can be first evaluated as
    `data >> mutate(x=f.col, y=f.col * 2)`
    """

    def getitem(self, parent, ref, level):
        """Interpret f[ref]"""
        # parent must be a dict
        if level == 1 and ref not in parent:
            raise ContextAutoEvalError(str(ref))

        return super().getitem(parent, ref, level)

    def getattr(self, parent, ref, level):
        """Evaluate f.a"""

        if level == 1:
            return self.getitem(parent, ref, level)

        return super().getattr(parent, ref, level)

    @property
    def ref(self) -> ContextBase:
        """Defines how `item` in `f[item]` is evaluated.

        This function should return a `ContextBase` object."""
        return Context.SELECT


class Context(Enum):
    """Context enumerator for types of contexts"""

    PENDING = ContextPending()
    SELECT = ContextSelect()
    EVAL = ContextEval()
    AUTOEVAL = ContextAutoEval()

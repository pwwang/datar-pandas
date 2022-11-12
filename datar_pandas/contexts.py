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
        if isinstance(parent, (dict, DataFrame)):
            return self.getitem(parent, ref, level)

        self._save_used_ref(parent, ref, level)
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

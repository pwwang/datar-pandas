from typing import Any, cast

from datar.apis.base import any_na

# numpy implmentation of any_na
from datar_numpy.api import sets as _  # noqa: F401

from ...common import is_null, is_scalar
from ...factory import func_bootstrap


def _any_na_impl(x: Any) -> bool:
    nulls = is_null(x)
    if is_scalar(x):  # pragma: no cover
        return bool(nulls)
    if isinstance(nulls, bool):  # pragma: no cover
        return nulls
    return any(cast(Any, nulls))


func_bootstrap(
    any_na,
    func=_any_na_impl,
    kind="agg",
)

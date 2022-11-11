from datar.apis.base import any_na

from ...common import is_null, is_scalar
from ...factory import func_bootstrap

func_bootstrap(
    any_na,
    func=lambda x: is_null(x) if is_scalar(x) else is_null(x).any(),
    kind="agg",
)

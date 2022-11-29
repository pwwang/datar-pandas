"""Functions from tidyr"""

from datar.apis.tidyr import full_seq

from ...pandas import Series
from ...factory import func_bootstrap
from ...utils import as_series
from ..base.seq import seq


@full_seq.register(object, backend="pandas")
def _full_seq_obj(x, period, tol=1e-6):
    x = as_series(x)
    return full_seq.dispatch(Series)(x, period, tol=tol)


@func_bootstrap(full_seq, exclude={"period", "tol"})
def _full_seq_bootstrap(x, period, tol=1e-6):
    """Create the full sequence of values in a vector

    Args:
        x: A numeric vector.
        period: Gap between each observation. The existing data will be
            checked to ensure that it is actually of this periodicity.
        tol: Numerical tolerance for checking periodicity.

    Returns:
        The full sequence
    """

    minx = x.min()  # na not counted
    maxx = x.max()

    if (
        ((x - minx) % period > tol) & (period - ((x - minx) % period) > tol)
    ).any():
        raise ValueError("`x` is not a regular sequence.")

    if period - ((maxx - minx) % period) <= tol:
        maxx += tol

    return seq(minx, maxx, by=period)

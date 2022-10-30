from ...defaults import USE_MODIN

if USE_MODIN:  # pragma: no cover
    from modin.pandas.groupby import (
        DataFrameGroupBy,
        GroupBy,
        SeriesGroupBy,
    )
else:
    from pandas.core.groupby import (
        DataFrameGroupBy,
        GroupBy,
        SeriesGroupBy,
    )

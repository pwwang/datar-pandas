from datar import get_option

from pandas.testing import (  # noqa: F401
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from pandas.api.types import (  # noqa: F401
    is_array_like,
    is_bool,
    is_number,
    is_scalar,
    is_categorical_dtype,
    is_complex_dtype,
    is_integer,
    is_integer_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_list_like,
    union_categoricals,
    CategoricalDtype,
)

if get_option("use_modin"):  # pragma: no cover
    from modin.pandas import (  # noqa: F401
        Categorical,
        CategoricalIndex,
        MultiIndex,
        DataFrame,
        Grouper,
        Index,
        Interval,
        RangeIndex,
        Series,
        Timestamp,
        __version__,
        crosstab,
        concat,
        cut,
        qcut,
        isna,
        isnull,
        merge,
        notnull,
        pivot_table,
        read_csv,
        to_datetime,
        unique,
    )
    from modin.pandas.base import BasePandasDataset as NDFrame  # noqa: F401
    from modin.pandas.groupby import DataFrameGroupBy, SeriesGroupBy  # noqa: F401
    try:
        from modin.pandas.groupby import GroupBy  # noqa: F401
    except ImportError:
        GroupBy = DataFrameGroupBy

    PandasObject = (NDFrame, DataFrameGroupBy)

    def get_obj(grouped):
        return grouped._df

else:
    from pandas import (  # noqa: F401
        Categorical,
        CategoricalIndex,
        MultiIndex,
        DataFrame,
        Grouper,
        Index,
        Interval,
        RangeIndex,
        Series,
        Timestamp,
        __version__,
        crosstab,
        concat,
        cut,
        qcut,
        isna,
        isnull,
        merge,
        notnull,
        pivot_table,
        read_csv,
        to_datetime,
        unique,
    )

    from pandas.core.base import PandasObject  # noqa: F401
    from pandas.core.generic import NDFrame
    from pandas.core.groupby import (  # noqa: F401
        DataFrameGroupBy,
        GroupBy,
        SeriesGroupBy,
    )

    def get_obj(grouped):
        return grouped.obj

    def is_categorical_dtype(x):  # noqa: F811
        # pandas2.1
        # is_categorical_dtype is deprecated and will be removed in a future version.
        # Use isinstance(dtype, CategoricalDtype) instead
        return isinstance(getattr(x, "dtype", None), CategoricalDtype)

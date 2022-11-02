from datar import get_option

from pandas.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from pandas.api.types import (
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
)

if get_option("use_modin"):  # pragma: no cover
    from modin.pandas import (
        Categorical,
        CategoricalIndex,
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
        isna,
        isnull,
        merge,
        notnull,
        pivot_table,
        read_csv,
        to_datetime,
        unique,
    )
    from modin.pandas.base import BasePandasDataset as NDFrame
    from modin.pandas.groupby import (
        DataFrameGroupBy,
        GroupBy,
        SeriesGroupBy,
    )

    PandasObject = (NDFrame, DataFrameGroupBy)

else:
    from pandas import (
        Categorical,
        CategoricalIndex,
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
        isna,
        isnull,
        merge,
        notnull,
        pivot_table,
        read_csv,
        to_datetime,
        unique,
    )

    from pandas.core.base import PandasObject
    from pandas.core.generic import NDFrame
    from pandas.core.groupby import (
        DataFrameGroupBy,
        GroupBy,
        SeriesGroupBy,
    )

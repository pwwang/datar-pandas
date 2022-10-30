from ...defaults import USE_MODIN

if USE_MODIN:
    from modin.pandas.base import BasePandasDataset
    from modin.pandas.groupby import DataFrameGroupBy

    PandasObject = (BasePandasDataset, DataFrameGroupBy)
else:
    from pandas.core.base import PandasObject

from ...defaults import USE_MODIN

if USE_MODIN:
    from modin.pandas.base import BasePandasDataset as NDFrame
else:
    from pandas.core.generic import NDFrame

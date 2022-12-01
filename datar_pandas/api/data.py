from typing import TYPE_CHECKING

from datar.apis.data import load_dataset

from ..pandas import read_csv

if TYPE_CHECKING:
    from datar.data import Metadata


@load_dataset.register(str, backend="pandas")
def load_dataset(name: str, meta: "Metadata"):
    return read_csv(meta.source, index_col=0 if meta.index else False)

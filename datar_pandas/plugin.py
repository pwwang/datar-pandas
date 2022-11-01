from typing import TYPE_CHECKING

from simplug import Simplug


if TYPE_CHECKING:
    from datar.data.metadata import Metadata

plugin = Simplug("datar")


@plugin.impl
def setup():
    from datar.core.options import add_option
    add_option("use_modin", False)


@plugin.impl
def load_dataset(name: str, meta: "Metadata"):
    from .pandas import read_csv

    return read_csv(meta.source, index_col=0 if meta.index else False)


@plugin.impl
def base_api():
    ...


@plugin.impl
def get_versions():
    import pandas
    from datar.core.options import get_option
    from .version import __version__

    out = {
        "datar-pandas": __version__,
        "pandas": pandas.__version__,
    }

    if get_option("use_modin"):
        import modin

        out["modin"] = modin.__version__

    return out


@plugin.impl
def c_getitem(item):
    ...


@plugin.impl
def operate(op, x, y=None):
    from .operators import operate as operate_
    return operate_(op, x, y)

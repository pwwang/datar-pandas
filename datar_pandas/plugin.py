from typing import TYPE_CHECKING, Mapping

import pdtypes
from datar.core.plugin import plugin

# Attach version to the plugin
from .version import __version__  # noqa: F401
from .pandas import read_csv

if TYPE_CHECKING:
    from .pandas import DataFrame


@plugin.impl
def setup():
    from datar.core.options import add_option
    pdtypes.patch()
    add_option("use_modin", False)
    add_option("dplyr_summarise_inform", True)


@plugin.impl
def load_dataset(name: str, metadata: Mapping) -> "DataFrame":
    if name not in metadata:
        raise AttributeError(
            f"No such dataset: {name}. "
            "Use datar.data.descr_datasets() to see all available datasets."
        )

    meta = metadata[name]
    return read_csv(meta.source, index_col=0 if meta.index else False)


@plugin.impl
def base_api():
    from .api.base import (  # noqa: F401
        arithm,
        asis,
        bessel,
        complex,
        cum,
        factor,
        funs,
        null,
        random,
        seq,
        special,
        string,
        table,
        trig,
        verbs,
        which,
    )
    from .api.base.asis import as_pd_date

    return {"as_pd_date": as_pd_date}


@plugin.impl
def dplyr_api():
    from .api.dplyr import (  # noqa: F401
        across,
        arrange,
        bind,
        context,
        count_tally,
        desc,
        distinct,
        filter_,
        funs,
        glimpse,
        group_by,
        group_data,
        group_iter,
        if_else,
        join,
        lead_lag,
        mutate,
        order_by,
        pick,
        pull,
        rank,
        recode,
        relocate,
        rename,
        rows,
        select,
        sets,
        slice_,
        summarise,
        tidyselect,
    )


@plugin.impl
def tibble_api():
    from .api.tibble import tibble, verbs  # noqa: F401


@plugin.impl
def tidyr_api():
    from .api.tidyr import (  # noqa: F401
        chop,
        complete,
        drop_na,
        expand,
        extract,
        fill,
        funs,
        nest,
        pack,
        pivot_long,
        pivot_wide,
        replace_na,
        separate,
        uncount,
        unite,
    )


@plugin.impl
def forcats_api():
    from .api.forcats import (  # noqa: F401
        fct_multi,
        lvl_addrm,
        lvl_order,
        lvl_value,
        lvls,
        misc,
    )


@plugin.impl
def misc_api():
    from .api.misc import (
        itemgetter,
        attrgetter,
        pd_cat,
        pd_dt,
        pd_str,
        flatten,
    )
    return {
        "itemgetter": itemgetter,
        "attrgetter": attrgetter,
        "pd_cat": pd_cat,
        "pd_dt": pd_dt,
        "pd_str": pd_str,
        "flatten": flatten,
    }


@plugin.impl
def get_versions():
    import pandas
    from datar.core.options import get_option

    out = {
        "datar-pandas": __version__,
        "pandas": pandas.__version__,
    }

    if get_option("use_modin"):  # pragma: no cover
        import modin
        out["modin"] = modin.__version__

    return out


@plugin.impl
def c_getitem(item):
    from .collections import Collection
    return Collection(item)


@plugin.impl
def operate(op, x, y=None):
    from .operators import operate as operate_
    return operate_(op, x, y)

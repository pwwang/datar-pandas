from typing import TYPE_CHECKING

import pdtypes
from simplug import Simplug


if TYPE_CHECKING:
    from datar.data.metadata import Metadata

plugin = Simplug("datar")


def _base_impl():
    from .api.base import (
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
def setup():
    from datar.core.options import add_option
    pdtypes.patch()
    add_option("use_modin", False)
    add_option("dplyr_summarise_inform", True)


@plugin.impl
def load_dataset(name: str, meta: "Metadata"):
    from .pandas import read_csv
    return read_csv(meta.source, index_col=0 if meta.index else False)


@plugin.impl
def base_api():
    return _base_impl()


@plugin.impl
def dplyr_api():
    _base_impl()
    from .api.dplyr import (
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
    _base_impl()
    from .api.tibble import tibble, verbs


@plugin.impl
def tidyr_api():
    _base_impl()
    from .api.tidyr import (
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
    _base_impl()
    from .api.forcats import (
        fct_multi,
        lvl_addrm,
        lvl_order,
        lvl_value,
        lvls,
        misc,
    )


@plugin.impl
def other_api():
    _base_impl()
    from .api.other import (
        itemgetter,
        attrgetter,
        pd_cat,
        pd_dt,
        pd_str,
    )
    return {
        "itemgetter": itemgetter,
        "attrgetter": attrgetter,
        "pd_cat": pd_cat,
        "pd_dt": pd_dt,
        "pd_str": pd_str,
    }


@plugin.impl
def get_versions():
    import pandas
    from datar.core.options import get_option
    from .version import __version__

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

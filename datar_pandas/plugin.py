import pdtypes
from simplug import Simplug

plugin = Simplug("datar")


@plugin.impl
def setup():
    from datar.core.options import add_option
    pdtypes.patch()
    add_option("use_modin", False)
    add_option("dplyr_summarise_inform", True)


@plugin.impl
def data_api():
    from .api import data


@plugin.impl
def base_api():
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
def dplyr_api():
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
    from .api.tibble import tibble, verbs


@plugin.impl
def tidyr_api():
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

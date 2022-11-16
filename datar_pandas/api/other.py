from pipda import register_verb, register_func
from datar_numpy.utils import make_array

from ..pandas import Series, PandasObject, SeriesGroupBy
from ..contexts import Context
from ..factory import func_factory
from ..utils import as_series
from ..tibble import Tibble, TibbleGrouped
from ..collections import Collection


def _itemgetter_post(__out, x, subscr, __args_raw=None):
    rawx = __args_raw["x"]
    idx = __out.index.get_level_values(0)
    out = __out.reset_index(drop=True, level=0)
    return out.groupby(
        idx,
        sort=rawx.sort,
        dropna=rawx.dropna,
        observed=rawx.observed,
    )


@register_func(cls=object, dispatchable=True, pipeable=True)
def itemgetter(x, subscr):
    """Itemgetter as a function for verb

    In datar expression, we can do:
    >>> arr = [1,2,3]
    >>> tibble(x=2) >> mutate(y=arr[f.x])

    Since `arr[f.x]` won't compile. We need to use the `itemgetter` operator:
    >>> tibble(x=2) >> mutate(y=itemgetter(arr, f.x))

    Args:
        data: The data to be get items from
        subscr: The subscripts
    """
    x = make_array(x)
    if isinstance(subscr, Collection):
        subscr.expand(pool=x.size)

    return x[make_array(subscr)]


@itemgetter.register(PandasObject, backend="pandas")
def _itemgetter_pobj(x, subscr):
    if isinstance(x, TibbleGrouped):
        if isinstance(subscr, SeriesGroupBy):
            df = Tibble.from_args(x=x, subscr=subscr)
            return df.apply(
                lambda subdf: itemgetter(subdf['x'], subdf['subscr'])
            )

        return x._datar["grouped"].apply(
            lambda subdf: itemgetter(subdf, subscr)
        )

    if isinstance(x, SeriesGroupBy):
        return x.apply(lambda ser: itemgetter(ser, subscr))

    if isinstance(x, PandasObject):
        if isinstance(subscr, Collection):
            subscr.expand(x.shape[0])

        subscr = make_array(subscr)
        return x.iloc[subscr, :] if x.ndim == 2 else x.iloc[subscr]

    # x is not a PandasObject
    # then subscr must be a PandasObject
    # as the function is registered for PandasObject
    x = as_series(x)
    return x.iloc[make_array(subscr)]


class _MethodAccessor:
    """Method holder for `_Accessor` objects"""

    def __init__(self, accessor, method):
        self.accessor = accessor
        self.method = method

    def __call__(self, *args, **kwds):
        out = self.accessor.sgb.apply(
            lambda x: getattr(
                getattr(x, self.accessor.name),
                self.method
            )(*args, **kwds)
        )

        try:
            return out.groupby(
                self.accessor.sgb.grouper,
                observed=self.accessor.sgb.observed,
                sort=self.accessor.sgb.sort,
                dropna=self.accessor.sgb.dropna,
            )
        except (AttributeError, ValueError, TypeError):  # pragma: no cover
            return out


class _Accessor:
    """Accessor for special columns, such as `.str`, `.cat` and `.dt`, etc

    This is used for SeriesGroupBy object, since `sgb.str` cannot be evaluated
    immediately.
    """
    def __init__(self, sgb: SeriesGroupBy, name: str):
        self.sgb = sgb
        self.name = name

    def __getitem__(self, key):
        return _MethodAccessor(self, "__getitem__")(key)

    def __getattr__(self, name):
        # See if name is a method
        accessor = getattr(Series, self.name)  # Series.str
        attr_or_method = getattr(accessor, name, None)

        if callable(attr_or_method):
            # x.str.lower()
            return _MethodAccessor(self, name)

        # x.cat.categories
        out = self.sgb.apply(
            lambda x: getattr(getattr(x, self.name), name)
        )

        try:
            return out.groupby(
                self.sgb.grouper,
                observed=self.sgb.observed,
                sort=self.sgb.sort,
                dropna=self.sgb.dropna,
            )
        except (AttributeError, ValueError, TypeError):  # pragma: no cover
            return out


@func_factory(kind="agg")
def attrgetter(x, attr):
    """Attrgetter as a function for verb

    This is helpful when we want to access to an accessor
    (ie. CategoricalAccessor) from a SeriesGroupBy object
    """
    return getattr(x, attr)


@attrgetter.register(SeriesGroupBy)
def _attrgetter_sgb(x, attr):
    return _Accessor(x, attr)


@register_verb(PandasObject, context=Context.EVAL)
def pd_str(x):
    """Pandas' str accessor for a Series (x.str)

    This is helpful when x is a SeriesGroupBy object
    """
    return attrgetter(x, "str", __ast_fallback="normal")


@register_verb(PandasObject, context=Context.EVAL)
def pd_cat(x):
    """Pandas' cat accessor for a Series (x.cat)

    This is helpful when x is a SeriesGroupBy object
    """
    return attrgetter(x, "cat", __ast_fallback="normal")


@register_verb(PandasObject, context=Context.EVAL)
def pd_dt(x):
    """Pandas' dt accessor for a Series (x.dt)

    This is helpful when x is a SeriesGroupBy object
    """
    return attrgetter(x, "dt", __ast_fallback="normal")

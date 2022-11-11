from pipda import register_verb

from ..pandas import Series, PandasObject, SeriesGroupBy
from ..contexts import Context
from ..factory import func_factory
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


@func_factory(post=_itemgetter_post)
def itemgetter(x, subscr, __args_raw=None):
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
    # allow f[:2] to work
    # subscr = evaluate_expr(subscr, x, Context.EVAL)
    if isinstance(subscr, Collection):
        subscr.expand(pool=x.size)

    if isinstance(subscr, Series):
        subscr = subscr.values
    out = x.iloc[subscr]
    if isinstance(__args_raw["x"], PandasObject):
        return out
    return out.values


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

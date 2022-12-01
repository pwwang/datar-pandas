"""Provide shortcuts to register functions for different types but """
from __future__ import annotations

import inspect
from functools import singledispatch, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
from pipda import register_func
from datar.core.utils import arg_match

from .utils import NO_DEFAULT
from .pandas import DataFrame, Series, PandasObject, SeriesGroupBy
from .tibble import Tibble, TibbleGrouped, TibbleRowwise

if TYPE_CHECKING:
    from inspect import Signature, BoundArguments


def _preprocess_data_args(
    args: Tuple,
    kwargs: Mapping,
    exclude: Set[str],
    signature: Signature,
) -> Tuple[BoundArguments, Tibble]:
    """Preprocess the data arguments.

    Args:
        args: The args to be processed
        kwargs: The kwargs to be processed
        exclude: The data arguments
        signature: The signature of the function

    Returns:
        The data, args, kwargs and args frame
    """
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()

    args_raw = bound.arguments.copy()
    args_df = Tibble.from_args(
        **{
            key: (
                val
                if bound.signature.parameters[key].kind
                != inspect.Parameter.VAR_POSITIONAL
                else None
                if len(val) == 0
                else Tibble.from_pairs([str(i) for i in range(len(val))], val)
            )
            for key, val in bound.arguments.items()
            if key not in exclude
        }
    )

    # inject __args_raw and __args_frame
    for arg in bound.arguments:
        if arg == "__args_frame":
            bound.arguments[arg] = args_df
        elif arg == "__args_raw":
            bound.arguments[arg] = args_raw
        elif arg in args_df or args_df.columns.str.startswith(f"{arg}$").any():
            if (
                bound.signature.parameters[arg].kind
                != inspect.Parameter.VAR_POSITIONAL
            ):
                bound.arguments[arg] = args_df[arg]
            elif len(bound.arguments[arg]) > 0:
                # nest frames
                bound.arguments[arg] = tuple(
                    args_df[arg].to_dict("series").values()
                )
            # else:  # nothing passed to *args

    return bound, args_df


def _with_hooks(
    func: Callable = None,
    pre: Callable = None,
    post: str | Callable = None,
) -> Callable:
    """Apply hooks to a function

    Args:
        func: The function to be wrapped
        pre: The pre hook, takes the `*args`, `**kwargs`. If it
            returns None, the original `*args` and `**kwargs` will be used.
            Otherwise, it should return a tuple of `args` and `kwargs`
        post: The post hook, takes the `__out`, and `*args`, `**kwargs`.
            It should return the modified `__out`.

    Returns:
        The wrapped function
    """
    if func is None:
        return lambda fun: _with_hooks(fun, pre, post)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if pre:
            arguments = pre(*args, **kwargs)
            if arguments is not None:
                args, kwargs = arguments
        out = func(*args, **kwargs)
        if post == "transform":
            grouped = [
                arg for arg in args
                if isinstance(arg, (SeriesGroupBy, TibbleGrouped))
            ]
            if not grouped:
                return out
            grouped = grouped[0]
            is_rowwise = False
            if isinstance(grouped, TibbleGrouped):  # pragma: no cover
                if isinstance(grouped, TibbleRowwise):
                    is_rowwise = True
                grouped = grouped._datar["grouped"]
                if getattr(grouped, "is_rowwise", False):
                    is_rowwise = True
            elif isinstance(grouped, SeriesGroupBy):
                is_rowwise = getattr(grouped, "is_rowwise", False)

            out = out.groupby(
                grouped.grouper,
                sort=grouped.sort,
                observed=grouped.observed,
                dropna=grouped.dropna,
            )
            if is_rowwise:
                out.is_rowwise = True

        elif callable(post):
            out = post(out, *args, **kwargs)
        return out

    return wrapper


def _deconstruct_df(df: DataFrame) -> List[DataFrame | Series]:
    """Deconstruct a dataframe into a list of series and dataframes

    Examples:
        >>> _deconstruct_df(tibble(a=1, b=2))
        >>> # [Series([1], name="a"), Series([2], name="b")]
        >>> _deconstruct_df(tibble(a=tibble(x=1, y=2), c=3))
        >>> # [DataFrame({"x": [1], "y": [2]}), Series([3], name="c")]

    Args:
        df: The dataframe to be deconstructed

    Returns:
        The deconstructed list
    """
    cnames = [col.split("$", 1)[0] for col in df.columns]
    unames, idx = np.unique(cnames, return_index=True)
    return [df[unames[ix]] for ix in np.argsort(idx)]


def _bootstrap_agg_func(
    registered: Callable,
    func: Callable,
    pre: Callable,
    post: Callable,
) -> Callable:

    @registered.register(Series, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _series_agg(*args, **kwargs):
        if isinstance(func, str) and hasattr(args[0], func):
            return getattr(args[0], func)(*args[1:], **kwargs)
        return func(*args, **kwargs)

    @registered.register(DataFrame, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _df_agg(*args, **kwargs):  # pragma: no cover
        return args[0].agg(func, 0, *args[1:], **kwargs).to_frame().T

    @registered.register(SeriesGroupBy, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _seriesgroupby_agg(*args, **kwargs):
        return args[0].agg(func, *args[1:], **kwargs)

    @registered.register(TibbleGrouped, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _tibblegrouped_agg(*args, **kwargs):  # pragma: no cover
        return Tibble(
            args[0]._datar["grouped"].agg(func, *args[1:], **kwargs),
            copy=False,
        )

    @registered.register(TibbleRowwise, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _tibblerowwise_agg(*args, **kwargs):
        return args[0].agg(func, 1, *args[1:], **kwargs)

    return registered


def _bootstrap_transform_func(
    registered: Callable,
    func: Callable,
    pre: Callable,
    post: Callable,
) -> Callable:

    @registered.register(Series, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _series_transform(*args, **kwargs):
        if isinstance(func, str) and hasattr(args[0], func):  # pragma: no cover
            out = getattr(args[0], func)(*args[1:], **kwargs)
        else:
            out = func(*args, **kwargs)
        if not isinstance(out, Series):
            out = Series(out, index=args[0].index)
        return out

    @registered.register(DataFrame, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _df_transform(*args, **kwargs):  # pragma: no cover
        return args[0].transform(func, 0, *args[1:], **kwargs)

    @registered.register(SeriesGroupBy, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _seriesgroupby_transform(*args, **kwargs):
        return args[0].transform(func, *args[1:], **kwargs)

    @registered.register(TibbleGrouped, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _tibblegrouped_transform(*args, **kwargs):  # pragma: no cover
        return args[0]._datar["grouped"].transform(func, *args[1:], **kwargs)

    @registered.register(TibbleRowwise, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _tibblerowwise_transform(*args, **kwargs):  # pragma: no cover
        return args[0].transform(func, 1, *args[1:], **kwargs)

    return registered


def _bootstrap_apply_func(
    registered: Callable,
    func: Callable,
    pre: Callable,
    post: Callable,
    exclude: str | Sequence[str] = (),
    signature: inspect.Signature = None,
) -> Callable:

    signature = signature or inspect.signature(func)

    if isinstance(exclude, str):
        exclude = {exclude}
    else:
        exclude = set(exclude)

    @singledispatch
    def apply_df(data, bound, exclude, func):
        """The frame apply function"""
        for key in bound.arguments:
            # When key is not in data, that means the value is None
            if key in exclude:
                continue

            try:
                dt = data[key]
            except KeyError:
                continue
            if (
                bound.signature.parameters[key].kind
                == inspect.Parameter.VAR_POSITIONAL
            ):
                dt = _deconstruct_df(dt)
            bound.arguments[key] = dt

        return func(*bound.args, **bound.kwargs)

    @apply_df.register(TibbleGrouped)
    def _apply_df_grouped(data, bound, exclude, func):
        return data._datar["grouped"].apply(
            apply_df.dispatch(object),
            bound,
            exclude,
            func,
        )

    @registered.register(PandasObject, backend="pandas")
    @_with_hooks(pre=pre, post=post)
    def _pandasobject_apply(*args, **kwargs):
        bound, args_frame = _preprocess_data_args(
            args,
            kwargs,
            exclude,
            signature,
        )
        return apply_df(args_frame, bound, exclude, func)

    registered.apply_df = apply_df
    return registered


def func_factory(
    func: Callable = None,
    *,
    kind: str = "apply",
    name: str = None,
    qualname: str = None,
    doc: str = None,
    module: str = None,
    pipeable: bool = False,
    dispatchable: str = None,
    ast_fallback: str = "normal_warning",
    pre: Callable = None,
    post: Callable = None,
    **kwargs,  # Other kind-specific arguments.
) -> Callable:
    """A factory to register functions.

    The function applies same function `func` on different data types. For
    example, DataFrame, Series, SeriesGroupBy, etc.

    Args:
        func: The function to apply to different types of data.
            If not provided, this function will return a decorator.
        kind: The kind of the focal function. Can be one of "agg", "transform",
            "apply".
        name: The name of the function, used to overwrite `func`'s name.
        qualname: The qualified name of the function, used to overwrite
            `func`'s qualname.
        doc: The docstring of the function, used to overwrite `func`'s doc.
        module: The module of the function, used to overwrite `func`'s module.
        pipeable: Whether the function is pipeable.
        dispatchable: The arguments to dispatch on. If not provided, the
            function will be dispatched on the first argument if kind is
            "agg", "aggregation" or "transform". Otherwise "args", meaning
            the function will be dispatched on positional arguments.
        ast_fallback: The AST fallback mode. Can be one of `normal`,
            `normal_warning`, `piping`, `piping_warning` and `raise`.
            See also `help(pipda.register_func)`.
        pre: The pre hook, takes the `*args`, `**kwargs`. If it
            returns None, the original `*args` and `**kwargs` will be used.
            Otherwise, it should return a tuple of `args` and `kwargs`
        post: The post hook, takes the `out`, `*args`, `**kwargs`.
            It should return the modified `out`.
            For "apply" kind, it could also be `transform` indicating that the
            output should be a transformation that has the same shape as the
            input.
            For "transform" kind, it is "transform" by default.
        **kwargs: Other kind-specific arguments.

    Returns:
        A function that registers the function if `func` is not provided.
        Otherwise the registered function.
    """
    if func is None:
        return lambda fun: func_factory(
            fun,
            kind=kind,
            name=name,
            qualname=qualname,
            doc=doc,
            module=module,
            pipeable=pipeable,
            ast_fallback=ast_fallback,
            pre=pre,
            post=post,
            **kwargs,
        )

    if not dispatchable:
        if kind in {"agg", "aggregation", "transform"}:
            dispatchable = "first"
        else:
            dispatchable = "args"

    registered = register_func(
        func,
        name=name,
        qualname=qualname,
        doc=doc,
        module=module,
        pipeable=pipeable,
        dispatchable=dispatchable,
        ast_fallback=ast_fallback,
    )
    registered.init_pre = pre
    registered.init_post = post
    registered.init_func = func

    return func_bootstrap(registered, func=func, kind=kind, **kwargs)


def func_bootstrap(
    registered: Callable,
    *,
    func: Callable = None,
    kind: str = "apply",
    pre: Callable = NO_DEFAULT,
    post: Callable = NO_DEFAULT,
    **kwargs: Any,
) -> Callable:
    """Bootstrap a function

    When kind is "agg" or "transform", the type of the first argument will be
    used for dispatching. When kind is "apply", the types of all arguments
    except `exclude` will be used for dispatching.

    Args:
        registered: The registered function.
        func: The implementations for all types.
        kind: The kind of the focal function. Can be one of "agg", "transform",
            "apply".
        pre: The pre hook, takes the `*args`, `**kwargs`. If it
            returns None, the original `*args` and `**kwargs` will be used.
            Otherwise, it should return a tuple of `args` and `kwargs`
        post: The post hook, takes the `__out`, `*args`, `**kwargs`.
            It should return the modified `__out`.
            For "apply" kind, it could also be `transform` indicating that the
            output should be a transformation that has the same shape as the
            input.
            For "transform" kind, it is "transform" by default.
        **kwargs: Other kind-specific arguments.
            For "apply", `exclude` is used to exclude arguments for
            broadcasting and dispatching. `signature` might be needed to bind
            the arguments if `inspect.signature()` cannot get the signature.

    Returns:
        The bootstrapped function.
    """
    if not getattr(registered, "_pipda_functype", False):  # pragma: no cover
        raise TypeError("Can only bootstrap a registered function")

    if func is None:
        return lambda fun: func_bootstrap(
            registered,
            func=fun,
            kind=kind,
            pre=pre,
            post=post,
            **kwargs,
        )

    kind = arg_match(kind, "kind", ["apply", "transform", "agg", "aggregation"])

    if func is NO_DEFAULT:  # pragma: no cover
        func = getattr(registered, "init_func", None)

    if pre is NO_DEFAULT:
        pre = getattr(registered, "init_pre", None)

    if post is NO_DEFAULT:
        post = getattr(registered, "init_post", None)

    if kind == "transform" and post is None:
        post = "transform"

    bsfunc = (
        _bootstrap_agg_func
        if kind in ("agg", "aggregation")
        else _bootstrap_transform_func
        if kind == "transform"
        else _bootstrap_apply_func
    )

    return bsfunc(registered, func, pre, post, **kwargs)

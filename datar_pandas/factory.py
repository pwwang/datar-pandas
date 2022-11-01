"""Provide shortcuts to register functions for different types but """
from __future__ import annotations

import inspect
from functools import singledispatch, wraps
from typing import TYPE_CHECKING, Any, Callable, Mapping, Set, Tuple, Type

from pipda import Verb, register_verb
from pipda.context import ContextType

from .contexts import Context
from .utils import NO_DEFAULT
from .pandas import DataFrame, Series, PandasObject, SeriesGroupBy
from .tibble import Tibble, TibbleGrouped, TibbleRowwise

if TYPE_CHECKING:
    from inspect import Signature, BoundArguments

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


def _preprocess_data_args(
    data_args: Set[str],
    signature: Signature,
    data: Any,
    args: Tuple,
    kwargs: Mapping,
) -> Tuple[BoundArguments, Tibble]:
    """Preprocess the data arguments.

    Args:
        data_args: The data arguments
        signature: The signature of the function
        data: The data to be processed
        args: The args to be processed
        kwargs: The kwargs to be processed

    Returns:
        The data, args, kwargs and args frame
    """
    bound = signature.bind(data, *args, **kwargs)
    bound.apply_defaults()

    diff_args = data_args - set(bound.arguments)
    if diff_args:
        raise ValueError(f"Data argument doesn't exist: {diff_args}.")

    args_raw = bound.arguments.copy()
    args_df = Tibble.from_args(
        **{
            key: (
                val
                if bound.signature.parameters[key].kind
                != inspect.Parameter.VAR_POSITIONAL
                else None
                if len(val) == 0
                else Tibble.from_pairs(
                    [str(i) for i in range(len(val))],
                    val
                )
            )
            for key, val in bound.arguments.items()
            if key in data_args
        }
    )

    # inject __args_raw and __args_frame
    for arg in bound.arguments:
        if arg == "__args_frame":
            bound.arguments[arg] = args_df
        elif arg == "__args_raw":
            bound.arguments[arg] = args_raw
        elif (
            arg in args_df
            or args_df.columns.str.startswith(f"{arg}$").any()
        ):
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
    post: Callable = None,
) -> Callable:
    """Apply hooks to a function

    Args:
        func: The function to be wrapped
        pre: The pre hook, takes the `__data` and `*args`, `**kwargs`. If it
            returns None, the original `*args` and `**kwargs` will be used.
            Otherwise, it should return a tuple of `__data`, `args` and `kwargs`
        post: The post hook, takes the `out`, `__data` and `*args`, `**kwargs`.
            It should return the modified `out`.

    Returns:
        The wrapped function
    """
    if func is None:
        return lambda fun: _with_hooks(fun, pre, post)

    @wraps(func)
    def wrapper(__data, *args, **kwargs):
        if pre:
            arguments = pre(__data, *args, **kwargs)
            if arguments is not None:
                __data, args, kwargs = arguments
        out = func(__data, *args, **kwargs)
        if post:
            out = post(out, __data, *args, **kwargs)
        return out

    return wrapper


class BootstrappableVerb(Verb):

    @classmethod
    def from_verb(
        cls: Type[BootstrappableVerb],
        verb: Verb,
    ) -> BootstrappableVerb:
        """Create a BootstrappableVerb from a Verb"""
        inst = cls(
            verb._generic,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            verb.dep,
            verb.ast_fallback,
        )
        inst.contexts = verb.contexts
        inst.extra_contexts = verb.extra_contexts
        inst._generic = verb._generic
        inst.func = verb.func
        inst.registry = verb.registry
        inst.dispatch = verb.dispatch
        inst._signature = verb._signature
        return inst

    @cached_property
    def kind_apply(self) -> Callable:
        """The frame apply function"""
        @singledispatch
        def _df_apply(
            data: PandasObject,
            bound: BoundArguments,
            data_args: Set[str],
            func: Callable = None,
        ) -> Any:
            """Apply a function to a dataframe"""
            for arg in data_args:
                if (
                    bound.signature.parameters[arg].kind
                    == inspect.Parameter.VAR_POSITIONAL
                ):
                    # nest frames?
                    bound.arguments[arg] = data[arg].to_dict("series").values()
                else:
                    bound.arguments[arg] = data[arg]

            return func(*bound.args, **bound.kwargs)

        @_df_apply.register(TibbleGrouped)
        def _(
            data: TibbleGrouped,
            bound: BoundArguments,
            data_args: Set[str],
            func: Callable = None,
        ) -> Any:
            """Apply a function to a grouped dataframe"""
            def to_apply(subdf):
                for arg in data_args:
                    # copy the bound arguments for parallel processing?
                    if (
                        bound.signature.parameters[arg].kind
                        == inspect.Parameter.VAR_POSITIONAL
                    ):
                        bound.arguments[arg] = (
                            subdf[arg].to_dict("series").values()
                        )
                    else:
                        bound.arguments[arg] = subdf[arg]

                if "__args_frame" in bound.arguments:
                    bound.arguments["__args_frame"] = subdf

                return func(*bound.args, **bound.kwargs)

            return data._datar["grouped"].apply(to_apply)

        return _df_apply

    def _bootstrap_pandas_agg(
        self,
        func: str | Callable,
        context: ContextType,
        extra_contexts: Mapping[str, ContextType],
        pre: Callable,
        post: Callable,
        # no extra arguments for agg
    ) -> BootstrappableVerb:
        """Register a function for pandas agg"""

        @self.register(Series, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            if isinstance(func, str) and hasattr(__data, func):
                return getattr(__data, func)(*args, **kwargs)
            return func(__data, *args, **kwargs)

        @self.register(DataFrame, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            return __data.agg(func, 0, *args, **kwargs).to_frame().T

        @self.register(SeriesGroupBy, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            return __data.agg(func, *args, **kwargs)

        @self.register(TibbleGrouped, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            return Tibble(
                __data._datar["grouped"].agg(func, *args, **kwargs),
                copy=False,
            )

        @self.register(TibbleRowwise, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            return __data.agg(func, 1, *args, **kwargs)

        return self

    def _bootstrap_pandas_transform(
        self,
        func: str | Callable,
        context: ContextType,
        extra_contexts: Mapping[str, ContextType],
        pre: Callable,
        post: Callable,
    ) -> BootstrappableVerb:
        """Register a function for pandas agg"""

        @self.register(Series, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            if isinstance(func, str) and hasattr(__data, func):
                out = getattr(__data, func)(*args, **kwargs)
            else:
                out = func(__data, *args, **kwargs)
            if not isinstance(out, Series):
                out = Series(out, index=__data.index)
            return out

        @self.register(DataFrame, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            return __data.transform(func, 0, *args, **kwargs)

        @self.register(SeriesGroupBy, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            out = __data.transform(func, *args, **kwargs)
            grouper = __data.grouper
            return out.groupby(
                grouper,
                sort=__data.sort,
                dropna=__data.dropna,
                observed=__data.observed,
            )

        @self.register(TibbleGrouped, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            return __data._datar["grouped"].transform(func, *args, **kwargs)

        @self.register(TibbleRowwise, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            return __data.transform(func, 1, *args, **kwargs)

        return self

    def _bootstrap_pandas_apply(
        self,
        func: str | Callable,
        context: ContextType,
        extra_contexts: Mapping[str, ContextType],
        pre: Callable,
        post: Callable,
        data_args: str | Set[str] = None,
        signature: Signature = None,
    ) -> BootstrappableVerb:
        """Register a function for pandas apply"""

        if not signature:
            if func is self.fact_func:
                signature = self.signature
            else:
                try:
                    signature = inspect.signature(func)
                except (ValueError, TypeError):
                    signature = self.signature

        data_args = data_args or {list(signature.parameters)[0]}
        if isinstance(data_args, str):
            data_args = {data_args}

        @self.register(object, context, extra_contexts)
        @_with_hooks(pre=pre, post=post)
        def _(__data, *args, **kwargs):
            bound, args_frame = _preprocess_data_args(
                data_args,
                signature,
                __data,
                args,
                kwargs,
            )

            return self.kind_apply(args_frame, bound, data_args, func)

        return self

    def bootstrap(
        self,
        func: str | Callable = None,
        *,
        kind: str = "apply",
        context: ContextType = NO_DEFAULT,
        extra_contexts: Mapping[str, ContextType] = NO_DEFAULT,
        pre: Callable = NO_DEFAULT,
        post: Callable = NO_DEFAULT,
        **kwargs: Any,
    ) -> BootstrappableVerb:
        """Bootstrap a verb for new types, including Series, DataFrame, etc.

        Args:
            func: The function to register. If not provided, use the function
                of the same name in the current module.
                If it is `NO_DEFAULT`, the function registered by
                `func_factory()` or `func_generic()` will be used.
            kind: The kind of the function, can be "apply", "agg", "transform".
            context: The context to use for the function.
            extra_contexts: The extra contexts to use for the function.
            pre: A function to be called before the function.
            post: A function to be called after the function.
            **kwargs: kind-specific arguments.

        Returns:
            The decorator to register the function.
        """
        if func is None:
            return lambda fun: self.bootstrap(
                fun,
                kind=kind,
                context=context,
                extra_contexts=extra_contexts,
                pre=pre,
                post=post,
                **kwargs,
            )

        if func is NO_DEFAULT:
            func = self.fact_func

        if context is NO_DEFAULT:
            context = self.contexts["_"]

        if extra_contexts is NO_DEFAULT:
            extra_contexts = self.extra_contexts["_"]

        if pre is NO_DEFAULT:
            pre = self.fact_pre

        if post is NO_DEFAULT:
            post = self.fact_post

        if kind == "agg":
            return self._bootstrap_pandas_agg(
                func,
                context=context,
                extra_contexts=extra_contexts,
                pre=pre,
                post=post,
                **kwargs,
            )

        if kind == "transform":
            return self._bootstrap_pandas_transform(
                func,
                context=context,
                extra_contexts=extra_contexts,
                pre=pre,
                post=post,
                **kwargs,
            )

        return self._bootstrap_pandas_apply(
            func,
            context=context,
            extra_contexts=extra_contexts,
            pre=pre,
            post=post,
            **kwargs,
        )


def func_generic(
    func: str | Callable = None,
    *,
    context: ContextType = Context.EVAL,
    extra_contexts: Mapping[str, ContextType] = None,
    dep: bool = False,
    ast_fallback: str = "normal_warning",
    name: str = None,
    qualname: str = None,
    doc: str = None,
    module: str = None,
    signature: Signature = None,
    pre: Callable = None,
    post: Callable = None,
) -> BootstrappableVerb:
    """Register a generic function to handle basic data types,
    without bootstrapping.

    Args:
        func: The focal function to apply to the data. If not provided, the
            function will be used as the focal function.
        context: The context to register the function.
        extra_contexts: Extra contexts to register the function.
        dep: Whether the function is dependent. Dependent functions should not
            have the data argument passed in explicitly.
        name: The name of the function.
        qualname: The qualified name of the function.
        doc: The docstring of the function.
        module: The module of the function.
        signature: The signature of the function.
        pre: The pre hook, takes the `__data` and `*args`, `**kwargs`. If it
            returns None, the original `*args` and `**kwargs` will be used.
            Otherwise, it should return a tuple of `__data`, `args` and `kwargs`
        post: The post hook, takes the `out`, `__data` and `*args`, `**kwargs`.
            It should return the modified `out`.

    Returns:
        The decorator to register the function if `func` is not provided.
        Otherwise, the registered function.
    """
    if func is None:
        return lambda fun: func_generic(
            fun,
            context=context,
            extra_contexts=extra_contexts,
            dep=dep,
            ast_fallback=ast_fallback,
            name=name,
            qualname=qualname,
            doc=doc,
            module=module,
            signature=signature,
            pre=pre,
            post=post,
        )

    verb = BootstrappableVerb(
        _with_hooks(func, pre, post) if pre or post else func,
        types=[object],
        context=context,
        extra_contexts=extra_contexts or {},
        name=name,
        qualname=qualname,
        doc=doc,
        module=module,
        signature=signature,
        dep=dep,
        ast_fallback=ast_fallback,
    )
    # for bootstrapping
    verb.fact_pre = pre
    verb.fact_post = post
    verb.fact_func = func
    return verb


def func_factory(
    func: str | Callable = None,
    *,
    kind: str = "apply",
    context: ContextType = Context.EVAL,
    extra_contexts: Mapping[str, ContextType] = None,
    dep: bool = False,
    ast_fallback: str = "normal_warning",
    name: str = None,
    qualname: str = None,
    doc: str = None,
    module: str = None,
    signature: Signature = None,
    pre: Callable = None,
    post: Callable = None,
    **kwargs,  # Other kind-specific arguments.
) -> BootstrappableVerb:
    """A factory to register functions.

    The function applies same function `func` on different data types. For
    example, DataFrame, Series, SeriesGroupBy, etc.

    Args:
        func: The function to apply to different types of data.
            If not provided, this function will return a decorator.
        kind: The kind of the focal function. Can be one of "agg", "transform",
            "apply".
        context: The context to register the function.
        extra_contexts: Extra contexts to register the function.
        dep: Whether the verb is dependent. If True, the data argument will not
            be passed in explicitly.
        ast_fallback: The AST fallback mode. Can be one of `normal`,
            `normal_warning`, `piping`, `piping_warning` and `raise`.
            See also `help(pipda.register_verb)`.
        name: The name of the function, used to overwrite `func`'s name.
        qualname: The qualified name of the function, used to overwrite
            `func`'s qualname.
        module: The module of the function, used to overwrite `func`'s module.
        doc: The docstring of the function, used to overwrite `func`'s doc.
        signature: The signature of the function. Only needed when
            signature is not available for the function (e.g. `np.sqrt`); and
            `extra_contexts` is provided or `kind` is `apply`
        pre: The pre hook, takes the `__data` and `*args`, `**kwargs`. If it
            returns None, the original `*args` and `**kwargs` will be used.
            Otherwise, it should return a tuple of `__data`, `args` and `kwargs`
        post: The post hook, takes the `out`, `__data` and `*args`, `**kwargs`.
            It should return the modified `out`.
        **kwargs: Other kind-specific arguments.

    Returns:
        A function that registers the function if `func` is not provided.
        Otherwise the registered function.
    """
    if func is None:
        return lambda fun: func_factory(
            fun,
            kind=kind,
            context=context,
            extra_contexts=extra_contexts,
            dep=dep,
            ast_fallback=ast_fallback,
            name=name,
            qualname=qualname,
            doc=doc,
            module=module,
            signature=signature,
            pre=pre,
            post=post,
            **kwargs,
        )

    verb = func_generic(
        func,
        context=context,
        extra_contexts=extra_contexts,
        dep=dep,
        ast_fallback=ast_fallback,
        name=name,
        qualname=qualname,
        doc=doc,
        module=module,
        signature=signature,
        pre=pre,
        post=post,
    )

    return verb.bootstrap(func, kind=kind, **kwargs)


def func_dispatched(
    func: Callable,
    context: ContextType = Context.EVAL,
    extra_contexts: Mapping[str, ContextType] = None
) -> Verb:
    """Register an already-dispatched function

    Args:
        func: The dispatched function
        context: The context to register the function.
        extra_contexts: Extra contexts to register the function.

    Returns:
        The registered function.
    """
    verb = register_verb(
        None,
        context=context,
        extra_contexts=extra_contexts,
        name=func.__name__,
        qualname=func.__qualname__,
        doc=func.__doc__,
        module=func.__module__,
        func="none",
    )
    for type_, method in func.registry.items():
        verb.register(type_)(method)
    return verb


def func_bootstrap(
    verb: Verb,
    *,
    func: Callable = None,
    kind: str = "apply",
    context: ContextType = NO_DEFAULT,
    extra_contexts: Mapping[str, ContextType] = NO_DEFAULT,
    pre: Callable = NO_DEFAULT,
    post: Callable = NO_DEFAULT,
    **kwargs: Any,
) -> BootstrappableVerb:
    """Bootstrap a verb"""
    if not isinstance(verb, Verb):
        raise TypeError("`verb` must be a pipda Verb")

    if func is None:
        return lambda fun: func_bootstrap(
            verb,
            func=fun,
            kind=kind,
            context=context,
            extra_contexts=extra_contexts,
            pre=pre,
            post=post,
            **kwargs,
        )

    if not isinstance(verb, BootstrappableVerb):
        verb = BootstrappableVerb.from_verb(verb)

    return verb.bootstrap(
        func,
        kind=kind,
        context=context,
        extra_contexts=extra_contexts,
        pre=pre,
        post=post,
        **kwargs,
    )

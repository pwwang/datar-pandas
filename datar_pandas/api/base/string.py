from datar.apis.base import (
    grep,
    grepl,
    gsub,
    chartr,
    endswith,
    nchar,
    nzchar,
    paste,
    paste0,
    sprintf,
    startswith,
    strsplit,
    strtoi,
    sub,
    substr,
    substring,
    tolower,
    toupper,
    trimws,
)
from datar_numpy.api import string as _  # noqa: F401

from ...factory import func_bootstrap
from ...tibble import Tibble, TibbleGrouped, TibbleRowwise


func_bootstrap(
    tolower,
    func=tolower.dispatch(object, backend="numpy"),
    kind="transform",
)


func_bootstrap(
    toupper,
    func=toupper.dispatch(object, backend="numpy"),
    kind="transform",
)


func_bootstrap(
    grep,
    func=grep.dispatch(object, backend="numpy"),
    exclude={"pattern", "ignore_case", "value", "fixed", "invert"},
    post="transform",
)


func_bootstrap(
    grepl,
    func=grepl.dispatch(object, backend="numpy"),
    exclude={"pattern", "ignore_case", "fixed", "invert"},
    post="transform",
)


func_bootstrap(
    sub,
    func=sub.dispatch(object, backend="numpy"),
    exclude={"pattern", "replacement", "ignore_case", "fixed"},
    post="transform",
)


func_bootstrap(
    gsub,
    func=gsub.dispatch(object, backend="numpy"),
    exclude={"pattern", "replacement", "ignore_case", "fixed"},
)


func_bootstrap(
    chartr,
    func=chartr.dispatch(object, backend="numpy"),
    exclude={"old", "new"},
    post="transform",
)


func_bootstrap(
    endswith,
    func=endswith.dispatch(object, backend="numpy"),
    kind="transform",
)


func_bootstrap(
    nchar,
    func=nchar.dispatch(object, backend="numpy"),
    kind="transform",
)


func_bootstrap(
    nzchar,
    func=nzchar.dispatch(object, backend="numpy"),
    kind="transform",
)


def _paste(frame, sep, collapse):
    out = frame.apply(
        lambda col: col.astype(str).str.cat(sep=sep),
        axis=1,
    )
    if isinstance(frame, TibbleGrouped):
        grouped = frame._datar["grouped"]
        out = out.groupby(
            grouped.grouper,
            sort=grouped.sort,
            observed=grouped.observed,
            dropna=grouped.dropna,
        )
        if isinstance(frame, TibbleRowwise):
            out.is_rowwise = True
            return out
        if collapse is None:
            return out
        return out.apply(lambda x: collapse.join(x))
    return collapse.join(out) if collapse else out


func_bootstrap(paste, exclude={"sep", "collapse"}, func=paste)
func_bootstrap(paste0, exclude="collapse", func=paste0)


@paste.apply_df.register(Tibble)
@paste.apply_df.register(TibbleGrouped)
def _paste_apply_df(data, bound, exclude, func):
    return _paste(
        data,
        sep=bound.kwargs["sep"],
        collapse=bound.kwargs["collapse"],
    )


@paste0.apply_df.register(Tibble)
@paste0.apply_df.register(TibbleGrouped)
def _paste0_apply_df(data, bound, exclude, func):
    return _paste(data, sep="", collapse=bound.kwargs["collapse"])


func_bootstrap(sprintf, func=sprintf)


@sprintf.apply_df.register(Tibble)
@sprintf.apply_df.register(TibbleGrouped)
def _sprintf_apply_df(data, bound, exclude, func):
    out = data.apply(
        lambda col: col.values[0] % tuple(col.values[1:]),
        axis=1,
    )
    if isinstance(data, TibbleGrouped):
        grouped = data._datar["grouped"]
        out = out.groupby(
            grouped.grouper,
            sort=grouped.sort,
            observed=grouped.observed,
            dropna=grouped.dropna,
        )
        if isinstance(data, TibbleRowwise):
            out.is_rowwise = True
    return out


func_bootstrap(
    substr,
    func=substr.dispatch(object, backend="numpy"),
    kind="transform",
)


func_bootstrap(
    substring,
    func=substring.dispatch(object, backend="numpy"),
    kind="transform",
)


func_bootstrap(
    strsplit,
    func=strsplit.dispatch(object, backend="numpy"),
    exclude="fixed",
)


func_bootstrap(
    strtoi,
    func=strtoi.dispatch(object, backend="numpy"),
    kind="transform",
)


func_bootstrap(
    startswith,
    func=startswith.dispatch(object, backend="numpy"),
    kind="transform",
)


func_bootstrap(
    trimws,
    func=trimws.dispatch(object, backend="numpy"),
    kind="transform",
)

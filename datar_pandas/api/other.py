from pipda import register_verb

from ..utils import PandasData


@register_verb(None)
def use_pandas(_data):
    """Wrap the data with PandasData, so that the implementation of this
    backend is used.

    Unlike `DataFrame`, `singledispatch` can simply distinguish the data type
    and dispatch the right implementation. But in some cases, for example,
    `enframe([1, 2, 3])`, we don't know which implementation to dispatch,
    since the data type registered is `object`.

    To solve this, we can wrap the data with `PandasData`, so that the
    implementation of this backend is used.

    Examples:
        >>> # Could use other backends, this backend is not guaranteed
        >>> c[1:3] >> enframe()
        >>> # Force to use this backend
        >>> c[1:3] >> use_pandas() >> enframe()

    Args:
        _data: The data

    Returns:
        The data wrapped with PandasData
    """
    return PandasData(_data)

import pytest  # noqa: F401

from pipda import evaluate_expr
from datar import f
# Load hook specs so that datar_pandas can be imported
from datar.core import plugin as _  # noqa: F401
from datar_pandas.tibble import Tibble
from datar_pandas.contexts import ContextEval


def test_context_eval():
    context = ContextEval()

    df = Tibble({'x': [1, 2, 3], 'y': [4, 5, 6]})
    df._datar["used_refs"] = set()

    evaluate_expr(f.x, df, context)
    assert 'x' in df._datar["used_refs"]

    evaluate_expr(f.x.name, df, context)
    assert df._datar["used_refs"] == {'x'}

    df._datar["used_refs"] = set()
    evaluate_expr(f.x + f[f.y], df, context)
    assert df._datar["used_refs"] == {'x', 'y'}

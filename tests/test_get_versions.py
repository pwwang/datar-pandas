from datar import get_versions


def test_get_versions():
    out = get_versions(prnt=False)
    assert "datar-pandas" in out
    assert "pandas" in out

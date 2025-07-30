import pytest  # noqa: F401

from datar import f
from datar.core.names import NameNonUniqueError
from datar.base import mean, which, rep, factor
from datar.dplyr import reframe, group_by, cur_group_id
from datar.tibble import tibble
from datar_pandas.tibble import TibbleGrouped

from ..conftest import assert_iterable_equal


def test_allows_summaries():
    df = tibble(g=[1, 1, 1, 2, 2], x=[1, 2, 3, 4, 5])
    result = reframe(df, x=mean(f.x))
    expected = tibble(x=3.0)
    assert result.equals(expected)

    gdf = group_by(df, f.g)
    result = reframe(gdf, x=mean(f.x))
    assert_iterable_equal(result.g, [1, 2])
    assert_iterable_equal(result.x, [2.0, 4.5])


def test_allows_size_0_results():
    df = tibble(g=[1, 1, 1, 2, 2], x=[1, 2, 3, 4, 5])
    gdf = group_by(df, f.g)

    result = reframe(df, x=which(f.x > 5))
    assert result.shape == (0, 1)

    gres = reframe(gdf, x=which(f.x > 5))
    assert gres.shape == (0, 2)


def test_allows_size_greater_than_1_results():
    df = tibble(g=[1, 1, 1, 2, 2], x=[1, 2, 3, 4, 5])
    gdf = group_by(df, f.g)

    result = reframe(df, x=which(f.x > 2))
    expected = tibble(x=[2, 3, 4])
    assert result.equals(expected)

    gres = reframe(gdf, x=which(f.x > 2))
    assert_iterable_equal(gres.g, [1, 2, 2])
    assert_iterable_equal(gres.x, [2, 0, 1])


def test_allows_recycling_across_columns():
    df = tibble(g=[1, 1, 2, 2, 2], x=[1, 2, 3, 4, 5])

    result = reframe(df, a=[1, 2], b=1, c=[2, 3])
    expected = tibble(a=[1, 2], b=[1, 1], c=[2, 3])
    assert result.equals(expected)

    gdf = group_by(df, f.g)
    result = reframe(gdf, a=[1, 2], b=1, c=[2, 3])
    assert_iterable_equal(result.g, [1, 1, 2, 2])
    assert_iterable_equal(result.a, [1, 2, 1, 2])
    assert_iterable_equal(result.b, [1, 1, 1, 1])
    assert_iterable_equal(result.c, [2, 3, 2, 3])


def test_can_recycle_to_size_0():
    df = tibble(g=[1, 2], x=[1, 2])
    gdf = group_by(df, f.g)

    result = reframe(df, y=mean(f.x), z=which(f.x > 3))
    assert result.shape == (0, 2)

    gres = reframe(gdf, y=mean(f.x), z=which(f.x > 1))
    assert_iterable_equal(gres.g, [2])
    assert_iterable_equal(gres.y, [2])
    assert_iterable_equal(gres.z, [0])


def test_throws_recycling_errors():
    df = tibble(g=[1, 2], x=[1, 2])
    gdf = group_by(df, f.g)

    with pytest.raises(ValueError, match=r"`y` must be size \[1 2\], not 3"):
        reframe(df, x=[1, 2], y=[3, 4, 5])

    with pytest.raises(ValueError, match=r"Cannot recycle `y` with size 3 to 2"):
        reframe(gdf, x=[1, 2], y=[3, 4, 5])


def test_can_return_more_rows_than_original():
    df = tibble(x=[1, 2])

    result = reframe(df, x=rep(f.x, f.x))
    expected = tibble(x=[1, 2, 2])
    assert result.equals(expected)


def test_doesnt_message_about_regrouping_when_multiple_group_columns():
    df = tibble(a=[1, 1, 2, 2, 2], b=[1, 2, 1, 1, 2], x=range(1, 6))
    gdf = group_by(df, f.a, f.b)

    result = reframe(gdf, x=mean(f.x))
    assert_iterable_equal(result.x, [1., 2., 3.5, 5.0])


def test_doesnt_message_about_regrouping_when_multiple_rows_per_group():
    df = tibble(g=[1, 1, 2, 2, 2], x=range(1, 6))
    gdf = group_by(df, f.g)

    result = reframe(gdf, x=rep(f.x, f.x))
    assert_iterable_equal(result.g, [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    assert_iterable_equal(result.x, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])


# TODO: Not yet supported
# def test_allows_sequential_assignments():
#     df = tibble(g=[1, 2], x=[1, 2])

#     result = reframe(df, y=3, z=mean(f.x) + f.y)
#     expected = tibble(y=3, z=4.5)
#     assert result.equals(expected)

#     gdf = group_by(df, f.g)
#     result = reframe(gdf, y=3, z=mean(f.x) + f.y)
#     assert_iterable_equal(result.g, [1, 2])
#     assert_iterable_equal(result.y, [3, 3])
#     assert_iterable_equal(result.z, [4.5, 5.0])


def test_allows_overwriting_existing_columns():
    df = tibble(g=["a", "b"], x=[1, 2])

    result = reframe(df, x=3, z=f.x)
    expected = tibble(x=3, z=3)
    assert result.equals(expected)

    gdf = group_by(df, f.g)
    result = reframe(gdf, x=cur_group_id(), z=f.x)
    assert_iterable_equal(result.g, ["a", "b"])
    assert_iterable_equal(result.x, [0, 1])
    assert_iterable_equal(result.z, [0, 1])


def test_with_group_by_sorts_keys():
    df = tibble(g=[2, 1, 2, 0], x=[4, 2, 8, 5])
    df = group_by(df, f.g, _sort=True)

    out = reframe(df, x=mean(f.x))

    assert_iterable_equal(out.g, [0, 1, 2])
    assert_iterable_equal(out.x, [5.0, 2.0, 6.0])


def test_with_group_by_respects_drop():
    df = tibble(g=factor(["c", "a", "c"], levels=["a", "b", "c"]), x=[1, 4, 2])
    gdf = group_by(df, f.g, _drop=False, _sort=True)

    out = reframe(gdf, x=mean(f.x))

    assert_iterable_equal(out.g, ["a", "b", "c"])
    assert_iterable_equal(out.x, [4.0, float("nan"), 1.5])


def test_with_group_by_always_returns_ungrouped_tibble():
    df = tibble(a=[1, 1, 2, 2, 2], b=[1, 2, 1, 1, 2], x=range(1, 6))
    gdf = group_by(df, f.a, f.b)

    out = reframe(gdf, x=mean(f.x))
    assert_iterable_equal(out.a, [1, 1, 2, 2])
    assert_iterable_equal(out.b, [1, 2, 1, 2])
    assert_iterable_equal(out.x, [1.0, 2.0, 3.5, 5.0])
    assert not isinstance(out, TibbleGrouped)


def test_with_rowwise():
    df = tibble(x=[1, 2, 3, 4, 5, 6])
    rdf = df.rowwise()

    out = reframe(rdf, x=f.x)
    assert_iterable_equal(out.x, [1, 2, 3, 4, 5, 6])


def test_with_rowwise_respects_rowwise_group_columns():
    df = tibble(g=[1, 1, 2], x=[1, 2, 3])
    rdf = df.rowwise("g")

    out = reframe(rdf, x=f.x)
    assert_iterable_equal(out.g, [1, 1, 2])
    assert_iterable_equal(out.x, [1, 2, 3])

    # Always returns an ungrouped tibble
    assert not isinstance(out, TibbleGrouped)


def test_errors_with_duplicated_colnames():
    # Duplicate column names
    x = 1
    df = tibble(x, x, _name_repair="minimal")
    with pytest.raises(NameNonUniqueError):
        df >> reframe(f.x)

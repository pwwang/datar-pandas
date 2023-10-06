import pytest
from pipda import register_func
from datar_pandas.tibble import TibbleRowwise

from datar import f
from datar.base import complete_cases, is_character, c, mean, sum, NA
from datar.dplyr import (
    filter_,
    pick,
    summarise,
    mutate,
    starts_with,
    where,
    group_by,
    group_vars,
    pull,
    everything,
    rowwise,
    all_of,
    arrange,
)
from datar.tibble import drop_index, tibble
from pandas.testing import assert_frame_equal

from ..conftest import assert_iterable_equal


# pick + mutate ---------------------------------------------------------------
def test_pick_columns_from_data():
    df = tibble(x1=1, y=2, x2=3, z=4)
    expect = df[["z", "x1", "x2"]]

    out = df >> mutate(sel=pick(f.z, starts_with("x")))
    assert_frame_equal(out['sel'], expect)


def test_use_namespaced_call():
    df = tibble(x=1, y="y")
    out = df >> mutate(z=pick(where(is_character)))
    assert_frame_equal(out['z'], tibble(y="y"))


def test_returns_separate_dataframes_for_each_group():
    @register_func()
    def fn(df):
        return df >> mutate(res=f.x + mean(f.z)) >> pull(f.res)

    df = tibble(g=[1, 1, 2, 2, 2], x=c[1:6], z=c[11:16])
    gdf = group_by(df, f.g)

    expect = mutate(gdf, res=f.x + mean(f.z))

    out = gdf >> mutate(res=fn(pick(f.x, f.z)))
    assert_frame_equal(out, expect)
    assert_iterable_equal(group_vars(out), ["g"])


def test_wont_select_grouping_columns():
    df = tibble(g=1, x=2)

    gdf = group_by(df, f.g)
    out = gdf >> mutate(y=pick(everything()))
    assert_iterable_equal(group_vars(out["y"]), ["g"])

    rdf = rowwise(df, f.g)
    out = rdf >> mutate(y=pick(everything()))
    assert isinstance(out["y"], TibbleRowwise)


def test_cant_select_grouping_columns():
    df = tibble(g=1, x=2)
    gdf = group_by(df, f.g)

    with pytest.raises(ValueError):
        gdf >> mutate(y=pick(f.g))


def test_with_all_of():
    df = tibble(g=1, x=2, y=3)
    y = ["x"]
    out = df >> mutate(z=pick(all_of(y)))
    assert_frame_equal(out['z'], tibble(x=2))


def test_must_supply_one_selector():
    df = tibble(x=[2, 3, 4])

    with pytest.raises(ValueError):
        df >> mutate(y=pick())


def test_evaluation_on_current_data():
    df = tibble(g=[1, 2, 3], x=[1, 2, 3])
    gdf = group_by(df, f.g)

    with pytest.raises(KeyError):
        gdf >> mutate(x=None, y=pick(f.x))

    out = gdf >> mutate(y=f.x + 1, z=pick(f.x, f.y))
    assert_frame_equal(out[['x', 'y']], out['z'])


# pick + summarise ------------------------------------------------------------
def test_uses_current_columns():
    df = tibble(x=c[1:6], y=c[6:11])

    out = df >> summarise(x=sum(f.x), z=pick(f.x))
    assert_iterable_equal(out['x'], [15])


def test_can_pick_new_columns():
    df = tibble(x=c[1:6])

    out = df >> mutate(y=f.x + 1, z=pick(f.y))
    assert_frame_equal(out['z'], out[['y']])


# pick + arrange --------------------------------------------------------------
def test_can_arrange_with_pick():
    df = tibble(x=[2, 2, 1], y=[3, 1, 3])

    out = df >> arrange(pick(f.x, f.y))
    assert_iterable_equal(out['x'], [1, 2, 2])
    assert_iterable_equal(out['y'], [3, 1, 3])


# pick + filter ---------------------------------------------------------------
def test_can_pick_inside_filter():
    df = tibble(x=[1, 2, NA, 3], y=[2, NA, 5, 3])
    out = df >> filter_(complete_cases(pick(f.x, f.y))) >> drop_index()
    assert_frame_equal(out, tibble(x=[1., 3.], y=[2., 3.]))


# pick + group_by -------------------------------------------------------------
def test_can_pick_inside_group_by():
    df = tibble(a=c[1:4], b=c[2:5], c=c[3:6])
    gdf = df >> group_by(pick(f.a, f.c))
    assert_iterable_equal(group_vars(gdf), ["a", "c"])

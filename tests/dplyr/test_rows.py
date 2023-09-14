# tests grabbed from:
# https://github1s.com/tidyverse/dplyr/blob/master/tests/testthat/test-rows.R
import pytest

from datar.all import (
    tibble,
    seq,
    letters,
    NA,
    c,
    rows_insert,
    rows_append,
    rows_delete,
    rows_patch,
    rows_update,
    rows_upsert,
)
from datar_pandas.pandas import assert_frame_equal


@pytest.fixture
def data():
    return tibble(a=seq(1, 3), b=c(letters[[0, 1]], NA), c=[0.5, 1.5, 2.5])


# rows_insert
def test_rows_insert(data):
    out = rows_insert(data, tibble(a=4, b="z"), by="a")
    exp = tibble(a=seq(1, 4), b=c("a", "b", NA, "z"), c=c(0.5, 1.5, 2.5, NA))
    assert_frame_equal(out, exp)


def test_rows_insert_disallows_matched_keys_by_default():
    x = tibble(a=1, b=2)
    y = tibble(a=1, b=3)
    with pytest.raises(ValueError, match="insert duplicate"):
        rows_insert(x, y, by="a")

    y = tibble(a=[1, 1, 1], b=[3, 4, 5])
    with pytest.raises(ValueError, match="insert duplicate"):
        rows_insert(x, y, by="a")


def test_rows_insert_allows_matched_keys_by_conflict_arg():
    x = tibble(a=1, b=2)
    y = tibble(a=1, b=3)
    out = rows_insert(x, y, by="a", conflict="ignore")
    assert_frame_equal(out, x)

    y = tibble(a=[1, 2, 1], b=[3, 4, 5])
    out = rows_insert(x, y, by="a", conflict="ignore")
    exp = tibble(a=[1, 2], b=[2, 4])
    assert_frame_equal(out, exp)


def test_rows_insert_allows_x_dup_keys():
    x = tibble(a=[1, 1], b=[2, 3])
    y = tibble(a=2, b=4)

    out = rows_insert(x, y, by="a")
    exp = tibble(a=[1, 1, 2], b=[2, 3, 4])
    assert_frame_equal(out, exp)


def test_rows_insert_allows_y_dup_keys():
    x = tibble(a=2, b=4)
    y = tibble(a=[1, 1], b=[2, 3])
    out = rows_insert(x, y, by="a")
    exp = tibble(a=[2, 1, 1], b=[4, 2, 3])
    assert_frame_equal(out, exp)


def test_rows_insert_checks_x_y_have_by():
    x = tibble(a=1, b=2)
    y = tibble(a=1)

    with pytest.raises(ValueError, match="All `by` columns must exist"):
        rows_insert(x, y, by="c")

    with pytest.raises(ValueError, match="All `by` columns must exist"):
        rows_insert(x, y, by=["a", "b"])


def test_rows_insert_conflict_arg():
    x = tibble(a=1)
    y = tibble(b=2)

    with pytest.raises(ValueError, match="conflict"):
        rows_insert(x, y, by="a", conflict="foo")

    with pytest.raises(ValueError, match="conflict"):
        rows_insert(x, y, by="a", conflict=1)


# rows_append
def test_rows_append(data):
    x = tibble(a=1, b=2)
    y = tibble(a=1, b=3)
    out = rows_append(x, y)
    exp = tibble(a=[1, 1], b=[2, 3])
    assert_frame_equal(out, exp)

    y = tibble(a=[1, 2, 1], b=[3, 4, 5])
    out = rows_append(x, y)
    exp = tibble(a=[1, 1, 2, 1], b=[2, 3, 4, 5])


def test_rows_append_requires_y_cols_in_x():
    x = tibble(a=1, b=2)
    y = tibble(a=1, b=2, c=3)
    with pytest.raises(ValueError):
        rows_append(x, y)


def test_rows_append_doesnt_require_x_cols_in_y():
    x = tibble(a=1, b=2, c=3)
    y = tibble(a=1, b=2)
    out = rows_append(x, y)
    assert_frame_equal(out, tibble(a=[1, 1], b=[2, 2], c=[3, NA]))


# rows_update
def test_rows_update(data):
    out = rows_update(data, tibble(a=[2, 3], b="z"), by="a")
    exp = tibble(a=seq(1, 3), b=c("a", "z", "z"), c=data.c)
    assert_frame_equal(out, exp)

    out = rows_update(data, tibble(b="z", a=[2, 3]), by="a")
    exp = tibble(a=seq(1, 3), b=c("a", "z", "z"), c=data.c)
    assert_frame_equal(out, exp)


def test_rows_update_requires_y_keys_in_x_by_default():
    x = tibble(a=1, b=2)
    y = tibble(a=[2, 1, 3], b=[1, 1, 1])
    with pytest.raises(ValueError):
        rows_update(x, y, by="a")


def test_rows_update_allows_y_keys_not_in_x_by_unmatched_arg():
    x = tibble(a=1, b=2)
    y = tibble(a=[2, 1, 3], b=[1, 1, 1])
    out = rows_update(x, y, by="a", unmatched="ignore")
    assert_frame_equal(out, tibble(a=1, b=1))


def test_rows_update_allow_x_dup_keys():
    x = tibble(a=[1, 2, 1, 3], b=[2, 3, 4, 5], c=letters[:4])
    y = tibble(a=[1, 3], b=[99, 88])
    out = rows_update(x, y, by="a")
    exp = tibble(a=[1, 2, 1, 3], b=[99, 3, 99, 88], c=letters[:4])
    assert_frame_equal(out, exp)


def test_rows_update_disallow_y_dup_keys():
    x = tibble(a=2, b=4)
    y = tibble(a=[2, 2], b=[2, 3])

    with pytest.raises(ValueError, match="must be unique"):
        rows_update(x, y, by="a")


def test_rows_update_unmatched_arg():
    x = tibble(a=1)
    y = tibble(a=1)

    with pytest.raises(ValueError, match="unmatched"):
        rows_update(x, y, by="a", unmatched="foo")

    with pytest.raises(ValueError, match="unmatched"):
        rows_update(x, y, by="a", unmatched=1)


# rows_patch
def test_rows_patch(data):
    out = rows_patch(data, tibble(a=[2, 3], b="z"), by="a")
    exp = tibble(a=seq(1, 3), b=c("a", "b", "z"), c=data.c)
    assert_frame_equal(out, exp)

    out = rows_patch(data, tibble(b="z", a=[2, 3]), by="a")
    exp = tibble(a=seq(1, 3), b=c("a", "b", "z"), c=data.c)
    assert_frame_equal(out, exp)


def test_rows_patch_requires_y_keys_in_x_by_default():
    x = tibble(a=1, b=2)
    y = tibble(a=[2, 1, 3], b=[1, 1, 1])
    with pytest.raises(ValueError):
        rows_patch(x, y, by="a")


def test_rows_patch_allows_y_keys_not_in_x_by_unmatched_arg():
    x = tibble(a=1, b=NA)
    y = tibble(a=[2, 1, 3], b=[1, 1, 1])
    out = rows_patch(x, y, by="a", unmatched="ignore")
    assert_frame_equal(out, tibble(a=1, b=1.))


def test_rows_patch_allow_x_dup_keys():
    x = tibble(a=[1, 2, 1, 3], b=[NA, 3, 4, NA], c=letters[:4])
    y = tibble(a=[1, 3], b=[99, 88])
    out = rows_patch(x, y, by="a")
    exp = tibble(a=[1, 2, 1, 3], b=[99., 3, 4, 88], c=letters[:4])
    assert_frame_equal(out, exp)


def test_rows_patch_disallow_y_dup_keys():
    x = tibble(a=2, b=4)
    y = tibble(a=[2, 2], b=[2, 3])

    with pytest.raises(ValueError, match="must be unique"):
        rows_patch(x, y, by="a")


def test_rows_patch_unmatched_arg():
    x = tibble(a=1)
    y = tibble(a=1)

    with pytest.raises(ValueError, match="unmatched"):
        rows_patch(x, y, by="a", unmatched="foo")

    with pytest.raises(ValueError, match="unmatched"):
        rows_patch(x, y, by="a", unmatched=1)


# rows_upsert
def test_rows_upsert(data):
    out = rows_upsert(data, tibble(a=[2, 3, 4], b="z"), by="a")
    exp = tibble(a=seq(1, 4), b=c("a", "z", "z", "z"), c=c(data.c.values, NA))
    assert_frame_equal(out, exp)


def test_rows_upsert_allow_x_dup_keys():
    x = tibble(a=[1, 2, 1, 3], b=[NA, 3, 4, NA], c=letters[:4])
    y = tibble(a=[1, 3, 4], b=[99, 88, 100])
    out = rows_upsert(x, y, by="a")
    exp = tibble(
        a=[1, 2, 1, 3, 4],
        b=[99., 3, 99, 88, 100],
        c=c(letters[:4], NA),
    )
    assert_frame_equal(out, exp)


def test_rows_upsert_disallow_y_dup_keys():
    x = tibble(a=2, b=4)
    y = tibble(a=[2, 2], b=[2, 3])

    with pytest.raises(ValueError, match="must be unique"):
        rows_upsert(x, y, by="a")


# rows_delete
def test_rows_delete(data):
    out = data >> rows_delete(tibble(a=[2, 3]), by="a")
    assert_frame_equal(out, data.iloc[[0], :])

    out = data >> rows_delete(tibble(a=[2, 3], b="b"), by="a")
    assert_frame_equal(out, data.iloc[[0], :])

    with pytest.raises(ValueError, match="delete missing"):
        data >> rows_delete(tibble(a=[2, 3], b="b"), by=["a", "b"])


def test_rows_delete_ignores_extra_y_cols(caplog):
    x = tibble(a=1)
    y = tibble(a=1, b=2)
    out = rows_delete(x, y)
    assert "Ignoring extra columns" in caplog.text
    assert out.shape[0] == 0

    out = rows_delete(x, y, by="a")
    assert "Ignoring extra columns" in caplog.text
    assert out.shape[0] == 0


def test_rows_delete_requires_y_keys_in_x_by_default():
    x = tibble(a=1, b=2)
    y = tibble(a=[2, 1, 3], b=[1, 1, 1])
    with pytest.raises(ValueError):
        rows_delete(x, y, by="a")


def test_rows_delete_allows_y_keys_not_in_x_by_unmatched_arg():
    x = tibble(a=1, b=2)
    y = tibble(a=[2, 1, 3], b=[1, 1, 1])
    out = rows_delete(x, y, by="a", unmatched="ignore")
    assert out.shape[0] == 0


def test_rows_delete_allow_x_dup_keys():
    x = tibble(a=[1, 2, 1, 3], b=[NA, 3, 4, NA], c=letters[:4])
    y = tibble(a=[1, 3])
    out = rows_delete(x, y, by="a")
    exp = x.iloc[[1], :]
    assert_frame_equal(out, exp)


def test_rows_delete_allow_y_dup_keys():
    x = tibble(a=[1, 2, 3], b=[4, 5, 6])
    y = tibble(a=[1, 1])
    out = rows_delete(x, y, by="a")
    exp = x.iloc[[1, 2], :]
    assert_frame_equal(out, exp)


def test_rows_errors(data):
    # by must be string or strings
    with pytest.raises(ValueError, match="must be a string"):
        data >> rows_delete(tibble(a=[2, 3]), by=1)

    # Insert
    with pytest.raises(ValueError):
        data >> rows_insert(tibble(a=3, b="z"))

    # with pytest.raises(ValueError):
    #     (
    #         data.iloc[
    #             [0, 0],
    #         ]
    #         >> rows_insert(tibble(a=3))
    #     )

    with pytest.raises(ValueError):
        data >> rows_insert(tibble(a=4, b="z"), by="e")

    with pytest.raises(ValueError):
        data >> rows_insert(tibble(d=4))

    # Update
    with pytest.raises(ValueError):
        rows_update(data, tibble(a=[2, 3], b="z"), by=["a", "b"])

    # Variants: patch
    with pytest.raises(ValueError):
        rows_patch(data, tibble(a=[2, 3], b="z"), by=["a", "b"])

    # Delete and truncate
    with pytest.raises(ValueError):
        data >> rows_delete(tibble(a=[2, 3, 4]))

    with pytest.raises(ValueError):
        data >> rows_delete(tibble(a=[2, 3], b="b"), by=["a", "b"])

    # works
    # rows_delete(data, tibble(a = [2,3]))
    # rows_delete(data, tibble(a = [2,3], b = "b"))

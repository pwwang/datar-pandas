import pytest  # noqa: F401

import numpy as np
from datar.tibble import tibble
from datar.base import (
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
    letters,
    LETTERS,
)
from datar_pandas.pandas import Series, get_obj
from ..conftest import assert_, assert_iterable_equal


def test_grep():
    out = grep("[a-z]", Series(["a"]))
    assert_iterable_equal(out, [0])
    out = grep("[a-z]", Series(["a"]), invert=True)
    assert_iterable_equal(out, [])

    out = grep("[a-z]", Series(letters))
    assert_iterable_equal(out, list(range(26)))

    txt = Series(["arm", "foot", "lefroo", "bafoobar"])
    out = grep("foo", txt)
    assert_iterable_equal(out, [1, 3])

    out = grep("foo", txt, value=True)
    assert_iterable_equal(out, ["foot", "bafoobar"])

    # fixed
    out = grep(".", Series(["a"]))
    assert_iterable_equal(out, [0])
    out = grep(".", Series(["a"]), fixed=True)
    assert_iterable_equal(out, [])

    sgb = Series(["a", "b", "c"]).groupby([1, 1, 2])
    assert_iterable_equal(list(grep("a", sgb).obj)[0], [0])
    assert_iterable_equal(list(grep("a", sgb).obj)[1], [])


def test_grepl(caplog):
    txt = Series(["arm", "foot", "lefroo", "bafoobar"])
    out = grepl(["foo"], txt)
    assert_iterable_equal(out, [False, True, False, True])

    assert_iterable_equal(grepl("a", Series(["a"])), [True])

    # np.nan
    out = grepl("a", Series(["a", np.nan]))
    assert_iterable_equal(out, [True, False])


def test_sub():
    txt = Series(["arm", "foot", "lefroo", "bafoobarfoo"])
    out = sub("foo", "bar", txt)
    assert_iterable_equal(out, ["arm", "bart", "lefroo", "babarbarfoo"])

    # fixed
    out = sub(".", "a", Series(["b"]))
    assert_iterable_equal(out, ["a"])
    out = sub(".", "a", Series(["b"]), fixed=True)
    assert_iterable_equal(out, ["b"])

    out = sub(".", "a", Series(["b"]).groupby([1]))
    assert_iterable_equal(list(get_obj(out))[0], ["a"])


def test_gsub():
    txt = Series(["arm", "foot", "lefroo", "bafoobarfoo"])
    out = gsub("foo", "bar", txt)
    assert_iterable_equal(out, ["arm", "bart", "lefroo", "babarbarbar"])


def test_nchar():
    s = Series(["abc王"])
    assert_iterable_equal(nchar(s), [5])
    assert_iterable_equal(nchar(s, type_="bytes"), [6])
    assert_iterable_equal(nchar(s, type_="width"), [5])

    out = nchar(Series([b"abc", np.nan, "a"]), type_="bytes")
    assert_iterable_equal(out, [3, 2, 1])


def test_nzchar():
    assert_iterable_equal(nzchar(Series([""])), [False])
    assert_iterable_equal(nzchar(Series(["1"])), [True])
    assert_iterable_equal(nzchar(Series(["1", np.nan])), [True, False])
    assert_iterable_equal(
        nzchar(Series(["1", np.nan]), keep_na=True), [True, np.nan]
    )


def test_paste():

    df = tibble(a=letters[:3], b=LETTERS[:3]).rowwise()
    out = paste0(df)
    assert out.is_rowwise
    assert_iterable_equal(get_obj(out), ["aA", "bB", "cC"])

    df = tibble(x=[1, 2, 3], y=[4, 5, 5])
    out = paste(df)
    assert_iterable_equal(out, ["1 4", "2 5", "3 5"])

    gf = df.group_by("y")
    out = paste0(gf).obj
    assert_iterable_equal(out, ["14", "25", "35"])

    out = paste0(gf, collapse="|")
    assert_iterable_equal(out, ["14", "25|35"])


def test_sprintf():
    df = tibble(x=["%d", "%.2f"], y=[1.1, 2.345]).group_by("x")
    assert_iterable_equal(sprintf(df.x, df.y).obj, ["1", "2.35"])

    rf = tibble(x=["%d", "%.2f"], y=[1.1, 2.345]).rowwise()
    out = sprintf(rf.x, rf.y)
    assert_iterable_equal(get_obj(out), ["1", "2.35"])
    assert_(out.is_rowwise)


def test_substr():
    assert_iterable_equal(substr(Series(["abcd"]), 1, 3), ["bc"])
    assert_iterable_equal(substring(Series(["abcd"]), 1), ["bcd"])
    # assert_iterable_equal(substr(Series([np.nan]), 1, 3), [np.nan])
    # assert_iterable_equal(
    #     substr(Series([np.nan]), [1, 2], 3), [np.nan, np.nan]
    # )
    assert_iterable_equal(substr(Series(["abce", "efgh"]), 1, 3), ["bc", "fg"])
    assert_iterable_equal(
        substring(Series(["abce", "efgh"]), 1), ["bce", "fgh"]
    )
    # assert_iterable_equal(
    #     substr(Series(["abce", "efgh", np.nan]), 1, 3), ["bc", "fg", np.nan]
    # )
    assert_iterable_equal(substr(Series(["abcd"]), 1, 3), ["bc"])
    assert_iterable_equal(substring(Series(["abcd"]), 1), ["bcd"])
    # assert_iterable_equal(substring(Series([np.nan]), 1), [np.nan])

    tg = tibble(x=["abcd", "efgh"], g=[1, 2]).group_by("g")
    assert_iterable_equal(substr(tg.x, 1, 3).obj, ["bc", "fg"])
    assert_iterable_equal(substring(tg.x, 1).obj, ["bcd", "fgh"])

    tr = tibble(x=["abcd", "efgh"], g=[1, 2]).rowwise()
    out = substr(tr.x, 1, 3)
    assert_iterable_equal(get_obj(out), ["bc", "fg"])
    assert_(out.is_rowwise)

    out = substring(tr.x, 1)
    assert_iterable_equal(get_obj(out), ["bcd", "fgh"])
    assert_(out.is_rowwise)


def test_strsplit():
    x = Series(["a.b.c"])
    assert_iterable_equal(strsplit(x, ".", fixed=True)[0], ["a", "b", "c"])
    out = strsplit(x, ".", fixed=False)
    assert_iterable_equal(out[0], [""] * 6)


def test_starts_endswith():
    assert_iterable_equal(startswith(Series(["abc"]), "a"), [True])
    assert_iterable_equal(endswith(Series(["abc"]), "c"), [True])
    assert_iterable_equal(
        startswith(Series(["abc", "def"]), "a"), [True, False]
    )
    assert_iterable_equal(endswith(Series(["abc", "def"]), "c"), [True, False])


def test_strtoi():
    assert_iterable_equal(strtoi(Series(["8"])), [8])
    assert_iterable_equal(strtoi(Series(["0b111"])), [7])
    assert_iterable_equal(strtoi(Series(["0xf"])), [15])
    assert_iterable_equal(strtoi(Series(["8", "0b111", "0xf"])), [8, 7, 15])


def test_chartr():
    assert_iterable_equal(chartr(Series(["a"]), "A", "abc"), ["Abc"])
    assert_iterable_equal(
        chartr("a", "A", Series(["abc", "ade"])), ["Abc", "Ade"]
    )

    tg = tibble(x=["abc", "ade"], g=[1, 2]).group_by("g")
    assert_iterable_equal(chartr("a", "A", tg.x).obj, ["Abc", "Ade"])

    tr = tibble(x=["abc", "ade"], g=[1, 2]).rowwise()
    out = chartr("a", "A", tr.x)
    assert_iterable_equal(get_obj(out), ["Abc", "Ade"])
    assert_(out.is_rowwise)


def test_transform_case():
    assert_iterable_equal(tolower(Series(["aBc"])), ["abc"])
    assert_iterable_equal(toupper(Series(["aBc"])), ["ABC"])
    assert_iterable_equal(tolower(Series(["aBc", "DeF"])), ["abc", "def"])
    assert_iterable_equal(toupper(Series(["aBc", "DeF"])), ["ABC", "DEF"])
    assert_iterable_equal(
        tolower(Series(["aBc", "DeF"]).groupby([1, 2])).obj,
        ["abc", "def"],
    )
    assert_iterable_equal(
        toupper(Series(["aBc", "DeF"]).groupby([1, 2])).obj,
        ["ABC", "DEF"],
    )


def test_trimws():
    assert_iterable_equal(trimws(Series([" a "])), ["a"])
    assert_iterable_equal(trimws(Series([" a ", " b "]), "both"), ["a", "b"])
    assert_iterable_equal(trimws(Series([" a ", " b "]), "left"), ["a ", "b "])
    assert_iterable_equal(
        trimws(Series([" a ", " b "]), "right"), [" a", " b"]
    )
    assert_iterable_equal(
        trimws(Series([" a ", " b "]).groupby([1, 2]), "right").obj,
        [" a", " b"],
    )

import pytest
from a_foobar import foobar_word, foobar


# ---------------------------------------------------------------------------
# foobar_word — all 16 distinct word patterns + fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        # plain number (no divisor matches)
        (1, "1"),
        (2, "2"),
        (13, "13"),
        # single-divisor words
        (3, "Foo"),
        (9, "Foo"),
        (5, "Bar"),
        (25, "Bar"),
        (7, "Bop"),
        (49, "Bop"),
        (11, "Bap"),
        (121, "Bap"),
        # two-divisor combos
        (15, "FooBar"),  # 3 × 5
        (21, "FooBop"),  # 3 × 7
        (33, "FooBap"),  # 3 × 11
        (35, "BarBop"),  # 5 × 7
        (55, "BarBap"),  # 5 × 11
        (77, "BopBap"),  # 7 × 11
        # three-divisor combos
        (105, "FooBarBop"),  # 3 × 5 × 7
        (165, "FooBarBap"),  # 3 × 5 × 11
        (231, "FooBopBap"),  # 3 × 7 × 11
        (385, "BarBopBap"),  # 5 × 7 × 11
        # all four divisors
        (1155, "FooBarBopBap"),  # 3 × 5 × 7 × 11
        # zero is divisible by everything
        (0, "FooBarBopBap"),
    ],
)
def test_foobar_word_patterns(n, expected):
    assert foobar_word(n) == expected


def test_foobar_word_returns_str_for_plain_number():
    result = foobar_word(4)
    assert isinstance(result, str)
    assert result == "4"


def test_foobar_word_negative_plain():
    # Python modulo: -1 % 3 == 2, so no match → returns the number as string
    assert foobar_word(-1) == "-1"


def test_foobar_word_negative_divisible():
    # -3 % 3 == 0 in Python
    assert foobar_word(-3) == "Foo"
    assert foobar_word(-35) == "BarBop"


# ---------------------------------------------------------------------------
# foobar — output via capsys
# ---------------------------------------------------------------------------


def test_foobar_three_lines(capsys):
    foobar(1, 4)
    assert capsys.readouterr().out == "1\n2\nFoo\n"


def test_foobar_empty_range_prints_nothing(capsys):
    foobar(5, 5)
    assert capsys.readouterr().out == ""


def test_foobar_reversed_range_prints_nothing(capsys):
    foobar(10, 3)
    assert capsys.readouterr().out == ""


def test_foobar_single_element(capsys):
    foobar(3, 4)
    assert capsys.readouterr().out == "Foo\n"


def test_foobar_known_sequence(capsys):
    # Cross-check a stretch that hits all four divisors
    foobar(1, 12)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [
        "1",
        "2",
        "Foo",
        "4",
        "Bar",
        "Foo",
        "Bop",
        "8",
        "Foo",
        "Bar",
        "Bap",
    ]


def test_foobar_foobarbopbap(capsys):
    foobar(1155, 1156)
    assert capsys.readouterr().out == "FooBarBopBap\n"

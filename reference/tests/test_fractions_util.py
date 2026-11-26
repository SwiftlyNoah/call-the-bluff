import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fractions_util import (
    StrategyError,
    format_fraction,
    normalize_strategy,
    parse_game_rational,
    parse_strategy_prob,
)

ACTS = {"I": ["a", "b"]}


def test_game_rationals():
    assert parse_game_rational("-1/18") == Fraction(-1, 18)
    assert parse_game_rational("0.25") == Fraction(1, 4)
    assert parse_game_rational("-2") == -2
    for bad in ["1e-3", "1/0", " 1", "1.", ".5", "1/-2", "nan", "inf", "1_000", 0.5, 1, ""]:
        with pytest.raises(ValueError):
            parse_game_rational(bad)


def test_strategy_probs_accept_float_repr_exactly():
    x = 0.1
    assert parse_strategy_prob(repr(x)) == Fraction(repr(x))
    assert parse_strategy_prob("1e-05") == Fraction(1, 100000)
    assert parse_strategy_prob("2/3") == Fraction(2, 3)
    assert parse_strategy_prob(repr(5e-324)) == Fraction("5e-324")
    assert parse_strategy_prob("1.7976931348623157e+308") == Fraction(repr(1.7976931348623157e308))
    # Exponents beyond 3 digits are rejected before Fraction would build a huge integer.
    for bad in ["-0.1", "nan", "inf", "1/0", "", 0.5, "0x1p-2", "1e-999999999", "1e0001"]:
        with pytest.raises(ValueError):
            parse_strategy_prob(bad)


def test_row_summing_to_099_is_rejected():
    with pytest.raises(StrategyError, match="not 1 within 1e-6"):
        normalize_strategy(ACTS, {"I": {"a": "0.5", "b": "0.49"}})


def test_row_just_outside_tolerance_rejected_just_inside_normalized_exactly():
    with pytest.raises(StrategyError):
        normalize_strategy(ACTS, {"I": {"a": "0.5", "b": "0.4999989"}})
    out = normalize_strategy(ACTS, {"I": {"a": "0.5", "b": "0.4999991"}})
    total = Fraction("0.9999991")
    assert out["I"] == {"a": Fraction("0.5") / total, "b": Fraction("0.4999991") / total}
    assert sum(out["I"].values()) == 1
    # exactly at the boundary (|sum - 1| == 1e-6) is accepted
    out = normalize_strategy(ACTS, {"I": {"a": "0.5", "b": "0.499999"}})
    assert sum(out["I"].values()) == 1


def test_float_repr_rows_normalize():
    a, b = 1 / 3, 2 / 3
    out = normalize_strategy(ACTS, {"I": {"a": repr(a), "b": repr(b)}})
    assert sum(out["I"].values()) == 1
    assert out["I"]["a"] == Fraction(a) / (Fraction(a) + Fraction(b))


@pytest.mark.parametrize("probs, msg", [
    ({}, "missing from strategy"),
    ({"I": {"a": "1"}}, "missing actions"),
    ({"I": {"a": "1", "b": "0", "c": "0"}}, "unknown actions"),
    ({"I": {"a": "1", "b": "0"}, "J": {"x": "1"}}, "not in the game"),
    ({"I": {"a": "1.1", "b": "-0.1"}}, "nonnegative"),
    ({"I": {"a": 1.0, "b": "0"}}, "nonnegative"),
    ({"I": {"a": "0", "b": "0"}}, "not 1 within"),
])
def test_rejections(probs, msg):
    with pytest.raises(StrategyError, match=msg):
        normalize_strategy(ACTS, probs)


def test_format_fraction():
    assert format_fraction(Fraction(-1, 18)) == "-1/18"
    assert format_fraction(Fraction(4, 2)) == "2"
    assert format_fraction(Fraction(0)) == "0"

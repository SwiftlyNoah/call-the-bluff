"""Exact number parsing and strategy normalization (CONTRACTS.md sections 1 and 2).

Kept tiny on purpose: this is part of the trusted base, so each function does one thing and
`reference/tests/test_fractions_util.py` pins its behavior.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from fractions import Fraction

# Game files: an integer, a fraction p/q (q > 0) or a finite decimal. No exponents, no floats.
GAME_RATIONAL_RE = re.compile(r"^-?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/0*[1-9][0-9]*)$")
# Strategy files: nonnegative decimals, possibly in exponent form as Python's repr prints a
# float ("1e-05"), or fractions. repr of a double is exact once read as a decimal. The exponent
# has at most 3 digits (a double's repr never needs more): "1e-999999999" would otherwise make
# Fraction build a billion-digit integer and hang the checker.
STRATEGY_PROB_RE = re.compile(
    r"^\+?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]{1,3})?|[0-9]+/0*[1-9][0-9]*)$"
)

NORMALIZATION_TOLERANCE = Fraction(1, 10**6)


class StrategyError(ValueError):
    """A strategy file that must be rejected (never repaired)."""


def parse_game_rational(text: object) -> Fraction:
    """Parse an exact rational string from a game file, rejecting anything else."""
    if not isinstance(text, str) or not GAME_RATIONAL_RE.match(text):
        raise ValueError(f"not an exact rational string: {text!r}")
    return Fraction(text)


def parse_strategy_prob(text: object) -> Fraction:
    """Parse a probability string from a strategy file exactly (never through a float)."""
    if not isinstance(text, str) or not STRATEGY_PROB_RE.match(text):
        raise ValueError(f"not a nonnegative exact number string: {text!r}")
    return Fraction(text)


def format_fraction(x: Fraction) -> str:
    """Canonical string for a Fraction: "n" or "n/d" in lowest terms."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def normalize_strategy(
    infoset_actions: Mapping[str, Sequence[str]],
    probs: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, Fraction]]:
    """Normalize a strategy's `probs` against a game's infosets.

    Every infoset of the game must appear with exactly its actions (no missing, no extra), each
    probability must parse as a nonnegative exact number, and each row must sum to 1 within 1e-6
    (compared exactly). The row is then divided by its exact sum. Anything else raises
    StrategyError: rows that fail are rejected, never repaired.
    """
    errors: list[str] = []
    out: dict[str, dict[str, Fraction]] = {}
    extra = sorted(set(probs) - set(infoset_actions))
    if extra:
        errors.append(f"strategy names infosets not in the game: {extra[:10]}")
    for infoset, actions in infoset_actions.items():
        row = probs.get(infoset)
        if row is None:
            errors.append(f"infoset {infoset!r} missing from strategy")
            continue
        if not isinstance(row, Mapping):
            errors.append(f"infoset {infoset!r}: row is not an object")
            continue
        missing = [a for a in actions if a not in row]
        unknown = [a for a in row if a not in actions]
        if missing or unknown:
            errors.append(
                f"infoset {infoset!r}: missing actions {missing}, unknown actions {unknown}"
            )
            continue
        parsed: dict[str, Fraction] = {}
        for a in actions:
            try:
                parsed[a] = parse_strategy_prob(row[a])
            except ValueError as e:
                errors.append(f"infoset {infoset!r} action {a!r}: {e}")
        if len(parsed) != len(actions):
            continue
        total = sum(parsed.values(), Fraction(0))
        if abs(total - 1) > NORMALIZATION_TOLERANCE:
            errors.append(
                f"infoset {infoset!r}: row sums to {float(total)!r} ({format_fraction(total)}), "
                f"not 1 within 1e-6"
            )
            continue
        out[infoset] = {a: parsed[a] / total for a in actions}
    if errors:
        raise StrategyError("; ".join(errors))
    return out

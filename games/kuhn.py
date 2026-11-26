"""Kuhn poker game file generator.

Rules (Kuhn 1950, in the standard form used by OpenSpiel's kuhn_poker):
  - Deck J < Q < K. Each player antes 1 and gets one private card; the third
    card is unused. All 6 ordered deals have probability 1/6.
  - Player 0 acts first: check or bet 1.
      * check, then player 1 checks: showdown for the pot of 2 (winner nets +1).
      * check, then player 1 bets 1: player 0 folds (nets -1) or calls
        (showdown, winner nets +2).
      * bet, then player 1 folds (player 0 nets +1) or calls (showdown, +-2).
  - Higher card wins the showdown.

Game value to player 0: -1/18 (Kuhn 1950).

Infoset id format: "<own card>:<public betting history>", history letters
  c = check (no bet outstanding) or call (bet outstanding), b = bet, f = fold.
  Examples: "J:" (player 0 with J, first to act), "Q:b" (player 1 with Q facing
  a bet), "K:cb" (player 0 with K after check, bet).
Action labels: "check"/"bet" when no bet is outstanding, "fold"/"call" when one is.
Chance outcome labels: "<p0 card><p1 card>", e.g. "JQ".

Outputs (in games/out/):
  kuhn.game.json          the game
  kuhn.ref-strategy.json  the known equilibrium family at alpha = 1/3, exact

Run: uv run games/kuhn.py
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GameBuilder, check_game, game_doc, q, write_json

CARDS = "JQK"
OUT = Path(__file__).resolve().parent / "out"


def build() -> dict:
    g = GameBuilder()

    def showdown(c0: str, c1: str, amount: int) -> Fraction:
        return Fraction(amount if CARDS.index(c0) > CARDS.index(c1) else -amount)

    def deal(c0: str, c1: str):
        def root():
            return g.decision(0, f"{c0}:", [
                ("check", lambda: g.decision(1, f"{c1}:c", [
                    ("check", lambda: g.terminal(showdown(c0, c1, 1))),
                    ("bet", lambda: g.decision(0, f"{c0}:cb", [
                        ("fold", lambda: g.terminal(-1)),
                        ("call", lambda: g.terminal(showdown(c0, c1, 2))),
                    ])),
                ])),
                ("bet", lambda: g.decision(1, f"{c1}:b", [
                    ("fold", lambda: g.terminal(1)),
                    ("call", lambda: g.terminal(showdown(c0, c1, 2))),
                ])),
            ])
        return root

    g.chance([(c0 + c1, Fraction(1, 6), deal(c0, c1)) for c0, c1 in permutations(CARDS, 2)])
    return game_doc(
        "kuhn",
        "3-card Kuhn poker (J, Q, K), ante 1, one bet of size 1, player 0 acts first. "
        "Infoset ids are '<own card>:<history>' with c=check/call, b=bet, f=fold.",
        g.nodes,
        known_value_p0="-1/18",
        known_value_source="Kuhn 1950",
    )


def ref_strategy(alpha: Fraction = Fraction(1, 3)) -> dict:
    """Kuhn's equilibrium family (Kuhn 1950; see also the OpenSpiel docs and
    Wikipedia 'Kuhn poker'), parameterized by alpha in [0, 1/3]:
      player 0: J bets alpha; Q checks, calls a bet with alpha + 1/3;
                K bets 3*alpha, always calls; J always folds to a bet.
      player 1: K bets after a check and calls; Q checks after a check and calls
                a bet with 1/3; J bets after a check with 1/3 and folds to a bet.
    Every member gives player 0 the value -1/18.
    """
    assert 0 <= alpha <= Fraction(1, 3)
    one, zero = Fraction(1), Fraction(0)

    def row(a: str, pa: Fraction, b: str) -> dict:
        return {a: q(pa), b: q(1 - pa)}

    probs = {
        "J:": row("check", 1 - alpha, "bet"),
        "Q:": row("check", one, "bet"),
        "K:": row("check", 1 - 3 * alpha, "bet"),
        "J:cb": row("fold", one, "call"),
        "Q:cb": row("fold", 1 - (alpha + Fraction(1, 3)), "call"),
        "K:cb": row("fold", zero, "call"),
        "J:c": row("check", Fraction(2, 3), "bet"),
        "Q:c": row("check", one, "bet"),
        "K:c": row("check", zero, "bet"),
        "J:b": row("fold", one, "call"),
        "Q:b": row("fold", Fraction(2, 3), "call"),
        "K:b": row("fold", zero, "call"),
    }
    return {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": f"games/kuhn.py ref_strategy: Kuhn 1950 closed-form equilibrium, alpha={q(alpha)}",
        "probs": probs,
    }


def main() -> None:
    doc = build()
    counts = check_game(doc)
    # Known counts: 12 infosets (6 per player), 30 terminal histories (6 deals x 5).
    assert counts["infosets"] == 12 and counts["infosets_p0"] == 6 and counts["infosets_p1"] == 6, counts
    assert counts["terminal"] == 30 and counts["decision"] == 24 and counts["chance"] == 1, counts
    write_json(OUT / "kuhn.game.json", doc)
    write_json(OUT / "kuhn.ref-strategy.json", ref_strategy())
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

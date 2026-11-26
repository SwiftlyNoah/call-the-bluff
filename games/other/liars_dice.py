"""Liar's Dice with one die each and a small die.

KIND: bluffing game with private information and an open-ended (but finite) bidding sequence; no
betting money, the stake is fixed at 1. Chance acts only at the start.

Rules as generated:
  - Chance rolls p0's die, then p1's die, each uniform on 1..F (two chance nodes, depths 0 and 1).
    Each player sees only their own die.
  - A bid is "QxV": "there are at least Q dice showing V" among both dice, Q in {1, 2}, V in 1..F.
    Bids are ordered by (Q, V): 1x1 < 1x2 < ... < 1xF < 2x1 < ... < 2xF. There are no wild faces.
  - p0 must open with a bid. Then players alternate; on a turn a player either makes a strictly
    higher bid or calls "liar" on the previous bid. After the top bid 2xF the only legal action is
    "liar" (a one-action decision node).
  - On "liar", count the dice showing V. If count >= Q the bidder wins 1 and the caller loses 1;
    otherwise the caller wins 1.

Infoset label: "P{player} d{own die} |{bid history}", for example "P1 d2 |1x3".
Known values (computed by games/other/solve.py, certified by exact best response, not closed form):
  F=3: -1/9, F=4: -1/8.
"""

from __future__ import annotations

from fractions import Fraction

from gamelib import Tree, game_doc


def bids(faces: int) -> list[tuple[int, int]]:
    return [(q, v) for q in (1, 2) for v in range(1, faces + 1)]


def bid_label(b: tuple[int, int]) -> str:
    return f"{b[0]}x{b[1]}"


def build(faces: int) -> Tree:
    t = Tree()
    all_bids = bids(faces)
    p_face = Fraction(1, faces)

    def turn(dice: tuple[int, int], history: list[int]) -> int:
        """history: indices into all_bids, strictly increasing. Player to act alternates, p0 first."""
        player = len(history) % 2
        infoset = f"P{player} d{dice[player]} |" + ",".join(bid_label(all_bids[i]) for i in history)
        actions = []
        if history:
            actions.append(("liar", lambda: t.terminal(resolve(dice, history))))
        start = history[-1] + 1 if history else 0
        for i in range(start, len(all_bids)):
            actions.append((bid_label(all_bids[i]), lambda i=i: turn(dice, history + [i])))
        return t.decision(player, infoset, actions)

    def resolve(dice: tuple[int, int], history: list[int]) -> int:
        q, v = all_bids[history[-1]]
        bidder = (len(history) - 1) % 2  # the player who made the last bid
        true = sum(1 for d in dice if d == v) >= q
        bidder_wins = true
        p0_wins = (bidder == 0) == bidder_wins
        return 1 if p0_wins else -1

    t.chance(
        [
            (
                f"p0:{d0}",
                p_face,
                lambda d0=d0: t.chance(
                    [
                        (f"p1:{d1}", p_face, lambda d1=d1: turn((d0, d1), []))
                        for d1 in range(1, faces + 1)
                    ]
                ),
            )
            for d0 in range(1, faces + 1)
        ]
    )
    return t


# Values found by games/other/solve.py: a float sequence-form LP whose profile, rounded to rationals,
# has exploitability exactly 0 under exact best response. Not derived by hand; see README.md.
_SRC = (
    "computed, not closed form: sequence-form LP rounded to a rational profile whose exploitability "
    "is exactly 0 by exact best response (games/other/solve.py; profile in games/out/other/eq/)"
)
KNOWN = {3: (Fraction(-1, 9), _SRC), 4: (Fraction(-1, 8), _SRC)}


def liars_dice(faces: int) -> dict:
    name = f"liars_dice_1x{faces}"
    kv = KNOWN.get(faces)
    return game_doc(
        name,
        f"Liar's Dice, one {faces}-sided die each, bids QxV with Q in {{1,2}}, no wilds, p0 opens, "
        "raise or call liar, stake 1. Chance at depths 0 and 1 (p0's die, then p1's).",
        build(faces),
        known_value=kv[0] if kv else None,
        source=kv[1] if kv else None,
    )


GAMES = [lambda: liars_dice(3), lambda: liars_dice(4)]

"""Goofspiel (the game of pure strategy) with N cards.

KIND: multi-round simultaneous-move game. Each round both players bid at the same time, written as
p0 moving first and p1 moving in an infoset that does not reveal p0's current bid. After a round
both bids are revealed (public), so the game is imperfect-information only within a round.

Rules as generated:
  - Each player holds cards 1..N; the prize deck is 1..N.
  - Prize order: variant "fixed" reveals prizes in the order 1, 2, ..., N (no chance at all);
    variant "random" has chance reveal each round's prize uniformly from the prizes left, at the
    start of that round (chance at depths 0, 3, 6, ...).
  - Each round: the prize is shown, p0 and p1 each play one unused card simultaneously; the higher
    card wins the prize; equal cards discard it. Played cards are gone.
  - The last round is forced (one card each) and is played automatically, with no decision node.
  - Payoff to p0: (sum of prizes p0 won) - (sum of prizes p1 won).

Infoset label: "P{player} prizes={prizes so far} p0={p0's past bids} p1={p1's past bids}". The
current round's bids are never in the label, which is what makes the round simultaneous.
Known value: 0 for both variants, by symmetry (README.md has the argument; solve.py certifies it).
"""

from __future__ import annotations

from fractions import Fraction

from gamelib import Tree, game_doc


def build(n: int, random_order: bool) -> Tree:
    t = Tree()
    deck = tuple(range(1, n + 1))

    def round_start(prizes: tuple, b0: tuple, b1: tuple) -> int:
        left = [c for c in deck if c not in prizes]
        if random_order:
            return t.chance(
                [
                    (f"prize:{c}", Fraction(1, len(left)), lambda c=c: bidding(prizes + (c,), b0, b1))
                    for c in left
                ]
            )
        return bidding(prizes + (left[0],), b0, b1)

    def label(p: int, prizes: tuple, b0: tuple, b1: tuple) -> str:
        def j(xs: tuple) -> str:
            return ",".join(map(str, xs))

        return f"P{p} prizes={j(prizes)} p0={j(b0)} p1={j(b1)}"

    def bidding(prizes: tuple, b0: tuple, b1: tuple) -> int:
        hand0 = [c for c in deck if c not in b0]
        hand1 = [c for c in deck if c not in b1]
        if len(hand0) == 1:  # forced last round
            return t.terminal(score(prizes, b0 + (hand0[0],), b1 + (hand1[0],)))
        return t.decision(
            0,
            label(0, prizes, b0, b1),
            [
                (
                    f"bid:{c0}",
                    lambda c0=c0: t.decision(
                        1,
                        label(1, prizes, b0, b1),
                        [(f"bid:{c1}", lambda c1=c1: after(prizes, b0 + (c0,), b1 + (c1,))) for c1 in hand1],
                    ),
                )
                for c0 in hand0
            ],
        )

    def after(prizes: tuple, b0: tuple, b1: tuple) -> int:
        return round_start(prizes, b0, b1)

    def score(prizes: tuple, b0: tuple, b1: tuple) -> int:
        s = 0
        for pr, x, y in zip(prizes, b0, b1):
            s += pr if x > y else -pr if y > x else 0
        return s

    round_start((), (), ())
    return t


def goofspiel(n: int, random_order: bool) -> dict:
    variant = "random" if random_order else "fixed"
    order = (
        "prize order by chance, each round uniform over the prizes left"
        if random_order
        else f"prize order fixed 1..{n}"
    )
    return game_doc(
        f"goofspiel{n}_{variant}",
        f"Goofspiel with {n} cards, {order}; simultaneous bids via infosets, bids revealed after "
        "each round, ties discard the prize, last round forced, payoff = p0 prize sum - p1 prize sum.",
        build(n, random_order),
        known_value=0,
        source="symmetric game (swapping the players maps the game to itself with negated payoff), "
        "so the value is 0; argument in games/other/README.md, certified exactly by solve.py",
    )


GAMES = [
    lambda: goofspiel(3, False),
    lambda: goofspiel(3, True),
    lambda: goofspiel(4, False),
]

"""Leduc hold'em game file generator.

Follows OpenSpiel's leduc_poker (open_spiel/games/leduc_poker.cc, default
parameters), which implements the game of Southey et al. 2005, "Bayes' Bluff:
Opponent Modelling in Poker":
  - Deck of 6 cards: two suits (s, h) of J < Q < K.
  - Each player antes 1 and gets one private card (all 30 ordered deals have
    probability 1/30; OpenSpiel deals them as 1/6 then 1/5, the same law).
  - Two betting rounds. Player 0 acts first in BOTH rounds. Fixed raise size 2
    in round 1 and 4 in round 2, at most 2 raises per round.
  - Actions (OpenSpiel's legal actions): fold only when facing a bet, call
    (a check when nothing is owed) always, raise while fewer than 2 raises
    have happened this round. A raise first matches the outstanding bet and
    then adds the round's raise size.
  - A round ends when a call (or check) follows the round's first action, so
    "cc", "rc", "crc", "rrc", "crrc" end a round; a fold ends the game.
  - After round 1 one public card is dealt from the 4 remaining (1/4 each).
  - Showdown: a private card that pairs the public card wins; otherwise the
    higher private rank wins; equal ranks split the pot (payoff 0).
  - Payoffs are net: the winner wins the loser's contribution (ante included).

Infoset id format:
  round 1: "<own card>:<round-1 history>"                   e.g. "Qs:cr"
  round 2: "<own card>|<public card>:<round-1>/<round-2>"   e.g. "Qs|Kh:rc/r"
  cards are rank then suit (Js, Jh, Qs, Qh, Ks, Kh); history letters are
  f = fold, c = call or check, r = raise.
Action labels: "fold", "call", "raise" (in that order, omitting illegal ones,
matching OpenSpiel's action order).
Chance labels: deal "<p0 card>,<p1 card>" (e.g. "Js,Qh"); public card "<card>".

Expected counts (checked in main): 936 infosets, 468 per player, the count
OpenSpiel reports for leduc_poker (and the usual figure in the CFR literature,
e.g. Lanctot et al. 2009, "Monte Carlo Sampling for Regret Minimization").
Derivation: per player 3 round-1 histories x 6 cards = 18 plus
5 round-1 endings x 3 round-2 histories x 6 x 5 card pairs = 450.

Run: uv run games/leduc.py
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GameBuilder, check_game, game_doc, write_json

RANKS = "JQK"
SUITS = "sh"
CARDS = [r + s for r in RANKS for s in SUITS]
RAISE_SIZE = {1: 2, 2: 4}
MAX_RAISES = 2
ANTE = 1
OUT = Path(__file__).resolve().parent / "out"


def showdown_sign(c0: str, c1: str, pub: str) -> int:
    """+1 if player 0 wins, -1 if player 1 wins, 0 on a split."""
    p0_pair, p1_pair = c0[0] == pub[0], c1[0] == pub[0]
    if p0_pair != p1_pair:
        return 1 if p0_pair else -1
    r0, r1 = RANKS.index(c0[0]), RANKS.index(c1[0])
    return (r0 > r1) - (r0 < r1)


def build() -> dict:
    g = GameBuilder()

    def betting(c0, c1, pub, rnd, h1, h2, money, stakes, raises, player):
        """Decision node for `player`, or the node reached after the round ends."""
        own = (c0, c1)[player]
        hist = h1 if rnd == 1 else h2
        iset = f"{own}:{h1}" if rnd == 1 else f"{own}|{pub}:{h1}/{h2}"
        actions = []
        if stakes > money[player]:
            actions.append(("fold", lambda: g.terminal(-money[0] if player == 0 else money[1])))

        def after(letter, new_money, new_stakes, new_raises, ends_round):
            nh1, nh2 = (h1 + letter, h2) if rnd == 1 else (h1, h2 + letter)

            def build_child():
                if not ends_round:
                    return betting(c0, c1, pub, rnd, nh1, nh2, new_money, new_stakes, new_raises, 1 - player)
                if rnd == 1:
                    rest = [c for c in CARDS if c not in (c0, c1)]
                    return g.chance([
                        (p, Fraction(1, len(rest)),
                         (lambda p=p: betting(c0, c1, p, 2, nh1, nh2, new_money, new_stakes, 0, 0)))
                        for p in rest
                    ])
                assert new_money[0] == new_money[1]
                return g.terminal(showdown_sign(c0, c1, pub) * new_money[0])
            return build_child

        call_money = list(money)
        call_money[player] = stakes
        actions.append(("call", after("c", call_money, stakes, raises, ends_round=len(hist) > 0)))
        if raises < MAX_RAISES:
            new_stakes = stakes + RAISE_SIZE[rnd]
            raise_money = list(money)
            raise_money[player] = new_stakes
            actions.append(("raise", after("r", raise_money, new_stakes, raises + 1, ends_round=False)))
        return g.decision(player, iset, actions)

    deals = [(c0, c1) for c0 in CARDS for c1 in CARDS if c0 != c1]
    g.chance([
        (f"{c0},{c1}", Fraction(1, len(deals)),
         (lambda c0=c0, c1=c1: betting(c0, c1, None, 1, "", "", [ANTE, ANTE], ANTE, 0, 0)))
        for c0, c1 in deals
    ])
    return game_doc(
        "leduc",
        "Leduc hold'em as in OpenSpiel leduc_poker (Southey et al. 2005): 6 cards (J, Q, K in suits s, h), "
        "ante 1, raises 2 then 4, at most 2 raises per round, player 0 first in both rounds, pair beats "
        "high card. Infoset ids: '<card>:<r1>' and '<card>|<public>:<r1>/<r2>' with f=fold, c=call/check, r=raise.",
        g.nodes,
    )


def main() -> None:
    doc = build()
    counts = check_game(doc)
    assert counts["infosets"] == 936 and counts["infosets_p0"] == 468 and counts["infosets_p1"] == 468, counts
    write_json(OUT / "leduc.game.json", doc)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

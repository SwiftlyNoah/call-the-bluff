"""Two-round dice bluff: a poker-like betting game with chance at five depths (0, 1, 4, 5, 6).

KIND: bluffing game with private information revealed in stages; chance acts before round 1 AND in
the middle of the game (before round 2), with non-uniform (loaded) probabilities in round 2.

Rules as generated:
  - Both players ante 1.
  - Chance rolls p0's first die, then p1's first die: fair, faces 1..3. Each sees only their own.
  - Round 1 betting, bet size 1, p0 acts first:
      p0: check or bet. After check, p1: check or bet. Facing a bet: fold or call.
      (So the sequences are: check-check, bet-call, bet-fold, check-bet-call, check-bet-fold.)
    A fold ends the game: the folder loses the ante plus what they put in.
  - Chance rolls p0's second die, then p1's second die: LOADED, P(1)=1/2, P(2)=1/3, P(3)=1/6.
    Each sees only their own.
  - Round 2 betting, same structure, bet size 2, p0 acts first.
  - Showdown: the higher total of one's two dice wins the other's contribution; equal totals tie (0).

Infoset label: "P{player} d={own dice} |{round 1 actions}|{round 2 actions}", actions abbreviated
k=check, b=bet, c=call, f=fold.
Known value (computed by games/other/solve.py, certified by exact best response, not closed form):
  -5980/142263, about -0.0420348.
"""

from __future__ import annotations

from fractions import Fraction

from gamelib import Tree, game_doc

FAIR = [(1, Fraction(1, 3)), (2, Fraction(1, 3)), (3, Fraction(1, 3))]
LOADED = [(1, Fraction(1, 2)), (2, Fraction(1, 3)), (3, Fraction(1, 6))]
BET = {1: 1, 2: 2}

# The sequences that end a betting round: ("fold", who folded) or ("next", None).
ROUND_ENDS = {
    "kk": ("next", None),
    "bc": ("next", None),
    "bf": ("fold", 1),
    "kbc": ("next", None),
    "kbf": ("fold", 0),
}
WORDS = {"k": "check", "b": "bet", "c": "call", "f": "fold"}


def legal(seq: str) -> list[str]:
    if seq in ("", "k"):
        return ["k", "b"]
    if seq in ("b", "kb"):
        return ["f", "c"]
    raise ValueError(seq)


def build() -> Tree:
    t = Tree()

    def deal(dice0: tuple, dice1: tuple, dist, then) -> int:
        return t.chance(
            [
                (
                    f"p0:{a}",
                    pa,
                    lambda a=a: t.chance(
                        [(f"p1:{b}", pb, lambda b=b: then(dice0 + (a,), dice1 + (b,))) for b, pb in dist]
                    ),
                )
                for a, pa in dist
            ]
        )

    def betting(rnd: int, dice0, dice1, hist: list[str], seq: str, pot: tuple[int, int]) -> int:
        if seq in ROUND_ENDS:
            what, arg = ROUND_ENDS[seq]
            if what == "fold":
                folder = arg
                # the folder loses everything they put in; p0's payoff is +/- that amount
                return t.terminal(pot[1] if folder == 1 else -pot[0])
            if rnd == 1:
                return deal(
                    dice0,
                    dice1,
                    LOADED,
                    lambda d0, d1: betting(2, d0, d1, hist + [seq], "", pot),
                )
            s0, s1 = sum(dice0), sum(dice1)
            return t.terminal(pot[1] if s0 > s1 else -pot[0] if s1 > s0 else 0)
        player = len(seq) % 2
        own = dice0 if player == 0 else dice1
        infoset = f"P{player} d={','.join(map(str, own))} |" + "|".join(hist + [seq])
        acts = []
        for a in legal(seq):
            add = BET[rnd] if a in ("b", "c") else 0
            new_pot = (pot[0] + add, pot[1]) if player == 0 else (pot[0], pot[1] + add)
            acts.append(
                (WORDS[a], lambda a=a, new_pot=new_pot: betting(rnd, dice0, dice1, hist, seq + a, new_pot))
            )
        return t.decision(player, infoset, acts)

    deal((), (), FAIR, lambda d0, d1: betting(1, d0, d1, [], "", (1, 1)))
    return t


def dice_bluff() -> dict:
    return game_doc(
        "dice_bluff_2round",
        "Two-round dice bluff: ante 1, fair 3-sided private die each, betting round (bet 1), then a "
        "loaded 3-sided private die each (1/2, 1/3, 1/6), betting round (bet 2), higher dice total "
        "wins. Chance at depths 0, 1 and mid-game.",
        build(),
        known_value=Fraction(-5980, 142263),
        source="computed, not closed form: sequence-form LP rounded to a rational profile whose "
        "exploitability is exactly 0 by exact best response (games/other/solve.py; profile in "
        "games/out/other/eq/)",
    )


GAMES = [dice_bluff]

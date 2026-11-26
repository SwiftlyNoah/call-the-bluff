"""Direct rules simulators ("referees") for the differential tests.

Each referee is the game written as a plain sequential program, in a different style from the tree
generators: a Python generator that plays one game and yields requests to its caller.

  yield ("chance", {label: prob})            -> receives the outcome label
  yield ("move", player, observation, legal) -> receives the chosen action label
  return payoff_p0

The observation is what the acting player knows, written in the referee's own words (a tuple), so
the test can check that the game file's infosets partition histories exactly as the referee's
observations do, without the two sharing any label code. The referees import nothing from the
generators.
"""

from __future__ import annotations

from fractions import Fraction

# ------------------------------------------------------------------ matrix games


def matrix_referee(actions: list[str], payoff: dict[tuple[str, str], int]):
    def play():
        a = yield ("move", 0, ("matrix",), tuple(actions))
        b = yield ("move", 1, ("matrix",), tuple(actions))  # p1 is not told a
        return payoff[(a, b)]

    return play


def pennies():
    return matrix_referee(["H", "T"], {(a, b): 1 if a == b else -1 for a in "HT" for b in "HT"})


def _rps_payoff(rock_bonus: int) -> dict:
    beats = {"R": "S", "P": "R", "S": "P"}
    table = {}
    for a in "RPS":
        for b in "RPS":
            if a == b:
                table[(a, b)] = 0
            elif beats[a] == b:
                table[(a, b)] = rock_bonus if a == "R" else 1
            else:
                table[(a, b)] = -1
    return table


def rps():
    return matrix_referee(["R", "P", "S"], _rps_payoff(1))


def rps_biased():
    return matrix_referee(["R", "P", "S"], _rps_payoff(2))


# ------------------------------------------------------------------ Liar's Dice


def liars_dice(faces: int):
    ladder = [f"{q}x{v}" for q in (1, 2) for v in range(1, faces + 1)]
    roll = {str(f): Fraction(1, faces) for f in range(1, faces + 1)}

    def play():
        dice = []
        for who in (0, 1):
            outcome = yield ("chance", {f"p{who}:{k}": p for k, p in roll.items()})
            dice.append(int(outcome.split(":")[1]))
        said: list[str] = []
        turn = 0
        while True:
            top = ladder.index(said[-1]) if said else -1
            legal = (["liar"] if said else []) + ladder[top + 1 :]
            choice = yield ("move", turn, ("die", dice[turn], "said", tuple(said)), tuple(legal))
            if choice == "liar":
                q, v = (int(x) for x in said[-1].split("x"))
                claim_holds = dice.count(v) >= q
                bidder = 1 - turn
                winner = bidder if claim_holds else turn
                return 1 if winner == 0 else -1
            said.append(choice)
            turn = 1 - turn

    return play


# ------------------------------------------------------------------ Goofspiel


def goofspiel(n: int, random_order: bool):
    def play():
        hands = [list(range(1, n + 1)), list(range(1, n + 1))]
        prizes_left = list(range(1, n + 1))
        shown: list[int] = []
        played: list[list[int]] = [[], []]
        score = [0, 0]
        for _ in range(n):
            if random_order:
                pick = yield ("chance", {f"prize:{c}": Fraction(1, len(prizes_left)) for c in prizes_left})
                prize = int(pick.split(":")[1])
            else:
                prize = prizes_left[0]
            prizes_left.remove(prize)
            shown.append(prize)
            if len(hands[0]) == 1:
                card = [hands[0][0], hands[1][0]]
            else:
                public = (tuple(shown), tuple(played[0]), tuple(played[1]))
                card = []
                for who in (0, 1):
                    c = yield ("move", who, public, tuple(f"bid:{x}" for x in hands[who]))
                    card.append(int(c.split(":")[1]))
            for who in (0, 1):
                hands[who].remove(card[who])
                played[who].append(card[who])
            if card[0] != card[1]:
                score[0 if card[0] > card[1] else 1] += prize
        return score[0] - score[1]

    return play


# ------------------------------------------------------------------ two-round dice bluff


def dice_bluff():
    fair = {1: Fraction(1, 3), 2: Fraction(1, 3), 3: Fraction(1, 3)}
    loaded = {1: Fraction(1, 2), 2: Fraction(1, 3), 3: Fraction(1, 6)}

    def play():
        put_in = [1, 1]  # antes
        dice: list[list[int]] = [[], []]
        log: list[tuple[str, ...]] = []
        for rnd, (dist, bet) in enumerate([(fair, 1), (loaded, 2)]):
            for who in (0, 1):
                o = yield ("chance", {f"p{who}:{k}": p for k, p in dist.items()})
                dice[who].append(int(o.split(":")[1]))
            acts: list[str] = []
            to_act, facing_bet = 0, False
            while True:
                legal = ("fold", "call") if facing_bet else ("check", "bet")
                obs = (tuple(dice[to_act]), tuple(log), tuple(acts))
                a = yield ("move", to_act, obs, legal)
                acts.append(a)
                if a == "fold":
                    return put_in[1] if to_act == 1 else -put_in[0]
                if a in ("bet", "call"):
                    put_in[to_act] += bet
                if a == "call" or acts == ["check", "check"]:
                    break
                facing_bet = a == "bet"
                to_act = 1 - to_act
            log.append(tuple(acts))
        t0, t1 = sum(dice[0]), sum(dice[1])
        if t0 == t1:
            return 0
        return put_in[1] if t0 > t1 else -put_in[0]

    return play


REFEREES = {
    "matching_pennies": pennies,
    "rps": rps,
    "rps_biased": rps_biased,
    "liars_dice_1x3": lambda: liars_dice(3),
    "liars_dice_1x4": lambda: liars_dice(4),
    "goofspiel3_fixed": lambda: goofspiel(3, False),
    "goofspiel3_random": lambda: goofspiel(3, True),
    "goofspiel4_fixed": lambda: goofspiel(4, False),
    "dice_bluff_2round": dice_bluff,
}

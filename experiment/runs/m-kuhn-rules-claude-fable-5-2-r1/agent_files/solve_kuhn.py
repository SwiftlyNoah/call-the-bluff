#!/usr/bin/env python3
"""Solve Kuhn poker with CFR+ and write the average strategy to strategy.json.

The game is implemented from its rules here. The solver is CFR+ (regret matching+,
alternating updates, linearly weighted averaging). After solving, the written file is
read back and evaluated exactly (rational arithmetic): the value for player 0 and the
exploitability, the latter computed two independent ways.
"""

import itertools
import json
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

CARDS = ("J", "Q", "K")
RANK = {"J": 0, "Q": 1, "K": 2}
DEALS = list(itertools.permutations(CARDS, 2))  # (player 0's card, player 1's card)
ACTION_LETTER = {"check": "c", "bet": "b", "fold": "f", "call": "k"}


# ---------------------------------------------------------------- game rules

def is_terminal(history):
    """Histories are strings of action letters: c check, b bet, f fold, k call."""
    return history in ("cc", "cbf", "cbk", "bf", "bk")


def player_to_act(history):
    return len(history) % 2


def legal_actions(history):
    """Actions for the player to act: check or bet if no bet is open, else fold or call."""
    if history.endswith("b"):
        return ("fold", "call")
    return ("check", "bet")


def payoff_p0(history, deal):
    """Net payoff to player 0 at a terminal history."""
    showdown = 1 if RANK[deal[0]] > RANK[deal[1]] else -1
    if history == "cc":  # both checked: showdown for the antes
        return showdown
    if history in ("cbk", "bk"):  # bet and call: showdown for ante + bet
        return 2 * showdown
    if history == "bf":  # player 1 folded to player 0's bet
        return 1
    if history == "cbf":  # player 0 folded to player 1's bet
        return -1
    raise ValueError(history)


def infoset_name(card, history):
    """Card, a colon, then the check and bet letters so far (only c and b can precede a decision)."""
    return f"{card}:{history}"


def all_infosets():
    """Every information set name mapped to its actions, found by walking the tree."""
    found = {}

    def walk(history):
        if is_terminal(history):
            return
        for card in CARDS:
            found[infoset_name(card, history)] = legal_actions(history)
        for action in legal_actions(history):
            walk(history + ACTION_LETTER[action])

    walk("")
    return found


INFOSETS = all_infosets()


# ---------------------------------------------------------------- CFR+

class Solver:
    def __init__(self):
        self.regret = {name: [0.0] * len(actions) for name, actions in INFOSETS.items()}
        self.strategy_sum = {name: [0.0] * len(actions) for name, actions in INFOSETS.items()}
        self.pending = {}  # this pass's counterfactual regrets, summed over an information set's deals

    def current_strategy(self, name):
        positive = self.regret[name]  # CFR+ keeps regrets nonnegative
        total = sum(positive)
        if total > 0:
            return [r / total for r in positive]
        return [1.0 / len(positive)] * len(positive)

    def cfr(self, history, deal, updating, reach_own, reach_other, weight):
        """Returns the expected payoff to `updating`. reach_own is the updating player's
        reach probability, reach_other is everyone else's (opponent and chance)."""
        if is_terminal(history):
            value = payoff_p0(history, deal)
            return value if updating == 0 else -value
        player = player_to_act(history)
        name = infoset_name(deal[player], history)
        actions = INFOSETS[name]
        strategy = self.current_strategy(name)

        if player != updating:
            return sum(
                p * self.cfr(history + ACTION_LETTER[a], deal, updating, reach_own, reach_other * p, weight)
                for a, p in zip(actions, strategy)
                if p > 0
            )

        values = [
            self.cfr(history + ACTION_LETTER[a], deal, updating, reach_own * p, reach_other, weight)
            for a, p in zip(actions, strategy)
        ]
        node_value = sum(p * v for p, v in zip(strategy, values))
        pending = self.pending.setdefault(name, [0.0] * len(actions))
        sums = self.strategy_sum[name]
        for i, v in enumerate(values):
            pending[i] += reach_other * (v - node_value)
            sums[i] += weight * reach_own * strategy[i]
        return node_value

    def run(self, iterations):
        chance = 1.0 / len(DEALS)
        for t in range(1, iterations + 1):
            for updating in (0, 1):
                self.pending = {}
                for deal in DEALS:
                    self.cfr("", deal, updating, 1.0, chance, float(t))
                # Regret matching+: add the whole information set's regret, then clip at zero.
                # Doing this per deal would clip partial sums and change the strategy mid-pass.
                for name, deltas in self.pending.items():
                    regret = self.regret[name]
                    for i, delta in enumerate(deltas):
                        regret[i] = max(regret[i] + delta, 0.0)

    def average_strategy(self):
        result = {}
        for name, sums in self.strategy_sum.items():
            total = sum(sums)
            result[name] = [s / total for s in sums] if total > 0 else [1.0 / len(sums)] * len(sums)
        return result


# ---------------------------------------------------------------- exact evaluation

def parse_prob(text):
    return Fraction(text)  # accepts decimals ("0.25") and fractions ("1/3")


def load_strategy(path):
    data = json.loads(Path(path).read_text())
    return {name: {a: parse_prob(p) for a, p in probs.items()} for name, probs in data["probs"].items()}


def validate(strategy):
    if set(strategy) != set(INFOSETS):
        raise ValueError("information sets do not match the game")
    for name, actions in INFOSETS.items():
        probs = strategy[name]
        if set(probs) != set(actions):
            raise ValueError(f"{name}: wrong actions")
        if any(p < 0 for p in probs.values()) or sum(probs.values()) != 1:
            raise ValueError(f"{name}: not a probability distribution")


def expected_value_p0(strategy):
    def walk(history, deal):
        if is_terminal(history):
            return Fraction(payoff_p0(history, deal))
        name = infoset_name(deal[player_to_act(history)], history)
        return sum(p * walk(history + ACTION_LETTER[a], deal) for a, p in strategy[name].items())

    return sum(walk("", deal) for deal in DEALS) / len(DEALS)


def best_response_value(strategy, responder):
    """Value to `responder` of its best response against the other player's strategy.

    Walks the public tree for each responder card, carrying the opponent's reach
    probability for each card the opponent may hold, so the responder chooses one action
    per information set rather than per deal."""

    def walk(history, card, opp_reach):
        if is_terminal(history):
            total = Fraction(0)
            for opp_card, reach in opp_reach.items():
                deal = (card, opp_card) if responder == 0 else (opp_card, card)
                value = payoff_p0(history, deal)
                total += reach * (value if responder == 0 else -value)
            return total
        actions = legal_actions(history)
        if player_to_act(history) == responder:
            return max(walk(history + ACTION_LETTER[a], card, opp_reach) for a in actions)
        total = Fraction(0)
        for a in actions:
            reach = {c: r * strategy[infoset_name(c, history)][a] for c, r in opp_reach.items()}
            total += walk(history + ACTION_LETTER[a], card, reach)
        return total

    chance = Fraction(1, len(DEALS))
    return sum(walk("", card, {c: chance for c in CARDS if c != card}) for card in CARDS)


def best_response_value_bruteforce(strategy, responder):
    """Same number by enumerating every pure strategy of the responder (an independent check)."""
    own = [name for name in INFOSETS if player_to_act(name.split(":")[1]) == responder]
    best = None
    for choice in itertools.product(*(INFOSETS[name] for name in own)):
        pure = dict(strategy)
        for name, action in zip(own, choice):
            pure[name] = {a: Fraction(int(a == action)) for a in INFOSETS[name]}
        value = expected_value_p0(pure)
        value = value if responder == 0 else -value
        best = value if best is None else max(best, value)
    return best


def exploitability(strategy):
    return (best_response_value(strategy, 0) + best_response_value(strategy, 1)) / 2


# ---------------------------------------------------------------- output

def to_strings(probs, places=12):
    """Decimal strings that sum to exactly 1: round all but the largest, which takes the rest."""
    quantum = Decimal(1).scaleb(-places)
    rounded = [Decimal(repr(p)).quantize(quantum) for p in probs]
    largest = max(range(len(probs)), key=lambda i: probs[i])
    rounded[largest] = Decimal(1) - sum(r for i, r in enumerate(rounded) if i != largest)
    return [format(r.normalize(), "f") for r in rounded]


def write_strategy(path, average, iterations):
    probs = {}
    for name in sorted(INFOSETS, key=lambda n: (len(n.split(":")[1]), n.split(":")[1], RANK[n[0]])):
        probs[name] = dict(zip(INFOSETS[name], to_strings(average[name])))
    data = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": f"solve_kuhn.py: CFR+ (alternating updates, linear averaging), {iterations} iterations",
        "probs": probs,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    out = Path(__file__).resolve().parent / "strategy.json"

    solver = Solver()
    solver.run(iterations)
    write_strategy(out, solver.average_strategy(), iterations)

    strategy = load_strategy(out)  # evaluate what was actually written
    validate(strategy)
    value = expected_value_p0(strategy)
    br0 = best_response_value(strategy, 0)
    br1 = best_response_value(strategy, 1)
    assert br0 == best_response_value_bruteforce(strategy, 0)
    assert br1 == best_response_value_bruteforce(strategy, 1)
    expl = (br0 + br1) / 2

    print(f"wrote {out}")
    for name, probs in strategy.items():
        print(f"  {name:5s} " + "  ".join(f"{a}={float(p):.6f}" for a, p in probs.items()))
    print(f"iterations            {iterations}")
    print(f"value to player 0     {float(value)!r}   (known game value -1/18 = {-1/18!r})")
    print(f"best response of p0   {float(br0)!r}")
    print(f"best response of p1   {float(br1)!r}")
    print(f"exploitability        {float(expl)!r}")


if __name__ == "__main__":
    main()

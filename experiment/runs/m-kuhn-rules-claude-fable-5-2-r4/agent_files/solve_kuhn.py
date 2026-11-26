#!/usr/bin/env python3
"""Solve Kuhn poker with CFR+ and write the average strategy to strategy.json.

The game is implemented from its rules. The solver is CFR+ (regret-matching+,
alternating updates, linearly weighted strategy averaging), run on the full
game tree with exact chance enumeration, so the run is deterministic.

After writing strategy.json the file is read back and evaluated with exact
rational arithmetic: the value for player 0 and the exploitability (the mean of
the two best-response values) are computed from what is on disk.
"""

import argparse
import itertools
import json
from decimal import Decimal
from fractions import Fraction

CARDS = "JQK"
RANK = {c: i for i, c in enumerate(CARDS)}
DEALS = list(itertools.permutations(CARDS, 2))  # (player 0 card, player 1 card)

# ---------------------------------------------------------------- the game


def is_terminal(history):
    return history in ("cc", "bc", "bf", "cbc", "cbf")


def player_to_act(history):
    return len(history) % 2


def actions(history):
    """Action labels available at a decision node, with the history letter each appends."""
    if history in ("", "c"):
        return [("check", "c"), ("bet", "b")]
    if history in ("b", "cb"):
        return [("fold", "f"), ("call", "c")]
    raise ValueError(history)


def payoff_p0(history, cards):
    """Net payoff to player 0 at a terminal history."""
    higher = 1 if RANK[cards[0]] > RANK[cards[1]] else -1
    if history == "cc":
        return higher  # showdown for the antes
    if history in ("bc", "cbc"):
        return 2 * higher  # showdown after a bet and a call
    if history == "bf":
        return 1  # player 1 folded
    if history == "cbf":
        return -1  # player 0 folded
    raise ValueError(history)


def infoset_name(history, cards):
    # Histories at decision nodes contain only checks and bets, so the letters
    # used internally ("c", "b") are the ones the naming scheme asks for.
    return cards[player_to_act(history)] + ":" + history


def all_infosets():
    out = {}
    for history in ("", "c", "b", "cb"):
        for card in CARDS:
            out[card + ":" + history] = [label for label, _ in actions(history)]
    return out


# ---------------------------------------------------------------- CFR+


class Solver:
    def __init__(self):
        self.infosets = all_infosets()
        self.regret = {k: [0.0] * len(v) for k, v in self.infosets.items()}
        self.strategy_sum = {k: [0.0] * len(v) for k, v in self.infosets.items()}

    def current(self, key):
        r = self.regret[key]
        total = sum(r)
        if total > 0:
            return [x / total for x in r]
        return [1.0 / len(r)] * len(r)

    def cfr(self, history, cards, updating, reach_self, reach_other, weight):
        """Return the expected payoff to `updating`. `reach_other` includes chance."""
        if is_terminal(history):
            v = payoff_p0(history, cards)
            return v if updating == 0 else -v
        key = infoset_name(history, cards)
        acts = actions(history)
        sigma = self.current(key)
        if player_to_act(history) != updating:
            return sum(
                sigma[i]
                * self.cfr(history + letter, cards, updating, reach_self, reach_other * sigma[i], weight)
                for i, (_, letter) in enumerate(acts)
            )
        values = [
            self.cfr(history + letter, cards, updating, reach_self * sigma[i], reach_other, weight)
            for i, (_, letter) in enumerate(acts)
        ]
        node = sum(s * v for s, v in zip(sigma, values))
        # An information set holds several deals. Their counterfactual regrets are
        # summed here and applied (with the clip at zero) once per pass, in run(),
        # so the strategy is the same at every node of an information set.
        pending = self.pending.setdefault(key, [0.0] * len(acts))
        total = self.strategy_sum[key]
        for i in range(len(acts)):
            pending[i] += reach_other * (values[i] - node)
            total[i] += weight * reach_self * sigma[i]
        return node

    def run(self, iterations):
        chance = 1.0 / len(DEALS)
        for t in range(1, iterations + 1):
            for updating in (0, 1):
                self.pending = {}
                for cards in DEALS:
                    self.cfr("", cards, updating, 1.0, chance, float(t))
                for key, delta in self.pending.items():
                    regret = self.regret[key]
                    for i, d in enumerate(delta):
                        regret[i] = max(regret[i] + d, 0.0)

    def average(self):
        out = {}
        for key, labels in self.infosets.items():
            s = self.strategy_sum[key]
            total = sum(s)
            probs = [x / total for x in s] if total > 0 else [1.0 / len(s)] * len(s)
            out[key] = dict(zip(labels, probs))
        return out


# ---------------------------------------------------------------- output


def to_decimal_strings(probs, places=12):
    """Round to fixed decimals so that each information set sums to exactly 1."""
    quantum = Decimal(1).scaleb(-places)
    out = {}
    for key, dist in probs.items():
        labels = list(dist)
        rounded = [Decimal(repr(dist[a])).quantize(quantum) for a in labels]
        rounded = [min(max(x, Decimal(0)), Decimal(1)) for x in rounded]
        biggest = max(range(len(labels)), key=lambda i: rounded[i])
        rounded[biggest] += Decimal(1) - sum(rounded)
        assert sum(rounded) == 1 and all(0 <= x <= 1 for x in rounded)
        out[key] = {a: format(x, "f") for a, x in zip(labels, rounded)}
    return out


# ---------------------------------------------------------------- exact evaluation


def load_strategy(path):
    with open(path) as f:
        data = json.load(f)
    expected = all_infosets()
    probs = {}
    for key, labels in expected.items():
        dist = {a: Fraction(data["probs"][key][a]) for a in labels}
        assert set(data["probs"][key]) == set(labels), key
        assert sum(dist.values()) == 1 and all(p >= 0 for p in dist.values()), key
        probs[key] = dist
    assert set(data["probs"]) == set(expected)
    return probs


def value_p0(probs, history="", cards=None):
    if cards is None:
        return sum(value_p0(probs, "", deal) for deal in DEALS) / len(DEALS)
    if is_terminal(history):
        return Fraction(payoff_p0(history, cards))
    dist = probs[infoset_name(history, cards)]
    return sum(dist[label] * value_p0(probs, history + letter, cards) for label, letter in actions(history))


def best_response_value(probs, br):
    """Value to player `br` of its best response against the other player's strategy in `probs`."""
    chance = Fraction(1, len(DEALS))

    def walk(history, states):
        # `states`: (deal, probability that chance and the opponent reach this node),
        # for every deal consistent with what the best responder has seen.
        if is_terminal(history):
            total = Fraction(0)
            for cards, reach in states:
                v = payoff_p0(history, cards)
                total += reach * (v if br == 0 else -v)
            return total
        acts = actions(history)
        if player_to_act(history) == br:
            return max(walk(history + letter, states) for _, letter in acts)
        total = Fraction(0)
        for label, letter in acts:
            nxt = [(cards, reach * probs[infoset_name(history, cards)][label]) for cards, reach in states]
            total += walk(history + letter, nxt)
        return total

    # The best responder sees only its own card, so each card is a separate subproblem.
    return sum(walk("", [(d, chance) for d in DEALS if d[br] == card]) for card in CARDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200000)
    parser.add_argument("--out", default="strategy.json")
    args = parser.parse_args()

    solver = Solver()
    solver.run(args.iterations)
    doc = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": f"solve_kuhn.py: CFR+ (alternating updates, linear averaging), {args.iterations} iterations",
        "probs": to_decimal_strings(solver.average()),
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    probs = load_strategy(args.out)
    value = value_p0(probs)
    br0 = best_response_value(probs, 0)
    br1 = best_response_value(probs, 1)
    exploitability = (br0 + br1) / 2
    print(f"wrote {args.out} after {args.iterations} iterations")
    for key in sorted(probs, key=lambda k: (len(k), k)):
        print(f"  {key:5s} " + "  ".join(f"{a}={float(p):.6f}" for a, p in probs[key].items()))
    print(f"value to player 0       : {float(value)!r}  (known game value -1/18 = {-1/18!r})")
    print(f"best response of player 0: {float(br0)!r}")
    print(f"best response of player 1: {float(br1)!r}")
    print(f"exploitability          : {float(exploitability)!r}")


if __name__ == "__main__":
    main()

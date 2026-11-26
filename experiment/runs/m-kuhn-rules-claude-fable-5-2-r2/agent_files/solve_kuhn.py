#!/usr/bin/env python3
"""Solve Kuhn poker with CFR+ and write the average strategy to strategy.json.

Usage: python3 solve_kuhn.py [iterations]

The game is implemented from its rules. The solver is CFR+ (regret matching+,
alternating updates, linearly weighted averaging). After solving, the strategy
is written to strategy.json, read back from the file, and evaluated exactly
(game value and exploitability) with rational arithmetic.
"""

import itertools
import json
import sys
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction

CARDS = "JQK"
RANK = {"J": 0, "Q": 1, "K": 2}
DEALS = list(itertools.permutations(CARDS, 2))  # (player 0 card, player 1 card)
DEAL_PROB = Fraction(1, len(DEALS))

# ---------------------------------------------------------------------------
# Game definition
# ---------------------------------------------------------------------------
# A history is a string of action letters. "c" is a check when no bet is
# pending and a fold when one is; "b" is a bet when no bet is pending and a
# call when one is. Info set names only ever contain non-terminal histories
# ("", "c", "b", "cb"), whose letters are all genuine checks and bets.

TERMINALS = {"cc", "cbc", "cbb", "bc", "bb"}


def is_terminal(history):
    return history in TERMINALS


def player_to_act(history):
    return len(history) % 2


def actions(history):
    """(letter, label) pairs available at a non-terminal history."""
    if history.endswith("b"):
        return (("c", "fold"), ("b", "call"))
    return (("c", "check"), ("b", "bet"))


def payoff_p0(history, cards):
    """Net payoff to player 0 at a terminal history."""
    showdown = 1 if RANK[cards[0]] > RANK[cards[1]] else -1
    if history == "cc":
        return showdown  # antes only
    if history == "bb" or history == "cbb":
        return 2 * showdown  # ante + bet each
    if history == "bc":
        return 1  # player 1 folded, player 0 wins the ante
    if history == "cbc":
        return -1  # player 0 folded, player 1 wins the ante
    raise ValueError(history)


def infoset(history, cards):
    return cards[player_to_act(history)] + ":" + history


def all_infosets():
    names = {}
    for cards in DEALS:
        for history in ("", "c", "b", "cb"):
            names[infoset(history, cards)] = [label for _, label in actions(history)]
    return names


# ---------------------------------------------------------------------------
# CFR+
# ---------------------------------------------------------------------------


class Node:
    __slots__ = ("regret", "pending", "strategy_sum")

    def __init__(self):
        self.regret = [0.0, 0.0]
        self.pending = [0.0, 0.0]  # this iteration's regret, summed over deals
        self.strategy_sum = [0.0, 0.0]

    def apply_pending(self):
        """Regret matching+: add the iteration's regret, then floor at zero."""
        for a in range(2):
            self.regret[a] = max(0.0, self.regret[a] + self.pending[a])
            self.pending[a] = 0.0

    def current(self):
        r0, r1 = self.regret
        total = r0 + r1
        if total > 0.0:
            return (r0 / total, r1 / total)
        return (0.5, 0.5)

    def average(self):
        s0, s1 = self.strategy_sum
        total = s0 + s1
        if total > 0.0:
            return (s0 / total, s1 / total)
        return (0.5, 0.5)


def cfr(nodes, history, cards, reach, updating, weight):
    """Return the value for `updating` of the subtree, accumulating that
    player's instantaneous regrets (in Node.pending) and weighted average
    strategy. The regrets of an info set are summed over every deal that
    reaches it before they are applied, in solve().

    reach[i] is player i's own contribution to the reach probability; chance
    is uniform over deals and is applied as a constant factor at the end, so
    it is left out here (it scales all regrets equally).
    """
    if is_terminal(history):
        value = payoff_p0(history, cards)
        return value if updating == 0 else -value

    player = player_to_act(history)
    node = nodes[infoset(history, cards)]
    strategy = node.current()
    letters = [letter for letter, _ in actions(history)]

    if player != updating:
        value = 0.0
        for a in range(2):
            if strategy[a] == 0.0:
                continue
            new_reach = list(reach)
            new_reach[player] *= strategy[a]
            value += strategy[a] * cfr(nodes, history + letters[a], cards, new_reach, updating, weight)
        return value

    action_values = [0.0, 0.0]
    for a in range(2):
        new_reach = list(reach)
        new_reach[player] *= strategy[a]
        action_values[a] = cfr(nodes, history + letters[a], cards, new_reach, updating, weight)
    value = strategy[0] * action_values[0] + strategy[1] * action_values[1]

    opponent_reach = reach[1 - player]
    for a in range(2):
        node.pending[a] += opponent_reach * (action_values[a] - value)
        node.strategy_sum[a] += weight * reach[player] * strategy[a]
    return value


def solve(iterations):
    nodes = {name: Node() for name in all_infosets()}
    for t in range(1, iterations + 1):
        for updating in (0, 1):
            for cards in DEALS:
                cfr(nodes, "", cards, [1.0, 1.0], updating, float(t))
            for node in nodes.values():
                node.apply_pending()
    return {name: node.average() for name, node in nodes.items()}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

QUANTUM = Decimal(1).scaleb(-12)


def to_decimal_pair(probs):
    """Round to 12 decimal places so the two strings sum to exactly 1."""
    first = Decimal(repr(probs[0])).quantize(QUANTUM, rounding=ROUND_HALF_EVEN)
    first = min(max(first, Decimal(0)), Decimal(1))
    return first, Decimal(1) - first


def fmt(d):
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def write_strategy(path, average, iterations):
    labels = all_infosets()
    order = sorted(labels, key=lambda n: (len(n.split(":")[1]), n.split(":")[1], RANK[n[0]]))
    probs = {}
    for name in order:
        pair = to_decimal_pair(average[name])
        probs[name] = {label: fmt(p) for label, p in zip(labels[name], pair)}
    doc = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": "solve_kuhn.py: CFR+ (alternating updates, linear averaging), %d iterations" % iterations,
        "probs": probs,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Exact evaluation of a strategy file (independent of the solver's state)
# ---------------------------------------------------------------------------


def load_strategy(path):
    with open(path) as f:
        doc = json.load(f)
    labels = all_infosets()
    probs = doc["probs"]
    if set(probs) != set(labels):
        raise ValueError("info sets differ: %s" % sorted(set(probs) ^ set(labels)))
    strategy = {}
    for name, expected in labels.items():
        if set(probs[name]) != set(expected):
            raise ValueError("actions differ at %s" % name)
        values = {label: Fraction(probs[name][label]) for label in expected}
        if any(v < 0 for v in values.values()) or sum(values.values()) != 1:
            raise ValueError("bad distribution at %s" % name)
        strategy[name] = values
    return strategy


def expected_value_p0(strategy, history, cards):
    if is_terminal(history):
        return Fraction(payoff_p0(history, cards))
    dist = strategy[infoset(history, cards)]
    return sum(
        dist[label] * expected_value_p0(strategy, history + letter, cards)
        for letter, label in actions(history)
    )


def game_value_p0(strategy):
    return sum(DEAL_PROB * expected_value_p0(strategy, "", cards) for cards in DEALS)


def best_response_value(strategy, br_player):
    """Exact value of br_player's best response to the other player's strategy.

    Works on the best responder's information sets: at each one, the value of
    an action is summed over the opponent cards consistent with it, weighted
    by chance and the opponent's reach probability, and the best action is
    taken. Perfect recall makes this bottom-up maximization exact.
    """

    def value(own_card, history, opp_reach):
        # opp_reach: {opponent card: chance prob * opponent reach prob}
        if is_terminal(history):
            total = Fraction(0)
            for opp_card, weight in opp_reach.items():
                cards = (own_card, opp_card) if br_player == 0 else (opp_card, own_card)
                v = payoff_p0(history, cards)
                total += weight * (v if br_player == 0 else -v)
            return total
        if player_to_act(history) == br_player:
            return max(value(own_card, history + letter, opp_reach) for letter, _ in actions(history))
        total = Fraction(0)
        for letter, label in actions(history):
            new_reach = {
                opp_card: weight * strategy[opp_card + ":" + history][label]
                for opp_card, weight in opp_reach.items()
            }
            total += value(own_card, history + letter, new_reach)
        return total

    total = Fraction(0)
    for own_card in CARDS:
        opp_reach = {c: DEAL_PROB for c in CARDS if c != own_card}
        total += value(own_card, "", opp_reach)
    return total


def brute_force_best_response_value(strategy, br_player):
    """Cross-check: enumerate every pure strategy of br_player."""
    labels = all_infosets()
    own = [n for n in labels if player_to_act(n.split(":")[1]) == br_player]
    best = None
    for choice in itertools.product(range(2), repeat=len(own)):
        trial = dict(strategy)
        for name, pick in zip(own, choice):
            trial[name] = {label: Fraction(int(i == pick)) for i, label in enumerate(labels[name])}
        v = game_value_p0(trial)
        v = v if br_player == 0 else -v
        if best is None or v > best:
            best = v
    return best


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    path = "strategy.json"

    average = solve(iterations)
    write_strategy(path, average, iterations)

    strategy = load_strategy(path)  # evaluate what was actually written
    value = game_value_p0(strategy)
    br0 = best_response_value(strategy, 0)
    br1 = best_response_value(strategy, 1)
    assert br0 == brute_force_best_response_value(strategy, 0)
    assert br1 == brute_force_best_response_value(strategy, 1)
    exploitability = (br0 + br1) / 2

    print("iterations:            %d" % iterations)
    for name, dist in strategy.items():
        print("  %-5s %s" % (name, "  ".join("%s=%.6f" % (k, float(v)) for k, v in dist.items())))
    print("value to player 0:     %.12f  (known game value -1/18 = %.12f)" % (float(value), -1 / 18))
    print("BR value, player 0:    %.12f" % float(br0))
    print("BR value, player 1:    %.12f" % float(br1))
    print("exploitability:        %.3e" % float(exploitability))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Solve Kuhn poker with CFR+ (linear averaging) and write strategy.json.

Game model
----------
Cards are indexed J=0, Q=1, K=2. A deal is an ordered pair (p0_card, p1_card)
with distinct cards; the 6 deals are equally likely.

Histories are strings over the actions c (check), b (bet), f (fold), k (call).
Decision nodes contain only c and b, which is what information set names use:
the acting player's card, a colon, then the history. Fold and call only occur
as the final action of a history, so they never appear in an information set name.

Payoffs are net, from player 0's perspective:
  fold by a player       -> the folder loses the ante: -1 for P0 folding, +1 for P1 folding
  showdown after check-check -> winner +1 / loser -1
  showdown after a call  -> winner +2 / loser -2
"""
import itertools
import json
import os
import sys
import time

CARDS = ["J", "Q", "K"]
LABEL = {"c": "check", "b": "bet", "f": "fold", "k": "call"}
DEALS = [(a, b) for a in range(3) for b in range(3) if a != b]
GAME_NAME = "kuhn"
FORMAT = "bluff-strategy-v1"


def is_terminal(hist):
    return hist.endswith("f") or hist.endswith("k") or hist == "cc"


def actions(hist):
    """Actions available at a non-terminal history, as letters."""
    if hist.endswith("b"):
        return ("f", "k")
    return ("c", "b")


def player_to_act(hist):
    """Player index (0 or 1) to act after this history. Player 0 acts first."""
    return len(hist) % 2


def info_key(cards, hist):
    player = player_to_act(hist)
    return CARDS[cards[player]] + ":" + hist


def terminal_utility_p0(cards, hist):
    sign = 1 if cards[0] > cards[1] else -1
    if hist.endswith("f"):
        folder = player_to_act(hist[:-1])
        return -1.0 if folder == 0 else 1.0
    if hist.endswith("k"):
        return 2.0 * sign
    return 1.0 * sign  # check-check showdown


class Node:
    __slots__ = ("regret", "strategy_sum")

    def __init__(self, n):
        self.regret = [0.0] * n
        self.strategy_sum = [0.0] * n


def regret_match(regret):
    positive = [max(r, 0.0) for r in regret]
    total = sum(positive)
    if total > 0:
        return [p / total for p in positive]
    return [1.0 / len(regret)] * len(regret)


nodes = {}


def get_node(key, n):
    node = nodes.get(key)
    if node is None:
        node = nodes[key] = Node(n)
    return node


def cfr(cards, hist, reach, t):
    """One CFR+ pass over the tree. Returns the expected utility to player 0.

    reach[p] is the probability that player p plays to this history (chance
    included through the deal loop, which is exact, not sampled).
    """
    if is_terminal(hist):
        return terminal_utility_p0(cards, hist)

    player = player_to_act(hist)
    acts = actions(hist)
    node = get_node(info_key(cards, hist), len(acts))
    strat = regret_match(node.regret)

    # Linear averaging: iteration t gets weight t.
    for i in range(len(acts)):
        node.strategy_sum[i] += t * reach[player] * strat[i]

    utils = []
    for i, a in enumerate(acts):
        new_reach = list(reach)
        new_reach[player] *= strat[i]
        utils.append(cfr(cards, hist + a, new_reach, t))

    node_util = sum(s * u for s, u in zip(strat, utils))

    # Counterfactual regret, weighted by the opponent's reach. Sign converts
    # player 0's utilities into the acting player's utilities.
    sign = 1.0 if player == 0 else -1.0
    opp_reach = reach[1 - player]
    for i in range(len(acts)):
        instant = opp_reach * sign * (utils[i] - node_util)
        node.regret[i] = max(0.0, node.regret[i] + instant)  # CFR+ floor

    return node_util


def train(iterations):
    for t in range(1, iterations + 1):
        for cards in DEALS:
            cfr(cards, "", [1.0, 1.0], t)


def average_profile():
    """Average strategy for every information set: key -> {action letter: prob}."""
    profile = {}
    for key, node in nodes.items():
        hist = key.split(":", 1)[1]
        acts = actions(hist)
        total = sum(node.strategy_sum)
        if total > 0:
            probs = [s / total for s in node.strategy_sum]
        else:
            probs = [1.0 / len(acts)] * len(acts)
        profile[key] = dict(zip(acts, probs))
    return profile


def all_info_keys():
    keys = []
    for card in CARDS:
        for hist in ("", "c", "b", "cb"):
            keys.append(card + ":" + hist)
    return keys


def expected_utility_p0(cards, hist, profile):
    if is_terminal(hist):
        return terminal_utility_p0(cards, hist)
    key = info_key(cards, hist)
    probs = profile[key]
    return sum(probs[a] * expected_utility_p0(cards, hist + a, profile)
               for a in actions(hist))


def game_value_p0(profile):
    return sum(expected_utility_p0(cards, "", profile) for cards in DEALS) / len(DEALS)


def best_response_value(player, opp_profile):
    """Exact best response by enumerating every pure strategy of `player`.

    Each player has 6 information sets with 2 actions, so 2^6 = 64 pure strategies.
    Returns the best expected payoff to `player` against opp_profile.
    """
    keys = [k for k in all_info_keys() if player_to_act(k.split(":", 1)[1]) == player]
    action_sets = [actions(k.split(":", 1)[1]) for k in keys]
    sign = 1.0 if player == 0 else -1.0
    best = float("-inf")
    for choice in itertools.product(*action_sets):
        profile = dict(opp_profile)
        for key, acts, pick in zip(keys, action_sets, choice):
            profile[key] = {a: (1.0 if a == pick else 0.0) for a in acts}
        value = sign * game_value_p0(profile)
        if value > best:
            best = value
    return best


def exploitability(profile):
    br0 = best_response_value(0, profile)
    br1 = best_response_value(1, profile)
    return (br0 + br1) / 2.0, br0, br1


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    start = time.time()
    train(iterations)
    elapsed = time.time() - start

    profile = average_profile()
    expl, br0, br1 = exploitability(profile)
    value = game_value_p0(profile)

    probs = {}
    for key in all_info_keys():
        hist = key.split(":", 1)[1]
        p = profile[key]
        probs[key] = {LABEL[a]: repr(float(p[a])) for a in actions(hist)}

    out = {
        "format": FORMAT,
        "game": GAME_NAME,
        "producer": (f"CFR+ with linear averaging, {iterations} iterations over all 6 deals "
                     "(exact chance), solved by solve_kuhn.py"),
        "probs": probs,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"iterations: {iterations} ({elapsed:.1f}s)")
    print(f"best response value P0: {br0:.10f}")
    print(f"best response value P1: {br1:.10f}")
    print(f"exploitability: {expl:.10f}")
    print(f"game value to P0: {value:.10f} (known value -1/18 = {-1/18:.10f})")
    for key in all_info_keys():
        print(f"  {key:6s} " + "  ".join(f"{LABEL[a]}={profile[key][a]:.6f}"
                                         for a in actions(key.split(':', 1)[1])))
    print(f"wrote {path}")
    print(json.dumps({"exploitability": expl, "value_p0": value}))


if __name__ == "__main__":
    main()

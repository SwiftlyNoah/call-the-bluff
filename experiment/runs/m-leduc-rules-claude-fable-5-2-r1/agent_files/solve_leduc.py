#!/usr/bin/env python3
"""Leduc hold'em solver: CFR+ (alternating updates, linear averaging) on the public tree.

Writes the average strategy for both players to strategy.json and prints the
exploitability and game value of exactly what was written.

Usage: python3 solve_leduc.py [iterations] [output path]
"""
import json
import sys
import time

CARDS = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]
NC = len(CARDS)
RANK = [c // 2 for c in range(NC)]
RAISE_SIZE = {1: 2, 2: 4}
MAX_RAISES = 2
DEAL_PROB = 1.0 / (NC * (NC - 1))  # probability of each ordered pair of private cards
PUB_PROB = 1.0 / (NC - 2)  # probability of each public card given the private cards
SCALE = 10 ** 12  # written probabilities are exact multiples of 1/SCALE


def showdown_sign(i, j, pub):
    """+1 if private card i beats private card j given the public card, -1 if it loses, 0 if tied."""
    pair_i = RANK[i] == RANK[pub]
    pair_j = RANK[j] == RANK[pub]
    if pair_i != pair_j:
        return 1 if pair_i else -1
    if RANK[i] != RANK[j]:
        return 1 if RANK[i] > RANK[j] else -1
    return 0


# SIGN[pub][i][j], zero for impossible deals (shared cards).
SIGN = [
    [
        [0 if len({i, j, pub}) < 3 else showdown_sign(i, j, pub) for j in range(NC)]
        for i in range(NC)
    ]
    for pub in range(NC)
]


class Node:
    __slots__ = (
        "kind", "player", "actions", "children", "regret", "strat_sum", "pub",
        "amount", "loser", "names",
    )


def build(rnd, h1, h2, pub, contrib, raises, player, decisions):
    """Build the public subtree at a decision point and return its root."""
    node = Node()
    node.kind = "decision"
    node.player = player
    node.pub = pub
    hist = h1 if rnd == 1 else h2
    opp = 1 - player
    to_call = contrib[opp] - contrib[player]
    actions = []
    children = []

    if to_call > 0:
        fold = Node()
        fold.kind = "fold"
        fold.loser = player
        fold.amount = contrib[player]
        fold.pub = pub
        actions.append("fold")
        children.append(fold)

    # call (or check)
    called = list(contrib)
    called[player] = contrib[opp]
    if hist == "":
        # A check as the round's first action: the other player still acts.
        if rnd == 1:
            child = build(1, h1 + "c", h2, pub, called, raises, opp, decisions)
        else:
            child = build(2, h1, h2 + "c", pub, called, raises, opp, decisions)
    elif rnd == 1:
        child = Node()
        child.kind = "chance"
        child.pub = None
        child.children = [
            build(2, h1 + "c", "", p, called, 0, 0, decisions) for p in range(NC)
        ]
    else:
        child = Node()
        child.kind = "showdown"
        child.pub = pub
        child.amount = called[0]
    actions.append("call")
    children.append(child)

    if raises < MAX_RAISES:
        raised = list(contrib)
        raised[player] = contrib[opp] + RAISE_SIZE[rnd]
        if rnd == 1:
            child = build(1, h1 + "r", h2, pub, raised, raises + 1, opp, decisions)
        else:
            child = build(2, h1, h2 + "r", pub, raised, raises + 1, opp, decisions)
        actions.append("raise")
        children.append(child)

    node.actions = actions
    node.children = children
    n = len(actions)
    node.regret = [[0.0] * n for _ in range(NC)]
    node.strat_sum = [[0.0] * n for _ in range(NC)]
    if rnd == 1:
        node.names = [f"{CARDS[c]}:{h1}" for c in range(NC)]
    else:
        node.names = [
            None if c == pub else f"{CARDS[c]}|{CARDS[pub]}:{h1}/{h2}" for c in range(NC)
        ]
    decisions.append(node)
    return node


def terminal_values(node, opp_reach, me):
    """Counterfactual values for player `me` at a fold or showdown node, one per private card."""
    if node.kind == "fold":
        total = sum(opp_reach)
        signed = -node.amount if node.loser == me else node.amount
        return [signed * (total - opp_reach[i]) for i in range(NC)]
    sign = SIGN[node.pub]
    amount = node.amount
    return [
        amount * sum(s * r for s, r in zip(sign[i], opp_reach)) for i in range(NC)
    ]


def chance_values(node, recurse, own, opp):
    """Sum the children of the public-card chance node; `recurse(child, own, opp)` gives values."""
    values = [0.0] * NC
    for p, child in enumerate(node.children):
        own_p = list(own)
        own_p[p] = 0.0
        opp_p = [r * PUB_PROB for r in opp]
        opp_p[p] = 0.0
        child_values = recurse(child, own_p, opp_p)
        for i in range(NC):
            if i != p:
                values[i] += child_values[i]
    return values


def regret_matching(regret):
    total = 0.0
    for r in regret:
        if r > 0.0:
            total += r
    if total > 0.0:
        return [r / total if r > 0.0 else 0.0 for r in regret]
    n = len(regret)
    return [1.0 / n] * n


def cfr(node, own, opp, me, weight):
    """One CFR+ traversal for player `me`; returns counterfactual values per private card."""
    kind = node.kind
    if kind != "decision":
        if kind == "chance":
            return chance_values(node, lambda c, a, b: cfr(c, a, b, me, weight), own, opp)
        return terminal_values(node, opp, me)

    n = len(node.actions)
    strategy = [regret_matching(node.regret[c]) for c in range(NC)]
    if node.player != me:
        values = [0.0] * NC
        for a in range(n):
            child_opp = [opp[c] * strategy[c][a] for c in range(NC)]
            child_values = cfr(node.children[a], own, child_opp, me, weight)
            for i in range(NC):
                values[i] += child_values[i]
        return values

    action_values = []
    for a in range(n):
        child_own = [own[c] * strategy[c][a] for c in range(NC)]
        action_values.append(cfr(node.children[a], child_own, opp, me, weight))
    values = [0.0] * NC
    for c in range(NC):
        if c == node.pub:
            continue
        strat = strategy[c]
        v = 0.0
        for a in range(n):
            v += strat[a] * action_values[a][c]
        values[c] = v
        regret = node.regret[c]
        strat_sum = node.strat_sum[c]
        w = weight * own[c]
        for a in range(n):
            r = regret[a] + action_values[a][c] - v
            regret[a] = r if r > 0.0 else 0.0
            strat_sum[a] += w * strat[a]
    return values


def average_strategy(node, c):
    total = sum(node.strat_sum[c])
    n = len(node.actions)
    if total > 0.0:
        return [s / total for s in node.strat_sum[c]]
    return [1.0 / n] * n


def rounded_strategy(node, c):
    """The average strategy as integers out of SCALE that sum to exactly SCALE."""
    probs = average_strategy(node, c)
    units = [int(round(p * SCALE)) for p in probs]
    big = max(range(len(units)), key=lambda a: units[a])
    units[big] += SCALE - sum(units)
    return units


def evaluate(node, own, opp, me, policy, best_response):
    """Values for `me` when the opponent follows `policy`; `me` best-responds or follows it too."""
    kind = node.kind
    if kind != "decision":
        if kind == "chance":
            return chance_values(
                node, lambda c, a, b: evaluate(c, a, b, me, policy, best_response), own, opp
            )
        return terminal_values(node, opp, me)
    n = len(node.actions)
    strategy = policy[id(node)]
    if node.player != me:
        values = [0.0] * NC
        for a in range(n):
            child_opp = [opp[c] * strategy[c][a] for c in range(NC)]
            child_values = evaluate(node.children[a], own, child_opp, me, policy, best_response)
            for i in range(NC):
                values[i] += child_values[i]
        return values
    action_values = [
        evaluate(node.children[a], own, opp, me, policy, best_response) for a in range(n)
    ]
    if best_response:
        return [max(action_values[a][c] for a in range(n)) for c in range(NC)]
    return [sum(strategy[c][a] * action_values[a][c] for a in range(n)) for c in range(NC)]


def measure(root, policy):
    """Return (exploitability, value to player 0) of `policy`."""
    ones = [1.0] * NC
    deal = [DEAL_PROB] * NC
    br0 = sum(evaluate(root, ones, deal, 0, policy, True))
    br1 = sum(evaluate(root, ones, deal, 1, policy, True))
    value0 = sum(evaluate(root, ones, deal, 0, policy, False))
    return (br0 + br1) / 2.0, value0


def unit_string(units):
    return "0." + str(units).zfill(12) if units < SCALE else "1.0"


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"

    decisions = []
    root = build(1, "", "", None, [1, 1], 0, 0, decisions)
    ones = [1.0] * NC
    deal = [DEAL_PROB] * NC

    start = time.time()
    checkpoint = 1
    for t in range(1, iterations + 1):
        for me in (0, 1):
            cfr(root, ones, deal, me, float(t))
        if t == checkpoint or t == iterations:
            policy = {id(n): [average_strategy(n, c) for c in range(NC)] for n in decisions}
            expl, value0 = measure(root, policy)
            print(
                f"iter {t:6d}  exploitability {expl:.3e}  value_p0 {value0:+.8f}"
                f"  ({time.time() - start:.1f}s)",
                flush=True,
            )
            checkpoint *= 2

    # Write the rounded strategy, then measure exactly what was written.
    probs = {}
    written_policy = {}
    for node in decisions:
        rows = []
        for c in range(NC):
            if node.names[c] is None:
                rows.append([1.0 / len(node.actions)] * len(node.actions))
                continue
            units = rounded_strategy(node, c)
            rows.append([u / SCALE for u in units])
            probs[node.names[c]] = {
                action: unit_string(u) for action, u in zip(node.actions, units)
            }
        written_policy[id(node)] = rows
    document = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": f"solve_leduc.py: CFR+ with alternating updates and linear averaging, {iterations} iterations",
        "probs": dict(sorted(probs.items())),
    }
    with open(out_path, "w") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")

    expl, value0 = measure(root, written_policy)
    print(f"wrote {out_path}: {len(probs)} information sets")
    print(f"final exploitability {expl:.6e}")
    print(f"final value_p0 {value0:+.10f}")


if __name__ == "__main__":
    main()

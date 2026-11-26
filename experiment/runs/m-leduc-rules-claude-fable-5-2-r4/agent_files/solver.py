#!/usr/bin/env python3
"""Leduc hold'em solver: CFR+ (alternating updates, linear averaging) on the public tree.

Each public node holds one row of regrets per private card, so an information set is
(private card, public node). Values are passed around as length-6 vectors indexed by the
traversing player's private card, already weighted by chance and the opponent's reach.

Usage: python3 solver.py [iterations]   (writes strategy.json next to this file)
"""
import json
import os
import sys
import time

import numpy as np

RANKS = "JQK"
SUITS = "sh"
CARDS = [r + s for r in RANKS for s in SUITS]  # Js Jh Qs Qh Ks Kh
NUM_CARDS = len(CARDS)
RAISE_SIZE = (2, 4)
MAX_RAISES = 2
ACTION_LABEL = {"f": "fold", "c": "call", "r": "raise"}


def rank(card):
    return card // 2


def showdown_sign(h0, h1, pub):
    """+1 if player 0 wins the showdown, -1 if player 1 wins, 0 for a split."""
    r0, r1, rp = rank(h0), rank(h1), rank(pub)
    if r0 == rp and r1 != rp:
        return 1
    if r1 == rp and r0 != rp:
        return -1
    return (r0 > r1) - (r0 < r1)


class Node:
    __slots__ = ("kind", "player", "actions", "children", "payoff", "regret", "strat_sum",
                 "hist1", "hist2", "pub")


def build(rnd, hist1, hist2, pub, contrib, raises, player):
    """Build the public subtree. contrib = chips each player has put in so far."""
    node = Node()
    node.kind = "decision"
    node.player = player
    node.hist1, node.hist2, node.pub = hist1, hist2, pub
    opp = 1 - player
    owed = contrib[opp] - contrib[player]
    actions = []
    if owed > 0:
        actions.append("f")
    actions.append("c")
    if raises < MAX_RAISES:
        actions.append("r")
    node.actions = actions
    node.children = []
    node.regret = np.zeros((NUM_CARDS, len(actions)))
    node.strat_sum = np.zeros((NUM_CARDS, len(actions)))
    hist = hist1 if rnd == 0 else hist2
    for a in actions:
        h1 = hist1 + a if rnd == 0 else hist1
        h2 = hist2 + a if rnd == 1 else hist2
        if a == "f":
            child = terminal_fold(player, contrib, pub)
        elif a == "c":
            new = list(contrib)
            new[player] += owed
            if len(hist) == 0:
                # Opening check: the other player still has to act.
                child = build(rnd, h1, h2, pub, new, raises, opp)
            elif rnd == 0:
                child = chance_node(h1, new)
            else:
                child = terminal_showdown(new, pub)
        else:
            new = list(contrib)
            new[player] += owed + RAISE_SIZE[rnd]
            child = build(rnd, h1, h2, pub, new, raises + 1, opp)
        node.children.append(child)
    return node


def chance_node(hist1, contrib):
    node = Node()
    node.kind = "chance"
    node.children = [build(1, hist1, "", p, contrib, 0, 0) for p in range(NUM_CARDS)]
    return node


def deal_weight(h0, h1, pub):
    """Chance probability of the deal; pub is None before the public card is dealt."""
    if h0 == h1:
        return 0.0
    if pub is None:
        return 1.0 / 30.0
    if pub == h0 or pub == h1:
        return 0.0
    return 1.0 / 120.0


def terminal(value_fn, pub):
    """payoff[i][h, o]: chance-weighted payoff to player i holding h against o."""
    node = Node()
    node.kind = "terminal"
    m = np.zeros((NUM_CARDS, NUM_CARDS))  # m[h0, h1] = payoff to player 0
    for h0 in range(NUM_CARDS):
        for h1 in range(NUM_CARDS):
            w = deal_weight(h0, h1, pub)
            if w:
                m[h0, h1] = w * value_fn(h0, h1)
    node.payoff = (m, -m.T.copy())
    return node


def terminal_fold(folder, contrib, pub):
    v0 = -contrib[0] if folder == 0 else contrib[1]
    return terminal(lambda h0, h1: v0, pub)


def terminal_showdown(contrib, pub):
    assert contrib[0] == contrib[1]
    return terminal(lambda h0, h1: contrib[0] * showdown_sign(h0, h1, pub), pub)


def current_strategy(node):
    pos = node.regret  # CFR+ keeps regrets non-negative
    total = pos.sum(axis=1, keepdims=True)
    n = pos.shape[1]
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), 1.0 / n)


def average_strategy(node):
    total = node.strat_sum.sum(axis=1, keepdims=True)
    n = node.strat_sum.shape[1]
    return np.where(total > 0, node.strat_sum / np.where(total > 0, total, 1.0), 1.0 / n)


def cfr(node, u, r_own, r_opp, weight):
    """Counterfactual values for player u, one per private card of u."""
    if node.kind == "terminal":
        return node.payoff[u] @ r_opp
    if node.kind == "chance":
        v = np.zeros(NUM_CARDS)
        for child in node.children:
            v += cfr(child, u, r_own, r_opp, weight)
        return v
    s = current_strategy(node)
    if node.player == u:
        vals = np.empty_like(s)
        for i, child in enumerate(node.children):
            vals[:, i] = cfr(child, u, r_own * s[:, i], r_opp, weight)
        v = (s * vals).sum(axis=1)
        node.regret = np.maximum(node.regret + vals - v[:, None], 0.0)
        node.strat_sum += weight * r_own[:, None] * s
        return v
    v = np.zeros(NUM_CARDS)
    for i, child in enumerate(node.children):
        v += cfr(child, u, r_own, r_opp * s[:, i], weight)
    return v


def best_response(node, u, r_opp):
    """Best-response counterfactual values for u against the opponent's average strategy."""
    if node.kind == "terminal":
        return node.payoff[u] @ r_opp
    if node.kind == "chance":
        v = np.zeros(NUM_CARDS)
        for child in node.children:
            v += best_response(child, u, r_opp)
        return v
    if node.player == u:
        vals = [best_response(child, u, r_opp) for child in node.children]
        return np.max(np.stack(vals, axis=1), axis=1)
    s = average_strategy(node)
    v = np.zeros(NUM_CARDS)
    for i, child in enumerate(node.children):
        v += best_response(child, u, r_opp * s[:, i])
    return v


def expected_value(node, r0, r1):
    """Player 0's expected payoff when both play their average strategies."""
    if node.kind == "terminal":
        return float(r0 @ node.payoff[0] @ r1)
    if node.kind == "chance":
        return sum(expected_value(child, r0, r1) for child in node.children)
    s = average_strategy(node)
    total = 0.0
    for i, child in enumerate(node.children):
        if node.player == 0:
            total += expected_value(child, r0 * s[:, i], r1)
        else:
            total += expected_value(child, r0, r1 * s[:, i])
    return total


def exploitability(root):
    ones = np.ones(NUM_CARDS)
    br0 = best_response(root, 0, ones).sum()
    br1 = best_response(root, 1, ones).sum()
    return (br0 + br1) / 2.0, br0, br1


def decision_nodes(node, out):
    if node.kind == "decision":
        out.append(node)
    if node.kind != "terminal":
        for child in node.children:
            decision_nodes(child, out)
    return out


def exact_decimals(probs, digits=12):
    """Round to `digits` decimals so the printed decimals sum to exactly 1."""
    scale = 10 ** digits
    ints = [int(round(float(p) * scale)) for p in probs]
    ints[max(range(len(ints)), key=lambda i: ints[i])] += scale - sum(ints)
    assert sum(ints) == scale and min(ints) >= 0
    out = []
    for n in ints:
        out.append("1.0" if n == scale else "0." + str(n).zfill(digits))
    return out


def write_strategy(root, path, producer):
    probs = {}
    for node in decision_nodes(root, []):
        avg = average_strategy(node)
        for card in range(NUM_CARDS):
            if node.pub is None:
                name = f"{CARDS[card]}:{node.hist1}"
            elif card == node.pub:
                continue  # impossible: the public card cannot also be the private card
            else:
                name = f"{CARDS[card]}|{CARDS[node.pub]}:{node.hist1}/{node.hist2}"
            assert name not in probs
            dec = exact_decimals(avg[card])
            probs[name] = {ACTION_LABEL[a]: d for a, d in zip(node.actions, dec)}
    doc = {"format": "bluff-strategy-v1", "game": "leduc", "producer": producer, "probs": probs}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    return len(probs)


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    root = build(0, "", "", None, [1, 1], 0, 0)
    ones = np.ones(NUM_CARDS)
    start = time.time()
    checkpoints = {10 ** k for k in range(1, 7)} | {iterations}
    for t in range(1, iterations + 1):
        for u in (0, 1):
            cfr(root, u, ones, ones, float(t))
        if t in checkpoints or t % 5000 == 0:
            expl, br0, br1 = exploitability(root)
            print(f"iter {t:6d}  exploitability {expl:.3e}  br0 {br0:+.6f}  br1 {br1:+.6f}"
                  f"  value_p0 {expected_value(root, ones, ones):+.6f}"
                  f"  {time.time() - start:.1f}s", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy.json")
    count = write_strategy(
        root, out, f"solver.py: CFR+ with alternating updates and linear averaging, {iterations} iterations")
    print(f"wrote {out} with {count} information sets")


if __name__ == "__main__":
    main()

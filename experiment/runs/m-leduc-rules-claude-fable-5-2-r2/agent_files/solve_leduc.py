#!/usr/bin/env python3
"""Leduc hold'em solver: CFR+ (alternating updates, linear averaging) on the public tree.

Every public node holds arrays indexed by the acting player's private card (round 1: shape
(6,)) or by public card and private card (round 2: shape (6, 6)), so one traversal of the
small public betting tree covers all 120 deals at once.

Usage: python3 solve_leduc.py [iterations] [output path]
"""
import json
import sys
import time
from decimal import Decimal

import numpy as np

CARDS = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]
RANK = [0, 0, 1, 1, 2, 2]
N = len(CARDS)
ANTE = 1
RAISE_SIZE = (2, 4)
MAX_RAISES = 2
LETTER = {"fold": "f", "call": "c", "raise": "r"}

# ---------------------------------------------------------------------------------------
# Chance weights and showdown tables
# ---------------------------------------------------------------------------------------
# PAIR[i, j]: probability weight of (own card i, opponent card j) when no public card matters.
PAIR = np.zeros((N, N))
# DEAL[k, i, j]: probability of public card k, own card i, opponent card j.
DEAL = np.zeros((N, N, N))
# SHOW[k, i, j]: DEAL weight times +1 / 0 / -1 for own card i against opponent card j.
SHOW = np.zeros((N, N, N))
# VALID[k, i]: 1 when a player can hold card i while the public card is k.
VALID = np.ones((N, N)) - np.eye(N)


def showdown_sign(own, opp, public):
    if RANK[own] == RANK[public]:
        return 1
    if RANK[opp] == RANK[public]:
        return -1
    return (RANK[own] > RANK[opp]) - (RANK[own] < RANK[opp])


for i in range(N):
    for j in range(N):
        if i == j:
            continue
        PAIR[i, j] = 1.0 / (N * (N - 1))
        for k in range(N):
            if k == i or k == j:
                continue
            DEAL[k, i, j] = 1.0 / (N * (N - 1) * (N - 2))
            SHOW[k, i, j] = DEAL[k, i, j] * showdown_sign(i, j, k)


# ---------------------------------------------------------------------------------------
# Public tree
# ---------------------------------------------------------------------------------------
class Node:
    __slots__ = ("kind", "rnd", "hist", "player", "actions", "children", "pay0",
                 "regret", "strat_sum", "avg")


def build(rnd, hist1, hist2, contrib, raises, player):
    """Decision node of `player` in round `rnd` (0 or 1) with the given chip contributions."""
    node = Node()
    node.kind = "decision"
    node.rnd = rnd
    node.hist = (hist1, hist2)
    node.player = player
    opp = 1 - player
    acted = len(hist2 if rnd else hist1)
    node.actions = []
    node.children = []
    if contrib[player] < contrib[opp]:
        node.actions.append("fold")
        leaf = Node()
        leaf.kind = "fold"
        leaf.rnd = rnd
        # The folder loses what they put in; the other player wins exactly that.
        leaf.pay0 = -contrib[0] if player == 0 else contrib[1]
        node.children.append(leaf)
    # call (or check)
    node.actions.append("call")
    c2 = list(contrib)
    c2[player] = contrib[opp]
    if acted == 0:
        child = build(rnd, hist1 + ("c" if rnd == 0 else ""), hist2 + ("c" if rnd else ""),
                      c2, raises, opp)
    elif rnd == 0:
        child = Node()
        child.kind = "chance"
        child.rnd = 0
        child.children = [build(1, hist1 + "c", "", c2, 0, 0)]
    else:
        child = Node()
        child.kind = "showdown"
        child.rnd = 1
        child.pay0 = c2[0]  # both contributions are equal here
    node.children.append(child)
    if raises < MAX_RAISES:
        node.actions.append("raise")
        c2 = list(contrib)
        c2[player] = contrib[opp] + RAISE_SIZE[rnd]
        node.children.append(build(rnd, hist1 + ("r" if rnd == 0 else ""),
                                   hist2 + ("r" if rnd else ""), c2, raises + 1, opp))
    shape = (N, len(node.actions)) if rnd == 0 else (N, N, len(node.actions))
    node.regret = np.zeros(shape)
    node.strat_sum = np.zeros(shape)
    node.avg = None
    return node


def decision_nodes(node, out=None):
    if out is None:
        out = []
    if node.kind == "decision":
        out.append(node)
    if node.kind in ("decision", "chance"):
        for c in node.children:
            decision_nodes(c, out)
    return out


# ---------------------------------------------------------------------------------------
# CFR+
# ---------------------------------------------------------------------------------------
def current_strategy(node):
    r = node.regret  # kept non-negative by CFR+
    tot = r.sum(axis=-1, keepdims=True)
    n = r.shape[-1]
    return np.where(tot > 0, r / np.where(tot > 0, tot, 1.0), 1.0 / n)


def terminal_value(node, p, reach_opp):
    sign = 1.0 if p == 0 else -1.0
    if node.kind == "fold":
        if node.rnd == 0:
            return sign * node.pay0 * (PAIR @ reach_opp)
        return sign * node.pay0 * np.einsum("kij,kj->ki", DEAL, reach_opp)
    # showdown: SHOW is from the point of view of the hand on the i axis
    return node.pay0 * np.einsum("kij,kj->ki", SHOW, reach_opp)


def cfr(node, p, reach_own, reach_opp, weight):
    """Counterfactual values of player p for every private card (and public card)."""
    if node.kind == "chance":
        v = cfr(node.children[0], p, reach_own[None, :] * VALID, reach_opp[None, :] * VALID,
                weight)
        return v.sum(axis=0)
    if node.kind != "decision":
        return terminal_value(node, p, reach_opp)
    s = current_strategy(node)
    if node.player == p:
        vals = [cfr(c, p, reach_own * s[..., a], reach_opp, weight)
                for a, c in enumerate(node.children)]
        vals = np.stack(vals, axis=-1)
        v = (s * vals).sum(axis=-1)
        node.regret = np.maximum(node.regret + vals - v[..., None], 0.0)
        node.strat_sum += weight * reach_own[..., None] * s
        return v
    v = 0.0
    for a, c in enumerate(node.children):
        v = v + cfr(c, p, reach_own, reach_opp * s[..., a], weight)
    return v


def set_average(nodes):
    for node in nodes:
        tot = node.strat_sum.sum(axis=-1, keepdims=True)
        n = node.strat_sum.shape[-1]
        node.avg = np.where(tot > 0, node.strat_sum / np.where(tot > 0, tot, 1.0), 1.0 / n)


def best_response(node, p, reach_opp):
    """Best-response counterfactual values of p against the opponent's average strategy."""
    if node.kind == "chance":
        return best_response(node.children[0], p, reach_opp[None, :] * VALID).sum(axis=0)
    if node.kind != "decision":
        return terminal_value(node, p, reach_opp)
    if node.player == p:
        vals = np.stack([best_response(c, p, reach_opp) for c in node.children], axis=-1)
        return vals.max(axis=-1)
    v = 0.0
    for a, c in enumerate(node.children):
        v = v + best_response(c, p, reach_opp * node.avg[..., a])
    return v


def profile_value(node, reach0, reach1):
    """Expected payoff to player 0 when both play the average strategy."""
    if node.kind == "chance":
        return profile_value(node.children[0], reach0[None, :] * VALID, reach1[None, :] * VALID)
    if node.kind != "decision":
        return float((reach0 * terminal_value(node, 0, reach1)).sum())
    total = 0.0
    for a, c in enumerate(node.children):
        if node.player == 0:
            total += profile_value(c, reach0 * node.avg[..., a], reach1)
        else:
            total += profile_value(c, reach0, reach1 * node.avg[..., a])
    return total


def exploitability(root, nodes):
    set_average(nodes)
    ones = np.ones(N)
    br0 = float(best_response(root, 0, ones).sum())
    br1 = float(best_response(root, 1, ones).sum())
    return (br0 + br1) / 2.0, br0, br1


# ---------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------
QUANT = Decimal(1).scaleb(-15)


def prob_strings(probs):
    """Decimal strings that sum to exactly 1 (the largest entry absorbs the rounding)."""
    dec = [Decimal(repr(float(x))).quantize(QUANT) for x in probs]
    big = max(range(len(dec)), key=lambda a: dec[a])
    dec[big] = Decimal(1) - sum(d for a, d in enumerate(dec) if a != big)
    assert all(d >= 0 for d in dec) and sum(dec) == 1
    return [format(d, "f") for d in dec]


def strategy_table(nodes):
    probs = {}
    for node in nodes:
        h1, h2 = node.hist
        for i in range(N):
            if node.rnd == 0:
                name = f"{CARDS[i]}:{h1}"
                probs[name] = dict(zip(node.actions, prob_strings(node.avg[i])))
            else:
                for k in range(N):
                    if k == i:
                        continue
                    name = f"{CARDS[i]}|{CARDS[k]}:{h1}/{h2}"
                    probs[name] = dict(zip(node.actions, prob_strings(node.avg[k, i])))
    return probs


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    root = build(0, "", "", [ANTE, ANTE], 0, 0)
    nodes = decision_nodes(root)
    ones = np.ones(N)
    start = time.time()
    report = 1
    for t in range(1, iterations + 1):
        for p in (0, 1):
            cfr(root, p, ones, ones, float(t))
        if t == report or t == iterations:
            expl, br0, br1 = exploitability(root, nodes)
            print(f"iter {t:7d}  exploitability {expl:.3e}  br0 {br0:+.6f}  br1 {br1:+.6f}"
                  f"  {time.time() - start:6.1f}s", flush=True)
            report = min(report * 2, report + 5000)
    expl, br0, br1 = exploitability(root, nodes)
    value = profile_value(root, ones, ones)
    probs = strategy_table(nodes)
    doc = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": f"solve_leduc.py: CFR+ with alternating updates and linear averaging, "
                    f"{iterations} iterations",
        "probs": probs,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path}: {len(probs)} information sets")
    print(f"solver-side numbers (before rounding to 15 decimals): exploitability {expl:.6e}, "
          f"value to player 0 {value:+.9f}")


if __name__ == "__main__":
    main()

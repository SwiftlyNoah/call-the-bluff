#!/usr/bin/env python3
"""Leduc hold'em solver: CFR+ (regret matching+, alternating updates, linear averaging).

The betting tree is walked once per player per iteration. Every node carries arrays indexed by
the deal [player 0 card, player 1 card, public card], so all 120 deals are handled at once.

Usage: python3 solve_leduc.py [iterations] [output path]
"""
import json
import sys
import time

import numpy as np

CARDS = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]
N = len(CARDS)
RANK = [i // 2 for i in range(N)]
ANTE = 1
RAISE_SIZE = (2, 4)
MAX_RAISES = 2
LABEL = {"f": "fold", "c": "call", "r": "raise"}

# Chance weight of each deal [c0, c1, pub]: 1/120 when the three cards are distinct, else 0.
DEAL = np.zeros((N, N, N))
# Showdown result for player 0: +1 win, -1 loss, 0 split.
SHOW = np.zeros((N, N, N))
for a in range(N):
    for b in range(N):
        for p in range(N):
            if a == b or a == p or b == p:
                continue
            DEAL[a, b, p] = 1.0 / (N * (N - 1) * (N - 2))
            if RANK[a] == RANK[p]:
                SHOW[a, b, p] = 1.0
            elif RANK[b] == RANK[p]:
                SHOW[a, b, p] = -1.0
            elif RANK[a] > RANK[b]:
                SHOW[a, b, p] = 1.0
            elif RANK[a] < RANK[b]:
                SHOW[a, b, p] = -1.0


class Node:
    __slots__ = ("kind", "player", "rnd", "actions", "children", "r1", "r2", "util0",
                 "regret", "strat_sum")


def play_round(hist, size, contrib):
    """Replay one round's actions. Returns (status, player to act, raises made)."""
    player, raises = 0, 0
    for k, a in enumerate(hist):
        if a == "f":
            return "fold", player, raises
        if a == "r":
            contrib[player] = contrib[1 - player] + size
            raises += 1
        else:
            contrib[player] = contrib[1 - player]
            if k >= 1:
                return "over", player, raises
        player = 1 - player
    return "open", player, raises


def build(r1, r2):
    contrib = [ANTE, ANTE]
    status, player, raises = play_round(r1, RAISE_SIZE[0], contrib)
    rnd = 0
    if status == "over":
        rnd = 1
        status, player, raises = play_round(r2, RAISE_SIZE[1], contrib)
    node = Node()
    node.r1, node.r2, node.rnd = r1, r2, rnd
    if status == "fold":
        # `player` folded and loses what they put in.
        node.kind = "terminal"
        node.util0 = (-contrib[0] if player == 0 else contrib[1]) * np.ones((N, N, N))
        return node
    if status == "over":
        node.kind = "terminal"
        node.util0 = SHOW * contrib[0]
        return node
    node.kind = "decision"
    node.player = player
    acts = []
    if contrib[player] < contrib[1 - player]:
        acts.append("f")
    acts.append("c")
    if raises < MAX_RAISES:
        acts.append("r")
    node.actions = acts
    node.children = [build(r1 + a, r2) if rnd == 0 else build(r1, r2 + a) for a in acts]
    shape = (N, len(acts)) if rnd == 0 else (N, N, len(acts))
    node.regret = np.zeros(shape)
    node.strat_sum = np.zeros(shape)
    return node


def expand(node, per_infoset):
    """Per-infoset array (trailing axes dropped) -> array broadcastable over deals."""
    if node.rnd == 0:
        return per_infoset[:, None, None] if node.player == 0 else per_infoset[None, :, None]
    return per_infoset[:, None, :] if node.player == 0 else per_infoset[None, :, :]


def collapse(node, deals):
    """Deal array -> per-infoset sums for the player acting at node."""
    if node.rnd == 0:
        return deals.sum(axis=(1, 2)) if node.player == 0 else deals.sum(axis=(0, 2))
    return deals.sum(axis=1) if node.player == 0 else deals.sum(axis=0)


def own_reach(node, reach):
    full = np.broadcast_to(reach, (N, N, N))
    if node.rnd == 0:
        return full[:, 0, 0] if node.player == 0 else full[0, :, 0]
    return full[:, 0, :] if node.player == 0 else full[0, :, :]


def normalize(x):
    total = x.sum(axis=-1, keepdims=True)
    k = x.shape[-1]
    return np.where(total > 0, x / np.where(total > 0, total, 1.0), 1.0 / k)


def cfr(node, i, reach_i, reach_o, weight):
    """Returns, per deal, (chance * opponent reach) * expected payoff to player i."""
    if node.kind == "terminal":
        return reach_o * (node.util0 if i == 0 else -node.util0)
    sigma = normalize(node.regret)
    if node.player != i:
        total = 0.0
        for k, child in enumerate(node.children):
            total = total + cfr(child, i, reach_i, reach_o * expand(node, sigma[..., k]), weight)
        return total
    vals = [cfr(child, i, reach_i * expand(node, sigma[..., k]), reach_o, weight)
            for k, child in enumerate(node.children)]
    value = 0.0
    for k, v in enumerate(vals):
        value = value + expand(node, sigma[..., k]) * v
    cf_node = collapse(node, value)
    node.strat_sum += weight * own_reach(node, reach_i)[..., None] * sigma
    for k, v in enumerate(vals):
        node.regret[..., k] = np.maximum(node.regret[..., k] + collapse(node, v) - cf_node, 0.0)
    return value


def best_response(node, i, reach_o, policy):
    """Per-deal value to player i of a best response against `policy` (opponent's strategy)."""
    if node.kind == "terminal":
        return reach_o * (node.util0 if i == 0 else -node.util0)
    if node.player != i:
        sigma = policy[id(node)]
        total = 0.0
        for k, child in enumerate(node.children):
            total = total + best_response(child, i, reach_o * expand(node, sigma[..., k]), policy)
        return total
    vals = [np.broadcast_to(best_response(child, i, reach_o, policy), (N, N, N))
            for child in node.children]
    cf = np.stack([collapse(node, v) for v in vals], axis=-1)
    best = cf.argmax(axis=-1)
    out = 0.0
    for k, v in enumerate(vals):
        out = out + expand(node, (best == k).astype(float)) * v
    return out


def profile_value(node, reach, policy):
    """Per-deal expected payoff to player 0 when both follow `policy`."""
    if node.kind == "terminal":
        return reach * node.util0
    sigma = policy[id(node)]
    total = 0.0
    for k, child in enumerate(node.children):
        total = total + profile_value(child, reach * expand(node, sigma[..., k]), policy)
    return total


def decision_nodes(node, out=None):
    out = [] if out is None else out
    if node.kind == "decision":
        out.append(node)
        for child in node.children:
            decision_nodes(child, out)
    return out


def average_policy(nodes):
    return {id(n): normalize(n.strat_sum) for n in nodes}


def exploitability(root, policy):
    br0 = float(np.sum(best_response(root, 0, DEAL, policy)))
    br1 = float(np.sum(best_response(root, 1, DEAL, policy)))
    return (br0 + br1) / 2.0, br0, br1


UNITS = 10 ** 12


def quantize(probs):
    """Probabilities -> decimal strings with 12 places that sum to exactly 1."""
    units = [int(round(float(p) * UNITS)) for p in probs]
    top = max(range(len(units)), key=lambda k: units[k])
    units[top] += UNITS - sum(units)
    assert all(0 <= u <= UNITS for u in units) and sum(units) == UNITS
    return ["1.000000000000" if u == UNITS else "0.%012d" % u for u in units]


def export(nodes, policy, path, producer):
    probs = {}
    for n in nodes:
        sigma = policy[id(n)]
        for card in range(N):
            if n.rnd == 0:
                name = "%s:%s" % (CARDS[card], n.r1)
                strs = quantize(sigma[card])
                probs[name] = {LABEL[a]: s for a, s in zip(n.actions, strs)}
                continue
            for pub in range(N):
                if pub == card:
                    continue
                name = "%s|%s:%s/%s" % (CARDS[card], CARDS[pub], n.r1, n.r2)
                strs = quantize(sigma[card, pub])
                probs[name] = {LABEL[a]: s for a, s in zip(n.actions, strs)}
    doc = {"format": "bluff-strategy-v1", "game": "leduc", "producer": producer, "probs": probs}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    return len(probs)


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    root = build("", "")
    nodes = decision_nodes(root)
    ones = np.ones((1, 1, 1))
    start = time.time()
    report = 1
    for t in range(1, iterations + 1):
        for i in (0, 1):
            cfr(root, i, ones, DEAL, float(t))
        if t == report or t == iterations:
            policy = average_policy(nodes)
            expl, br0, br1 = exploitability(root, policy)
            value = float(np.sum(profile_value(root, DEAL, policy)))
            print("iter %7d  exploitability %.3e  br0 %+.8f  br1 %+.8f  value_p0 %+.8f  %.1fs"
                  % (t, expl, br0, br1, value, time.time() - start), flush=True)
            report *= 2
    policy = average_policy(nodes)
    count = export(nodes, policy, path,
                   "solve_leduc.py: CFR+ (alternating updates, linear averaging), %d iterations"
                   % iterations)
    print("wrote %d information sets to %s" % (count, path))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent exact check of strategy.json against game.json.

Uses only the standard library and exact rational arithmetic, and shares no
code with solver.py. Prints the value of the strategy profile and both
best-response values.
"""
import json
import sys
from fractions import Fraction

game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
strat_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
game = json.load(open(game_path))
strat = json.load(open(strat_path))
nodes = {n["id"]: n for n in game["nodes"]}
root = game["root"]

# Format checks.
assert strat["format"] == "bluff-strategy-v1", strat["format"]
assert strat["game"] == game["name"], (strat["game"], game["name"])
infosets = {}
for n in nodes.values():
    if n["kind"] == "decision":
        infosets.setdefault(n["infoset"], []).append(n)
probs = {k: {a: Fraction(s) for a, s in v.items()} for k, v in strat["probs"].items()}
assert set(probs) == set(infosets), "infoset names differ"
for name, members in infosets.items():
    labels = [a["label"] for a in members[0]["actions"]]
    assert set(probs[name]) == set(labels), name
    assert all(isinstance(s, str) for s in strat["probs"][name].values()), name
    assert all(p >= 0 for p in probs[name].values()), name
    assert sum(probs[name].values()) == 1, (name, sum(probs[name].values()))

# Tree checks: every non-root node has exactly one parent, all reachable.
parents = {}
for n in nodes.values():
    for e in n.get("outcomes", []) + n.get("actions", []):
        assert e["child"] not in parents
        parents[e["child"]] = n["id"]
assert root not in parents and len(parents) == len(nodes) - 1


def edges(n):
    """(probability or None, child id, label) for each move at node n."""
    if n["kind"] == "chance":
        return [(Fraction(o["prob"]), o["child"], o["label"]) for o in n["outcomes"]]
    return [(probs[n["infoset"]][a["label"]], a["child"], a["label"]) for a in n["actions"]]


def profile_value(i):
    n = nodes[i]
    if n["kind"] == "terminal":
        return Fraction(n["payoff_p0"])
    return sum(p * profile_value(c) for p, c, _ in edges(n))


def best_response(p):
    """Exact value to player p of a best response to the other player's strategy."""
    sign = 1 if p == 0 else -1
    # Reach probability of each node counting only chance and the opponent.
    reach = {root: Fraction(1)}
    stack = [root]
    while stack:
        i = stack.pop()
        n = nodes[i]
        if n["kind"] == "terminal":
            continue
        for pr, c, _ in edges(n):
            mine = n["kind"] == "decision" and n["player"] == p
            reach[c] = reach[i] if mine else reach[i] * pr
            stack.append(c)
    chosen = {}
    cache = {}

    def value(i):
        if i in cache:
            return cache[i]
        n = nodes[i]
        if n["kind"] == "terminal":
            r = sign * Fraction(n["payoff_p0"])
        elif n["kind"] == "decision" and n["player"] == p:
            label = pick(n["infoset"])
            r = value(next(a["child"] for a in n["actions"] if a["label"] == label))
        else:
            r = sum(pr * value(c) for pr, c, _ in edges(n))
        cache[i] = r
        return r

    def pick(name):
        if name not in chosen:
            best = None
            for a in infosets[name][0]["actions"]:
                total = Fraction(0)
                for h in infosets[name]:
                    child = next(x["child"] for x in h["actions"] if x["label"] == a["label"])
                    total += reach[h["id"]] * value(child)
                if best is None or total > best[0]:
                    best = (total, a["label"])
            chosen[name] = best[1]
        return chosen[name]

    return value(root)


v = profile_value(root)
b0 = best_response(0)
b1 = best_response(1)
assert b0 >= v and b1 >= -v
print(json.dumps({
    "infosets": len(infosets),
    "value_p0": float(v),
    "br_value_p0": float(b0),
    "br_value_p1": float(b1),
    "exploitability": float((b0 + b1) / 2),
}))

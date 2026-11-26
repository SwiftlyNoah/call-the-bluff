#!/usr/bin/env python3
"""CFR+ solver for a two-player zero-sum extensive-form game read from game.json.

Writes the average strategy to strategy.json and reports exploitability (exact best responses)
and the expected value for player 0.
"""
import json
import sys
from fractions import Fraction

ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000


def load(path="game.json"):
    g = json.load(open(path))
    nodes = {n["id"]: n for n in g["nodes"]}
    for n in nodes.values():
        if n["kind"] == "chance":
            n["_probs"] = [float(Fraction(o["prob"])) for o in n["outcomes"]]
            n["_children"] = [o["child"] for o in n["outcomes"]]
        elif n["kind"] == "decision":
            n["_children"] = [a["child"] for a in n["actions"]]
            n["_labels"] = [a["label"] for a in n["actions"]]
        elif n["kind"] == "terminal":
            n["_u"] = float(Fraction(n["payoff_p0"]))
        else:
            raise ValueError(f"unknown node kind {n['kind']}")
    infosets = {}
    for n in nodes.values():
        if n["kind"] == "decision":
            key = n["infoset"]
            if key in infosets:
                p, labels = infosets[key]
                if p != n["player"] or labels != n["_labels"]:
                    raise ValueError(f"inconsistent infoset {key}")
            else:
                infosets[key] = (n["player"], n["_labels"])
    return g, nodes, infosets


def main():
    g, nodes, infosets = load()
    root = g["root"]
    regret = {k: [0.0] * len(v[1]) for k, v in infosets.items()}
    avg = {k: [0.0] * len(v[1]) for k, v in infosets.items()}

    def current(k):
        r = [max(x, 0.0) for x in regret[k]]
        s = sum(r)
        return [x / s for x in r] if s > 0 else [1.0 / len(r)] * len(r)

    cur = {}

    def cfr(nid, traverser, reach_self, reach_other, weight):
        """Returns value to `traverser`. reach_other includes chance."""
        n = nodes[nid]
        if n["kind"] == "terminal":
            return n["_u"] if traverser == 0 else -n["_u"]
        if n["kind"] == "chance":
            return sum(p * cfr(c, traverser, reach_self, reach_other * p, weight)
                       for p, c in zip(n["_probs"], n["_children"]))
        k = n["infoset"]
        sigma = cur[k]
        if n["player"] == traverser:
            vals = [cfr(c, traverser, reach_self * s, reach_other, weight)
                    for s, c in zip(sigma, n["_children"])]
            v = sum(s * x for s, x in zip(sigma, vals))
            r = regret[k]
            for i in range(len(r)):
                r[i] += reach_other * (vals[i] - v)
            for i in range(len(sigma)):
                avg[k][i] += weight * reach_self * sigma[i]
            return v
        return sum(s * cfr(c, traverser, reach_self, reach_other * s, weight)
                   for s, c in zip(sigma, n["_children"]))

    for t in range(1, ITERATIONS + 1):
        for p in (0, 1):  # alternating updates
            cur = {k: current(k) for k in regret}  # fixed strategy for this traversal
            cfr(root, p, 1.0, 1.0, float(t))  # linear averaging (CFR+)
            for k in regret:
                if infosets[k][0] == p:
                    regret[k] = [max(x, 0.0) for x in regret[k]]

    strategy = {}
    for k, s in avg.items():
        tot = sum(s)
        strategy[k] = [x / tot for x in s] if tot > 0 else [1.0 / len(s)] * len(s)

    ev = expected_value(nodes, root, strategy)
    br0 = best_response_value(nodes, root, infosets, strategy, 0)
    br1 = best_response_value(nodes, root, infosets, strategy, 1)
    expl = (br0 + br1) / 2

    out = {
        "format": "bluff-strategy-v1",
        "game": g.get("name", "kuhn"),
        "producer": f"solve.py: CFR+ (alternating, linear averaging), {ITERATIONS} iterations",
        "probs": {k: {lab: repr(p) for lab, p in zip(infosets[k][1], strategy[k])}
                  for k in sorted(strategy)},
    }
    json.dump(out, open("strategy.json", "w"), indent=1)
    for k in sorted(strategy):
        print(k, dict(zip(infosets[k][1], [round(x, 4) for x in strategy[k]])))
    print(f"value_p0={ev!r} br0={br0!r} br1={br1!r} exploitability={expl!r}")


def expected_value(nodes, nid, strat):
    n = nodes[nid]
    if n["kind"] == "terminal":
        return n["_u"]
    probs = n["_probs"] if n["kind"] == "chance" else strat[n["infoset"]]
    return sum(p * expected_value(nodes, c, strat) for p, c in zip(probs, n["_children"]))


def best_response_value(nodes, root, infosets, strat, br):
    """Exact best-response value for player `br` (in br's own payoff) against strat."""
    # Collect nodes of br's infosets with depth and opponent*chance reach.
    members, depth = {}, {}

    def walk(nid, reach, d):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return
        if n["kind"] == "chance":
            for p, c in zip(n["_probs"], n["_children"]):
                walk(c, reach * p, d + 1)
            return
        k = n["infoset"]
        if n["player"] == br:
            members.setdefault(k, []).append((nid, reach))
            depth[k] = max(depth.get(k, 0), d)
            for c in n["_children"]:
                walk(c, reach, d + 1)
        else:
            for p, c in zip(strat[k], n["_children"]):
                walk(c, reach * p, d + 1)

    walk(root, 1.0, 0)
    policy = {}

    def val(nid):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return n["_u"] if br == 0 else -n["_u"]
        if n["kind"] == "chance":
            return sum(p * val(c) for p, c in zip(n["_probs"], n["_children"]))
        k = n["infoset"]
        if n["player"] == br:
            return val(n["_children"][policy[k]])
        return sum(p * val(c) for p, c in zip(strat[k], n["_children"]))

    # Deepest infosets first; with perfect recall, children only reach deeper infosets.
    for k in sorted(members, key=lambda k: -depth[k]):
        nacts = len(infosets[k][1])
        scores = [sum(r * val(nodes[nid]["_children"][a]) for nid, r in members[k])
                  for a in range(nacts)]
        policy[k] = max(range(nacts), key=lambda a: scores[a])
    return val(root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CFR+ solver for a two-player zero-sum extensive-form game in bluff-game-v1 format."""
import json, sys
from fractions import Fraction

def load(path):
    g = json.load(open(path))
    nodes = {n["id"]: n for n in g["nodes"]}
    for n in nodes.values():
        if n["kind"] == "chance":
            n["_p"] = [float(Fraction(o["prob"])) for o in n["outcomes"]]
            n["_c"] = [o["child"] for o in n["outcomes"]]
        elif n["kind"] == "decision":
            n["_a"] = [a["label"] for a in n["actions"]]
            n["_c"] = [a["child"] for a in n["actions"]]
        else:
            n["_u"] = float(Fraction(n["payoff_p0"]))
    infosets = {}
    for n in nodes.values():
        if n["kind"] == "decision":
            key = n["infoset"]
            if key in infosets:
                assert infosets[key] == (n["player"], n["_a"]), f"inconsistent infoset {key}"
            else:
                infosets[key] = (n["player"], n["_a"])
    return g, nodes, infosets

def cfr_plus(nodes, root, infosets, iters):
    regret = {k: [0.0] * len(a) for k, (_, a) in infosets.items()}
    avg = {k: [0.0] * len(a) for k, (_, a) in infosets.items()}

    def cur(k):
        r = regret[k]; s = sum(r)
        return [x / s for x in r] if s > 0 else [1.0 / len(r)] * len(r)

    def walk(nid, i, pi_i, pi_o, t):
        # returns utility for player i; updates player i's regrets (alternating updates)
        n = nodes[nid]
        if n["kind"] == "terminal":
            return n["_u"] if i == 0 else -n["_u"]
        if n["kind"] == "chance":
            return sum(p * walk(c, i, pi_i, pi_o * p, t) for p, c in zip(n["_p"], n["_c"]))
        k = n["infoset"]; sig = cur(k)
        if n["player"] == i:
            vals = [walk(c, i, pi_i * s, pi_o, t) for s, c in zip(sig, n["_c"])]
            v = sum(s * x for s, x in zip(sig, vals))
            r = regret[k]
            for a in range(len(r)):
                r[a] = max(0.0, r[a] + pi_o * (vals[a] - v))
            return v
        else:
            # opponent node: accumulate opponent average strategy (linear weighting)
            av = avg[k]
            for a in range(len(av)):
                av[a] += t * pi_o * sig[a]
            return sum(s * walk(c, i, pi_i, pi_o * s, t) for s, c in zip(sig, n["_c"]))

    for t in range(1, iters + 1):
        for i in (0, 1):
            walk(root, i, 1.0, 1.0, t)
    out = {}
    for k, (_, acts) in infosets.items():
        s = sum(avg[k])
        out[k] = [x / s for x in avg[k]] if s > 0 else [1.0 / len(acts)] * len(acts)
    return out

def expected_value(nodes, nid, strat):
    n = nodes[nid]
    if n["kind"] == "terminal":
        return n["_u"]
    probs = n["_p"] if n["kind"] == "chance" else strat[n["infoset"]]
    return sum(p * expected_value(nodes, c, strat) for p, c in zip(probs, n["_c"]))

def best_response_value(nodes, root, infosets, strat, br):
    """Value to player br of a best response (respecting information sets) against strat."""
    members = {}  # infoset -> list of (node id, reach prob of chance*opponent)
    def collect(nid, reach):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return
        if n["kind"] == "chance":
            for p, c in zip(n["_p"], n["_c"]):
                collect(c, reach * p)
            return
        if n["player"] == br:
            members.setdefault(n["infoset"], []).append((nid, reach))
            for c in n["_c"]:
                collect(c, reach)
        else:
            for p, c in zip(strat[n["infoset"]], n["_c"]):
                collect(c, reach * p)
    collect(root, 1.0)
    choice, memo = {}, {}
    def val(nid):  # value to br at node, given br's best choices below
        if nid in memo:
            return memo[nid]
        n = nodes[nid]
        if n["kind"] == "terminal":
            v = n["_u"] if br == 0 else -n["_u"]
        elif n["kind"] == "chance":
            v = sum(p * val(c) for p, c in zip(n["_p"], n["_c"]))
        elif n["player"] != br:
            v = sum(p * val(c) for p, c in zip(strat[n["infoset"]], n["_c"]))
        else:
            k = n["infoset"]
            if k not in choice:
                tot = [sum(r * val(nodes[m]["_c"][a]) for m, r in members[k]) for a in range(len(n["_c"]))]
                choice[k] = max(range(len(tot)), key=tot.__getitem__)
            v = val(n["_c"][choice[k]])
        memo[nid] = v
        return v
    return val(root)

def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    g, nodes, infosets = load(game_path)
    root = g["root"]
    strat = cfr_plus(nodes, root, infosets, iters)
    v0 = expected_value(nodes, root, strat)
    br0 = best_response_value(nodes, root, infosets, strat, 0)
    br1 = best_response_value(nodes, root, infosets, strat, 1)
    expl = (br0 + br1) / 2
    json.dump({
        "format": "bluff-strategy-v1",
        "game": g.get("name", "game"),
        "producer": f"CFR+ (alternating updates, linear averaging), {iters} iterations, solve.py",
        "probs": {k: {a: repr(p) for a, p in zip(infosets[k][1], strat[k])} for k in sorted(infosets)},
    }, open(out_path, "w"), indent=1)
    print(f"value_p0={v0!r} br_p0={br0!r} br_p1={br1!r} exploitability={expl!r}")
    for k in sorted(strat):
        print(k, {a: round(p, 4) for a, p in zip(infosets[k][1], strat[k])})

if __name__ == "__main__":
    main()

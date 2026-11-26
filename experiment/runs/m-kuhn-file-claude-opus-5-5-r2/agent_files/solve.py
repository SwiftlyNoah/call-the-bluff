#!/usr/bin/env python3
"""CFR+ solver for games in bluff-game-v1 format. Reads game.json, writes strategy.json."""
import json, sys
from fractions import Fraction

def num(s): return float(Fraction(s))

def load(path):
    g = json.load(open(path))
    nodes = {n["id"]: n for n in g["nodes"]}
    infosets = {}  # name -> (player, [labels])
    for n in g["nodes"]:
        if n["kind"] == "decision":
            labels = [a["label"] for a in n["actions"]]
            prev = infosets.setdefault(n["infoset"], (n["player"], labels))
            assert prev == (n["player"], labels), f"inconsistent infoset {n['infoset']}"
        elif n["kind"] == "chance":
            n["_p"] = [num(o["prob"]) for o in n["outcomes"]]
        elif n["kind"] == "terminal":
            n["_u"] = num(n["payoff_p0"])
    return g, nodes, infosets

def solve(nodes, root, infosets, iters):
    regret = {I: [0.0] * len(a) for I, (_, a) in infosets.items()}
    avg = {I: [0.0] * len(a) for I, (_, a) in infosets.items()}

    def regret_matching(I):
        r = [max(x, 0.0) for x in regret[I]]; s = sum(r)
        return [x / s for x in r] if s > 0 else [1.0 / len(r)] * len(r)

    cur = {}

    def cfr(nid, trav, pi_trav, pi_other, w):
        # returns value to player `trav`
        n = nodes[nid]
        k = n["kind"]
        if k == "terminal":
            return n["_u"] if trav == 0 else -n["_u"]
        if k == "chance":
            return sum(p * cfr(o["child"], trav, pi_trav, pi_other * p, w)
                       for p, o in zip(n["_p"], n["outcomes"]))
        I = n["infoset"]; sigma = cur[I]
        if n["player"] == trav:
            vals = [cfr(a["child"], trav, pi_trav * s, pi_other, w) for s, a in zip(sigma, n["actions"])]
            v = sum(s * x for s, x in zip(sigma, vals))
            r = regret[I]
            for i in range(len(r)):
                r[i] += pi_other * (vals[i] - v)
            # accumulate average strategy weighted by own reach (linear weights)
            a_ = avg[I]
            for i in range(len(a_)):
                a_[i] += w * pi_trav * sigma[i]
            return v
        return sum(s * cfr(a["child"], trav, pi_trav, pi_other * s, w) for s, a in zip(sigma, n["actions"]))

    for t in range(1, iters + 1):
        for p in (0, 1):
            cur = {I: regret_matching(I) for I in infosets}
            cfr(root, p, 1.0, 1.0, t)
            for I, (pl, _) in infosets.items():
                if pl == p:
                    regret[I] = [max(x, 0.0) for x in regret[I]]  # CFR+
    out = {}
    for I, a in avg.items():
        s = sum(a)
        out[I] = [x / s for x in a] if s > 0 else [1.0 / len(a)] * len(a)
    return out

def evaluate(nodes, root, infosets, strat):
    """Return (value to p0 under strat, BR value p0, BR value p1)."""
    def ev(nid):
        n = nodes[nid]
        if n["kind"] == "terminal": return n["_u"]
        if n["kind"] == "chance":
            return sum(p * ev(o["child"]) for p, o in zip(n["_p"], n["outcomes"]))
        return sum(s * ev(a["child"]) for s, a in zip(strat[n["infoset"]], n["actions"]))

    def br(brp):
        # reach of each node from chance and opponent only
        reach, members = {}, {}
        def walk(nid, r):
            n = nodes[nid]; reach[nid] = r
            if n["kind"] == "chance":
                for p, o in zip(n["_p"], n["outcomes"]): walk(o["child"], r * p)
            elif n["kind"] == "decision":
                if n["player"] == brp:
                    members.setdefault(n["infoset"], []).append(nid)
                    for a in n["actions"]: walk(a["child"], r)
                else:
                    for s, a in zip(strat[n["infoset"]], n["actions"]): walk(a["child"], r * s)
        walk(root, 1.0)
        choice, memo = {}, {}
        def val(nid):  # value to brp, not weighted by reach
            if nid in memo: return memo[nid]
            n = nodes[nid]
            if n["kind"] == "terminal": v = n["_u"] if brp == 0 else -n["_u"]
            elif n["kind"] == "chance":
                v = sum(p * val(o["child"]) for p, o in zip(n["_p"], n["outcomes"]))
            elif n["player"] == brp:
                I = n["infoset"]
                if I not in choice:
                    k = len(n["actions"])
                    tot = [sum(reach[h] * val(nodes[h]["actions"][i]["child"]) for h in members[I]) for i in range(k)]
                    choice[I] = max(range(k), key=lambda i: tot[i])
                v = val(n["actions"][choice[I]]["child"])
            else:
                v = sum(s * val(a["child"]) for s, a in zip(strat[n["infoset"]], n["actions"]))
            memo[nid] = v
            return v
        return val(root)
    return ev(root), br(0), br(1)

def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    g, nodes, infosets = load("game.json")
    sys.setrecursionlimit(100000)
    strat = solve(nodes, g["root"], infosets, iters)
    v, b0, b1 = evaluate(nodes, g["root"], infosets, strat)
    expl = (b0 + b1) / 2
    json.dump({
        "format": "bluff-strategy-v1",
        "game": g.get("name", "kuhn"),
        "producer": f"solve.py: CFR+ (alternating updates, linear averaging), {iters} iterations",
        "probs": {I: {lab: repr(p) for lab, p in zip(infosets[I][1], ps)} for I, ps in strat.items()},
    }, open("strategy.json", "w"), indent=1)
    print(f"value_p0={v!r} br_p0={b0!r} br_p1={b1!r} exploitability={expl!r}")
    for I, ps in sorted(strat.items()): print(I, [round(p, 4) for p in ps])

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent exact check of strategy.json against game.json.

Shares no code with solve.py. Uses exact rational arithmetic and a plain
recursive best response, so the numbers it prints are exact for the strategy
file as written (then shown as decimals).

Usage: python3 verify.py [game.json] [strategy.json]
"""
import json
import sys
from fractions import Fraction

sys.setrecursionlimit(100000)


def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    strat_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    g = json.load(open(game_path))
    s = json.load(open(strat_path))
    nodes = {n["id"]: n for n in g["nodes"]}
    root = g["root"]

    assert s["format"] == "bluff-strategy-v1", "bad strategy format"
    assert s["game"] == g["name"], "game name mismatch"

    # ---- format checks: every infoset, every action, sums to exactly 1 ----
    infosets = {}
    for n in nodes.values():
        if n["kind"] == "decision":
            infosets.setdefault(n["infoset"], []).append(n)
    probs = {}
    for name, members in infosets.items():
        labels = [a["label"] for a in members[0]["actions"]]
        assert name in s["probs"], f"missing infoset {name!r}"
        entry = s["probs"][name]
        assert set(entry) == set(labels), f"action mismatch at {name!r}"
        p = {a: Fraction(entry[a]) for a in labels}
        assert all(x >= 0 for x in p.values()), f"negative probability at {name!r}"
        assert sum(p.values()) == 1, f"probabilities at {name!r} sum to {sum(p.values())}"
        probs[name] = p
    extra = set(s["probs"]) - set(infosets)
    assert not extra, f"unknown infosets in strategy: {sorted(extra)[:5]}"

    # ---- expected value to player 0 when both follow the strategy ----
    def ev(nid):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return Fraction(n["payoff_p0"])
        if n["kind"] == "chance":
            return sum(Fraction(o["prob"]) * ev(o["child"]) for o in n["outcomes"])
        p = probs[n["infoset"]]
        return sum(p[a["label"]] * ev(a["child"]) for a in n["actions"] if p[a["label"]])

    value = ev(root)

    # ---- best response of `me` against the other player's strategy ----
    def best_response(me):
        sign = 1 if me == 0 else -1
        # reach probability of every node counting only chance and the opponent
        cf = {}

        def walk(nid, r):
            cf[nid] = r
            n = nodes[nid]
            if n["kind"] == "chance":
                for o in n["outcomes"]:
                    walk(o["child"], r * Fraction(o["prob"]))
            elif n["kind"] == "decision":
                for a in n["actions"]:
                    if n["player"] == me:
                        walk(a["child"], r)
                    else:
                        walk(a["child"], r * probs[n["infoset"]][a["label"]])

        walk(root, Fraction(1))
        choice = {}  # infoset -> index of the best action
        memo = {}

        def u(nid):
            """Reach-weighted best-response value of the subtree at nid."""
            if nid in memo:
                return memo[nid]
            n = nodes[nid]
            if n["kind"] == "terminal":
                r = sign * cf[nid] * Fraction(n["payoff_p0"])
            elif n["kind"] == "chance":
                r = sum(u(o["child"]) for o in n["outcomes"])
            elif n["player"] != me:
                r = sum(u(a["child"]) for a in n["actions"])
            else:
                name = n["infoset"]
                if name not in choice:
                    members = infosets[name]
                    totals = [sum(u(m["actions"][j]["child"]) for m in members) for j in range(len(n["actions"]))]
                    choice[name] = max(range(len(totals)), key=lambda j: totals[j])
                r = u(n["actions"][choice[name]]["child"])
            memo[nid] = r
            return r

        return u(root)

    br0 = best_response(0)
    br1 = best_response(1)
    expl = (br0 + br1) / 2
    print(f"infosets checked: {len(infosets)} (all present, all sum to exactly 1)")
    print(f"value_p0         = {float(value):+.12f}")
    print(f"br value p0      = {float(br0):+.12f}")
    print(f"br value p1      = {float(br1):+.12f}")
    print(f"exploitability   = {float(expl):.6e}")
    assert expl >= 0
    # the equilibrium value is bracketed by the two best responses
    print(f"game value lies in [{float(-br1):+.12f}, {float(br0):+.12f}]")


if __name__ == "__main__":
    main()

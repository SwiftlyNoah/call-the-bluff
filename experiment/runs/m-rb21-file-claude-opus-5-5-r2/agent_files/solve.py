#!/usr/bin/env python3
"""CFR+ solver for games in bluff-game-v1 format. Reads game.json, writes strategy.json."""
import json
import sys
from fractions import Fraction

sys.setrecursionlimit(100000)


def load(path):
    g = json.load(open(path))
    nodes = {n["id"]: n for n in g["nodes"]}
    for n in nodes.values():
        if n["kind"] == "chance":
            n["_p"] = [float(Fraction(o["prob"])) for o in n["outcomes"]]
            n["_c"] = [o["child"] for o in n["outcomes"]]
        elif n["kind"] == "decision":
            n["_c"] = [a["child"] for a in n["actions"]]
        else:
            n["_v"] = float(Fraction(n["payoff_p0"]))
    infosets = {}
    for n in nodes.values():
        if n["kind"] == "decision":
            labels = [a["label"] for a in n["actions"]]
            if n["infoset"] in infosets:
                assert infosets[n["infoset"]]["labels"] == labels
                assert infosets[n["infoset"]]["player"] == n["player"]
            else:
                infosets[n["infoset"]] = {"labels": labels, "player": n["player"]}
    return g, nodes, infosets


class CFRPlus:
    def __init__(self, nodes, infosets, root):
        self.nodes, self.root = nodes, root
        self.regret = {k: [0.0] * len(v["labels"]) for k, v in infosets.items()}
        self.avg = {k: [0.0] * len(v["labels"]) for k, v in infosets.items()}
        self.infosets = infosets

    def current(self, I):
        r = self.regret[I]
        r = [max(x, 0.0) for x in r]
        s = sum(r)
        if s > 0:
            return [x / s for x in r]
        return [1.0 / len(r)] * len(r)

    def walk(self, nid, p, reach_p, reach_o, w):
        """Return value for player p (traverser). reach_o includes chance."""
        n = self.nodes[nid]
        k = n["kind"]
        if k == "terminal":
            return n["_v"] if p == 0 else -n["_v"]
        if k == "chance":
            return sum(pr * self.walk(c, p, reach_p, reach_o * pr, w) for pr, c in zip(n["_p"], n["_c"]))
        I = n["infoset"]
        sigma = self.sigma[I]
        if n["player"] == p:
            vals = [self.walk(c, p, reach_p * s, reach_o, w) for s, c in zip(sigma, n["_c"])]
            v = sum(s * x for s, x in zip(sigma, vals))
            r = self.regret[I]
            for a in range(len(r)):
                r[a] += reach_o * (vals[a] - v)
            # strategy averaging (linear weight w) for traverser's own infosets
            av = self.avg[I]
            for a in range(len(av)):
                av[a] += w * reach_p * sigma[a]
            return v
        return sum(s * self.walk(c, p, reach_p, reach_o * s, w) for s, c in zip(sigma, n["_c"]))

    def iterate(self, t):
        for p in (0, 1):
            self.sigma = {I: self.current(I) for I in self.infosets}
            self.walk(self.root, p, 1.0, 1.0, float(t))
            # regret matching+: clip negatives after update (only player p's sets changed)
            for I, info in self.infosets.items():
                if info["player"] == p:
                    r = self.regret[I]
                    for a in range(len(r)):
                        if r[a] < 0:
                            r[a] = 0.0

    def average(self):
        out = {}
        for I, av in self.avg.items():
            s = sum(av)
            out[I] = [x / s for x in av] if s > 0 else [1.0 / len(av)] * len(av)
        return out


def evaluate(nodes, root, strat):
    """Expected payoff to p0 under strat."""
    def rec(nid):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return n["_v"]
        if n["kind"] == "chance":
            return sum(pr * rec(c) for pr, c in zip(n["_p"], n["_c"]))
        return sum(s * rec(c) for s, c in zip(strat[n["infoset"]], n["_c"]) if s > 0)
    return rec(root)


def best_response_value(nodes, root, infosets, strat, br):
    """Value to player br when best-responding to strat of the other player."""
    sign = 1.0 if br == 0 else -1.0
    # opponent+chance reach for every node
    reach = {}
    members = {}

    def down(nid, r):
        reach[nid] = r
        n = nodes[nid]
        if n["kind"] == "terminal":
            return
        if n["kind"] == "chance":
            for pr, c in zip(n["_p"], n["_c"]):
                down(c, r * pr)
            return
        if n["player"] == br:
            members.setdefault(n["infoset"], []).append(nid)
            for c in n["_c"]:
                down(c, r)
        else:
            for s, c in zip(strat[n["infoset"]], n["_c"]):
                down(c, r * s)
    down(root, 1.0)

    choice = {}
    memo = {}

    def val(nid):
        # reach-weighted value (counterfactual): reach * payoff, summed
        if nid in memo:
            return memo[nid]
        n = nodes[nid]
        if n["kind"] == "terminal":
            v = reach[nid] * sign * n["_v"]
        elif n["kind"] == "chance":
            v = sum(val(c) for c in n["_c"])
        elif n["player"] != br:
            v = sum(val(c) for c in n["_c"])
        else:
            I = n["infoset"]
            if I not in choice:
                k = len(infosets[I]["labels"])
                tot = [0.0] * k
                for m in members[I]:
                    for a, c in enumerate(nodes[m]["_c"]):
                        tot[a] += val(c)
                choice[I] = max(range(k), key=lambda a: tot[a])
            v = val(n["_c"][choice[I]])
        memo[nid] = v
        return v
    return val(root)


def main():
    g, nodes, infosets = load("game.json")
    root = g["root"]
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    solver = CFRPlus(nodes, infosets, root)
    for t in range(1, iters + 1):
        solver.iterate(t)
        if t % 200 == 0 or t == iters:
            s = solver.average()
            b0 = best_response_value(nodes, root, infosets, s, 0)
            b1 = best_response_value(nodes, root, infosets, s, 1)
            print(f"iter {t}: v0={evaluate(nodes, root, s):.6f} br0={b0:.6f} br1={b1:.6f} expl={(b0 + b1) / 2:.6g}", flush=True)
    s = solver.average()
    out = {
        "format": "bluff-strategy-v1",
        "game": g["name"],
        "producer": f"CFR+ (alternating updates, linear averaging), {iters} iterations",
        "probs": {I: {lab: repr(p) for lab, p in zip(infosets[I]["labels"], s[I])} for I in infosets},
    }
    json.dump(out, open("strategy.json", "w"), indent=1)
    print("wrote strategy.json")


if __name__ == "__main__":
    main()

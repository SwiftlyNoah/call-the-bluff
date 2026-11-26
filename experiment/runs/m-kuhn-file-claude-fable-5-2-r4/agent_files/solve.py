#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in bluff-game-v1 format.

Reads game.json, runs CFR+ (alternating updates, regret matching+, linearly
weighted averaging), writes the average strategy to strategy.json, then re-reads
that file and reports its exact exploitability and value in rational arithmetic.

Usage: python3 solve.py [game.json] [strategy.json] [--iters N] [--target EPS]
"""
import argparse
import json
import sys
from decimal import Decimal
from fractions import Fraction

DECIMALS = 12  # digits written per probability


# ---------------------------------------------------------------- game loading
class Game:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        if raw.get("format") != "bluff-game-v1":
            raise ValueError("unexpected game format: %r" % raw.get("format"))
        self.name = raw["name"]
        self.root = raw["root"]
        self.nodes = {}
        self.infosets = {}  # name -> {"player", "labels", "nodes"}
        for n in raw["nodes"]:
            nid = n["id"]
            if nid in self.nodes:
                raise ValueError("duplicate node id %r" % nid)
            kind = n["kind"]
            if kind == "chance":
                node = {
                    "kind": "chance",
                    "children": [o["child"] for o in n["outcomes"]],
                    "probs_exact": [Fraction(o["prob"]) for o in n["outcomes"]],
                }
                node["probs"] = [float(p) for p in node["probs_exact"]]
                if sum(node["probs_exact"]) != 1:
                    raise ValueError("chance node %r probabilities do not sum to 1" % nid)
            elif kind == "decision":
                labels = [a["label"] for a in n["actions"]]
                node = {
                    "kind": "decision",
                    "player": n["player"],
                    "infoset": n["infoset"],
                    "children": [a["child"] for a in n["actions"]],
                }
                info = self.infosets.setdefault(
                    n["infoset"], {"player": n["player"], "labels": labels, "nodes": []}
                )
                if info["player"] != n["player"] or info["labels"] != labels:
                    raise ValueError("inconsistent infoset %r" % n["infoset"])
                info["nodes"].append(nid)
            elif kind == "terminal":
                node = {"kind": "terminal", "payoff_exact": Fraction(n["payoff_p0"])}
                node["payoff"] = float(node["payoff_exact"])
            else:
                raise ValueError("unknown node kind %r" % kind)
            self.nodes[nid] = node
        self._check_tree()

    def _check_tree(self):
        seen = set()
        stack = [self.root]
        while stack:
            nid = stack.pop()
            if nid in seen:
                raise ValueError("node %r reached twice: not a tree" % nid)
            seen.add(nid)
            stack.extend(self.nodes[nid].get("children", ()))
        if seen != set(self.nodes):
            raise ValueError("unreachable nodes: %r" % sorted(set(self.nodes) - seen))


# ------------------------------------------------------------------------ CFR+
class CFRPlus:
    def __init__(self, game):
        self.g = game
        self.regret = {k: [0.0] * len(v["labels"]) for k, v in game.infosets.items()}
        self.strat_sum = {k: [0.0] * len(v["labels"]) for k, v in game.infosets.items()}
        self.t = 0

    def current(self, name):
        r = self.regret[name]
        s = sum(r)
        if s > 0.0:
            return [x / s for x in r]
        return [1.0 / len(r)] * len(r)

    def _walk(self, nid, updater, reach_self, reach_other, weight):
        """Return the expected value for `updater`. reach_other includes chance."""
        node = self.g.nodes[nid]
        kind = node["kind"]
        if kind == "terminal":
            return node["payoff"] if updater == 0 else -node["payoff"]
        if kind == "chance":
            return sum(
                p * self._walk(c, updater, reach_self, reach_other * p, weight)
                for p, c in zip(node["probs"], node["children"])
            )
        name = node["infoset"]
        sigma = self.current(name)
        if node["player"] != updater:
            return sum(
                s * self._walk(c, updater, reach_self, reach_other * s, weight)
                for s, c in zip(sigma, node["children"])
                if s > 0.0 or reach_other == 0.0
            ) if any(sigma) else 0.0
        vals = [
            self._walk(c, updater, reach_self * s, reach_other, weight)
            for s, c in zip(sigma, node["children"])
        ]
        ev = sum(s * v for s, v in zip(sigma, vals))
        self._pending.setdefault(name, [0.0] * len(vals))
        pend = self._pending[name]
        for i, v in enumerate(vals):
            pend[i] += reach_other * (v - ev)
        ssum = self.strat_sum[name]
        for i, s in enumerate(sigma):
            ssum[i] += weight * reach_self * s
        return ev

    def iterate(self):
        self.t += 1
        for updater in (0, 1):
            self._pending = {}
            self._walk(self.g.root, updater, 1.0, 1.0, float(self.t))
            for name, pend in self._pending.items():
                r = self.regret[name]
                for i, d in enumerate(pend):
                    r[i] = max(r[i] + d, 0.0)

    def average(self):
        out = {}
        for name, s in self.strat_sum.items():
            tot = sum(s)
            out[name] = [x / tot for x in s] if tot > 0.0 else [1.0 / len(s)] * len(s)
        return out


# ------------------------------------------------- exact evaluation of a profile
def expected_value(game, probs, nid=None):
    """Player 0's expected payoff under the profile `probs` (name -> [Fraction])."""
    nid = game.root if nid is None else nid
    node = game.nodes[nid]
    if node["kind"] == "terminal":
        return node["payoff_exact"]
    if node["kind"] == "chance":
        w = node["probs_exact"]
    else:
        w = probs[node["infoset"]]
    return sum(p * expected_value(game, probs, c) for p, c in zip(w, node["children"]) if p)


def best_response_value(game, probs, br):
    """Value to player `br` of an exact best response against `probs`."""
    # Reach probability of every node counting only chance and the opponent.
    reach = {}

    def fill(nid, r):
        reach[nid] = r
        node = game.nodes[nid]
        if node["kind"] == "chance":
            for p, c in zip(node["probs_exact"], node["children"]):
                fill(c, r * p)
        elif node["kind"] == "decision":
            if node["player"] == br:
                for c in node["children"]:
                    fill(c, r)
            else:
                for p, c in zip(probs[node["infoset"]], node["children"]):
                    fill(c, r * p)

    fill(game.root, Fraction(1))
    choice = {}

    def choose(name):
        if name not in choice:
            info = game.infosets[name]
            totals = [
                sum(reach[h] * value(game.nodes[h]["children"][a]) for h in info["nodes"])
                for a in range(len(info["labels"]))
            ]
            choice[name] = max(range(len(totals)), key=lambda a: totals[a])
        return choice[name]

    def value(nid):
        node = game.nodes[nid]
        if node["kind"] == "terminal":
            return node["payoff_exact"] if br == 0 else -node["payoff_exact"]
        if node["kind"] == "chance":
            return sum(p * value(c) for p, c in zip(node["probs_exact"], node["children"]))
        if node["player"] == br:
            return value(node["children"][choose(node["infoset"])])
        return sum(
            p * value(c) for p, c in zip(probs[node["infoset"]], node["children"]) if p
        )

    return value(game.root)


def exploitability(game, probs):
    br0 = best_response_value(game, probs, 0)
    br1 = best_response_value(game, probs, 1)
    return (br0 + br1) / 2, br0, br1


# ------------------------------------------------------------- strategy file IO
def to_exact(dist):
    """Round a float distribution to DECIMALS places so that it sums to exactly 1."""
    q = Fraction(1, 10 ** DECIMALS)
    fr = [Fraction(round(max(x, 0.0) * 10 ** DECIMALS)) * q for x in dist]
    big = max(range(len(fr)), key=lambda i: fr[i])
    fr[big] += 1 - sum(fr)
    assert sum(fr) == 1 and all(f >= 0 for f in fr)
    return fr


def fmt(fr):
    s = format(Decimal(fr.numerator) / Decimal(fr.denominator), "f")
    assert Fraction(s) == fr
    return s


def write_strategy(game, exact, path, producer):
    doc = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": producer,
        "probs": {
            name: {lab: fmt(p) for lab, p in zip(game.infosets[name]["labels"], exact[name])}
            for name in sorted(game.infosets)
        },
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def read_strategy(game, path):
    with open(path) as f:
        doc = json.load(f)
    probs = {}
    for name, info in game.infosets.items():
        row = doc["probs"][name]
        if set(row) != set(info["labels"]):
            raise ValueError("infoset %r has wrong actions" % name)
        probs[name] = [Fraction(row[lab]) for lab in info["labels"]]
        if sum(probs[name]) != 1 or min(probs[name]) < 0:
            raise ValueError("infoset %r is not a distribution" % name)
    return probs


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="?", default="game.json")
    ap.add_argument("out", nargs="?", default="strategy.json")
    ap.add_argument("--iters", type=int, default=200000, help="maximum CFR+ iterations")
    ap.add_argument("--target", type=float, default=1e-7, help="stop at this exploitability")
    args = ap.parse_args()

    game = Game(args.game)
    print("game %r: %d nodes, %d infosets" % (game.name, len(game.nodes), len(game.infosets)))
    solver = CFRPlus(game)
    check = 100
    while solver.t < args.iters:
        solver.iterate()
        if solver.t == check or solver.t == args.iters:
            exact = {k: to_exact(v) for k, v in solver.average().items()}
            eps = exploitability(game, exact)[0]
            print("iter %7d  exploitability %.3e" % (solver.t, float(eps)))
            if eps <= args.target:
                break
            check *= 2

    exact = {k: to_exact(v) for k, v in solver.average().items()}
    write_strategy(
        game, exact, args.out,
        "solve.py: CFR+ (alternating updates, linear averaging), %d iterations" % solver.t,
    )

    # Verify what was actually written, in exact arithmetic.
    probs = read_strategy(game, args.out)
    eps, br0, br1 = exploitability(game, probs)
    val = expected_value(game, probs)
    print("wrote %s after %d iterations" % (args.out, solver.t))
    for name in sorted(probs):
        print("  %-8s %s" % (name, "  ".join(
            "%s=%.6f" % (lab, float(p)) for lab, p in zip(game.infosets[name]["labels"], probs[name]))))
    print("value to player 0 under the profile: %.12f (%s)" % (float(val), val.limit_denominator(10000)))
    print("best response value, player 0: %.12f" % float(br0))
    print("best response value, player 1: %.12f" % float(br1))
    print("exploitability: %.6e" % float(eps))
    print("game value bracket for player 0: [%.12f, %.12f]" % (float(-br1), float(br0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

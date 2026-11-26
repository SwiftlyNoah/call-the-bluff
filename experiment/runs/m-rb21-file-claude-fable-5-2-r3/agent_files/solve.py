#!/usr/bin/env python3
"""CFR+ solver for games in the bluff-game-v1 format.

Reads game.json, runs CFR+ (alternating updates, regret matching+, linearly
weighted averaging), writes the average strategy to strategy.json, then
reloads that file and reports its exact exploitability and value.

Usage: python3 solve.py [game.json] [strategy.json] [--iters N] [--target EPS]
"""
import argparse
import json
import sys
import time
from fractions import Fraction

SCALE = 10 ** 12  # probabilities are written as 12-digit decimals summing to exactly 1


class Game:
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.name = data["name"]
        self.root = data["root"]
        nodes = {n["id"]: n for n in data["nodes"]}
        self.kind = {}       # id -> 'c' | 'd' | 't'
        self.children = {}   # id -> list of child ids
        self.probs = {}      # chance id -> list of Fraction
        self.payoff = {}     # terminal id -> Fraction (to player 0)
        self.player = {}     # decision id -> 0 | 1
        self.iset = {}       # decision id -> infoset index
        self.iset_names = []
        self.iset_actions = []
        self.iset_player = []
        self.iset_nodes = []
        index = {}
        for nid, n in nodes.items():
            k = n["kind"]
            if k == "chance":
                self.kind[nid] = "c"
                self.children[nid] = [o["child"] for o in n["outcomes"]]
                self.probs[nid] = [Fraction(o["prob"]) for o in n["outcomes"]]
                if sum(self.probs[nid]) != 1:
                    raise ValueError("chance node %r probabilities do not sum to 1" % nid)
            elif k == "decision":
                self.kind[nid] = "d"
                self.children[nid] = [a["child"] for a in n["actions"]]
                labels = [a["label"] for a in n["actions"]]
                name = n["infoset"]
                if name not in index:
                    index[name] = len(self.iset_names)
                    self.iset_names.append(name)
                    self.iset_actions.append(labels)
                    self.iset_player.append(n["player"])
                    self.iset_nodes.append([])
                i = index[name]
                if self.iset_actions[i] != labels or self.iset_player[i] != n["player"]:
                    raise ValueError("inconsistent infoset %r" % name)
                self.iset[nid] = i
                self.iset_nodes[i].append(nid)
                self.player[nid] = n["player"]
            elif k == "terminal":
                self.kind[nid] = "t"
                self.payoff[nid] = Fraction(n["payoff_p0"])
            else:
                raise ValueError("unknown node kind %r" % k)


def run_cfr_plus(g, max_iters, target, check_every, time_limit):
    """Returns the average strategy as a list (per infoset) of float lists."""
    sys.setrecursionlimit(100000)
    n_sets = len(g.iset_names)
    regret = [[0.0] * len(a) for a in g.iset_actions]
    strat_sum = [[0.0] * len(a) for a in g.iset_actions]
    kind, children, iset, player = g.kind, g.children, g.iset, g.player
    cprob = {k: [float(p) for p in v] for k, v in g.probs.items()}
    payoff = {k: float(v) for k, v in g.payoff.items()}

    def current(i):
        r = regret[i]
        s = sum(r)
        if s > 0.0:
            return [x / s for x in r]
        n = len(r)
        return [1.0 / n] * n

    delta = [[0.0] * len(a) for a in g.iset_actions]
    pending = [False] * n_sets

    def walk(node, upd, r_upd, r_opp, r_chance, weight):
        k = kind[node]
        if k == "t":
            return payoff[node] if upd == 0 else -payoff[node]
        ch = children[node]
        if k == "c":
            v = 0.0
            for c, p in zip(ch, cprob[node]):
                v += p * walk(c, upd, r_upd, r_opp, r_chance * p, weight)
            return v
        i = iset[node]
        sigma = current(i)
        if player[node] == upd:
            vals = [walk(c, upd, r_upd * s, r_opp, r_chance, weight) for c, s in zip(ch, sigma)]
            v = 0.0
            for s, x in zip(sigma, vals):
                v += s * x
            cf = r_opp * r_chance
            if cf > 0.0:
                pending[i] = True
                acc = delta[i]
                for a, x in enumerate(vals):
                    acc[a] += cf * (x - v)
            return v
        ss = strat_sum[i]
        v = 0.0
        w = weight * r_opp
        for a, (c, s) in enumerate(zip(ch, sigma)):
            if s > 0.0:
                ss[a] += w * s
                v += s * walk(c, upd, r_upd, r_opp * s, r_chance, weight)
        return v

    def average():
        out = []
        for i in range(n_sets):
            s = sum(strat_sum[i])
            n = len(strat_sum[i])
            out.append([x / s for x in strat_sum[i]] if s > 0.0 else [1.0 / n] * n)
        return out

    start = time.time()
    t = 0
    while t < max_iters:
        t += 1
        for upd in (0, 1):
            walk(g.root, upd, 1.0, 1.0, 1.0, float(t))
            for i in range(n_sets):
                if pending[i]:
                    pending[i] = False
                    r, d = regret[i], delta[i]
                    for a in range(len(r)):
                        x = r[a] + d[a]
                        r[a] = x if x > 0.0 else 0.0
                        d[a] = 0.0
        if t % check_every == 0 or t == max_iters:
            avg = average()
            expl, val = evaluate(g, avg)
            print("iter %7d  exploitability %.3e  value_p0 %.9f  (%.1fs)"
                  % (t, expl, val, time.time() - start), flush=True)
            if expl <= target or time.time() - start > time_limit:
                break
    return average(), t


def evaluate(g, strat):
    """Exploitability and player-0 value of `strat` (list per infoset).

    Works with floats or Fractions, depending on the entries of `strat` and
    of `chance`/`payoff` (Fractions give an exact answer).
    """
    exact = isinstance(strat[0][0], Fraction)
    zero = Fraction(0) if exact else 0.0
    one = Fraction(1) if exact else 1.0
    cprob = g.probs if exact else {k: [float(p) for p in v] for k, v in g.probs.items()}
    payoff = g.payoff if exact else {k: float(v) for k, v in g.payoff.items()}
    kind, children, iset, player = g.kind, g.children, g.iset, g.player

    def value(node):
        k = kind[node]
        if k == "t":
            return payoff[node]
        if k == "c":
            return sum((p * value(c) for c, p in zip(children[node], cprob[node])), zero)
        return sum((s * value(c) for c, s in zip(children[node], strat[iset[node]]) if s), zero)

    def best_response(br):
        # reach[node] = product of chance and opponent probabilities on the path
        reach = {}
        stack = [(g.root, one)]
        while stack:
            node, r = stack.pop()
            reach[node] = r
            k = kind[node]
            if k == "t":
                continue
            ch = children[node]
            if k == "c":
                for c, p in zip(ch, cprob[node]):
                    stack.append((c, r * p))
            elif player[node] == br:
                for c in ch:
                    stack.append((c, r))
            else:
                for c, s in zip(ch, strat[iset[node]]):
                    stack.append((c, r * s))
        choice = {}
        memo = {}

        def val(node):
            """Reach-weighted value of the subtree for the best responder."""
            if node in memo:
                return memo[node]
            k = kind[node]
            if k == "t":
                v = reach[node] * (payoff[node] if br == 0 else -payoff[node])
            elif k == "c" or player[node] != br:
                v = zero
                for c in children[node]:
                    if reach[c]:
                        v += val(c)
            else:
                i = iset[node]
                if i not in choice:
                    best, best_a = None, 0
                    for a in range(len(g.iset_actions[i])):
                        tot = zero
                        for m in g.iset_nodes[i]:
                            tot += val(children[m][a])
                        if best is None or tot > best:
                            best, best_a = tot, a
                    choice[i] = best_a
                v = val(children[node][choice[i]])
            memo[node] = v
            return v

        return val(g.root)

    br0 = best_response(0)
    br1 = best_response(1)
    return (br0 + br1) / 2, value(g.root)


def to_decimals(p):
    """Round a probability vector to 12-digit decimal strings summing to exactly 1."""
    units = [int(round(x * SCALE)) for x in p]
    j = max(range(len(units)), key=lambda a: units[a])
    units[j] += SCALE - sum(units)
    out = []
    for u in units:
        if u < 0 or u > SCALE:
            raise ValueError("bad probability vector %r" % (p,))
        out.append("1" if u == SCALE else "0.%012d" % u)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="?", default="game.json")
    ap.add_argument("out", nargs="?", default="strategy.json")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--target", type=float, default=1e-6)
    ap.add_argument("--check-every", type=int, default=250)
    ap.add_argument("--time-limit", type=float, default=600.0)
    args = ap.parse_args()

    sys.setrecursionlimit(100000)
    g = Game(args.game)
    print("game %s: %d nodes, %d infosets" % (g.name, len(g.kind), len(g.iset_names)), flush=True)
    avg, iters = run_cfr_plus(g, args.iters, args.target, args.check_every, args.time_limit)

    probs = {}
    for i, name in enumerate(g.iset_names):
        probs[name] = dict(zip(g.iset_actions[i], to_decimals(avg[i])))
    with open(args.out, "w") as f:
        json.dump({
            "format": "bluff-strategy-v1",
            "game": g.name,
            "producer": "solve.py: CFR+ (alternating updates, linear averaging), %d iterations" % iters,
            "probs": probs,
        }, f, indent=1)

    # Verify the file as written, in exact rational arithmetic.
    with open(args.out) as f:
        written = json.load(f)["probs"]
    exact = []
    for i, name in enumerate(g.iset_names):
        row = [Fraction(written[name][a]) for a in g.iset_actions[i]]
        assert sum(row) == 1 and min(row) >= 0, name
        exact.append(row)
    expl, val = evaluate(g, exact)
    print("wrote %s after %d iterations" % (args.out, iters))
    print("exact exploitability of written file: %.12g" % float(expl))
    print("exact value for player 0 of written file: %.12g" % float(val))


if __name__ == "__main__":
    main()

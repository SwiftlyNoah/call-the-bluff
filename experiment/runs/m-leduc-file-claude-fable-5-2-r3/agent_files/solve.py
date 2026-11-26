#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in bluff-game-v1 format.

Reads game.json, runs CFR+ (alternating updates, regret matching+, linearly weighted
average strategy), writes the average strategy to strategy.json, and reports the
exploitability of the strategy exactly as it was written.

Usage: python3 solve.py [--game game.json] [--out strategy.json]
                        [--iters N] [--seconds S] [--target EPS]
"""
import argparse
import json
import sys
import time
from fractions import Fraction

sys.setrecursionlimit(100000)

CHANCE, DECISION, TERMINAL = 0, 1, 2
DIGITS = 12  # decimal places used for the probabilities written to the strategy file


class Game:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.name = raw.get("name", "game")
        self.root = raw["root"]
        nodes = {n["id"]: n for n in raw["nodes"]}
        if len(nodes) != len(raw["nodes"]):
            raise ValueError("duplicate node ids")
        self.kind = {}
        self.player = {}
        self.infoset = {}       # node id -> infoset index
        self.children = {}      # node id -> list of child ids
        self.probs = {}         # chance node id -> list of float probabilities
        self.probs_exact = {}   # chance node id -> list of Fractions
        self.payoff = {}        # terminal node id -> float payoff to player 0
        self.payoff_exact = {}
        self.iset_names = []    # infoset index -> name
        self.iset_player = []
        self.iset_actions = []  # infoset index -> list of action labels
        self.iset_nodes = []    # infoset index -> list of node ids
        index = {}
        for nid, n in nodes.items():
            kind = n["kind"]
            if kind == "chance":
                self.kind[nid] = CHANCE
                self.children[nid] = [o["child"] for o in n["outcomes"]]
                exact = [Fraction(o["prob"]) for o in n["outcomes"]]
                if sum(exact) != 1 or min(exact) < 0:
                    raise ValueError("chance node %r: bad probabilities" % nid)
                self.probs_exact[nid] = exact
                self.probs[nid] = [float(p) for p in exact]
            elif kind == "decision":
                self.kind[nid] = DECISION
                if n["player"] not in (0, 1):
                    raise ValueError("node %r: bad player" % nid)
                labels = [a["label"] for a in n["actions"]]
                name = n["infoset"]
                if name not in index:
                    index[name] = len(self.iset_names)
                    self.iset_names.append(name)
                    self.iset_player.append(n["player"])
                    self.iset_actions.append(labels)
                    self.iset_nodes.append([])
                i = index[name]
                if self.iset_player[i] != n["player"] or self.iset_actions[i] != labels:
                    raise ValueError("infoset %r is inconsistent" % name)
                self.iset_nodes[i].append(nid)
                self.player[nid] = n["player"]
                self.infoset[nid] = i
                self.children[nid] = [a["child"] for a in n["actions"]]
            elif kind == "terminal":
                self.kind[nid] = TERMINAL
                self.payoff_exact[nid] = Fraction(n["payoff_p0"])
                self.payoff[nid] = float(self.payoff_exact[nid])
            else:
                raise ValueError("node %r: unknown kind %r" % (nid, kind))
        # Every node must be reachable from the root exactly once (a tree).
        seen = set()
        stack = [self.root]
        while stack:
            nid = stack.pop()
            if nid in seen:
                raise ValueError("node %r has more than one parent" % nid)
            seen.add(nid)
            stack.extend(self.children.get(nid, ()))
        if len(seen) != len(nodes):
            raise ValueError("%d nodes are unreachable from the root" % (len(nodes) - len(seen)))


class CFRPlus:
    def __init__(self, game):
        self.g = game
        n = [len(a) for a in game.iset_actions]
        self.regret = [[0.0] * k for k in n]
        self.strat_sum = [[0.0] * k for k in n]
        self.current = [[1.0 / k] * k for k in n]
        self.delta = [[0.0] * k for k in n]
        self.t = 0
        # A player's own reach probability of an infoset depends only on that player's last
        # earlier (infoset, action), which perfect recall makes the same for every node of the
        # infoset. order[pl] lists pl's infosets with parents first.
        self.parent = [None] * len(n)   # infoset -> (earlier own infoset, action index) or None
        self.order = [[], []]
        seen = [False] * len(n)
        stack = [(game.root, None, None)]
        while stack:
            nid, last0, last1 = stack.pop()
            kind = game.kind[nid]
            if kind == TERMINAL:
                continue
            kids = game.children[nid]
            if kind == CHANCE:
                for c in kids:
                    stack.append((c, last0, last1))
                continue
            i = game.infoset[nid]
            pl = game.player[nid]
            last = last0 if pl == 0 else last1
            if not seen[i]:
                seen[i] = True
                self.parent[i] = last
                self.order[pl].append(i)
            elif self.parent[i] != last:
                raise ValueError("infoset %r: the game does not have perfect recall"
                                 % game.iset_names[i])
            for a, c in enumerate(kids):
                if pl == 0:
                    stack.append((c, (i, a), last1))
                else:
                    stack.append((c, last0, (i, a)))
        self.own_reach = [1.0] * len(n)

    def _walk(self, nid, me, pi_opp):
        """Value to `me` of the subtree at nid under the current strategies.

        pi_opp is the reach probability contributed by the opponent and chance; it is never
        zero here because zero-probability branches of theirs are not entered (they add nothing
        to the value or to any regret).
        """
        g = self.g
        kind = g.kind[nid]
        if kind == TERMINAL:
            return g.payoff[nid] if me == 0 else -g.payoff[nid]
        kids = g.children[nid]
        if kind == CHANCE:
            v = 0.0
            for p, c in zip(g.probs[nid], kids):
                if p > 0.0:
                    v += p * self._walk(c, me, pi_opp * p)
            return v
        i = g.infoset[nid]
        s = self.current[i]
        if g.player[nid] != me:
            v = 0.0
            for p, c in zip(s, kids):
                if p > 0.0:
                    v += p * self._walk(c, me, pi_opp * p)
            return v
        utils = [self._walk(c, me, pi_opp) for c in kids]
        v = 0.0
        for p, u in zip(s, utils):
            v += p * u
        d = self.delta[i]
        for a in range(len(kids)):
            d[a] += pi_opp * (utils[a] - v)
        return v

    def iterate(self):
        self.t += 1
        g = self.g
        weight = float(self.t)
        for me in (0, 1):
            # Average strategy: once per infoset, weighted by iteration and own reach.
            reach = self.own_reach
            for i in self.order[me]:
                par = self.parent[i]
                r = 1.0 if par is None else reach[par[0]] * self.current[par[0]][par[1]]
                reach[i] = r
                if r > 0.0:
                    w = weight * r
                    ss = self.strat_sum[i]
                    s = self.current[i]
                    for a in range(len(s)):
                        ss[a] += w * s[a]
            self._walk(g.root, me, 1.0)
            # Regret matching+: add this iteration's regrets, floor at zero, rebuild strategy.
            for i in self.order[me]:
                r = self.regret[i]
                d = self.delta[i]
                total = 0.0
                for a in range(len(r)):
                    x = r[a] + d[a]
                    if x < 0.0:
                        x = 0.0
                    r[a] = x
                    d[a] = 0.0
                    total += x
                if total > 0.0:
                    self.current[i] = [x / total for x in r]
                else:
                    self.current[i] = [1.0 / len(r)] * len(r)

    def average(self):
        out = []
        for ss in self.strat_sum:
            total = sum(ss)
            out.append([x / total for x in ss] if total > 0.0 else [1.0 / len(ss)] * len(ss))
        return out


def evaluate(game, strat, exact=False):
    """Return (value_p0, br0, br1) for a behavioral strategy (list per infoset index).

    br0 is player 0's best-response value against player 1's strategy and br1 is player 1's
    best-response value against player 0's strategy. With exact=True all arithmetic is done in
    Fractions (strat must then hold Fractions).
    """
    g = game
    zero = Fraction(0) if exact else 0.0
    one = Fraction(1) if exact else 1.0
    cprobs = g.probs_exact if exact else g.probs
    payoff = g.payoff_exact if exact else g.payoff

    def value(nid):
        kind = g.kind[nid]
        if kind == TERMINAL:
            return payoff[nid]
        ps = cprobs[nid] if kind == CHANCE else strat[g.infoset[nid]]
        v = zero
        for p, c in zip(ps, g.children[nid]):
            if p:
                v += p * value(c)
        return v

    def best_response(me):
        # Reach probability of every node counting only chance and the opponent.
        reach = {}
        stack = [(g.root, one)]
        while stack:
            nid, r = stack.pop()
            reach[nid] = r
            kind = g.kind[nid]
            if kind == TERMINAL:
                continue
            kids = g.children[nid]
            if kind == CHANCE:
                ps = cprobs[nid]
            elif g.player[nid] != me:
                ps = strat[g.infoset[nid]]
            else:
                ps = [one] * len(kids)
            for p, c in zip(ps, kids):
                stack.append((c, r * p))
        choice = {}

        def br(nid):
            """Reach-weighted value to `me` of the subtree at nid when `me` best-responds."""
            kind = g.kind[nid]
            if kind == TERMINAL:
                r = reach[nid]
                if not r:
                    return zero
                return r * (payoff[nid] if me == 0 else -payoff[nid])
            kids = g.children[nid]
            if kind == CHANCE or g.player[nid] != me:
                v = zero
                for c in kids:
                    v += br(c)
                return v
            i = g.infoset[nid]
            if i not in choice:
                # Pick the action that is best summed over every node of the infoset.
                choice[i] = None  # guards against imperfect-recall loops
                totals = [zero] * len(kids)
                for m in g.iset_nodes[i]:
                    for a, c in enumerate(g.children[m]):
                        totals[a] += br(c)
                best = 0
                for a in range(1, len(totals)):
                    if totals[a] > totals[best]:
                        best = a
                choice[i] = best
            if choice[i] is None:
                raise ValueError("game does not have perfect recall")
            return br(kids[choice[i]])

        return br(g.root)

    return value(g.root), best_response(0), best_response(1)


def to_decimal_strings(probs):
    """Round a distribution to DIGITS decimals so that the written values sum to exactly 1."""
    scale = 10 ** DIGITS
    ints = [max(0, int(round(p * scale))) for p in probs]
    big = max(range(len(ints)), key=lambda a: ints[a])
    ints[big] += scale - sum(ints)
    assert min(ints) >= 0 and sum(ints) == scale
    out = []
    for x in ints:
        if x == scale:
            out.append("1")
        elif x == 0:
            out.append("0")
        else:
            out.append(("0.%0*d" % (DIGITS, x)).rstrip("0"))
    return out


def write_strategy(game, avg, path, producer):
    probs = {}
    for i, name in enumerate(game.iset_names):
        probs[name] = dict(zip(game.iset_actions[i], to_decimal_strings(avg[i])))
    doc = {"format": "bluff-strategy-v1", "game": game.name, "producer": producer, "probs": probs}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def read_strategy(game, path, exact=False):
    with open(path) as f:
        doc = json.load(f)
    probs = doc["probs"]
    if set(probs) != set(game.iset_names):
        raise ValueError("strategy file does not cover exactly the game's infosets")
    out = []
    for i, name in enumerate(game.iset_names):
        row = probs[name]
        if set(row) != set(game.iset_actions[i]):
            raise ValueError("infoset %r: wrong action labels" % name)
        fr = [Fraction(row[a]) for a in game.iset_actions[i]]
        if sum(fr) != 1 or min(fr) < 0:
            raise ValueError("infoset %r: probabilities are not a distribution" % name)
        out.append(fr if exact else [float(x) for x in fr])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default="game.json")
    ap.add_argument("--out", default="strategy.json")
    ap.add_argument("--iters", type=int, default=20000, help="maximum number of CFR+ iterations")
    ap.add_argument("--seconds", type=float, default=600.0, help="wall-clock limit for iterating")
    ap.add_argument("--target", type=float, default=1e-5, help="stop once exploitability is below this")
    ap.add_argument("--check-every", type=int, default=250)
    args = ap.parse_args()

    game = Game(args.game)
    print("game %r: %d nodes, %d infosets" % (game.name, len(game.kind), len(game.iset_names)), flush=True)
    solver = CFRPlus(game)
    start = time.time()
    while solver.t < args.iters:
        solver.iterate()
        if solver.t % args.check_every == 0 or solver.t == args.iters:
            v, b0, b1 = evaluate(game, solver.average())
            expl = (b0 + b1) / 2
            print("iter %6d  %7.1fs  value_p0 %+.8f  exploitability %.3e"
                  % (solver.t, time.time() - start, v, expl), flush=True)
            if expl <= args.target or time.time() - start > args.seconds:
                break
    producer = "solve.py: CFR+ (alternating updates, linear averaging), %d iterations" % solver.t
    write_strategy(game, solver.average(), args.out, producer)

    # Report on the file as written, in exact rational arithmetic.
    v, b0, b1 = evaluate(game, read_strategy(game, args.out, exact=True), exact=True)
    expl = (b0 + b1) / 2
    print("wrote %s after %d iterations" % (args.out, solver.t))
    print("exact, from the written file:")
    print("  value_p0            %.12f" % float(v))
    print("  best response p0    %.12f" % float(b0))
    print("  best response p1    %.12f" % float(b1))
    print("  exploitability      %.12e" % float(expl))


if __name__ == "__main__":
    main()

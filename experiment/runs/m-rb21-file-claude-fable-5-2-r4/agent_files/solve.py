#!/usr/bin/env python3
"""CFR+ solver for games in the bluff-game-v1 format.

Reads game.json, runs CFR+ (alternating updates, regret matching+, linearly
weighted averaging), writes the average strategy to strategy.json, and reports
the exact best-response exploitability of the strategy that was written.

Usage: python3 solve.py [game.json] [strategy.json] [--iters N] [--target EPS]
"""
import argparse
import json
import sys
import time
from fractions import Fraction

CHANCE, DECISION, TERMINAL = 0, 1, 2


class Game:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.name = raw.get("name", "")
        self.root = raw["root"]
        nodes = {n["id"]: n for n in raw["nodes"]}
        self.kind = {}
        self.children = {}      # node id -> list of child ids
        self.probs = {}         # chance node id -> list of float probs
        self.probs_exact = {}   # chance node id -> list of Fraction probs
        self.payoff = {}        # terminal id -> float payoff to player 0
        self.payoff_exact = {}
        self.player = {}        # decision id -> player
        self.iset = {}          # decision id -> infoset index
        self.iset_names = []
        self.iset_player = []
        self.iset_actions = []  # infoset index -> action labels
        self.iset_nodes = []    # infoset index -> node ids
        index = {}
        for nid, n in nodes.items():
            k = n["kind"]
            if k == "chance":
                self.kind[nid] = CHANCE
                self.children[nid] = [o["child"] for o in n["outcomes"]]
                exact = [Fraction(o["prob"]) for o in n["outcomes"]]
                if sum(exact) != 1:
                    raise ValueError("chance node %r probabilities sum to %s" % (nid, sum(exact)))
                self.probs_exact[nid] = exact
                self.probs[nid] = [float(p) for p in exact]
            elif k == "decision":
                self.kind[nid] = DECISION
                labels = [a["label"] for a in n["actions"]]
                self.children[nid] = [a["child"] for a in n["actions"]]
                name = n["infoset"]
                if name not in index:
                    index[name] = len(self.iset_names)
                    self.iset_names.append(name)
                    self.iset_player.append(n["player"])
                    self.iset_actions.append(labels)
                    self.iset_nodes.append([])
                i = index[name]
                if self.iset_player[i] != n["player"] or self.iset_actions[i] != labels:
                    raise ValueError("inconsistent infoset %r" % name)
                self.iset[nid] = i
                self.iset_nodes[i].append(nid)
                self.player[nid] = n["player"]
            elif k == "terminal":
                self.kind[nid] = TERMINAL
                self.payoff_exact[nid] = Fraction(n["payoff_p0"])
                self.payoff[nid] = float(self.payoff_exact[nid])
            else:
                raise ValueError("unknown node kind %r" % k)
        # Pre-order (parents before children), computed iteratively.
        order = []
        stack = [self.root]
        while stack:
            nid = stack.pop()
            order.append(nid)
            if self.kind[nid] != TERMINAL:
                stack.extend(reversed(self.children[nid]))
        if len(order) != len(nodes):
            raise ValueError("tree has %d reachable nodes but file lists %d" % (len(order), len(nodes)))
        self.order = order


def regret_matching(regrets):
    total = 0.0
    for r in regrets:
        if r > 0:
            total += r
    n = len(regrets)
    if total > 0:
        return [r / total if r > 0 else 0.0 for r in regrets]
    return [1.0 / n] * n


class CFRPlus:
    def __init__(self, game):
        self.g = game
        self.regret = [[0.0] * len(a) for a in game.iset_actions]
        self.strat_sum = [[0.0] * len(a) for a in game.iset_actions]
        self.t = 0

    def current(self):
        return [regret_matching(r) for r in self.regret]

    def average(self):
        out = []
        for s in self.strat_sum:
            total = sum(s)
            n = len(s)
            out.append([x / total for x in s] if total > 0 else [1.0 / n] * n)
        return out

    def iterate(self):
        self.t += 1
        for p in (0, 1):
            self._update(p)

    def _update(self, p):
        """One alternating-update pass for player p over the whole tree."""
        g = self.g
        kind, children, iset = g.kind, g.children, g.iset
        sigma = self.current()
        n_isets = len(g.iset_names)
        # Top-down: reach of player p alone, and of everyone else (opponent and chance).
        reach_own = {g.root: 1.0}
        reach_opp = {g.root: 1.0}
        for nid in g.order:
            k = kind[nid]
            if k == TERMINAL:
                continue
            ro, rp = reach_own[nid], reach_opp[nid]
            ch = children[nid]
            if k == CHANCE:
                pr = g.probs[nid]
                for c, q in zip(ch, pr):
                    reach_own[c] = ro
                    reach_opp[c] = rp * q
            else:
                s = sigma[iset[nid]]
                if g.player[nid] == p:
                    for c, q in zip(ch, s):
                        reach_own[c] = ro * q
                        reach_opp[c] = rp
                else:
                    for c, q in zip(ch, s):
                        reach_own[c] = ro
                        reach_opp[c] = rp * q
        # Bottom-up: expected value to player p under the current strategies.
        sign = 1.0 if p == 0 else -1.0
        value = {}
        cf_action = [[0.0] * len(a) for a in g.iset_actions]
        cf_node = [0.0] * n_isets
        for nid in reversed(g.order):
            k = kind[nid]
            if k == TERMINAL:
                value[nid] = sign * g.payoff[nid]
            elif k == CHANCE:
                v = 0.0
                for c, q in zip(children[nid], g.probs[nid]):
                    v += q * value[c]
                value[nid] = v
            else:
                i = iset[nid]
                s = sigma[i]
                ch = children[nid]
                v = 0.0
                for c, q in zip(ch, s):
                    v += q * value[c]
                value[nid] = v
                if g.player[nid] == p:
                    w = reach_opp[nid]
                    if w != 0.0:
                        acc = cf_action[i]
                        for a, c in enumerate(ch):
                            acc[a] += w * value[c]
                        cf_node[i] += w * v
        # Regret matching+ update and linearly weighted average-strategy update.
        weight = float(self.t)
        own_reach_iset = [None] * n_isets
        for i in range(n_isets):
            if g.iset_player[i] != p:
                continue
            reg = self.regret[i]
            acc = cf_action[i]
            base = cf_node[i]
            for a in range(len(reg)):
                r = reg[a] + acc[a] - base
                reg[a] = r if r > 0 else 0.0
            own_reach_iset[i] = sum(reach_own[n] for n in g.iset_nodes[i])
        for i in range(n_isets):
            if g.iset_player[i] != p:
                continue
            w = own_reach_iset[i] * weight
            if w != 0.0:
                s = sigma[i]
                ss = self.strat_sum[i]
                for a in range(len(ss)):
                    ss[a] += w * s[a]


def best_response_value(game, strategy, br_player, exact=False):
    """Value to br_player of a best response against `strategy` (list per infoset).

    The responder's choice is made per information set, using the reach
    probability contributed by chance and the opponent at each of its nodes.
    """
    g = game
    zero = Fraction(0) if exact else 0.0
    one = Fraction(1) if exact else 1.0
    probs = g.probs_exact if exact else g.probs
    payoff = g.payoff_exact if exact else g.payoff
    sign = 1 if br_player == 0 else -1
    reach = {g.root: one}
    for nid in g.order:
        k = g.kind[nid]
        if k == TERMINAL:
            continue
        r = reach[nid]
        ch = g.children[nid]
        if k == CHANCE:
            for c, q in zip(ch, probs[nid]):
                reach[c] = r * q
        elif g.player[nid] == br_player:
            for c in ch:
                reach[c] = r
        else:
            for c, q in zip(ch, strategy[g.iset[nid]]):
                reach[c] = r * q
    value = {}
    choice = {}

    def decide(i):
        best_a, best_v = 0, None
        for a in range(len(g.iset_actions[i])):
            v = zero
            for n in g.iset_nodes[i]:
                w = reach[n]
                if w != 0:
                    v += w * node_value(g.children[n][a])
            if best_v is None or v > best_v:
                best_a, best_v = a, v
        choice[i] = best_a
        return best_a

    def node_value(nid):
        if nid in value:
            return value[nid]
        k = g.kind[nid]
        if k == TERMINAL:
            v = sign * payoff[nid]
        elif k == CHANCE:
            v = zero
            for c, q in zip(g.children[nid], probs[nid]):
                v += q * node_value(c)
        else:
            i = g.iset[nid]
            if g.player[nid] == br_player:
                a = choice[i] if i in choice else decide(i)
                v = node_value(g.children[nid][a])
            else:
                v = zero
                for c, q in zip(g.children[nid], strategy[i]):
                    if q != 0:
                        v += q * node_value(c)
        value[nid] = v
        return v

    return node_value(g.root)


def profile_value(game, strategy, exact=False):
    """Expected payoff to player 0 when both players follow `strategy`."""
    g = game
    probs = g.probs_exact if exact else g.probs
    payoff = g.payoff_exact if exact else g.payoff
    value = {}
    for nid in reversed(g.order):
        k = g.kind[nid]
        if k == TERMINAL:
            value[nid] = payoff[nid]
        else:
            w = probs[nid] if k == CHANCE else strategy[g.iset[nid]]
            v = 0
            for c, q in zip(g.children[nid], w):
                v += q * value[c]
            value[nid] = v
    return value[g.root]


def exploitability(game, strategy, exact=False):
    b0 = best_response_value(game, strategy, 0, exact)
    b1 = best_response_value(game, strategy, 1, exact)
    return (b0 + b1) / 2, b0, b1


def to_output(strategy):
    """Round to clean decimal probabilities that sum to exactly 1 per infoset."""
    scale = 10 ** 12
    out = []
    for s in strategy:
        ints = [int(round(x * scale)) for x in s]
        ints[max(range(len(ints)), key=ints.__getitem__)] += scale - sum(ints)
        out.append([Fraction(x, scale) for x in ints])
    return out


def frac_to_decimal(x, digits=12):
    n = int(x * 10 ** digits)
    s = str(n).rjust(digits + 1, "0")
    whole, frac = s[:-digits], s[-digits:].rstrip("0")
    return whole + ("." + frac if frac else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="?", default="game.json")
    ap.add_argument("out", nargs="?", default="strategy.json")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--target", type=float, default=1e-6, help="stop once exploitability is below this")
    ap.add_argument("--check-every", type=int, default=250)
    ap.add_argument("--time-limit", type=float, default=600.0, help="seconds")
    args = ap.parse_args()

    sys.setrecursionlimit(100000)
    game = Game(args.game)
    print("game %r: %d nodes, %d infosets (%d / %d)" % (
        game.name, len(game.order), len(game.iset_names),
        game.iset_player.count(0), game.iset_player.count(1)), flush=True)

    solver = CFRPlus(game)
    start = time.time()
    while solver.t < args.iters:
        solver.iterate()
        if solver.t % args.check_every == 0:
            expl, b0, b1 = exploitability(game, solver.average())
            print("iter %6d  exploitability %.3e  br0 %+.8f  br1 %+.8f  %.1fs" % (
                solver.t, expl, b0, b1, time.time() - start), flush=True)
            if expl < args.target or time.time() - start > args.time_limit:
                break

    final = to_output(solver.average())
    probs = {}
    for i, name in enumerate(game.iset_names):
        probs[name] = {label: frac_to_decimal(q) for label, q in zip(game.iset_actions[i], final[i])}
    with open(args.out, "w") as f:
        json.dump({
            "format": "bluff-strategy-v1",
            "game": game.name,
            "producer": "solve.py: CFR+ (alternating updates, linear averaging), %d iterations" % solver.t,
            "probs": probs,
        }, f, indent=1)

    # Re-read what was written and evaluate that, exactly, with rationals.
    with open(args.out) as f:
        written = json.load(f)["probs"]
    strat = [[Fraction(written[name][label]) for label in game.iset_actions[i]]
             for i, name in enumerate(game.iset_names)]
    for i, s in enumerate(strat):
        if sum(s) != 1 or min(s) < 0:
            raise ValueError("bad distribution at %r" % game.iset_names[i])
    expl, b0, b1 = exploitability(game, strat, exact=True)
    val = profile_value(game, strat, exact=True)
    print("wrote %s after %d iterations" % (args.out, solver.t))
    print("exact exploitability of written strategy: %.12g" % float(expl))
    print("  best response value p0: %.12g   p1: %.12g" % (float(b0), float(b1)))
    print("value to p0 under the written profile: %.12g" % float(val))


if __name__ == "__main__":
    main()

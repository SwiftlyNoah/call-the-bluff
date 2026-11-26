#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in bluff-game-v1 format.

Reads game.json, runs CFR+ (alternating updates, regret matching+, linearly
weighted averaging), writes the average strategy to strategy.json, then
re-reads that file and reports its exact exploitability and game value
computed with rational arithmetic.

Usage: python3 solve.py [--game game.json] [--out strategy.json] [--iters N]
                        [--target EPS] [--time-limit SECONDS]
"""
import argparse
import json
import sys
import time
from decimal import Decimal
from fractions import Fraction

import numpy as np


# ----------------------------------------------------------------------------
# Game loading
# ----------------------------------------------------------------------------

class Game:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        if raw.get("format") != "bluff-game-v1":
            raise ValueError("unexpected game format: %r" % raw.get("format"))
        self.name = raw["name"]
        self.root = raw["root"]
        self.nodes = {n["id"]: n for n in raw["nodes"]}
        if len(self.nodes) != len(raw["nodes"]):
            raise ValueError("duplicate node ids")

        # Infosets in order of first appearance: name -> (player, [labels], [node ids])
        self.infosets = {}
        for n in raw["nodes"]:
            kind = n["kind"]
            if kind == "decision":
                labels = [a["label"] for a in n["actions"]]
                if len(set(labels)) != len(labels) or not labels:
                    raise ValueError("bad action labels at node %r" % n["id"])
                info = self.infosets.get(n["infoset"])
                if info is None:
                    self.infosets[n["infoset"]] = (n["player"], labels, [n["id"]])
                else:
                    if info[0] != n["player"] or info[1] != labels:
                        raise ValueError("inconsistent infoset %r" % n["infoset"])
                    info[2].append(n["id"])
                if n["player"] not in (0, 1):
                    raise ValueError("bad player at node %r" % n["id"])
            elif kind == "chance":
                total = sum(Fraction(o["prob"]) for o in n["outcomes"])
                if total != 1:
                    raise ValueError("chance node %r probs sum to %s" % (n["id"], total))
            elif kind != "terminal":
                raise ValueError("unknown node kind %r" % kind)

        # Breadth-first order from the root (parents always before children).
        self.order = []
        self.depth = {self.root: 0}
        self.parent = {self.root: None}
        frontier = [self.root]
        while frontier:
            self.order.extend(frontier)
            nxt = []
            for nid in frontier:
                for c in self.children(nid):
                    if c in self.depth:
                        raise ValueError("node %r has more than one parent" % c)
                    self.depth[c] = self.depth[nid] + 1
                    self.parent[c] = nid
                    nxt.append(c)
            frontier = nxt
        if len(self.order) != len(self.nodes):
            raise ValueError("%d nodes unreachable from the root"
                             % (len(self.nodes) - len(self.order)))

    def children(self, nid):
        n = self.nodes[nid]
        if n["kind"] == "chance":
            return [o["child"] for o in n["outcomes"]]
        if n["kind"] == "decision":
            return [a["child"] for a in n["actions"]]
        return []

    def check_perfect_recall(self):
        """Return a list of infosets whose nodes have differing own-action histories."""
        seq = {self.root: ((), ())}
        bad = []
        for nid in self.order:
            n = self.nodes[nid]
            s = seq[nid]
            if n["kind"] == "decision":
                p = n["player"]
                for a in n["actions"]:
                    t = list(s)
                    t[p] = s[p] + ((n["infoset"], a["label"]),)
                    seq[a["child"]] = tuple(t)
            elif n["kind"] == "chance":
                for o in n["outcomes"]:
                    seq[o["child"]] = s
        for name, (p, _labels, nids) in self.infosets.items():
            if len({seq[nid][p] for nid in nids}) != 1:
                bad.append(name)
        return bad


# ----------------------------------------------------------------------------
# Vectorized CFR+
# ----------------------------------------------------------------------------

class CFRPlus:
    def __init__(self, game):
        g = self.game = game
        idx = {nid: i for i, nid in enumerate(g.order)}  # dense index, BFS order
        n_nodes = len(g.order)

        self.info_names = list(g.infosets)
        info_idx = {name: i for i, name in enumerate(self.info_names)}
        n_actions = np.array([len(g.infosets[nm][1]) for nm in self.info_names])
        self.info_off = np.concatenate(([0], np.cumsum(n_actions)[:-1])).astype(np.int64)
        self.n_slots = int(n_actions.sum())
        self.slot_info = np.repeat(np.arange(len(self.info_names)), n_actions)
        self.slot_player = np.array(
            [g.infosets[self.info_names[i]][0] for i in self.slot_info])
        self.n_actions = n_actions

        self.payoff = np.zeros(n_nodes)  # player 0's payoff at terminals
        e_par, e_child, e_slot, e_prob, e_player, e_depth = [], [], [], [], [], []
        for nid in g.order:
            n = g.nodes[nid]
            i = idx[nid]
            d = g.depth[nid]
            if n["kind"] == "terminal":
                self.payoff[i] = float(Fraction(n["payoff_p0"]))
            elif n["kind"] == "chance":
                for o in n["outcomes"]:
                    e_par.append(i); e_child.append(idx[o["child"]])
                    e_slot.append(-1); e_prob.append(float(Fraction(o["prob"])))
                    e_player.append(-1); e_depth.append(d)
            else:
                off = self.info_off[info_idx[n["infoset"]]]
                for k, a in enumerate(n["actions"]):
                    e_par.append(i); e_child.append(idx[a["child"]])
                    e_slot.append(off + k); e_prob.append(0.0)
                    e_player.append(n["player"]); e_depth.append(d)

        e_depth = np.array(e_depth)
        order = np.argsort(e_depth, kind="stable")
        self.e_par = np.array(e_par)[order]
        self.e_child = np.array(e_child)[order]
        self.e_slot = np.array(e_slot)[order]
        self.e_chance_prob = np.array(e_prob)[order]
        self.e_player = np.array(e_player)[order]
        e_depth = e_depth[order]
        self.n_nodes = n_nodes
        self.e_is_dec = self.e_slot >= 0
        self.e_dec_slot = self.e_slot[self.e_is_dec]
        # Edge ranges per parent depth.
        max_d = int(e_depth.max())
        bounds = np.searchsorted(e_depth, np.arange(max_d + 2))
        self.levels = [(int(bounds[d]), int(bounds[d + 1])) for d in range(max_d + 1)]
        self.e_mask = [self.e_player == p for p in (0, 1)]

        self.regret = np.zeros(self.n_slots)
        self.strat_sum = np.zeros(self.n_slots)
        self.t = 0

    def _normalize(self, x):
        """Per-infoset normalization; uniform where the infoset total is zero."""
        tot = np.add.reduceat(x, self.info_off)
        tot_s = tot[self.slot_info]
        uniform = 1.0 / self.n_actions[self.slot_info]
        return np.where(tot_s > 0, x / np.where(tot_s > 0, tot_s, 1.0), uniform)

    def current_strategy(self):
        return self._normalize(np.maximum(self.regret, 0.0))

    def average_strategy(self):
        return self._normalize(self.strat_sum)

    def _edge_probs(self, sigma):
        w = self.e_chance_prob.copy()
        w[self.e_is_dec] = sigma[self.e_dec_slot]
        return w

    def _update(self, p, weight):
        sigma = self.current_strategy()
        w = self._edge_probs(sigma)
        own = self.e_mask[p]
        w_opp = np.where(own, 1.0, w)   # opponent and chance probabilities
        w_own = np.where(own, w, 1.0)

        # Top-down reach probabilities.
        reach_opp = np.ones(self.n_nodes)
        reach_own = np.ones(self.n_nodes)
        for lo, hi in self.levels:
            par = self.e_par[lo:hi]
            ch = self.e_child[lo:hi]
            reach_opp[ch] = reach_opp[par] * w_opp[lo:hi]
            reach_own[ch] = reach_own[par] * w_own[lo:hi]

        # Bottom-up expected values for player p under the current profile.
        v = self.payoff if p == 0 else -self.payoff
        v = v.copy()
        for lo, hi in reversed(self.levels):
            v += np.bincount(self.e_par[lo:hi], weights=w[lo:hi] * v[self.e_child[lo:hi]],
                             minlength=self.n_nodes)

        # Counterfactual action values and regrets at p's infosets.
        slots = self.e_slot[own]
        q = np.bincount(slots, weights=reach_opp[self.e_par[own]] * v[self.e_child[own]],
                        minlength=self.n_slots)
        ev = np.add.reduceat(sigma * q, self.info_off)[self.slot_info]
        mine = self.slot_player == p
        self.regret[mine] = np.maximum(self.regret[mine] + (q - ev)[mine], 0.0)

        # Linearly weighted average strategy (own reach summed over the infoset's nodes).
        self.strat_sum += weight * np.bincount(
            slots, weights=reach_own[self.e_par[own]] * sigma[slots], minlength=self.n_slots)

    def iterate(self, n):
        for _ in range(n):
            self.t += 1
            self._update(0, self.t)
            self._update(1, self.t)

    # Fast float evaluation of a strategy given per-slot probabilities.
    def evaluate(self, sigma):
        """Return (value_p0, br_value_p0, br_value_p1) for the slot strategy sigma."""
        w = self._edge_probs(sigma)
        v = self.payoff.copy()
        for lo, hi in reversed(self.levels):
            v += np.bincount(self.e_par[lo:hi], weights=w[lo:hi] * v[self.e_child[lo:hi]],
                             minlength=self.n_nodes)
        value = v[0]
        br = []
        for p in (0, 1):
            own = self.e_mask[p]
            w_opp = np.where(own, 1.0, w)
            reach_opp = np.ones(self.n_nodes)
            for lo, hi in self.levels:
                reach_opp[self.e_child[lo:hi]] = reach_opp[self.e_par[lo:hi]] * w_opp[lo:hi]
            # Counterfactual-value-weighted backward pass; requires that all nodes of
            # an infoset sit at the same depth (checked by the caller before use).
            u = (self.payoff if p == 0 else -self.payoff) * reach_opp
            for lo, hi in reversed(self.levels):
                par = self.e_par[lo:hi]; ch = self.e_child[lo:hi]
                o = own[lo:hi]
                # Non-own nodes: sum children (reach already folded into leaves).
                u += np.bincount(par[~o], weights=u[ch[~o]], minlength=self.n_nodes)
                if o.any():
                    q = np.bincount(self.e_slot[lo:hi][o], weights=u[ch[o]],
                                    minlength=self.n_slots)
                    best = np.zeros(self.n_slots, dtype=bool)
                    # first maximizing slot of each touched infoset
                    touched = np.unique(self.slot_info[self.e_slot[lo:hi][o]])
                    for i in touched:
                        a = self.info_off[i]
                        k = int(np.argmax(q[a:a + self.n_actions[i]]))
                        best[a + k] = True
                    sel = best[self.e_slot[lo:hi]] & o
                    u += np.bincount(par[sel], weights=u[ch[sel]], minlength=self.n_nodes)
            br.append(u[0])
        return value, br[0], br[1]


# ----------------------------------------------------------------------------
# Exact (rational) evaluation of a strategy file
# ----------------------------------------------------------------------------

def exact_evaluate(game, probs):
    """probs: infoset name -> {label: Fraction}. Returns (value_p0, br0, br1) as Fractions.

    br_p is the best-response value for player p against the other player's strategy.
    """
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    nodes = game.nodes

    def edge_list(nid):
        n = nodes[nid]
        if n["kind"] == "chance":
            return [(Fraction(o["prob"]), o["child"]) for o in n["outcomes"]]
        s = probs[n["infoset"]]
        return [(s[a["label"]], a["child"]) for a in n["actions"]]

    def value(nid):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return Fraction(n["payoff_p0"])
        return sum(pr * value(c) for pr, c in edge_list(nid) if pr != 0)

    result = [value(game.root)]

    for p in (0, 1):
        sign = 1 if p == 0 else -1
        # Opponent+chance reach of every node.
        reach = {game.root: Fraction(1)}
        for nid in game.order:
            n = nodes[nid]
            if n["kind"] == "terminal":
                continue
            own = n["kind"] == "decision" and n["player"] == p
            for pr, c in edge_list(nid):
                reach[c] = reach[nid] if own else reach[nid] * pr
        best = {}
        memo = {}

        def br_node(nid):
            """Best-response value of the subtree at nid (not reach-weighted)."""
            if nid in memo:
                return memo[nid]
            n = nodes[nid]
            if n["kind"] == "terminal":
                r = sign * Fraction(n["payoff_p0"])
            elif n["kind"] == "decision" and n["player"] == p:
                r = br_node(n["actions"][br_action(n["infoset"])]["child"])
            else:
                r = sum((pr * br_node(c) for pr, c in edge_list(nid) if pr != 0), Fraction(0))
            memo[nid] = r
            return r

        def br_action(name):
            if name in best:
                return best[name]
            _pl, labels, nids = game.infosets[name]
            live = [nid for nid in nids if reach[nid] != 0]
            if not live:
                live_vals = [Fraction(0)] * len(labels)
            else:
                live_vals = [
                    sum(reach[nid] * br_node(nodes[nid]["actions"][k]["child"]) for nid in live)
                    for k in range(len(labels))]
            best[name] = max(range(len(labels)), key=lambda k: live_vals[k])
            return best[name]

        result.append(br_node(game.root))
    return tuple(result)


# ----------------------------------------------------------------------------
# Strategy output
# ----------------------------------------------------------------------------

def strategy_to_decimal_strings(solver, sigma, digits=15):
    """Round to `digits` decimals and fix the largest entry so each infoset sums to exactly 1."""
    g = solver.game
    quant = Decimal(1).scaleb(-digits)
    probs = {}
    for i, name in enumerate(solver.info_names):
        labels = g.infosets[name][1]
        a = int(solver.info_off[i])
        vals = [max(Decimal(repr(float(x))).quantize(quant), Decimal(0))
                for x in sigma[a:a + len(labels)]]
        k = max(range(len(vals)), key=lambda j: vals[j])
        vals[k] = Decimal(1) - sum(v for j, v in enumerate(vals) if j != k)
        assert all(0 <= v <= 1 for v in vals) and sum(vals) == 1, (name, vals)
        probs[name] = {lab: format(v, "f") for lab, v in zip(labels, vals)}
    return probs


def load_strategy(game, path):
    with open(path) as f:
        raw = json.load(f)
    probs = {}
    for name, (_p, labels, _nids) in game.infosets.items():
        entry = raw["probs"][name]
        if set(entry) != set(labels):
            raise ValueError("infoset %r has wrong action set" % name)
        fr = {lab: Fraction(entry[lab]) for lab in labels}
        if sum(fr.values()) != 1 or any(v < 0 for v in fr.values()):
            raise ValueError("infoset %r is not a probability distribution" % name)
        probs[name] = fr
    extra = set(raw["probs"]) - set(game.infosets)
    if extra:
        raise ValueError("unknown infosets in strategy: %r" % sorted(extra)[:5])
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default="game.json")
    ap.add_argument("--out", default="strategy.json")
    ap.add_argument("--iters", type=int, default=100000, help="maximum CFR+ iterations")
    ap.add_argument("--target", type=float, default=1e-6,
                    help="stop once exploitability drops below this")
    ap.add_argument("--time-limit", type=float, default=240.0, help="seconds of CFR+ time")
    ap.add_argument("--check-every", type=int, default=500)
    args = ap.parse_args()

    t0 = time.time()
    game = Game(args.game)
    bad = game.check_perfect_recall()
    print("game %r: %d nodes, %d infosets, perfect recall: %s"
          % (game.name, len(game.nodes), len(game.infosets), "yes" if not bad else
             "NO (%d infosets)" % len(bad)), flush=True)
    same_depth = all(len({game.depth[n] for n in nids}) == 1
                     for _p, _l, nids in game.infosets.values())

    solver = CFRPlus(game)
    probs_of = lambda sigma: {
        name: {lab: Fraction(float(sigma[int(solver.info_off[i]) + k]))
               for k, lab in enumerate(game.infosets[name][1])}
        for i, name in enumerate(solver.info_names)}

    def evaluate(sigma):
        if same_depth and not bad:
            return solver.evaluate(sigma)
        return tuple(float(x) for x in exact_evaluate(game, probs_of(sigma)))

    start = time.time()
    while solver.t < args.iters:
        solver.iterate(min(args.check_every, args.iters - solver.t))
        value, br0, br1 = evaluate(solver.average_strategy())
        expl = (br0 + br1) / 2
        print("iter %7d  value_p0 %+.9f  br0 %+.9f  br1 %+.9f  exploitability %.3e  (%.1fs)"
              % (solver.t, value, br0, br1, expl, time.time() - start), flush=True)
        if expl < args.target or time.time() - start > args.time_limit:
            break

    probs = strategy_to_decimal_strings(solver, solver.average_strategy())
    out = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": "solve.py: CFR+ (alternating updates, regret matching+, linear averaging), "
                    "%d iterations" % solver.t,
        "probs": probs,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote %s" % args.out, flush=True)

    # Independent exact check of the file as written.
    value, br0, br1 = exact_evaluate(game, load_strategy(game, args.out))
    expl = (br0 + br1) / 2
    print("exact check of %s:" % args.out)
    print("  value_p0        = %.12f" % float(value))
    print("  br value p0     = %.12f" % float(br0))
    print("  br value p1     = %.12f" % float(br1))
    print("  exploitability  = %.6e" % float(expl))
    print("total time %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

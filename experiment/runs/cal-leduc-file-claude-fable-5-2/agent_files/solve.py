#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in bluff-game-v1 format.

Reads the game tree from game.json (nothing about the game is hard-coded), runs CFR+
(regret matching+, alternating updates, linearly weighted averaging) for both players,
writes the average strategy to strategy.json, and then re-reads that file and reports its
exact exploitability and value using rational arithmetic.

Usage: python3 solve.py [--game game.json] [--out strategy.json]
                        [--max-iters N] [--target EPS] [--max-seconds S]
"""
import argparse
import json
import sys
import time
from fractions import Fraction

import numpy as np

DIGITS = 15  # decimal digits written per probability


# ----------------------------------------------------------------------------------------
# Game loading
# ----------------------------------------------------------------------------------------
class Game:
    """The tree, re-indexed in breadth-first order so every depth level is a slice."""

    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.name = raw.get("name", "")
        by_id = {n["id"]: n for n in raw["nodes"]}

        order = [raw["root"]]          # raw ids in BFS order
        parent = [-1]                  # BFS index of the parent
        depth = [0]
        edge = [None]                  # incoming edge: ("c", prob) or ("d", infoset, action idx)
        i = 0
        while i < len(order):
            n = by_id[order[i]]
            if n["kind"] == "chance":
                for o in n["outcomes"]:
                    order.append(o["child"])
                    parent.append(i)
                    depth.append(depth[i] + 1)
                    edge.append(("c", Fraction(o["prob"])))
            elif n["kind"] == "decision":
                for k, a in enumerate(n["actions"]):
                    order.append(a["child"])
                    parent.append(i)
                    depth.append(depth[i] + 1)
                    edge.append(("d", n["infoset"], k))
            elif n["kind"] != "terminal":
                raise ValueError("unknown node kind %r" % n["kind"])
            i += 1
        if len(order) != len(by_id) or len(set(order)) != len(order):
            raise ValueError("game.json is not a tree reachable from the root")

        N = self.N = len(order)
        self.nodes = [by_id[r] for r in order]
        self.parent = parent
        self.edge = edge
        index_of = {r: i for i, r in enumerate(order)}
        self.children = []
        for n in self.nodes:
            if n["kind"] == "chance":
                self.children.append([index_of[o["child"]] for o in n["outcomes"]])
            elif n["kind"] == "decision":
                self.children.append([index_of[a["child"]] for a in n["actions"]])
            else:
                self.children.append([])
        self.kind = [n["kind"] for n in self.nodes]
        self.payoff = [Fraction(n["payoff_p0"]) if n["kind"] == "terminal" else None
                       for n in self.nodes]
        self.chance_probs = [[Fraction(o["prob"]) for o in n["outcomes"]]
                             if n["kind"] == "chance" else None for n in self.nodes]
        for i, n in enumerate(self.nodes):
            if n["kind"] == "chance" and sum(self.chance_probs[i]) != 1:
                raise ValueError("chance probabilities at node %r do not sum to 1" % n["id"])

        # Information sets.
        self.infosets = []             # names, in order of first appearance
        self.iset_index = {}
        self.iset_player = []
        self.iset_actions = []
        self.iset_nodes = []
        self.node_iset = [-1] * N
        for i, n in enumerate(self.nodes):
            if n["kind"] != "decision":
                continue
            name = n["infoset"]
            labels = [a["label"] for a in n["actions"]]
            if name not in self.iset_index:
                self.iset_index[name] = len(self.infosets)
                self.infosets.append(name)
                self.iset_player.append(n["player"])
                self.iset_actions.append(labels)
                self.iset_nodes.append([])
            k = self.iset_index[name]
            if self.iset_player[k] != n["player"] or self.iset_actions[k] != labels:
                raise ValueError("inconsistent information set %r" % name)
            self.iset_nodes[k].append(i)
            self.node_iset[i] = k
        self.depth = depth
        self.check_perfect_recall()

    def check_perfect_recall(self):
        """Each player must remember their own past (infoset, action) sequence."""
        last = [[None] * self.N for _ in range(2)]
        for i in range(1, self.N):
            p = self.parent[i]
            for pl in range(2):
                last[pl][i] = last[pl][p]
            e = self.edge[i]
            if e[0] == "d":
                k = self.iset_index[e[1]]
                last[self.iset_player[k]][i] = (k, e[2])
        self.perfect_recall = True
        for k, nodes in enumerate(self.iset_nodes):
            pl = self.iset_player[k]
            if len({last[pl][n] for n in nodes}) != 1:
                self.perfect_recall = False
        if not self.perfect_recall:
            print("WARNING: the game does not have perfect recall; CFR has no guarantee",
                  file=sys.stderr)


# ----------------------------------------------------------------------------------------
# Exact / generic evaluation (works with floats or Fractions)
# ----------------------------------------------------------------------------------------
def best_response_value(game, probs, br_player, zero=Fraction(0), one=Fraction(1), conv=None):
    """Value, to br_player, of br_player's best response against the other player's
    strategy in `probs` (list indexed by infoset of lists of probabilities)."""
    conv = conv or (lambda x: x)
    N = game.N
    sign = 1 if br_player == 0 else -1
    # Counterfactual reach: chance and opponent probabilities only.
    reach = [zero] * N
    reach[0] = one
    for i in range(N):
        ch = game.children[i]
        if not ch:
            continue
        r = reach[i]
        if game.kind[i] == "chance":
            for c, p in zip(ch, game.chance_probs[i]):
                reach[c] = r * conv(p)
        else:
            k = game.node_iset[i]
            if game.iset_player[k] == br_player:
                for c in ch:
                    reach[c] = r
            else:
                for c, p in zip(ch, probs[k]):
                    reach[c] = r * p

    val = [None] * N
    choice = {}

    def value(i):
        v = val[i]
        if v is not None:
            return v
        kind = game.kind[i]
        if kind == "terminal":
            v = sign * conv(game.payoff[i])
        elif kind == "chance":
            v = zero
            for c, p in zip(game.children[i], game.chance_probs[i]):
                v += conv(p) * value(c)
        else:
            k = game.node_iset[i]
            if game.iset_player[k] == br_player:
                v = value(game.children[i][best_action(k)])
            else:
                v = zero
                for c, p in zip(game.children[i], probs[k]):
                    if p:
                        v += p * value(c)
        val[i] = v
        return v

    def best_action(k):
        a = choice.get(k)
        if a is None:
            best = None
            for j in range(len(game.iset_actions[k])):
                tot = zero
                for h in game.iset_nodes[k]:
                    if reach[h]:
                        tot += reach[h] * value(game.children[h][j])
                if best is None or tot > best:
                    best, a = tot, j
            choice[k] = a
        return a

    return value(0)


def profile_value(game, probs, zero=Fraction(0), conv=None):
    """Expected payoff to player 0 when both players follow `probs`."""
    conv = conv or (lambda x: x)
    val = [None] * game.N
    for i in range(game.N - 1, -1, -1):       # BFS order: children come after parents
        kind = game.kind[i]
        if kind == "terminal":
            val[i] = conv(game.payoff[i])
        else:
            ps = game.chance_probs[i] if kind == "chance" else probs[game.node_iset[i]]
            v = zero
            for c, p in zip(game.children[i], ps):
                v += conv(p) * val[c]
            val[i] = v
    return val[0]


def float_exploitability(game, probs):
    b0 = best_response_value(game, probs, 0, 0.0, 1.0, float)
    b1 = best_response_value(game, probs, 1, 0.0, 1.0, float)
    return (b0 + b1) / 2, b0, b1


# ----------------------------------------------------------------------------------------
# Vectorized CFR+
# ----------------------------------------------------------------------------------------
class CFRPlus:
    def __init__(self, game):
        g = self.g = game
        N = g.N
        nI = self.nI = len(g.infosets)
        A = self.A = max(len(a) for a in g.iset_actions)
        self.valid = np.zeros((nI, A), dtype=bool)
        for k, acts in enumerate(g.iset_actions):
            self.valid[k, :len(acts)] = True
        self.uniform = self.valid / self.valid.sum(axis=1, keepdims=True)

        self.par = np.array(g.parent, dtype=np.int64)
        self.par[0] = 0
        owner = np.full(N, -1, dtype=np.int64)       # who owns the incoming edge (-1 chance)
        sig = np.zeros(N, dtype=np.int64)            # flat (infoset, action) of incoming edge
        chance_p = np.ones(N)
        for i in range(1, N):
            e = g.edge[i]
            if e[0] == "c":
                chance_p[i] = float(e[1])
            else:
                k = g.iset_index[e[1]]
                owner[i] = g.iset_player[k]
                sig[i] = k * A + e[2]
        self.owner, self.sig, self.chance_p = owner, sig, chance_p
        self.is_dec_edge = owner >= 0
        self.payoff0 = np.array([float(p) if p is not None else 0.0 for p in g.payoff])

        # Depth levels as slices, plus reduceat offsets for summing children into parents.
        depth = np.array(g.depth)
        self.levels = []
        for d in range(depth.max() + 1):
            idx = np.nonzero(depth == d)[0]
            self.levels.append((int(idx[0]), int(idx[-1]) + 1))
        self.internal = []       # internal[d]: internal node indices at depth d
        self.starts = [None]     # starts[d]: offsets of first children within level d
        nchild = np.array([len(c) for c in g.children])
        for d, (s, e) in enumerate(self.levels):
            self.internal.append(s + np.nonzero(nchild[s:e] > 0)[0])
        for d in range(1, len(self.levels)):
            s, e = self.levels[d]
            first = np.array([g.children[i][0] for i in self.internal[d - 1]], dtype=np.int64)
            self.starts.append(first - s)

        self.own = []            # per player: nodes reached by that player's action
        self.rep = []            # per player: (flat sigma indices, one representative child)
        for pl in range(2):
            nodes = np.nonzero(owner == pl)[0]
            self.own.append((nodes, self.par[nodes], sig[nodes]))
            flat, child = [], []
            for k in range(nI):
                if g.iset_player[k] == pl:
                    h = g.iset_nodes[k][0]
                    for j, c in enumerate(g.children[h]):
                        flat.append(k * A + j)
                        child.append(c)
            self.rep.append((np.array(flat, dtype=np.int64), np.array(child, dtype=np.int64)))

        self.R = np.zeros((nI, A))     # cumulative regrets (kept >= 0)
        self.S = np.zeros((nI, A))     # weighted cumulative strategy
        self.t = 0

    def current_strategy(self):
        tot = self.R.sum(axis=1, keepdims=True)
        return np.where(tot > 0, self.R / np.where(tot > 0, tot, 1.0), self.uniform)

    def iterate(self):
        self.t += 1
        t = self.t
        par, levels = self.par, self.levels
        Rflat, Sflat = self.R.reshape(-1), self.S.reshape(-1)
        for pl in range(2):
            sigma = self.current_strategy().reshape(-1)
            p_edge = np.where(self.is_dec_edge, sigma[self.sig], self.chance_p)
            mine = self.owner == pl
            w_cf = np.where(mine, 1.0, p_edge)
            w_own = np.where(mine, p_edge, 1.0)
            r_cf = np.ones(self.g.N)
            r_own = np.ones(self.g.N)
            for s, e in levels[1:]:
                pp = par[s:e]
                r_cf[s:e] = r_cf[pp] * w_cf[s:e]
                r_own[s:e] = r_own[pp] * w_own[s:e]
            v = self.payoff0.copy() if pl == 0 else -self.payoff0
            for d in range(len(levels) - 1, 0, -1):
                s, e = levels[d]
                if len(self.starts[d]):
                    v[self.internal[d - 1]] = np.add.reduceat(p_edge[s:e] * v[s:e], self.starts[d])
            nodes, pnodes, sidx = self.own[pl]
            regret = np.bincount(sidx, weights=r_cf[pnodes] * (v[nodes] - v[pnodes]),
                                 minlength=Rflat.size)
            np.maximum(Rflat + regret, 0.0, out=Rflat)
            flat, child = self.rep[pl]
            Sflat[flat] += t * r_own[child]

    def average_strategy(self):
        tot = self.S.sum(axis=1, keepdims=True)
        return np.where(tot > 0, self.S / np.where(tot > 0, tot, 1.0), self.uniform)


# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------
def to_decimal_strings(ps):
    """Round probabilities to DIGITS decimals so that they sum to exactly 1."""
    D = 10 ** DIGITS
    ints = [max(0, int(round(float(p) * D))) for p in ps]
    j = max(range(len(ints)), key=lambda i: ints[i])
    ints[j] += D - sum(ints)
    assert all(x >= 0 for x in ints) and sum(ints) == D
    return ["%d.%0*d" % (x // D, DIGITS, x % D) for x in ints]


def write_strategy(game, avg, path, producer):
    probs = {}
    for k, name in enumerate(game.infosets):
        acts = game.iset_actions[k]
        probs[name] = dict(zip(acts, to_decimal_strings(avg[k, :len(acts)])))
    with open(path, "w") as f:
        json.dump({"format": "bluff-strategy-v1", "game": game.name, "producer": producer,
                   "probs": probs}, f, indent=1)
        f.write("\n")


def load_strategy(game, path):
    with open(path) as f:
        data = json.load(f)
    out = []
    for k, name in enumerate(game.infosets):
        row = data["probs"][name]
        acts = game.iset_actions[k]
        if set(row) != set(acts):
            raise ValueError("wrong action set at %r" % name)
        ps = [Fraction(row[a]) for a in acts]
        if sum(ps) != 1 or min(ps) < 0:
            raise ValueError("probabilities at %r are not a distribution" % name)
        out.append(ps)
    if set(data["probs"]) != set(game.infosets):
        raise ValueError("strategy has unknown information sets")
    return out


def exact_report(game, path):
    probs = load_strategy(game, path)
    b0 = best_response_value(game, probs, 0)
    b1 = best_response_value(game, probs, 1)
    return {"exploitability": float((b0 + b1) / 2), "br_value_p0": float(b0),
            "br_value_p1": float(b1), "value_p0": float(profile_value(game, probs))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default="game.json")
    ap.add_argument("--out", default="strategy.json")
    ap.add_argument("--max-iters", type=int, default=200000)
    ap.add_argument("--target", type=float, default=1e-6,
                    help="stop once the exploitability of the average strategy is below this")
    ap.add_argument("--max-seconds", type=float, default=420.0)
    ap.add_argument("--check-every", type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    game = Game(args.game)
    print("game %r: %d nodes, %d information sets, perfect recall: %s"
          % (game.name, game.N, len(game.infosets), game.perfect_recall), flush=True)
    cfr = CFRPlus(game)

    def as_lists(avg):
        return [[float(x) for x in avg[k, :len(a)]] for k, a in enumerate(game.iset_actions)]

    expl = None
    while cfr.t < args.max_iters:
        cfr.iterate()
        if cfr.t % args.check_every == 0 or cfr.t == args.max_iters:
            expl, b0, b1 = float_exploitability(game, as_lists(cfr.average_strategy()))
            print("iter %7d  exploitability %.3e  (BR0 %+.8f, BR1 %+.8f)  %.1fs"
                  % (cfr.t, expl, b0, b1, time.time() - t0), flush=True)
            if expl <= args.target or time.time() - t0 > args.max_seconds:
                break

    producer = ("solve.py: CFR+ (regret matching+, alternating updates, linear averaging), "
                "%d iterations" % cfr.t)
    write_strategy(game, cfr.average_strategy(), args.out, producer)
    rep = exact_report(game, args.out)
    rep["iterations"] = cfr.t
    rep["seconds"] = round(time.time() - t0, 1)
    print("wrote %s" % args.out)
    print("exact evaluation of the written file: " + json.dumps(rep))


if __name__ == "__main__":
    main()

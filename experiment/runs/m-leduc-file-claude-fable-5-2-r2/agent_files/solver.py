#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in bluff-game-v1 format.

Reads game.json, runs CFR+ (alternating updates, regret matching+, linearly
weighted average strategy), and writes the average strategy to strategy.json.

Nothing about the game is hard-coded: the tree, the information sets, the
chance probabilities and the payoffs all come from the game file.

Usage: python3 solver.py [--game game.json] [--out strategy.json]
                         [--target 1e-5] [--max-iters 100000] [--max-seconds 300]
"""
import argparse
import json
import sys
import time
from fractions import Fraction

import numpy as np


# ----------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------

class Game:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        if raw.get("format") != "bluff-game-v1":
            raise ValueError(f"unexpected game format: {raw.get('format')!r}")
        self.name = raw["name"]
        nodes = raw["nodes"]
        ids = [n["id"] for n in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node ids")
        index = {nid: i for i, nid in enumerate(ids)}  # node id -> dense index
        n = len(nodes)
        self.n = n
        self.root = index[raw["root"]]

        self.kind = [None] * n          # 'c', 'd', 't'
        self.player = [-1] * n          # acting player of decision nodes
        self.children = [()] * n        # dense child indices
        self.chance_prob = [()] * n     # float chance probabilities
        self.payoff = np.zeros(n)       # payoff to player 0 at terminals
        self.node_infoset = [-1] * n    # infoset index of decision nodes

        self.infoset_names = []         # infoset index -> name
        self.infoset_actions = []       # infoset index -> tuple of labels
        self.infoset_player = []
        self.infoset_nodes = []
        infoset_index = {}

        for i, nd in enumerate(nodes):
            kind = nd["kind"]
            if kind == "terminal":
                self.kind[i] = "t"
                self.payoff[i] = float(Fraction(nd["payoff_p0"]))
            elif kind == "chance":
                self.kind[i] = "c"
                probs = [Fraction(o["prob"]) for o in nd["outcomes"]]
                if sum(probs) != 1 or any(p < 0 for p in probs):
                    raise ValueError(f"chance node {nd['id']}: bad probabilities")
                self.children[i] = tuple(index[o["child"]] for o in nd["outcomes"])
                self.chance_prob[i] = tuple(float(p) for p in probs)
            elif kind == "decision":
                self.kind[i] = "d"
                player = nd["player"]
                if player not in (0, 1):
                    raise ValueError(f"node {nd['id']}: bad player {player!r}")
                labels = tuple(a["label"] for a in nd["actions"])
                if len(set(labels)) != len(labels) or not labels:
                    raise ValueError(f"node {nd['id']}: bad action labels")
                name = nd["infoset"]
                k = infoset_index.get(name)
                if k is None:
                    k = len(self.infoset_names)
                    infoset_index[name] = k
                    self.infoset_names.append(name)
                    self.infoset_actions.append(labels)
                    self.infoset_player.append(player)
                    self.infoset_nodes.append([])
                elif self.infoset_actions[k] != labels or self.infoset_player[k] != player:
                    raise ValueError(f"infoset {name!r}: inconsistent player or actions")
                self.player[i] = player
                self.node_infoset[i] = k
                self.infoset_nodes[k].append(i)
                self.children[i] = tuple(index[a["child"]] for a in nd["actions"])
            else:
                raise ValueError(f"node {nd['id']}: unknown kind {kind!r}")

        # Slots: one per (infoset, action), laid out contiguously per infoset.
        self.infoset_offset = []
        off = 0
        for acts in self.infoset_actions:
            self.infoset_offset.append(off)
            off += len(acts)
        self.num_slots = off
        self.num_infosets = len(self.infoset_names)

        # Depths (also checks that the nodes form a tree rooted at root).
        depth = [-1] * n
        depth[self.root] = 0
        order = [self.root]
        for u in order:
            for c in self.children[u]:
                if depth[c] != -1:
                    raise ValueError("node has more than one parent: not a tree")
                depth[c] = depth[u] + 1
                order.append(c)
        if len(order) != n:
            raise ValueError("some nodes are unreachable from the root")
        self.depth = depth
        self.order = order  # BFS order: parents before children

        self._check_perfect_recall()

    def _check_perfect_recall(self):
        """Each player must remember their own past infosets and actions.

        CFR's guarantees need this. We raise if it is violated.
        """
        seq = [None] * self.n  # per node: (own-history of p0, own-history of p1)
        seq[self.root] = ((), ())
        seen = {}
        for u in self.order:
            if self.kind[u] == "d":
                p = self.player[u]
                k = self.node_infoset[u]
                if seen.setdefault(k, seq[u][p]) != seq[u][p]:
                    raise ValueError(
                        f"infoset {self.infoset_names[k]!r} violates perfect recall")
                for a, c in enumerate(self.children[u]):
                    s = list(seq[u])
                    s[p] = s[p] + ((k, a),)
                    seq[c] = tuple(s)
            else:
                for c in self.children[u]:
                    seq[c] = seq[u]


# ----------------------------------------------------------------------------
# Vectorized CFR+
# ----------------------------------------------------------------------------

class CFRPlus:
    def __init__(self, game):
        g = self.g = game
        maxd = max(g.depth)
        # Edges grouped by the depth of their parent.
        per_level = [[] for _ in range(maxd + 1)]
        for u in range(g.n):
            if g.kind[u] == "c":
                for c, pr in zip(g.children[u], g.chance_prob[u]):
                    per_level[g.depth[u]].append((u, c, -1, -1, pr))
            elif g.kind[u] == "d":
                off = g.infoset_offset[g.node_infoset[u]]
                for a, c in enumerate(g.children[u]):
                    per_level[g.depth[u]].append((u, c, g.player[u], off + a, 1.0))
        self.levels = []
        for edges in per_level:
            if not edges:
                continue
            par = np.array([e[0] for e in edges])
            child = np.array([e[1] for e in edges])
            owner = np.array([e[2] for e in edges])
            slot = np.array([e[3] for e in edges])
            prob = np.array([e[4] for e in edges])
            upar, par_local = np.unique(par, return_inverse=True)
            self.levels.append(dict(
                par=par, child=child, prob=prob, upar=upar, par_local=par_local,
                dec=np.nonzero(owner >= 0)[0], dec_slot=slot[owner >= 0],
                own=[np.nonzero(owner == p)[0] for p in (0, 1)],
                notown=[np.nonzero(owner != p)[0] for p in (0, 1)],
            ))
        # All decision edges of each player, for regret and average updates.
        self.pe = []
        for p in (0, 1):
            par, child, slot = [], [], []
            for u in range(g.n):
                if g.kind[u] == "d" and g.player[u] == p:
                    off = g.infoset_offset[g.node_infoset[u]]
                    for a, c in enumerate(g.children[u]):
                        par.append(u); child.append(c); slot.append(off + a)
            self.pe.append((np.array(par), np.array(child), np.array(slot)))

        self.slot_infoset = np.concatenate([
            np.full(len(acts), k) for k, acts in enumerate(g.infoset_actions)])
        self.slot_nact = np.array([
            len(g.infoset_actions[k]) for k in self.slot_infoset], dtype=float)
        self.regret = np.zeros(g.num_slots)
        self.strategy_sum = np.zeros(g.num_slots)
        self.t = 0

    def _normalize(self, x):
        """Per-infoset normalization of nonnegative slot weights; uniform if all zero."""
        tot = np.bincount(self.slot_infoset, weights=x, minlength=self.g.num_infosets)
        tot = tot[self.slot_infoset]
        pos = tot > 0
        return np.where(pos, x / np.where(pos, tot, 1.0), 1.0 / self.slot_nact)

    def current_strategy(self):
        return self._normalize(self.regret)

    def average_strategy(self):
        return self._normalize(self.strategy_sum)

    def _update(self, p):
        g = self.g
        sigma = self.current_strategy()
        own = np.zeros(g.n)   # product of player p's own action probabilities
        cf = np.zeros(g.n)    # product of opponent and chance probabilities
        own[g.root] = 1.0
        cf[g.root] = 1.0
        weights = []
        for L in self.levels:
            w = L["prob"].copy()
            w[L["dec"]] = sigma[L["dec_slot"]]
            weights.append(w)
            po = own[L["par"]]
            pc = cf[L["par"]]
            o, no = L["own"][p], L["notown"][p]
            po[o] *= w[o]
            pc[no] *= w[no]
            own[L["child"]] = po
            cf[L["child"]] = pc
        v = g.payoff.copy()   # expected payoff to player 0 under sigma
        for L, w in zip(reversed(self.levels), reversed(weights)):
            v[L["upar"]] = np.bincount(
                L["par_local"], weights=w * v[L["child"]], minlength=len(L["upar"]))
        par, child, slot = self.pe[p]
        sign = 1.0 if p == 0 else -1.0
        inst = cf[par] * sign * (v[child] - v[par])
        delta = np.bincount(slot, weights=inst, minlength=g.num_slots)
        # Average strategy: weight iteration t linearly, by own reach probability.
        self.strategy_sum += self.t * np.bincount(
            slot, weights=own[par] * sigma[slot], minlength=g.num_slots)
        # Regret matching+: clip cumulative regrets at zero.
        self.regret = np.maximum(self.regret + delta, 0.0)

    def iterate(self):
        self.t += 1
        self._update(0)
        self._update(1)


# ----------------------------------------------------------------------------
# Evaluation (plain Python, separate from the vectorized code above)
# ----------------------------------------------------------------------------

def strategy_value(g, sigma):
    """Expected payoff to player 0 when both players follow sigma (slot array)."""
    v = [0.0] * g.n
    for u in reversed(g.order):
        if g.kind[u] == "t":
            v[u] = g.payoff[u]
        elif g.kind[u] == "c":
            v[u] = sum(pr * v[c] for pr, c in zip(g.chance_prob[u], g.children[u]))
        else:
            off = g.infoset_offset[g.node_infoset[u]]
            v[u] = sum(sigma[off + a] * v[c] for a, c in enumerate(g.children[u]))
    return v[g.root]


def best_response_value(g, sigma, p):
    """Value to player p of p's best response against sigma (the opponent's part)."""
    # cf[u]: probability of reaching u from chance and the opponent alone.
    cf = [0.0] * g.n
    cf[g.root] = 1.0
    for u in g.order:
        if g.kind[u] == "c":
            for pr, c in zip(g.chance_prob[u], g.children[u]):
                cf[c] = cf[u] * pr
        elif g.kind[u] == "d":
            if g.player[u] == p:
                for c in g.children[u]:
                    cf[c] = cf[u]
            else:
                off = g.infoset_offset[g.node_infoset[u]]
                for a, c in enumerate(g.children[u]):
                    cf[c] = cf[u] * sigma[off + a]
    sign = 1.0 if p == 0 else -1.0
    choice = {}
    memo = {}

    def val(u):
        r = memo.get(u)
        if r is not None:
            return r
        if g.kind[u] == "t":
            r = sign * g.payoff[u]
        elif g.kind[u] == "c":
            r = sum(pr * val(c) for pr, c in zip(g.chance_prob[u], g.children[u]))
        elif g.player[u] != p:
            off = g.infoset_offset[g.node_infoset[u]]
            r = sum(sigma[off + a] * val(c) for a, c in enumerate(g.children[u]))
        else:
            r = val(g.children[u][best(g.node_infoset[u])])
        memo[u] = r
        return r

    def best(k):
        a = choice.get(k)
        if a is None:
            nodes = g.infoset_nodes[k]
            totals = [sum(cf[h] * val(g.children[h][a]) for h in nodes)
                      for a in range(len(g.infoset_actions[k]))]
            a = max(range(len(totals)), key=totals.__getitem__)
            choice[k] = a
        return a

    return val(g.root)


def exploitability(g, sigma):
    return (best_response_value(g, sigma, 0) + best_response_value(g, sigma, 1)) / 2


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

DIGITS = 12


def exact_decimals(probs):
    """Round probabilities to DIGITS decimals so that they sum to exactly 1."""
    scale = 10 ** DIGITS
    ints = [max(0, int(round(float(x) * scale))) for x in probs]
    ints[max(range(len(ints)), key=ints.__getitem__)] += scale - sum(ints)
    assert sum(ints) == scale and all(i >= 0 for i in ints)
    out = []
    for i in ints:
        whole, frac = divmod(i, scale)
        out.append(f"{whole}.{frac:0{DIGITS}d}")
    return out


def write_strategy(g, sigma, path, producer):
    probs = {}
    rounded = np.zeros(g.num_slots)
    for k, name in enumerate(g.infoset_names):
        off = g.infoset_offset[k]
        acts = g.infoset_actions[k]
        strs = exact_decimals(sigma[off:off + len(acts)])
        probs[name] = dict(zip(acts, strs))
        rounded[off:off + len(acts)] = [float(s) for s in strs]
    with open(path, "w") as f:
        json.dump({"format": "bluff-strategy-v1", "game": g.name,
                   "producer": producer, "probs": probs}, f, indent=1)
        f.write("\n")
    return rounded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="game.json")
    ap.add_argument("--out", default="strategy.json")
    ap.add_argument("--target", type=float, default=1e-5,
                    help="stop when exploitability falls below this")
    ap.add_argument("--max-iters", type=int, default=100000)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--check-every", type=int, default=250)
    args = ap.parse_args()

    g = Game(args.game)
    print(f"game {g.name!r}: {g.n} nodes, {g.num_infosets} infosets, "
          f"{g.num_slots} infoset-actions", file=sys.stderr)
    solver = CFRPlus(g)
    start = time.time()
    expl = None
    while solver.t < args.max_iters:
        solver.iterate()
        if solver.t % args.check_every == 0:
            avg = solver.average_strategy()
            expl = exploitability(g, avg)
            print(f"iter {solver.t:7d}  exploitability {expl:.3e}  "
                  f"value_p0 {strategy_value(g, avg):+.8f}  "
                  f"{time.time() - start:6.1f}s", file=sys.stderr)
            if expl < args.target or time.time() - start > args.max_seconds:
                break

    producer = (f"solver.py: CFR+ (alternating updates, linear averaging), "
                f"{solver.t} iterations")
    final = write_strategy(g, solver.average_strategy(), args.out, producer)
    # Report numbers for the strategy exactly as written (after rounding).
    print(json.dumps({
        "iterations": solver.t,
        "exploitability": exploitability(g, final),
        "br_value_p0": best_response_value(g, final, 0),
        "br_value_p1": best_response_value(g, final, 1),
        "value_p0": strategy_value(g, final),
        "seconds": round(time.time() - start, 1),
    }))


if __name__ == "__main__":
    main()

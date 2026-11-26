#!/usr/bin/env python3
"""Generic CFR+ solver for bluff-game-v1 trees. Reads game.json, writes strategy.json.

Uses alternating-update CFR+ (regret matching+, linear averaging) with array-based
forward/backward passes over a topologically ordered node list. Also computes the
exact best-response exploitability of the resulting average strategy.
"""
import json
import sys
import time
from fractions import Fraction

import numpy as np


def load(path):
    g = json.load(open(path))
    nodes = {nd["id"]: nd for nd in g["nodes"]}
    order = []  # parents before children (BFS from root)
    parent = {g["root"]: -1}
    stack = [g["root"]]
    while stack:
        nid = stack.pop()
        order.append(nid)
        nd = nodes[nid]
        kids = []
        if nd["kind"] == "chance":
            kids = [o["child"] for o in nd["outcomes"]]
        elif nd["kind"] == "decision":
            kids = [a["child"] for a in nd["actions"]]
        for c in kids:
            parent[c] = nid
            stack.append(c)
    return g, nodes, order


class Game:
    def __init__(self, path):
        g, nodes, order = load(path)
        self.name = g["name"]
        idx = {nid: i for i, nid in enumerate(order)}
        N = len(order)
        self.N = N
        self.kind = []
        self.player = np.full(N, -1)
        self.payoff = np.zeros(N)
        self.children = [[] for _ in range(N)]
        self.chance_p = [None] * N
        self.node_iset = np.full(N, -1)
        self.isets = []  # (name, player, labels)
        iset_idx = {}
        self.iset_nodes = []
        for i, nid in enumerate(order):
            nd = nodes[nid]
            self.kind.append(nd["kind"])
            if nd["kind"] == "terminal":
                self.payoff[i] = float(Fraction(nd["payoff_p0"]))
            elif nd["kind"] == "chance":
                self.children[i] = [idx[o["child"]] for o in nd["outcomes"]]
                self.chance_p[i] = [float(Fraction(o["prob"])) for o in nd["outcomes"]]
            else:
                self.children[i] = [idx[a["child"]] for a in nd["actions"]]
                labels = [a["label"] for a in nd["actions"]]
                name = nd["infoset"]
                if name not in iset_idx:
                    iset_idx[name] = len(self.isets)
                    self.isets.append((name, nd["player"], labels))
                    self.iset_nodes.append([])
                k = iset_idx[name]
                assert self.isets[k][1] == nd["player"] and self.isets[k][2] == labels
                self.node_iset[i] = k
                self.player[i] = nd["player"]
                self.iset_nodes[k].append(i)
        self.M = len(self.isets)

    def uniform(self):
        return [np.full(len(l), 1.0 / len(l)) for (_, _, l) in self.isets]

    def reaches(self, strat):
        """Return (reach_p0, reach_p1, reach_chance) per node."""
        N = self.N
        r = np.ones((3, N))
        for i in range(N):
            k = self.kind[i]
            if k == "terminal":
                continue
            if k == "chance":
                for c, p in zip(self.children[i], self.chance_p[i]):
                    r[:, c] = r[:, i]
                    r[2, c] *= p
            else:
                pl = self.player[i]
                s = strat[self.node_iset[i]]
                for c, p in zip(self.children[i], s):
                    r[:, c] = r[:, i]
                    r[pl, c] *= p
        return r

    def values(self, strat):
        """Expected p0 payoff of each subtree under strat."""
        v = self.payoff.copy()
        for i in range(self.N - 1, -1, -1):
            k = self.kind[i]
            if k == "terminal":
                continue
            ch = self.children[i]
            if k == "chance":
                v[i] = sum(p * v[c] for c, p in zip(ch, self.chance_p[i]))
            else:
                s = strat[self.node_iset[i]]
                v[i] = sum(p * v[c] for c, p in zip(ch, s))
        return v

    def best_response_value(self, strat, br):
        """Value to player `br` when it best-responds to strat (other player's part)."""
        sign = 1.0 if br == 0 else -1.0
        opp = 1 - br
        r = self.reaches(strat)
        # weight of node = opponent reach * chance reach
        w = r[opp] * r[2]
        v = sign * self.payoff.copy()  # values from br's perspective
        choice = {}
        # process in reverse topological order; when hitting a br node, decide
        # its infoset (all nodes of infoset have their children already computed
        # since with perfect recall descendants of any infoset node come later in
        # order... we instead compute infoset choice lazily on first encounter,
        # requiring all its nodes' children values to be final).
        # To be safe: process by depth descending.
        depth = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            for c in self.children[i]:
                depth[c] = depth[i] + 1
        for d in range(depth.max(), -1, -1):
            layer = np.nonzero(depth == d)[0]
            # first decide br infosets whose nodes are in this layer
            for i in layer:
                if self.kind[i] == "decision" and self.player[i] == br:
                    k = self.node_iset[i]
                    if k in choice:
                        continue
                    tot = np.zeros(len(self.isets[k][2]))
                    for h in self.iset_nodes[k]:
                        assert depth[h] == d
                        tot += w[h] * np.array([v[c] for c in self.children[h]])
                    choice[k] = int(np.argmax(tot))
            for i in layer:
                k = self.kind[i]
                if k == "terminal":
                    continue
                ch = self.children[i]
                if k == "chance":
                    v[i] = sum(p * v[c] for c, p in zip(ch, self.chance_p[i]))
                elif self.player[i] == br:
                    v[i] = v[ch[choice[self.node_iset[i]]]]
                else:
                    s = strat[self.node_iset[i]]
                    v[i] = sum(p * v[c] for c, p in zip(ch, s))
        return v[0]

    def exploitability(self, strat):
        b0 = self.best_response_value(strat, 0)
        b1 = self.best_response_value(strat, 1)
        return (b0 + b1) / 2, b0, b1


def cfr_plus(game, iters, log_every=1000):
    M = game.M
    regret = [np.zeros(len(l)) for (_, _, l) in game.isets]
    avg = [np.zeros(len(l)) for (_, _, l) in game.isets]
    strat = game.uniform()

    def current():
        out = []
        for R in regret:
            pos = np.maximum(R, 0)
            s = pos.sum()
            out.append(pos / s if s > 0 else np.full(len(R), 1.0 / len(R)))
        return out

    def avg_strat():
        out = []
        for A in avg:
            s = A.sum()
            out.append(A / s if s > 0 else np.full(len(A), 1.0 / len(A)))
        return out

    t0 = time.time()
    for t in range(1, iters + 1):
        for p in (0, 1):
            strat = current()
            r = game.reaches(strat)
            v = game.values(strat)
            sign = 1.0 if p == 0 else -1.0
            opp = 1 - p
            for k in range(M):
                if game.isets[k][1] != p:
                    continue
                inst = np.zeros(len(regret[k]))
                own = 0.0
                for h in game.iset_nodes[k]:
                    cv = np.array([v[c] for c in game.children[h]])
                    inst += r[opp, h] * r[2, h] * sign * (cv - v[h])
                    own += r[p, h]
                regret[k] = np.maximum(regret[k] + inst, 0.0)
                # linear averaging weighted by own reach
                avg[k] += t * own * strat[k]
        if t % log_every == 0:
            e = game.exploitability(avg_strat())
            print(f"iter {t} expl {e[0]:.6g} ({time.time()-t0:.1f}s)", flush=True)
    return avg_strat()


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    game = Game("game.json")
    print(f"{game.N} nodes, {game.M} infosets")
    s = cfr_plus(game, iters)
    expl, b0, b1 = game.exploitability(s)
    v0 = game.values(s)[0]
    print(f"final exploitability {expl:.8g} (BR0 {b0:.8g}, BR1 {b1:.8g}); value p0 {v0:.8g}")
    probs = {}
    for (name, _, labels), ps in zip(game.isets, s):
        ps = ps / ps.sum()
        probs[name] = {a: repr(float(x)) for a, x in zip(labels, ps)}
    json.dump(
        {"format": "bluff-strategy-v1", "game": game.name,
         "producer": f"CFR+ (alternating, linear averaging), {iters} iterations; exploitability {expl:.3g}",
         "probs": probs},
        open("strategy.json", "w"), indent=1)
    print("wrote strategy.json")


if __name__ == "__main__":
    main()

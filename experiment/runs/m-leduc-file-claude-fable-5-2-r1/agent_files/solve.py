#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in bluff-game-v1 format.

Reads game.json, runs CFR+ (regret matching+, alternating updates, linearly
weighted average strategy), and writes the average strategy to strategy.json.

Usage: python3 solve.py [--game game.json] [--out strategy.json]
                        [--target 1e-6] [--max-iters 200000] [--max-seconds 300]

Nothing about the game is hard-coded: the tree, information sets, chance
probabilities and payoffs all come from the game file. The tree is flattened
into per-depth numpy arrays so one iteration is a handful of vector operations
per tree level.
"""
import argparse
import json
import sys
import time
from fractions import Fraction

import numpy as np

DECIMALS = 15  # digits written per probability


def load_game(path):
    with open(path) as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise ValueError(f"unexpected game format: {g.get('format')!r}")
    return g


class Tree:
    """The game tree flattened into arrays, nodes renumbered in depth order."""

    def __init__(self, g):
        by_id = {n["id"]: n for n in g["nodes"]}
        if len(by_id) != len(g["nodes"]):
            raise ValueError("duplicate node ids")
        root = g["root"]

        def kids(n):
            if n["kind"] == "chance":
                return [o["child"] for o in n["outcomes"]]
            if n["kind"] == "decision":
                return [a["child"] for a in n["actions"]]
            if n["kind"] == "terminal":
                return []
            raise ValueError(f"unknown node kind {n['kind']!r}")

        # Breadth-first order: nodes of one depth are contiguous.
        order, depth = [root], {root: 0}
        i = 0
        while i < len(order):
            nid = order[i]
            i += 1
            for c in kids(by_id[nid]):
                if c in depth:
                    raise ValueError(f"node {c} has more than one parent")
                depth[c] = depth[nid] + 1
                order.append(c)
        if len(order) != len(by_id):
            raise ValueError("some nodes are unreachable from the root")
        new = {nid: k for k, nid in enumerate(order)}
        n_nodes = len(order)
        self.n_nodes = n_nodes
        node_depth = np.array([depth[nid] for nid in order])
        self.n_levels = int(node_depth.max()) + 1
        # level_start[d] .. level_start[d+1] are the nodes at depth d
        self.level_start = np.searchsorted(node_depth, np.arange(self.n_levels + 1))

        # Information sets and their (infoset, action) slots, flattened.
        self.inf_names, self.inf_player, self.inf_actions, self.inf_offset = [], [], [], []
        inf_index = {}
        n_slots = 0
        self.terminal_payoff = np.zeros(n_nodes)  # payoff to player 0
        e_parent, e_child, e_owner, e_slot, e_prob = [], [], [], [], []
        node_inf = np.full(n_nodes, -1)
        for nid in order:
            n = by_id[nid]
            k = new[nid]
            if n["kind"] == "terminal":
                self.terminal_payoff[k] = float(Fraction(n["payoff_p0"]))
            elif n["kind"] == "chance":
                total = Fraction(0)
                for o in n["outcomes"]:
                    p = Fraction(o["prob"])
                    total += p
                    e_parent.append(k)
                    e_child.append(new[o["child"]])
                    e_owner.append(-1)
                    e_slot.append(0)
                    e_prob.append(float(p))
                if total != 1:
                    raise ValueError(f"chance node {nid} probabilities sum to {total}")
            else:
                name, player = n["infoset"], n["player"]
                labels = [a["label"] for a in n["actions"]]
                if player not in (0, 1):
                    raise ValueError(f"bad player at node {nid}")
                if name not in inf_index:
                    inf_index[name] = len(self.inf_names)
                    self.inf_names.append(name)
                    self.inf_player.append(player)
                    self.inf_actions.append(labels)
                    self.inf_offset.append(n_slots)
                    n_slots += len(labels)
                ii = inf_index[name]
                if self.inf_player[ii] != player or self.inf_actions[ii] != labels:
                    raise ValueError(f"inconsistent infoset {name!r}")
                node_inf[k] = ii
                for j, a in enumerate(n["actions"]):
                    e_parent.append(k)
                    e_child.append(new[a["child"]])
                    e_owner.append(player)
                    e_slot.append(self.inf_offset[ii] + j)
                    e_prob.append(0.0)

        self.n_inf = len(self.inf_names)
        self.n_slots = n_slots
        self.inf_player = np.array(self.inf_player)
        self.inf_offset = np.array(self.inf_offset)
        self.inf_nact = np.array([len(a) for a in self.inf_actions])
        self.slot_inf = np.repeat(np.arange(self.n_inf), self.inf_nact)
        self.slot_player = self.inf_player[self.slot_inf]
        self.node_inf = node_inf

        # Edges were appended parent by parent in BFS order, so they are already
        # sorted by parent, hence grouped by parent depth.
        self.e_parent = np.array(e_parent)
        self.e_child = np.array(e_child)
        self.e_owner = np.array(e_owner)
        self.e_slot = np.array(e_slot)
        self.e_prob = np.array(e_prob)
        self.e_is_chance = self.e_owner < 0
        self.edge_start = np.searchsorted(node_depth[self.e_parent], np.arange(self.n_levels + 1))
        self.own_edges = [np.nonzero(self.e_owner == p)[0] for p in (0, 1)]

        # One representative node per infoset (for the player's own reach).
        self.inf_rep = np.zeros(self.n_inf, dtype=np.int64)
        seen = np.zeros(self.n_inf, dtype=bool)
        for k in range(n_nodes):
            ii = node_inf[k]
            if ii >= 0 and not seen[ii]:
                seen[ii] = True
                self.inf_rep[ii] = k
        self._check_perfect_recall()

        # Padded (infoset, action) matrix index for argmax in the best response.
        max_a = int(self.inf_nact.max())
        self.pad_index = np.full((self.n_inf, max_a), -1)
        for ii in range(self.n_inf):
            self.pad_index[ii, : self.inf_nact[ii]] = self.inf_offset[ii] + np.arange(self.inf_nact[ii])
        # The level-wise best response needs each infoset to sit on one depth.
        d_of_inf = {}
        self.inf_single_depth = True
        for k in range(n_nodes):
            ii = node_inf[k]
            if ii >= 0 and d_of_inf.setdefault(ii, node_depth[k]) != node_depth[k]:
                self.inf_single_depth = False

    def _check_perfect_recall(self):
        """Every node of an infoset must share the owner's own action sequence."""
        own_seq = [None] * self.n_nodes  # (seq id for p0, seq id for p1)
        own_seq[0] = (0, 0)
        seq_ids = {}
        inf_seq = {}
        self.perfect_recall = True
        for e in range(len(self.e_parent)):
            par, ch, owner = self.e_parent[e], self.e_child[e], self.e_owner[e]
            s = list(own_seq[par])
            if owner >= 0:
                ii = self.node_inf[par]
                if inf_seq.setdefault(ii, s[owner]) != s[owner]:
                    self.perfect_recall = False
                s[owner] = seq_ids.setdefault((owner, s[owner], int(self.e_slot[e])), len(seq_ids) + 1)
            own_seq[ch] = tuple(s)

    # ---- tree passes -------------------------------------------------------

    def edge_weights(self, sigma):
        """Probability of each edge under behavioral strategy sigma (per slot)."""
        return np.where(self.e_is_chance, self.e_prob, sigma[self.e_slot])

    def reach(self, w, exclude=None):
        """Top-down reach probabilities; edges owned by `exclude` count as 1."""
        if exclude is not None:
            w = np.where(self.e_owner == exclude, 1.0, w)
        r = np.ones(self.n_nodes)
        for d in range(self.n_levels - 1):
            a, b = self.edge_start[d], self.edge_start[d + 1]
            r[self.e_child[a:b]] = r[self.e_parent[a:b]] * w[a:b]
        return r

    def values(self, w):
        """Bottom-up expected payoff to player 0 at every node under weights w."""
        v = self.terminal_payoff.copy()
        for d in range(self.n_levels - 2, -1, -1):
            a, b = self.edge_start[d], self.edge_start[d + 1]
            s, t = self.level_start[d], self.level_start[d + 1]
            v[s:t] += np.bincount(self.e_parent[a:b] - s, w[a:b] * v[self.e_child[a:b]], minlength=t - s)
        return v

    def uniform(self):
        return 1.0 / self.inf_nact[self.slot_inf]

    def normalize(self, x):
        """Normalize a nonnegative per-slot array within each infoset."""
        tot = np.bincount(self.slot_inf, x, minlength=self.n_inf)[self.slot_inf]
        return np.where(tot > 0, x / np.where(tot > 0, tot, 1.0), self.uniform())

    def best_response_value(self, sigma, p):
        """Value to player p of p's best response against sigma (other player)."""
        assert self.inf_single_depth and self.perfect_recall
        sign = 1.0 if p == 0 else -1.0
        w = self.edge_weights(sigma)
        cf = self.reach(w, exclude=p)  # chance and opponent reach
        u = sign * self.terminal_payoff * cf  # reach-weighted values
        mine = self.e_owner == p
        for d in range(self.n_levels - 2, -1, -1):
            a, b = self.edge_start[d], self.edge_start[d + 1]
            s, t = self.level_start[d], self.level_start[d + 1]
            uc = u[self.e_child[a:b]]
            m = mine[a:b]
            keep = np.ones(b - a)
            if m.any():
                q = np.bincount(self.e_slot[a:b][m], uc[m], minlength=self.n_slots)
                qp = np.where(self.pad_index >= 0, q[self.pad_index], -np.inf)
                best = self.pad_index[np.arange(self.n_inf), qp.argmax(axis=1)]
                chosen = np.zeros(self.n_slots)
                chosen[best] = 1.0
                keep = np.where(m, chosen[self.e_slot[a:b]], 1.0)
            u[s:t] += np.bincount(self.e_parent[a:b] - s, keep * uc, minlength=t - s)
        return u[0]

    def exploitability(self, sigma):
        b0 = self.best_response_value(sigma, 0)
        b1 = self.best_response_value(sigma, 1)
        return (b0 + b1) / 2.0, b0, b1


def cfr_plus(tree, target, max_iters, max_seconds, check_every, log=print):
    R = np.zeros(tree.n_slots)  # regret-matching+ cumulative regrets
    S = np.zeros(tree.n_slots)  # weighted sum of reach * strategy
    t0 = time.time()
    it = 0
    expl = float("inf")
    while it < max_iters:
        it += 1
        for p in (0, 1):
            sigma = tree.normalize(R)
            w = tree.edge_weights(sigma)
            v = tree.values(w)  # payoff to player 0
            cf = tree.reach(w, exclude=p)  # chance * opponent reach
            own = tree.reach(np.where(tree.e_owner == p, w, 1.0))  # p's own reach
            sign = 1.0 if p == 0 else -1.0
            e = tree.own_edges[p]
            # counterfactual value of each action, summed over the infoset's nodes
            q = sign * np.bincount(tree.e_slot[e], cf[tree.e_parent[e]] * v[tree.e_child[e]], minlength=tree.n_slots)
            iv = np.bincount(tree.slot_inf, sigma * q, minlength=tree.n_inf)
            mine = tree.slot_player == p
            R = np.where(mine, np.maximum(R + q - iv[tree.slot_inf], 0.0), R)
            S += np.where(mine, it * own[tree.inf_rep][tree.slot_inf] * sigma, 0.0)
        if it % check_every == 0 or it == max_iters:
            avg = tree.normalize(S)
            expl, b0, b1 = tree.exploitability(avg)
            el = time.time() - t0
            log(f"iter {it:7d}  exploitability {expl:.3e}  (br0 {b0:+.6f}, br1 {b1:+.6f})  {el:6.1f}s")
            if expl <= target or el > max_seconds:
                break
    return tree.normalize(S), it, expl


def to_decimal_strings(probs):
    """Round a probability vector to DECIMALS digits so the strings sum to exactly 1."""
    scale = 10**DECIMALS
    ints = [max(0, int(round(float(x) * scale))) for x in probs]
    k = max(range(len(ints)), key=lambda j: ints[j])
    ints[k] += scale - sum(ints)
    assert all(x >= 0 for x in ints) and sum(ints) == scale
    out = []
    for x in ints:
        s = f"{x // scale}.{x % scale:0{DECIMALS}d}".rstrip("0")
        out.append(s + "0" if s.endswith(".") else s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="game.json")
    ap.add_argument("--out", default="strategy.json")
    ap.add_argument("--target", type=float, default=1e-6, help="stop at this exploitability")
    ap.add_argument("--max-iters", type=int, default=200000)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--check-every", type=int, default=500)
    args = ap.parse_args()

    g = load_game(args.game)
    tree = Tree(g)
    print(
        f"game {g.get('name')!r}: {tree.n_nodes} nodes, {tree.n_inf} infosets, "
        f"{tree.n_slots} infoset-actions, depth {tree.n_levels - 1}, perfect recall: {tree.perfect_recall}"
    )
    if not tree.perfect_recall:
        sys.exit("game does not have perfect recall; CFR is not guaranteed to converge")
    if not tree.inf_single_depth:
        sys.exit("an information set spans several depths; not supported by this solver")

    avg, iters, _ = cfr_plus(tree, args.target, args.max_iters, args.max_seconds, args.check_every)

    probs = {}
    final = np.zeros(tree.n_slots)
    for ii, name in enumerate(tree.inf_names):
        o, k = tree.inf_offset[ii], tree.inf_nact[ii]
        strs = to_decimal_strings(avg[o : o + k])
        probs[name] = dict(zip(tree.inf_actions[ii], strs))
        final[o : o + k] = [float(s) for s in strs]

    # Report numbers for the strategy exactly as written (after rounding).
    expl, b0, b1 = tree.exploitability(final)
    value = tree.values(tree.edge_weights(final))[0]
    out = {
        "format": "bluff-strategy-v1",
        "game": g.get("name"),
        "producer": f"solve.py: CFR+ (alternating updates, linear averaging), {iters} iterations",
        "probs": probs,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}: {len(probs)} infosets after {iters} iterations")
    print(f"exploitability {expl:.6e}  (br0 {b0:+.9f}, br1 {b1:+.9f})  value_p0 {value:+.9f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Solve a bluff-game-v1 game (game.json) with CFR+ and write the average strategy.

Usage: python3 -I solve_cfr.py [game.json] [strategy.json] [iterations]

CFR+ with alternating updates (one player traverses per iteration), regret
matching+ (regrets floored at 0), linear averaging of the strategy (iteration t
has weight t). Exploitability of the final average strategy is computed exactly
with best-response traversals over the game tree.
"""
import json
import sys
from collections import defaultdict
from fractions import Fraction

sys.setrecursionlimit(100000)


def load_game(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise SystemExit(f"unsupported game format: {g.get('format')!r}")
    nodes = {n["id"]: n for n in g["nodes"]}
    return g, nodes


def num(s):
    return float(Fraction(s))


class Game:
    def __init__(self, nodes, root):
        self.nodes = nodes
        self.root = root
        self.infosets = {}       # name -> dict(player, labels, nodes)
        self.info_of = {}        # decision node id -> infoset name
        for nid, n in nodes.items():
            if n["kind"] != "decision":
                continue
            labels = [a["label"] for a in n["actions"]]
            name = n["infoset"]
            if name not in self.infosets:
                self.infosets[name] = {"player": n["player"], "labels": labels, "nodes": []}
            info = self.infosets[name]
            if info["player"] != n["player"] or info["labels"] != labels:
                raise SystemExit(f"inconsistent infoset {name!r}")
            info["nodes"].append(nid)
            self.info_of[nid] = name
        # Terminal payoffs (player 0) and chance/decision children, as floats.
        self.payoff = {}
        self.children = {}
        self.probs = {}
        for nid, n in nodes.items():
            if n["kind"] == "terminal":
                self.payoff[nid] = num(n["payoff_p0"])
            elif n["kind"] == "chance":
                self.children[nid] = [o["child"] for o in n["outcomes"]]
                self.probs[nid] = [num(o["prob"]) for o in n["outcomes"]]
            elif n["kind"] == "decision":
                self.children[nid] = [a["child"] for a in n["actions"]]
            else:
                raise SystemExit(f"unknown node kind {n['kind']!r}")
        # Topological order (parents before children) for top-down passes.
        self.order = []
        stack = [root]
        while stack:
            x = stack.pop()
            self.order.append(x)
            stack.extend(self.children.get(x, []))


class CFRPlus:
    def __init__(self, game):
        self.g = game
        self.regret = {}
        self.strat_sum = {}
        for name, info in game.infosets.items():
            k = len(info["labels"])
            self.regret[name] = [0.0] * k
            self.strat_sum[name] = [0.0] * k

    def current(self, name):
        r = self.regret[name]
        pos = [x if x > 0.0 else 0.0 for x in r]
        s = sum(pos)
        if s > 0.0:
            return [x / s for x in pos]
        k = len(pos)
        return [1.0 / k] * k

    def traverse(self, nid, r0, r1, c, t, upd):
        """Return player 0's expected payoff from nid.

        r0, r1: reach probabilities of the two players, c: chance reach.
        upd: the player whose regrets and average strategy are updated.
        """
        g = self.g
        n_kind = g.nodes[nid]["kind"]
        if n_kind == "terminal":
            return g.payoff[nid]
        if n_kind == "chance":
            total = 0.0
            for child, p in zip(g.children[nid], g.probs[nid]):
                total += p * self.traverse(child, r0, r1, c * p, t, upd)
            return total

        player = g.nodes[nid]["player"]
        name = g.info_of[nid]
        sigma = self.current(name)
        children = g.children[nid]
        own_reach = r0 if player == 0 else r1
        opp_reach = r1 if player == 0 else r0

        if player == upd:
            ss = self.strat_sum[name]
            for a in range(len(sigma)):
                ss[a] += t * own_reach * sigma[a]

        vals = []
        for a, child in enumerate(children):
            if player == 0:
                vals.append(self.traverse(child, r0 * sigma[a], r1, c, t, upd))
            else:
                vals.append(self.traverse(child, r0, r1 * sigma[a], c, t, upd))
        node_val = sum(s * v for s, v in zip(sigma, vals))

        if player == upd:
            # cfv for the updating player, in that player's own perspective.
            sign = 1.0 if player == 0 else -1.0
            node_util = sign * node_val
            weight = opp_reach * c
            reg = self.regret[name]
            for a in range(len(sigma)):
                action_util = sign * vals[a]
                reg[a] = max(0.0, reg[a] + weight * (action_util - node_util))
        return node_val

    def run(self, iterations, log_every=0):
        g = self.g
        for t in range(1, iterations + 1):
            upd = (t - 1) % 2
            self.traverse(g.root, 1.0, 1.0, 1.0, t, upd)
            if log_every and t % log_every == 0:
                expl = exploitability(g, self.average())
                print(f"iter {t:7d}  exploitability {expl:.8f}", flush=True)

    def average(self):
        avg = {}
        for name, ss in self.strat_sum.items():
            s = sum(ss)
            k = len(ss)
            avg[name] = [x / s for x in ss] if s > 0 else [1.0 / k] * k
        return avg


def expected_value(g, avg):
    """Player 0's expected payoff when both players follow avg."""
    def val(nid):
        kind = g.nodes[nid]["kind"]
        if kind == "terminal":
            return g.payoff[nid]
        if kind == "chance":
            return sum(p * val(c) for c, p in zip(g.children[nid], g.probs[nid]))
        sigma = avg[g.info_of[nid]]
        return sum(s * val(c) for s, c in zip(sigma, g.children[nid]))
    return val(g.root)


def best_response_value(g, avg, br_player):
    """Expected payoff (to br_player) when br_player best-responds to avg.

    The opponent follows avg. Reach of the opponent and chance is computed
    top-down; then the best action at each infoset of br_player is chosen by
    the reach-weighted value of its continuation, deepest infosets first (via
    memoized recursion, since the tree has no cycles).
    """
    sign = 1.0 if br_player == 0 else -1.0
    # q[n]: probability of reaching n through chance and the opponent only.
    q = defaultdict(float)
    q[g.root] = 1.0
    for nid in g.order:
        kind = g.nodes[nid]["kind"]
        if kind == "chance":
            for c, p in zip(g.children[nid], g.probs[nid]):
                q[c] += q[nid] * p
        elif kind == "decision" and g.nodes[nid]["player"] != br_player:
            sigma = avg[g.info_of[nid]]
            for c, s in zip(g.children[nid], sigma):
                q[c] += q[nid] * s
        elif kind == "decision":
            for c in g.children[nid]:
                q[c] += q[nid]

    memo_val = {}
    best_action = {}

    def choose(name):
        if name in best_action:
            return best_action[name]
        info = g.infosets[name]
        k = len(info["labels"])
        totals = [0.0] * k
        for nid in info["nodes"]:
            w = q[nid]
            if w == 0.0:
                # Still need the continuation values to be well-defined for
                # tie-breaking; skip weight-zero nodes in the choice.
                continue
            for a, c in enumerate(g.children[nid]):
                totals[a] += w * val(c)
        a_best = max(range(k), key=lambda a: (totals[a], -a))
        best_action[name] = a_best
        return a_best

    def val(nid):
        if nid in memo_val:
            return memo_val[nid]
        kind = g.nodes[nid]["kind"]
        if kind == "terminal":
            res = sign * g.payoff[nid]
        elif kind == "chance":
            res = sum(p * val(c) for c, p in zip(g.children[nid], g.probs[nid]))
        elif g.nodes[nid]["player"] == br_player:
            a = choose(g.info_of[nid])
            res = val(g.children[nid][a])
        else:
            sigma = avg[g.info_of[nid]]
            res = sum(s * val(c) for s, c in zip(sigma, g.children[nid]))
        memo_val[nid] = res
        return res

    return val(g.root)


def exploitability(g, avg):
    br0 = best_response_value(g, avg, 0)
    br1 = best_response_value(g, avg, 1)
    return (br0 + br1) / 2.0


def write_strategy(g, game_name, avg, path, producer):
    probs = {}
    for name in sorted(g.infosets):
        labels = g.infosets[name]["labels"]
        sigma = avg[name]
        s = sum(sigma)
        probs[name] = {lab: repr(x / s) for lab, x in zip(labels, sigma)}
    out = {
        "format": "bluff-strategy-v1",
        "game": game_name,
        "producer": producer,
        "probs": probs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")


def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 20000

    raw, nodes = load_game(game_path)
    g = Game(nodes, raw["root"])
    solver = CFRPlus(g)
    log_every = max(1, iterations // 10)
    solver.run(iterations, log_every=log_every)
    avg = solver.average()
    expl = exploitability(g, avg)
    value = expected_value(g, avg)
    print(f"final exploitability {expl:.10f}")
    print(f"value to player 0    {value:.10f}")
    producer = (f"CFR+ (alternating, linear averaging), {iterations} iterations, "
                f"python3 solve_cfr.py; exploitability {expl:.3g}")
    write_strategy(g, raw["name"], avg, out_path, producer)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

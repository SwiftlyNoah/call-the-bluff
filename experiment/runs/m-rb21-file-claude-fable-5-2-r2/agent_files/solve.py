#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in bluff-game-v1 format.

Reads game.json, runs CFR+ (regret matching+, alternating updates, linearly weighted
averaging) for both players, and writes the average strategy to strategy.json.

Usage: python3 solve.py [--game game.json] [--out strategy.json]
                        [--target 1e-5] [--max-iters 200000] [--max-seconds 600]
"""
import argparse
import json
import sys
import time
from fractions import Fraction

TERMINAL, CHANCE, DECISION = 0, 1, 2


class Game:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.name = raw.get("name", "")
        by_id = {n["id"]: n for n in raw["nodes"]}

        # Depth-first preorder from the root: every parent gets a smaller index than its children.
        order = []
        stack = [raw["root"]]
        seen = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                raise ValueError("node %r is reachable twice: not a tree" % (nid,))
            seen.add(nid)
            order.append(nid)
            node = by_id[nid]
            if node["kind"] == "chance":
                kids = [o["child"] for o in node["outcomes"]]
            elif node["kind"] == "decision":
                kids = [a["child"] for a in node["actions"]]
            else:
                kids = []
            stack.extend(reversed(kids))
        index = {nid: i for i, nid in enumerate(order)}

        n = len(order)
        self.n = n
        self.kind = [TERMINAL] * n
        self.children = [()] * n
        self.payoff = [0.0] * n  # player 0's payoff at terminals
        self.payoff_exact = [None] * n
        self.chance = [()] * n  # outcome probabilities at chance nodes
        self.chance_exact = [()] * n
        self.player = [-1] * n
        self.infoset = [-1] * n  # infoset index at decision nodes

        self.iset_names = []
        self.iset_player = []
        self.iset_actions = []
        self.iset_nodes = []
        iset_index = {}

        for i, nid in enumerate(order):
            node = by_id[nid]
            kind = node["kind"]
            if kind == "terminal":
                exact = Fraction(node["payoff_p0"])
                self.payoff_exact[i] = exact
                self.payoff[i] = float(exact)
            elif kind == "chance":
                self.kind[i] = CHANCE
                self.children[i] = tuple(index[o["child"]] for o in node["outcomes"])
                exact = tuple(Fraction(o["prob"]) for o in node["outcomes"])
                if sum(exact) != 1:
                    raise ValueError("chance node %r: probabilities sum to %s" % (nid, sum(exact)))
                self.chance_exact[i] = exact
                self.chance[i] = tuple(float(p) for p in exact)
            elif kind == "decision":
                self.kind[i] = DECISION
                self.children[i] = tuple(index[a["child"]] for a in node["actions"])
                labels = tuple(a["label"] for a in node["actions"])
                player = node["player"]
                name = node["infoset"]
                k = iset_index.get(name)
                if k is None:
                    k = len(self.iset_names)
                    iset_index[name] = k
                    self.iset_names.append(name)
                    self.iset_player.append(player)
                    self.iset_actions.append(labels)
                    self.iset_nodes.append([])
                elif self.iset_player[k] != player or self.iset_actions[k] != labels:
                    raise ValueError("infoset %r is inconsistent across its nodes" % (name,))
                self.iset_nodes[k].append(i)
                self.player[i] = player
                self.infoset[i] = k
            else:
                raise ValueError("unknown node kind %r" % (kind,))
        self.num_isets = len(self.iset_names)


def best_response_value(game, strategy, br_player, exact=False):
    """Value, to br_player, of br_player's best response against `strategy` (list of per-infoset
    probability lists; only the opponent's entries are used)."""
    n = game.n
    kind, children, infoset, player = game.kind, game.children, game.infoset, game.player
    chance = game.chance_exact if exact else game.chance
    payoff = game.payoff_exact if exact else game.payoff
    one = Fraction(1) if exact else 1.0
    sign = 1 if br_player == 0 else -1

    # Reach probability from chance and the opponent only.
    reach = [None] * n
    reach[0] = one
    for i in range(n):
        k = kind[i]
        if k == TERMINAL:
            continue
        r = reach[i]
        if k == CHANCE:
            for c, p in zip(children[i], chance[i]):
                reach[c] = r * p
        elif player[i] == br_player:
            for c in children[i]:
                reach[c] = r
        else:
            for c, p in zip(children[i], strategy[infoset[i]]):
                reach[c] = r * p

    # Reach-weighted values, bottom-up. An infoset of br_player is resolved once all of its
    # nodes have their children evaluated; with perfect recall that happens when its
    # lowest-indexed node is visited in the reverse sweep, provided we defer the choice:
    # we process infosets in order of decreasing minimum node index via an explicit recursion.
    value = [None] * n  # reach-weighted value of each node for br_player
    choice = {}

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    def node_value(i):
        v = value[i]
        if v is not None:
            return v
        k = kind[i]
        if k == TERMINAL:
            v = reach[i] * payoff[i] * sign
        elif k == CHANCE or player[i] != br_player:
            v = sum(node_value(c) for c in children[i])
        else:
            v = node_value(children[i][best_action(infoset[i])])
        value[i] = v
        return v

    def best_action(k):
        a = choice.get(k)
        if a is None:
            nodes = game.iset_nodes[k]
            best = None
            for j in range(len(game.iset_actions[k])):
                total = sum(node_value(children[i][j]) for i in nodes)
                if best is None or total > best:
                    best, a = total, j
            choice[k] = a
        return a

    return node_value(0)


def expected_value_p0(game, strategy, exact=False):
    chance = game.chance_exact if exact else game.chance
    payoff = game.payoff_exact if exact else game.payoff
    value = [None] * game.n
    for i in range(game.n - 1, -1, -1):
        k = game.kind[i]
        if k == TERMINAL:
            value[i] = payoff[i]
        else:
            probs = chance[i] if k == CHANCE else strategy[game.infoset[i]]
            value[i] = sum(p * value[c] for c, p in zip(game.children[i], probs))
    return value[0]


def exploitability(game, strategy, exact=False):
    br0 = best_response_value(game, strategy, 0, exact)
    br1 = best_response_value(game, strategy, 1, exact)
    return (br0 + br1) / 2, br0, br1


class CFRPlus:
    def __init__(self, game):
        self.game = game
        sizes = [len(a) for a in game.iset_actions]
        self.regret = [[0.0] * s for s in sizes]
        self.strategy_sum = [[0.0] * s for s in sizes]
        self.current = [[1.0 / s] * s for s in sizes]
        self.iterations = 0

    def _update(self, p, weight):
        game = self.game
        n = game.n
        kind, children, infoset, player = game.kind, game.children, game.infoset, game.player
        chance, payoff = game.chance, game.payoff
        current, regret, strategy_sum = self.current, self.regret, self.strategy_sum
        sign = 1.0 if p == 0 else -1.0

        own = [0.0] * n  # reach probability from player p's own actions
        opp = [0.0] * n  # reach probability from chance and the opponent
        own[0] = opp[0] = 1.0
        for i in range(n):
            k = kind[i]
            if k == TERMINAL:
                continue
            o, w = opp[i], own[i]
            if k == CHANCE:
                for c, q in zip(children[i], chance[i]):
                    opp[c] = o * q
                    own[c] = w
            elif player[i] == p:
                for c, q in zip(children[i], current[infoset[i]]):
                    opp[c] = o
                    own[c] = w * q
            else:
                for c, q in zip(children[i], current[infoset[i]]):
                    opp[c] = o * q
                    own[c] = w

        # Average strategy: weight this iteration's strategy by the player's own reach.
        for k, nodes in enumerate(game.iset_nodes):
            if game.iset_player[k] != p:
                continue
            w = own[nodes[0]] * weight
            if w > 0.0:
                acc = strategy_sum[k]
                for j, q in enumerate(current[k]):
                    acc[j] += w * q

        # Counterfactual values and regrets, bottom-up.
        value = [0.0] * n
        touched = set()
        for i in range(n - 1, -1, -1):
            k = kind[i]
            if k == TERMINAL:
                value[i] = payoff[i] * sign
            elif k == CHANCE:
                value[i] = sum(q * value[c] for c, q in zip(children[i], chance[i]))
            else:
                h = infoset[i]
                kids = children[i]
                v = 0.0
                for c, q in zip(kids, current[h]):
                    v += q * value[c]
                value[i] = v
                if player[i] == p:
                    o = opp[i]
                    if o > 0.0:
                        r = regret[h]
                        for j, c in enumerate(kids):
                            r[j] += o * (value[c] - v)
                        touched.add(h)

        # Regret matching+: clamp at zero, then play in proportion to positive regret.
        for h in touched:
            r = regret[h]
            total = 0.0
            for j, x in enumerate(r):
                if x < 0.0:
                    r[j] = 0.0
                else:
                    total += x
            if total > 0.0:
                current[h] = [x / total for x in r]
            else:
                current[h] = [1.0 / len(r)] * len(r)

    def iterate(self):
        self.iterations += 1
        for p in (0, 1):
            self._update(p, float(self.iterations))

    def average_strategy(self):
        result = []
        for acc in self.strategy_sum:
            total = sum(acc)
            if total > 0.0:
                result.append([x / total for x in acc])
            else:
                result.append([1.0 / len(acc)] * len(acc))
        return result


def write_strategy(game, strategy, path, producer):
    probs = {}
    for name, labels, row in zip(game.iset_names, game.iset_actions, strategy):
        probs[name] = {label: repr(float(q)) for label, q in zip(labels, row)}
    doc = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": producer,
        "probs": probs,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game", default="game.json")
    parser.add_argument("--out", default="strategy.json")
    parser.add_argument("--target", type=float, default=1e-5,
                        help="stop once exploitability is at or below this")
    parser.add_argument("--max-iters", type=int, default=200000)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--check-every", type=int, default=250)
    args = parser.parse_args()

    game = Game(args.game)
    print("game %r: %d nodes, %d infosets" % (game.name, game.n, game.num_isets), flush=True)

    solver = CFRPlus(game)
    start = time.time()
    best = None  # (exploitability, strategy, iterations)
    while True:
        solver.iterate()
        t = solver.iterations
        elapsed = time.time() - start
        done = t >= args.max_iters or elapsed >= args.max_seconds
        if t % args.check_every == 0 or done:
            avg = solver.average_strategy()
            expl, br0, br1 = exploitability(game, avg)
            print("iter %6d  %7.1fs  exploitability %.3e  (br0 %+.6f, br1 %+.6f)"
                  % (t, elapsed, expl, br0, br1), flush=True)
            if best is None or expl < best[0]:
                best = (expl, avg, t)
            if expl <= args.target:
                done = True
        if done:
            break

    expl, strategy, iters = best
    producer = "solve.py: CFR+ (alternating updates, linear averaging), %d iterations" % iters
    write_strategy(game, strategy, args.out, producer)
    print("wrote %s (average strategy after %d iterations)" % (args.out, iters))
    print("exploitability %.6e" % expl)
    print("value_p0 %.9f" % expected_value_p0(game, strategy))


if __name__ == "__main__":
    main()

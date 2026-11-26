#!/usr/bin/env python3
"""Solve a two-player zero-sum extensive-form game (bluff-game-v1) with CFR+.

Reads the game tree from game.json, runs CFR+ (regret matching+, alternating
updates, linearly weighted averaging) for both players, and writes the average
strategy to strategy.json. The exploitability of the strategy exactly as
written to the file is then computed in exact rational arithmetic.

Usage: python3 solve.py [--game game.json] [--out strategy.json]
                        [--iterations 200000] [--target 1e-7]
"""

import argparse
import itertools
import json
import sys
from fractions import Fraction

DECIMALS = 12  # digits written per probability


# ---------------------------------------------------------------- game loading

class Game:
    def __init__(self, raw):
        if raw.get("format") != "bluff-game-v1":
            raise ValueError("unsupported game format: %r" % raw.get("format"))
        self.name = raw["name"]
        self.root = raw["root"]
        self.nodes = {}
        for node in raw["nodes"]:
            if node["id"] in self.nodes:
                raise ValueError("duplicate node id %r" % node["id"])
            self.nodes[node["id"]] = node

        # infoset name -> {"player", "labels", "nodes"}
        self.infosets = {}
        parents = {}
        for node in self.nodes.values():
            kind = node["kind"]
            if kind == "chance":
                node["probs"] = [Fraction(o["prob"]) for o in node["outcomes"]]
                node["children"] = [o["child"] for o in node["outcomes"]]
                if sum(node["probs"]) != 1 or min(node["probs"]) < 0:
                    raise ValueError("chance node %r: bad probabilities" % node["id"])
            elif kind == "decision":
                node["children"] = [a["child"] for a in node["actions"]]
                labels = [a["label"] for a in node["actions"]]
                if len(set(labels)) != len(labels) or not labels:
                    raise ValueError("decision node %r: bad actions" % node["id"])
                if node["player"] not in (0, 1):
                    raise ValueError("decision node %r: bad player" % node["id"])
                info = self.infosets.setdefault(
                    node["infoset"],
                    {"player": node["player"], "labels": labels, "nodes": []})
                if info["player"] != node["player"] or info["labels"] != labels:
                    raise ValueError("infoset %r is inconsistent" % node["infoset"])
                info["nodes"].append(node["id"])
            elif kind == "terminal":
                node["payoff"] = Fraction(node["payoff_p0"])
                node["children"] = []
            else:
                raise ValueError("node %r: unknown kind %r" % (node["id"], kind))
            for child in node["children"]:
                if child not in self.nodes:
                    raise ValueError("node %r: unknown child %r" % (node["id"], child))
                if child in parents:
                    raise ValueError("node %r has more than one parent" % child)
                parents[child] = node["id"]
        if self.root not in self.nodes or self.root in parents:
            raise ValueError("bad root")

        # Depth-first order from the root; also proves every node is reachable.
        self.order = []
        stack = [self.root]
        while stack:
            node = self.nodes[stack.pop()]
            self.order.append(node["id"])
            stack.extend(reversed(node["children"]))
        if len(self.order) != len(self.nodes):
            raise ValueError("some nodes are unreachable from the root")


# ---------------------------------------------------------------------- CFR+

class CFRPlus:
    """CFR+ in floating point: regret matching+, alternating updates, and an
    average strategy weighted linearly by iteration number."""

    def __init__(self, game):
        self.game = game
        self.regret = {k: [0.0] * len(v["labels"]) for k, v in game.infosets.items()}
        self.average = {k: [0.0] * len(v["labels"]) for k, v in game.infosets.items()}
        self.iteration = 0
        # Float copies of the tree for speed.
        self.fnodes = {}
        for nid, node in game.nodes.items():
            kind = node["kind"]
            if kind == "terminal":
                self.fnodes[nid] = ("t", float(node["payoff"]))
            elif kind == "chance":
                self.fnodes[nid] = ("c", [float(p) for p in node["probs"]], node["children"])
            else:
                self.fnodes[nid] = ("d", node["player"], node["infoset"], node["children"])

    def current(self, infoset):
        regret = self.regret[infoset]
        total = sum(regret)
        if total > 0.0:
            return [r / total for r in regret]
        return [1.0 / len(regret)] * len(regret)

    def walk(self, nid, updater, reach_me, reach_other, weight):
        """Return the expected payoff to `updater`; update its regrets.

        reach_me is the updater's own reach probability, reach_other is the
        product of the opponent's and chance's reach probabilities."""
        node = self.fnodes[nid]
        if node[0] == "t":
            return node[1] if updater == 0 else -node[1]
        if node[0] == "c":
            value = 0.0
            for prob, child in zip(node[1], node[2]):
                if prob > 0.0:
                    value += prob * self.walk(child, updater, reach_me, reach_other * prob, weight)
            return value
        _, player, infoset, children = node
        sigma = self.sigma[infoset]
        if player != updater:
            value = 0.0
            for prob, child in zip(sigma, children):
                if prob > 0.0:
                    value += prob * self.walk(child, updater, reach_me, reach_other * prob, weight)
            return value
        values = [self.walk(child, updater, reach_me * prob, reach_other, weight)
                  for prob, child in zip(sigma, children)]
        value = sum(p * v for p, v in zip(sigma, values))
        regret = self.regret[infoset]
        average = self.average[infoset]
        for i, action_value in enumerate(values):
            # Accumulated here and clipped at zero once the whole pass is done.
            regret[i] += reach_other * (action_value - value)
            average[i] += weight * reach_me * sigma[i]
        return value

    def step(self):
        self.iteration += 1
        for updater in (0, 1):
            # One strategy snapshot per pass, so that every node of an infoset
            # sees the same strategy while regrets are being accumulated.
            self.sigma = {infoset: self.current(infoset) for infoset in self.regret}
            self.walk(self.game.root, updater, 1.0, 1.0, float(self.iteration))
            for infoset, info in self.game.infosets.items():
                if info["player"] == updater:
                    regret = self.regret[infoset]
                    for i, r in enumerate(regret):
                        if r < 0.0:
                            regret[i] = 0.0

    def average_strategy(self):
        """The average strategy as exact rationals with DECIMALS digits that
        sum to exactly 1 in every infoset."""
        scale = 10 ** DECIMALS
        strategy = {}
        for infoset, sums in self.average.items():
            total = sum(sums)
            if total > 0.0:
                probs = [s / total for s in sums]
            else:  # never reached by its own player: any choice is fine
                probs = [1.0 / len(sums)] * len(sums)
            units = [int(round(p * scale)) for p in probs]
            units[max(range(len(units)), key=units.__getitem__)] += scale - sum(units)
            assert min(units) >= 0 and sum(units) == scale
            strategy[infoset] = [Fraction(u, scale) for u in units]
        return strategy


# ------------------------------------------------- exact evaluation (Fractions)

def expected_value(game, strategy):
    """Player 0's expected payoff when both players follow `strategy`."""
    def walk(nid):
        node = game.nodes[nid]
        if node["kind"] == "terminal":
            return node["payoff"]
        if node["kind"] == "chance":
            probs = node["probs"]
        else:
            probs = strategy[node["infoset"]]
        return sum((p * walk(c) for p, c in zip(probs, node["children"]) if p), Fraction(0))
    return walk(game.root)


def best_response_value(game, strategy, br_player):
    """Value to `br_player` of its best response against `strategy`.

    Works infoset by infoset: at each of the responder's infosets it picks the
    action with the highest total counterfactual value over the infoset's
    nodes. Requires perfect recall for the responder."""
    sign = 1 if br_player == 0 else -1

    # Reach probability of each node from the opponent and chance only.
    reach = {game.root: Fraction(1)}
    for nid in game.order:
        node = game.nodes[nid]
        if node["kind"] == "terminal":
            continue
        if node["kind"] == "chance":
            probs = node["probs"]
        elif node["player"] == br_player:
            probs = [Fraction(1)] * len(node["children"])
        else:
            probs = strategy[node["infoset"]]
        for p, c in zip(probs, node["children"]):
            reach[c] = reach[nid] * p

    choice = {}
    in_progress = set()

    def choose(infoset):
        if infoset in choice:
            return choice[infoset]
        if infoset in in_progress:
            raise ValueError("imperfect recall at infoset %r" % infoset)
        in_progress.add(infoset)
        info = game.infosets[infoset]
        totals = [Fraction(0)] * len(info["labels"])
        for nid in info["nodes"]:
            for i, child in enumerate(game.nodes[nid]["children"]):
                totals[i] += reach[nid] * walk(child)
        in_progress.discard(infoset)
        choice[infoset] = max(range(len(totals)), key=totals.__getitem__)
        return choice[infoset]

    def walk(nid):
        node = game.nodes[nid]
        if node["kind"] == "terminal":
            return sign * node["payoff"]
        if node["kind"] == "decision" and node["player"] == br_player:
            return walk(node["children"][choose(node["infoset"])])
        probs = node["probs"] if node["kind"] == "chance" else strategy[node["infoset"]]
        return sum((p * walk(c) for p, c in zip(probs, node["children"]) if p), Fraction(0))

    return walk(game.root)


def brute_force_best_response_value(game, strategy, br_player, limit=200000):
    """Same quantity by enumerating every pure strategy of the responder.
    Returns None if there are more than `limit` of them."""
    mine = [k for k, v in game.infosets.items() if v["player"] == br_player]
    count = 1
    for k in mine:
        count *= len(game.infosets[k]["labels"])
        if count > limit:
            return None
    best = None
    for combo in itertools.product(*[range(len(game.infosets[k]["labels"])) for k in mine]):
        profile = dict(strategy)
        for k, a in zip(mine, combo):
            profile[k] = [Fraction(int(i == a)) for i in range(len(game.infosets[k]["labels"]))]
        value = expected_value(game, profile)
        if br_player == 1:
            value = -value
        if best is None or value > best:
            best = value
    return best


def exploitability(game, strategy):
    return (best_response_value(game, strategy, 0) + best_response_value(game, strategy, 1)) / 2


# ------------------------------------------------------------------------ I/O

def format_prob(prob):
    units = prob * 10 ** DECIMALS
    assert units.denominator == 1
    whole, frac = divmod(units.numerator, 10 ** DECIMALS)
    return "%d.%0*d" % (whole, DECIMALS, frac)


def write_strategy(game, strategy, path, producer):
    doc = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": producer,
        "probs": {
            infoset: {label: format_prob(p) for label, p in zip(info["labels"], strategy[infoset])}
            for infoset, info in game.infosets.items()
        },
    }
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=1)
        handle.write("\n")


def read_strategy(game, path):
    """Read a strategy file back and validate it against the game."""
    with open(path) as handle:
        doc = json.load(handle)
    if doc.get("format") != "bluff-strategy-v1" or doc.get("game") != game.name:
        raise ValueError("strategy file header does not match the game")
    probs = doc["probs"]
    if set(probs) != set(game.infosets):
        raise ValueError("strategy file infosets do not match the game")
    strategy = {}
    for infoset, info in game.infosets.items():
        if set(probs[infoset]) != set(info["labels"]):
            raise ValueError("infoset %r: actions do not match the game" % infoset)
        row = [Fraction(probs[infoset][label]) for label in info["labels"]]
        if min(row) < 0 or sum(row) != 1:
            raise ValueError("infoset %r: not a probability distribution" % infoset)
        strategy[infoset] = row
    return strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game", default="game.json")
    parser.add_argument("--out", default="strategy.json")
    parser.add_argument("--iterations", type=int, default=200000, help="maximum CFR+ iterations")
    parser.add_argument("--target", type=float, default=1e-7,
                        help="stop early once exploitability is at or below this")
    args = parser.parse_args()

    with open(args.game) as handle:
        game = Game(json.load(handle))
    print("game %r: %d nodes, %d infosets" % (game.name, len(game.nodes), len(game.infosets)))

    solver = CFRPlus(game)
    strategy = None
    next_check = 1000
    while solver.iteration < args.iterations:
        solver.step()
        if solver.iteration == next_check or solver.iteration == args.iterations:
            strategy = solver.average_strategy()
            eps = exploitability(game, strategy)
            print("iteration %7d  exploitability %.3e" % (solver.iteration, float(eps)))
            if eps <= args.target:
                break
            next_check *= 2
    if strategy is None:
        strategy = solver.average_strategy()

    producer = "solve.py: CFR+ (alternating updates, linear averaging), %d iterations" % solver.iteration
    write_strategy(game, strategy, args.out, producer)

    # Everything reported below is computed from the file as written.
    written = read_strategy(game, args.out)
    br0 = best_response_value(game, written, 0)
    br1 = best_response_value(game, written, 1)
    for player, fast in ((0, br0), (1, br1)):
        slow = brute_force_best_response_value(game, written, player)
        if slow is not None and slow != fast:
            raise AssertionError("best-response check failed for player %d" % player)
    value = expected_value(game, written)
    eps = (br0 + br1) / 2
    print("wrote %s after %d iterations" % (args.out, solver.iteration))
    print("value to player 0:        %.12f" % float(value))
    print("best response value p0:   %.12f" % float(br0))
    print("best response value p1:   %.12f" % float(br1))
    print("exploitability:           %.6e" % float(eps))
    for infoset in sorted(written):
        info = game.infosets[infoset]
        print("  P%d %-6s %s" % (info["player"], infoset, "  ".join(
            "%s=%.6f" % (label, float(p)) for label, p in zip(info["labels"], written[infoset]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

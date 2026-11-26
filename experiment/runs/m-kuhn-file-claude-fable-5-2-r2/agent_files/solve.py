#!/usr/bin/env python3
"""Solve a two-player zero-sum extensive-form game (bluff-game-v1) with CFR+.

Reads the game tree from game.json, runs CFR+ (regret matching+, alternating
updates, linearly weighted averaging), writes the average strategy to
strategy.json, then reads that file back and reports its exact exploitability
and game value, computed with rational arithmetic.

Usage: python3 solve.py [--game game.json] [--out strategy.json] [--iters N]
"""

import argparse
import json
import sys
from fractions import Fraction


# ---------------------------------------------------------------- game loading

class Game:
    def __init__(self, data):
        if data.get("format") != "bluff-game-v1":
            raise ValueError("unexpected game format: %r" % data.get("format"))
        self.name = data["name"]
        self.root = data["root"]
        self.nodes = {}
        for node in data["nodes"]:
            if node["id"] in self.nodes:
                raise ValueError("duplicate node id %r" % node["id"])
            self.nodes[node["id"]] = node
        self.infosets = {}  # name -> {"player": p, "actions": [labels], "nodes": [ids]}
        self._validate()

    def children(self, node):
        if node["kind"] == "chance":
            return [o["child"] for o in node["outcomes"]]
        if node["kind"] == "decision":
            return [a["child"] for a in node["actions"]]
        return []

    def _validate(self):
        parents = {}
        for nid, node in self.nodes.items():
            kind = node["kind"]
            if kind == "chance":
                probs = [Fraction(o["prob"]) for o in node["outcomes"]]
                if any(p < 0 for p in probs) or sum(probs) != 1:
                    raise ValueError("chance node %r probabilities do not sum to 1" % nid)
                node["_probs"] = probs
            elif kind == "decision":
                if node["player"] not in (0, 1):
                    raise ValueError("node %r has bad player" % nid)
                labels = [a["label"] for a in node["actions"]]
                if not labels or len(set(labels)) != len(labels):
                    raise ValueError("node %r has bad action labels" % nid)
                info = self.infosets.setdefault(
                    node["infoset"],
                    {"player": node["player"], "actions": labels, "nodes": []},
                )
                if info["player"] != node["player"] or info["actions"] != labels:
                    raise ValueError("infoset %r is inconsistent" % node["infoset"])
                info["nodes"].append(nid)
            elif kind == "terminal":
                node["_payoff"] = Fraction(node["payoff_p0"])
            else:
                raise ValueError("node %r has unknown kind %r" % (nid, kind))
            for child in self.children(node):
                if child not in self.nodes:
                    raise ValueError("node %r points at missing child %r" % (nid, child))
                if child in parents:
                    raise ValueError("node %r has more than one parent" % child)
                parents[child] = nid
        if self.root not in self.nodes or self.root in parents:
            raise ValueError("bad root")
        # Every node must be reachable from the root (a tree, no cycles).
        seen, stack = set(), [self.root]
        while stack:
            nid = stack.pop()
            if nid in seen:
                raise ValueError("cycle at node %r" % nid)
            seen.add(nid)
            stack.extend(self.children(self.nodes[nid]))
        if len(seen) != len(self.nodes):
            raise ValueError("%d nodes are unreachable" % (len(self.nodes) - len(seen)))
        self._check_perfect_recall()

    def _check_perfect_recall(self):
        """CFR's guarantees need perfect recall: all nodes of an infoset must
        share the acting player's own sequence of (infoset, action) choices."""
        history = {}

        def walk(nid, seqs):
            node = self.nodes[nid]
            if node["kind"] == "chance":
                for child in self.children(node):
                    walk(child, seqs)
            elif node["kind"] == "decision":
                p, name = node["player"], node["infoset"]
                if history.setdefault(name, seqs[p]) != seqs[p]:
                    raise ValueError("infoset %r violates perfect recall" % name)
                for a in node["actions"]:
                    nxt = list(seqs)
                    nxt[p] = seqs[p] + ((name, a["label"]),)
                    walk(a["child"], tuple(nxt))

        walk(self.root, ((), ()))


# ------------------------------------------------------------------------ CFR+

def regret_matching(regrets):
    total = sum(r for r in regrets if r > 0)
    if total > 0:
        return [r / total if r > 0 else 0.0 for r in regrets]
    return [1.0 / len(regrets)] * len(regrets)


def cfr_plus(game, iterations):
    nodes = game.nodes
    regret = {name: [0.0] * len(i["actions"]) for name, i in game.infosets.items()}
    strat_sum = {name: [0.0] * len(i["actions"]) for name, i in game.infosets.items()}

    def walk(nid, updating, reach_own, reach_other, weight):
        """Returns the node's expected payoff to `updating`. reach_own is the
        updating player's reach probability; reach_other covers the opponent
        and chance."""
        node = nodes[nid]
        kind = node["kind"]
        if kind == "terminal":
            v = float(node["_payoff"])
            return v if updating == 0 else -v
        if kind == "chance":
            return sum(
                float(p) * walk(o["child"], updating, reach_own, reach_other * float(p), weight)
                for o, p in zip(node["outcomes"], node["_probs"])
            )
        name = node["infoset"]
        sigma = regret_matching(regret[name])
        if node["player"] != updating:
            return sum(
                s * walk(a["child"], updating, reach_own, reach_other * s, weight)
                for a, s in zip(node["actions"], sigma)
            )
        values = [
            walk(a["child"], updating, reach_own * s, reach_other, weight)
            for a, s in zip(node["actions"], sigma)
        ]
        node_value = sum(s * v for s, v in zip(sigma, values))
        pending = regret_delta.setdefault(name, [0.0] * len(values))
        for k, v in enumerate(values):
            pending[k] += reach_other * (v - node_value)
        acc = strat_sum[name]
        for k, s in enumerate(sigma):
            acc[k] += weight * reach_own * s
        return node_value

    for t in range(1, iterations + 1):
        for updating in (0, 1):
            # Regret updates are collected over the whole traversal and applied
            # afterwards, so every node of an infoset sees the same strategy.
            regret_delta = {}
            walk(game.root, updating, 1.0, 1.0, float(t))
            for name, delta in regret_delta.items():
                r = regret[name]
                for k, d in enumerate(delta):
                    r[k] = max(r[k] + d, 0.0)

    average = {}
    for name, acc in strat_sum.items():
        total = sum(acc)
        n = len(acc)
        average[name] = [a / total for a in acc] if total > 0 else [1.0 / n] * n
    return average


# ---------------------------------------------------- exact evaluation (checks)

def expected_value(game, strategy):
    """Player 0's expected payoff when both players follow `strategy`."""
    def walk(nid):
        node = game.nodes[nid]
        if node["kind"] == "terminal":
            return node["_payoff"]
        if node["kind"] == "chance":
            return sum(p * walk(o["child"]) for o, p in zip(node["outcomes"], node["_probs"]))
        probs = strategy[node["infoset"]]
        return sum(probs[a["label"]] * walk(a["child"]) for a in node["actions"])
    return walk(game.root)


def best_response_value(game, strategy, br_player):
    """Value to br_player of its best response against `strategy` (the other
    player's part of it). Exact for games with perfect recall."""
    nodes = game.nodes
    # Counterfactual reach (chance and opponent only) of every node.
    reach = {}

    def fill(nid, r):
        reach[nid] = r
        node = nodes[nid]
        if node["kind"] == "chance":
            for o, p in zip(node["outcomes"], node["_probs"]):
                fill(o["child"], r * p)
        elif node["kind"] == "decision":
            for a in node["actions"]:
                if node["player"] == br_player:
                    fill(a["child"], r)
                else:
                    fill(a["child"], r * strategy[node["infoset"]][a["label"]])

    fill(game.root, Fraction(1))
    choice = {}

    def best_action(name):
        if name not in choice:
            info = game.infosets[name]
            totals = [Fraction(0)] * len(info["actions"])
            for nid in info["nodes"]:
                if reach[nid] == 0:
                    continue
                for k, a in enumerate(nodes[nid]["actions"]):
                    totals[k] += reach[nid] * value(a["child"])
            choice[name] = max(range(len(totals)), key=lambda k: totals[k])
        return choice[name]

    def value(nid):
        """Expected payoff to br_player from this node, given it is reached."""
        node = nodes[nid]
        if node["kind"] == "terminal":
            return node["_payoff"] if br_player == 0 else -node["_payoff"]
        if node["kind"] == "chance":
            return sum(p * value(o["child"]) for o, p in zip(node["outcomes"], node["_probs"]))
        if node["player"] == br_player:
            return value(node["actions"][best_action(node["infoset"])]["child"])
        probs = strategy[node["infoset"]]
        return sum(probs[a["label"]] * value(a["child"]) for a in node["actions"])

    return value(game.root)


def load_strategy(game, path):
    """Reads a strategy file strictly, as an independent reader would."""
    with open(path) as f:
        data = json.load(f)
    if data.get("format") != "bluff-strategy-v1" or data.get("game") != game.name:
        raise ValueError("bad strategy header")
    probs = data["probs"]
    if set(probs) != set(game.infosets):
        raise ValueError("strategy infosets do not match the game")
    strategy = {}
    for name, info in game.infosets.items():
        if set(probs[name]) != set(info["actions"]):
            raise ValueError("strategy actions do not match at %r" % name)
        row = {label: Fraction(probs[name][label]) for label in info["actions"]}
        if any(p < 0 for p in row.values()) or sum(row.values()) != 1:
            raise ValueError("probabilities at %r do not sum to exactly 1" % name)
        strategy[name] = row
    return strategy


def to_exact_rows(average):
    """Turns float rows into decimal strings that sum to exactly 1."""
    out = {}
    scale = 10 ** 12
    for name, row in average.items():
        units = [int(round(p * scale)) for p in row]
        # Put any rounding remainder on the largest entry.
        units[max(range(len(units)), key=lambda k: units[k])] += scale - sum(units)
        out[name] = units
    return out, scale


def format_units(units, scale):
    digits = len(str(scale)) - 1
    whole, frac = divmod(units, scale)
    text = "%d.%0*d" % (whole, digits, frac)
    text = text.rstrip("0")
    return text + "0" if text.endswith(".") else text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="game.json")
    parser.add_argument("--out", default="strategy.json")
    parser.add_argument("--iters", type=int, default=100000)
    args = parser.parse_args()

    with open(args.game) as f:
        game = Game(json.load(f))

    average = cfr_plus(game, args.iters)
    rows, scale = to_exact_rows(average)

    probs = {}
    for name in game.infosets:
        labels = game.infosets[name]["actions"]
        probs[name] = {label: format_units(u, scale) for label, u in zip(labels, rows[name])}
    output = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": "solve.py: CFR+ (alternating updates, linear averaging), %d iterations"
        % args.iters,
        "probs": probs,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=1)
        f.write("\n")

    # Check the file that was actually written, with exact arithmetic.
    strategy = load_strategy(game, args.out)
    value_p0 = expected_value(game, strategy)
    br0 = best_response_value(game, strategy, 0)
    br1 = best_response_value(game, strategy, 1)
    exploitability = (br0 + br1) / 2
    report = {
        "game": game.name,
        "infosets": len(game.infosets),
        "nodes": len(game.nodes),
        "iterations": args.iters,
        "value_p0": float(value_p0),
        "br_value_p0": float(br0),
        "br_value_p1": float(br1),
        "exploitability": float(exploitability),
    }
    json.dump(report, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

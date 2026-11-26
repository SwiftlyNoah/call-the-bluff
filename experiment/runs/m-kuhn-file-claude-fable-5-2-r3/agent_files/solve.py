#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in bluff-game-v1 format.

Reads game.json, runs CFR+ (alternating updates, regret-matching+, linearly weighted
average strategy), writes strategy.json, and then re-reads strategy.json and checks it
with exact rational arithmetic (game value and exploitability via best responses).

Usage: python3 solve.py [--game game.json] [--out strategy.json] [--iters 100000]
"""
import argparse
import json
import sys
from fractions import Fraction


# ---------------------------------------------------------------- game loading

class Game:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        if raw.get("format") != "bluff-game-v1":
            raise ValueError("unexpected game format: %r" % raw.get("format"))
        self.name = raw["name"]
        self.root = raw["root"]
        self.nodes = {}
        for n in raw["nodes"]:
            if n["id"] in self.nodes:
                raise ValueError("duplicate node id %r" % n["id"])
            self.nodes[n["id"]] = n
        # infoset name -> (player, [action labels]), and infoset -> node ids
        self.infosets = {}
        self.infoset_nodes = {}
        self._validate()

    def _validate(self):
        seen = set()
        stack = [self.root]
        while stack:
            i = stack.pop()
            if i in seen:
                raise ValueError("node %r reached twice: not a tree" % i)
            seen.add(i)
            n = self.nodes[i]
            kind = n["kind"]
            if kind == "terminal":
                n["_payoff"] = Fraction(n["payoff_p0"])
            elif kind == "chance":
                probs = [Fraction(o["prob"]) for o in n["outcomes"]]
                if any(p < 0 for p in probs) or sum(probs) != 1:
                    raise ValueError("chance node %r: bad probabilities" % i)
                n["_probs"] = probs
                n["_children"] = [o["child"] for o in n["outcomes"]]
                stack.extend(n["_children"])
            elif kind == "decision":
                if n["player"] not in (0, 1):
                    raise ValueError("node %r: bad player" % i)
                labels = [a["label"] for a in n["actions"]]
                if len(set(labels)) != len(labels) or not labels:
                    raise ValueError("node %r: bad action labels" % i)
                key = n["infoset"]
                if key in self.infosets:
                    if self.infosets[key] != (n["player"], labels):
                        raise ValueError("infoset %r is inconsistent" % key)
                else:
                    self.infosets[key] = (n["player"], labels)
                    self.infoset_nodes[key] = []
                self.infoset_nodes[key].append(i)
                n["_children"] = [a["child"] for a in n["actions"]]
                stack.extend(n["_children"])
            else:
                raise ValueError("node %r: unknown kind %r" % (i, kind))
        if seen != set(self.nodes):
            raise ValueError("unreachable nodes: %r" % sorted(set(self.nodes) - seen))


# ------------------------------------------------------------------------ CFR+

def run_cfr_plus(game, iters):
    """CFR+ with alternating updates. Returns the average strategy (floats)."""
    nodes = game.nodes
    for n in nodes.values():
        if n["kind"] == "terminal":
            n["_fpayoff"] = float(n["_payoff"])
        elif n["kind"] == "chance":
            n["_fprobs"] = [float(p) for p in n["_probs"]]
    regret = {k: [0.0] * len(v[1]) for k, v in game.infosets.items()}
    strat_sum = {k: [0.0] * len(v[1]) for k, v in game.infosets.items()}

    def current(key):
        r = regret[key]
        s = sum(r)  # entries are >= 0 under regret-matching+
        if s > 0:
            return [x / s for x in r]
        return [1.0 / len(r)] * len(r)

    def walk(i, upd, weight, p_upd, p_other):
        """Returns the expected payoff to player `upd` under the current strategies.

        p_upd: reach probability contributed by the updating player.
        p_other: reach probability contributed by the opponent and chance.
        """
        n = nodes[i]
        kind = n["kind"]
        if kind == "terminal":
            return n["_fpayoff"] if upd == 0 else -n["_fpayoff"]
        if kind == "chance":
            v = 0.0
            for p, c in zip(n["_fprobs"], n["_children"]):
                if p > 0:
                    v += p * walk(c, upd, weight, p_upd, p_other * p)
            return v
        key = n["infoset"]
        sigma = current(key)
        children = n["_children"]
        if n["player"] != upd:
            v = 0.0
            for s, c in zip(sigma, children):
                if s > 0:
                    v += s * walk(c, upd, weight, p_upd, p_other * s)
            return v
        utils = [walk(c, upd, weight, p_upd * s, p_other) for s, c in zip(sigma, children)]
        v = sum(s * u for s, u in zip(sigma, utils))
        # Accumulate instantaneous counterfactual regrets; the regret-matching+ floor is
        # applied once per infoset after the whole traversal (see below), so that all
        # nodes of an infoset are summed before clipping.
        pending = pend[key]
        for a, u in enumerate(utils):
            pending[a] += p_other * (u - v)
        ss = strat_sum[key]
        for a, s in enumerate(sigma):
            ss[a] += weight * p_upd * s
        return v

    for t in range(1, iters + 1):
        for upd in (0, 1):
            pend = {k: [0.0] * len(v[1]) for k, v in game.infosets.items() if v[0] == upd}
            # Strategies of the updating player must stay fixed during the traversal, so
            # regrets are only committed afterwards.
            walk(game.root, upd, float(t), 1.0, 1.0)
            for key, d in pend.items():
                r = regret[key]
                for a, x in enumerate(d):
                    r[a] = max(r[a] + x, 0.0)

    avg = {}
    for key, ss in strat_sum.items():
        tot = sum(ss)
        if tot > 0:
            avg[key] = [x / tot for x in ss]
        else:
            avg[key] = [1.0 / len(ss)] * len(ss)
    return avg


# ------------------------------------------------- exact evaluation of a strategy

def load_strategy(game, path):
    with open(path) as f:
        raw = json.load(f)
    probs = {}
    for key, (_, labels) in game.infosets.items():
        entry = raw["probs"][key]
        if set(entry) != set(labels):
            raise ValueError("infoset %r: action labels do not match the game" % key)
        p = [Fraction(entry[a]) for a in labels]
        if any(x < 0 for x in p) or sum(p) != 1:
            raise ValueError("infoset %r: probabilities invalid (sum %s)" % (key, sum(p)))
        probs[key] = p
    extra = set(raw["probs"]) - set(game.infosets)
    if extra:
        raise ValueError("unknown infosets in strategy: %r" % sorted(extra))
    return probs


def expected_value(game, strat, i=None):
    """Exact payoff to player 0 when both players follow `strat`."""
    n = game.nodes[game.root if i is None else i]
    if n["kind"] == "terminal":
        return n["_payoff"]
    if n["kind"] == "chance":
        return sum(p * expected_value(game, strat, c) for p, c in zip(n["_probs"], n["_children"]) if p)
    return sum(p * expected_value(game, strat, c) for p, c in zip(strat[n["infoset"]], n["_children"]) if p)


def best_response_value(game, strat, br):
    """Exact value, to player `br`, of a best response against `strat` (opponent's part).

    The best response is chosen per information set (not per node): for each infoset the
    action maximizing the reach-weighted sum of values over all of its nodes is picked,
    with infosets deeper in the tree resolved first (requires perfect recall).
    """
    nodes = game.nodes
    # reach[i]: probability that chance and the opponent play towards node i.
    reach = {}
    stack = [(game.root, Fraction(1))]
    while stack:
        i, r = stack.pop()
        reach[i] = r
        n = nodes[i]
        if n["kind"] == "terminal":
            continue
        if n["kind"] == "chance":
            ps = n["_probs"]
        elif n["player"] == br:
            ps = [Fraction(1)] * len(n["_children"])
        else:
            ps = strat[n["infoset"]]
        for p, c in zip(ps, n["_children"]):
            stack.append((c, r * p))

    choice = {}
    in_progress = set()

    def choose(key):
        if key in choice:
            return choice[key]
        if key in in_progress:
            raise ValueError("game does not have perfect recall (infoset %r)" % key)
        in_progress.add(key)
        n_actions = len(game.infosets[key][1])
        totals = [Fraction(0)] * n_actions
        for i in game.infoset_nodes[key]:
            if reach[i] == 0:
                continue
            for a, c in enumerate(nodes[i]["_children"]):
                totals[a] += reach[i] * value(c)
        best = max(range(n_actions), key=lambda a: totals[a])
        in_progress.discard(key)
        choice[key] = best
        return best

    def value(i):
        n = nodes[i]
        if n["kind"] == "terminal":
            return n["_payoff"] if br == 0 else -n["_payoff"]
        if n["kind"] == "chance":
            return sum(p * value(c) for p, c in zip(n["_probs"], n["_children"]) if p)
        if n["player"] == br:
            return value(n["_children"][choose(n["infoset"])])
        return sum(p * value(c) for p, c in zip(strat[n["infoset"]], n["_children"]) if p)

    return value(game.root)


def evaluate(game, strat):
    v0 = expected_value(game, strat)
    br0 = best_response_value(game, strat, 0)
    br1 = best_response_value(game, strat, 1)
    return v0, br0, br1, (br0 + br1) / 2


def to_decimal_strings(p, digits=12):
    """Round probabilities to `digits` decimals so that they sum to exactly 1."""
    scale = 10 ** digits
    units = [int(round(x * scale)) for x in p]
    big = max(range(len(units)), key=lambda a: units[a])
    units[big] += scale - sum(units)  # the largest entry absorbs the rounding error
    return ["%d.%0*d" % (u // scale, digits, u % scale) for u in units]


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="game.json")
    ap.add_argument("--out", default="strategy.json")
    ap.add_argument("--iters", type=int, default=100000)
    args = ap.parse_args()

    game = Game(args.game)
    avg = run_cfr_plus(game, args.iters)

    probs = {}
    for key in sorted(game.infosets):
        labels = game.infosets[key][1]
        probs[key] = dict(zip(labels, to_decimal_strings(avg[key])))
    out = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": "solve.py: CFR+ (alternating updates, linear averaging), %d iterations" % args.iters,
        "probs": probs,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")

    # Check the file that was actually written, with exact arithmetic.
    strat = load_strategy(game, args.out)
    v0, br0, br1, expl = evaluate(game, strat)
    print("game: %s, %d nodes, %d infosets, %d CFR+ iterations" % (game.name, len(game.nodes), len(game.infosets), args.iters))
    for key in sorted(strat):
        labels = game.infosets[key][1]
        print("  P%d %-6s %s" % (game.infosets[key][0], key, "  ".join("%s=%.6f" % (a, float(p)) for a, p in zip(labels, strat[key]))))
    print("value to player 0 under strategy.json: %.12g" % float(v0))
    print("best-response value, player 0 vs P1 strategy: %.12g" % float(br0))
    print("best-response value, player 1 vs P0 strategy: %.12g" % float(br1))
    print("exploitability (br0 + br1) / 2: %.6e" % float(expl))
    return 0


if __name__ == "__main__":
    sys.exit(main())

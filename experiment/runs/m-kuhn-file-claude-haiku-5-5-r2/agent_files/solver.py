#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in the bluff-game-v1 format.

Usage: python3 -I solver.py [--game game.json] [--out strategy.json]
                            [--max-iters N] [--tol X]

Reads the game tree from the game file (nothing is hard-coded), runs CFR+ for both
players with linear averaging, reports the exploitability of the average strategy
(exact best responses over the tree), and writes the average strategy.
"""
import argparse
import json
import sys
from fractions import Fraction


def children(node):
    kind = node["kind"]
    if kind == "chance":
        return [o["child"] for o in node["outcomes"]]
    if kind == "decision":
        return [a["child"] for a in node["actions"]]
    return []


def load_game(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise SystemExit(f"unsupported game format: {g.get('format')!r}")
    nodes = {}
    for n in g["nodes"]:
        if n["id"] in nodes:
            raise SystemExit(f"duplicate node id {n['id']}")
        nodes[n["id"]] = n
    root = g["root"]

    # The tree must be a tree: every node reachable from the root exactly once.
    seen = set()
    stack = [root]
    while stack:
        nid = stack.pop()
        if nid in seen:
            raise SystemExit(f"node {nid} has more than one parent (not a tree)")
        seen.add(nid)
        for c in children(nodes[nid]):
            if c not in nodes:
                raise SystemExit(f"node {nid} refers to missing child {c}")
            stack.append(c)
    if seen != set(nodes):
        raise SystemExit("some nodes are unreachable from the root")

    # Infosets: consistent player and action labels; probabilities sum to 1.
    infosets = {}  # name -> {"player": p, "labels": [...], "nodes": [ids]}
    for nid, n in nodes.items():
        if n["kind"] == "chance":
            total = sum(Fraction(o["prob"]) for o in n["outcomes"])
            if total != 1:
                raise SystemExit(f"chance node {nid} probabilities sum to {total}")
        elif n["kind"] == "decision":
            labels = [a["label"] for a in n["actions"]]
            name = n["infoset"]
            if name in infosets:
                info = infosets[name]
                if info["player"] != n["player"] or info["labels"] != labels:
                    raise SystemExit(f"inconsistent infoset {name!r}")
                info["nodes"].append(nid)
            else:
                infosets[name] = {"player": n["player"], "labels": labels, "nodes": [nid]}
        elif n["kind"] != "terminal":
            raise SystemExit(f"unknown node kind {n['kind']!r} at node {nid}")
    return g, nodes, root, infosets


class Game:
    def __init__(self, g, nodes, root, infosets):
        self.name = g["name"]
        self.nodes = nodes
        self.root = root
        self.infosets = infosets
        # Exact values converted to floats once, for the numerical solver.
        self.payoff = {}
        self.chance = {}
        for nid, n in nodes.items():
            if n["kind"] == "terminal":
                self.payoff[nid] = float(Fraction(n["payoff_p0"]))
            elif n["kind"] == "chance":
                self.chance[nid] = [(o["child"], float(Fraction(o["prob"]))) for o in n["outcomes"]]

    # ---- CFR+ ---------------------------------------------------------------
    def setup_cfr(self):
        self.regret = {I: [0.0] * len(v["labels"]) for I, v in self.infosets.items()}
        self.strat_sum = {I: [0.0] * len(v["labels"]) for I, v in self.infosets.items()}

    def current_strategy(self, I):
        r = self.regret[I]
        pos = [max(x, 0.0) for x in r]
        total = sum(pos)
        if total > 0:
            return [x / total for x in pos]
        return [1.0 / len(r)] * len(r)

    def cfr_iteration(self, t):
        delta = {I: [0.0] * len(v["labels"]) for I, v in self.infosets.items()}
        self._walk(self.root, 1.0, 1.0, 1.0, t, delta)
        for I, d in delta.items():
            r = self.regret[I]
            for a in range(len(r)):
                r[a] = max(r[a] + d[a], 0.0)  # CFR+: regret floored at zero

    def _walk(self, nid, p0, p1, pc, t, delta):
        """Return the expected payoff to player 0 at node nid under current strategies.

        p0, p1: each player's reach probability; pc: chance reach probability.
        """
        n = self.nodes[nid]
        kind = n["kind"]
        if kind == "terminal":
            return self.payoff[nid]
        if kind == "chance":
            total = 0.0
            for child, prob in self.chance[nid]:
                total += prob * self._walk(child, p0, p1, pc * prob, t, delta)
            return total

        player = n["player"]
        I = n["infoset"]
        sigma = self.current_strategy(I)
        own = p0 if player == 0 else p1
        opp = p1 if player == 0 else p0
        kids = children(n)
        # Linear averaging (weight t) of the strategy, weighted by own reach.
        ss = self.strat_sum[I]
        for a in range(len(kids)):
            ss[a] += t * own * sigma[a]

        values = []
        for a, child in enumerate(kids):
            if player == 0:
                values.append(self._walk(child, p0 * sigma[a], p1, pc, t, delta))
            else:
                values.append(self._walk(child, p0, p1 * sigma[a], pc, t, delta))
        node_value = sum(sigma[a] * values[a] for a in range(len(kids)))

        # Counterfactual regret: weighted by the opponents' and chance reach.
        cf = opp * pc
        sign = 1.0 if player == 0 else -1.0  # player 1's utility is the negation
        d = delta[I]
        for a in range(len(kids)):
            d[a] += cf * sign * (values[a] - node_value)
        return node_value

    def average_strategy(self, I):
        s = self.strat_sum[I]
        total = sum(s)
        if total > 0:
            return [x / total for x in s]
        return [1.0 / len(s)] * len(s)

    def average_profile(self):
        return {I: self.average_strategy(I) for I in self.infosets}

    # ---- Evaluation ---------------------------------------------------------
    def expected_value_p0(self, profile):
        def ev(nid):
            n = self.nodes[nid]
            if n["kind"] == "terminal":
                return self.payoff[nid]
            if n["kind"] == "chance":
                return sum(p * ev(c) for c, p in self.chance[nid])
            sigma = profile[n["infoset"]]
            return sum(sigma[a] * ev(c) for a, c in enumerate(children(n)))
        return ev(self.root)

    def best_response_value(self, br, profile):
        """Best-response value for player br (in br's own payoff units) against profile.

        Standard method: w(h) is the probability of reaching node h through chance and the
        opponent (not through br's own actions). For each br infoset, the best action
        maximizes sum over its nodes h of w(h) * V(child), and V is the conditional value
        under br's optimal play below. Perfect recall makes the infoset choices consistent.
        """
        sign = 1.0 if br == 0 else -1.0
        nodes = self.nodes

        weight = {}
        stack = [(self.root, 1.0)]
        while stack:
            nid, wt = stack.pop()
            weight[nid] = wt
            n = nodes[nid]
            if n["kind"] == "chance":
                for c, p in self.chance[nid]:
                    stack.append((c, wt * p))
            elif n["kind"] == "decision":
                sigma = profile[n["infoset"]]
                for a, c in enumerate(children(n)):
                    if n["player"] == br:
                        stack.append((c, wt))
                    else:
                        stack.append((c, wt * sigma[a]))

        value_memo = {}
        best_memo = {}

        def best_action(I):
            if I in best_memo:
                return best_memo[I]
            labels = self.infosets[I]["labels"]
            totals = [0.0] * len(labels)
            for nid in self.infosets[I]["nodes"]:
                kids = children(nodes[nid])
                for a, c in enumerate(kids):
                    totals[a] += weight[nid] * V(c)
            best = max(range(len(labels)), key=lambda a: (totals[a], -a))
            best_memo[I] = best
            return best

        def V(nid):
            if nid in value_memo:
                return value_memo[nid]
            n = nodes[nid]
            if n["kind"] == "terminal":
                res = sign * self.payoff[nid]
            elif n["kind"] == "chance":
                res = sum(p * V(c) for c, p in self.chance[nid])
            elif n["player"] == br:
                res = V(children(n)[best_action(n["infoset"])])
            else:
                sigma = profile[n["infoset"]]
                res = sum(sigma[a] * V(c) for a, c in enumerate(children(n)))
            value_memo[nid] = res
            return res

        return V(self.root)

    def exploitability(self, profile):
        b0 = self.best_response_value(0, profile)  # player 0's BR payoff vs profile
        b1 = self.best_response_value(1, profile)  # player 1's BR payoff vs profile
        return (b0 + b1) / 2.0, b0, b1


def write_strategy(path, game, profile, producer, round_den=None):
    probs = {}
    for I in sorted(game.infosets):
        labels = game.infosets[I]["labels"]
        p = profile[I]
        if round_den:
            # Round each probability to the nearest fraction with denominator <= round_den.
            # Keep the result only if each infoset sums to exactly 1.
            vals = [Fraction(x).limit_denominator(round_den) for x in p]
            if sum(vals) != 1:
                raise SystemExit(f"rounding to denominator {round_den} does not sum to 1 at {I!r}")
            probs[I] = {lab: str(v) for lab, v in zip(labels, vals)}
        else:
            # Repair floating-point drift so each infoset sums to exactly 1 (up to repr).
            vals = [float(x) for x in p]
            s = sum(vals)
            vals = [x / s for x in vals]
            probs[I] = {lab: repr(float(v)) for lab, v in zip(labels, vals)}
    out = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": producer,
        "probs": probs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="game.json")
    ap.add_argument("--out", default="strategy.json")
    ap.add_argument("--max-iters", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="stop once exploitability of the average strategy is below this")
    ap.add_argument("--check-every", type=int, default=500)
    ap.add_argument("--round-den", type=int, default=None,
                    help="write probabilities as fractions with denominator <= N (exact if the "
                         "rounded profile is an equilibrium); must sum to 1 per infoset")
    args = ap.parse_args()

    g, nodes, root, infosets = load_game(args.game)
    game = Game(g, nodes, root, infosets)
    game.setup_cfr()

    iters = 0
    expl = None
    for t in range(1, args.max_iters + 1):
        game.cfr_iteration(t)
        iters = t
        if t % args.check_every == 0 or t == args.max_iters:
            expl, _, _ = game.exploitability(game.average_profile())
            if expl < args.tol:
                break

    profile = game.average_profile()
    expl, b0, b1 = game.exploitability(profile)
    value = game.expected_value_p0(profile)
    producer = (f"CFR+ (linear averaging), {iters} iterations on {args.game}; "
                f"average strategy exploitability {expl!r}")
    if args.round_den:
        producer += f"; probabilities rounded to fractions with denominator <= {args.round_den}"
    write_strategy(args.out, game, profile, producer, round_den=args.round_den)

    print(f"game: {game.name}")
    print(f"iterations: {iters}")
    print(f"best response value p0: {b0!r}")
    print(f"best response value p1: {b1!r}")
    print(f"exploitability: {expl!r}")
    print(f"value to player 0: {value!r}")
    print(f"wrote {args.out}")
    for I in sorted(infosets):
        row = ", ".join(f"{lab}={profile[I][a]:.6f}" for a, lab in enumerate(infosets[I]["labels"]))
        print(f"  {I:8s} {row}")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in the bluff-game-v1 format.

Usage: python3 -I cfr_solver.py <game.json> <strategy.json> [max_iterations]

Reads the game tree from game.json (nothing is hard-coded), runs CFR+ with linear
averaging for both players, computes the exploitability of the average strategy with
exact best-response passes, and writes the average strategy in the bluff-strategy-v1
format.
"""
import json
import sys
from fractions import Fraction


def num(s):
    """Parse an exact rational string ("-2", "1/6", "0.25") into a float."""
    return float(Fraction(s))


def load_game(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise ValueError("unsupported game format: %r" % g.get("format"))
    nodes = {n["id"]: n for n in g["nodes"]}
    infosets = {}
    for n in g["nodes"]:
        if n["kind"] != "decision":
            continue
        labels = [a["label"] for a in n["actions"]]
        info = infosets.setdefault(n["infoset"], {"player": n["player"], "actions": labels})
        if info["player"] != n["player"] or info["actions"] != labels:
            raise ValueError("inconsistent infoset %r" % n["infoset"])
    return g, nodes, infosets


class CFRPlus:
    def __init__(self, nodes, infosets, root):
        self.nodes = nodes
        self.infosets = infosets
        self.root = root
        self.regret = {I: [0.0] * len(v["actions"]) for I, v in infosets.items()}
        self.avg = {I: [0.0] * len(v["actions"]) for I, v in infosets.items()}
        self.delta = None

    def current_strategy(self, I):
        r = self.regret[I]
        pos = [max(x, 0.0) for x in r]
        total = sum(pos)
        if total > 0:
            return [p / total for p in pos]
        return [1.0 / len(r)] * len(r)

    def traverse(self, nid, p0, p1, pc, t):
        """Return the expected payoff to player 0 (conditional on reaching nid).

        p0, p1: reach probabilities of the players from the root; pc: chance reach.
        Regret deltas and weighted strategy sums are accumulated as side effects.
        """
        n = self.nodes[nid]
        kind = n["kind"]
        if kind == "terminal":
            return num(n["payoff_p0"])
        if kind == "chance":
            value = 0.0
            for o in n["outcomes"]:
                pr = num(o["prob"])
                value += pr * self.traverse(o["child"], p0, p1, pc * pr, t)
            return value

        I = n["infoset"]
        player = n["player"]
        sigma = self.current_strategy(I)
        own_reach = p0 if player == 0 else p1
        for a in range(len(sigma)):
            self.avg[I][a] += t * own_reach * sigma[a]

        utils = []
        for a, act in enumerate(n["actions"]):
            if player == 0:
                utils.append(self.traverse(act["child"], p0 * sigma[a], p1, pc, t))
            else:
                utils.append(self.traverse(act["child"], p0, p1 * sigma[a], pc, t))
        value = sum(s * u for s, u in zip(sigma, utils))

        # Counterfactual reach of the opponent (and chance) and the sign converting
        # player 0's payoffs into the acting player's payoffs.
        opp_reach = (p1 if player == 0 else p0) * pc
        sign = 1.0 if player == 0 else -1.0
        for a in range(len(sigma)):
            self.delta[I][a] += sign * opp_reach * (utils[a] - value)
        return value

    def iterate(self, t):
        self.delta = {I: [0.0] * len(v) for I, v in self.regret.items()}
        self.traverse(self.root, 1.0, 1.0, 1.0, t)
        # CFR+: accumulated regrets are floored at zero after every iteration.
        for I in self.regret:
            for a in range(len(self.regret[I])):
                self.regret[I][a] = max(self.regret[I][a] + self.delta[I][a], 0.0)

    def average_strategy(self):
        out = {}
        for I, sums in self.avg.items():
            total = sum(sums)
            n = len(sums)
            if total > 0:
                out[I] = [s / total for s in sums]
            else:
                out[I] = [1.0 / n] * n
        return out


def evaluate(nodes, root, strat):
    """Expected payoff to player 0 when both players follow the behavioral strategy."""

    def ev(nid):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return num(n["payoff_p0"])
        if n["kind"] == "chance":
            return sum(num(o["prob"]) * ev(o["child"]) for o in n["outcomes"])
        s = strat[n["infoset"]]
        return sum(s[k] * ev(a["child"]) for k, a in enumerate(n["actions"]))

    return ev(root)


def best_response(nodes, infosets, root, player, strat):
    """Value to `player` of its best response against the fixed strategy `strat` of the
    opponent. Computed by policy iteration over pure strategies: each information set's
    action is improved by the opponent-and-chance-weighted values of its nodes."""
    sign = 1.0 if player == 0 else -1.0
    choice = {I: 0 for I, v in infosets.items() if v["player"] == player}

    # Probability that the opponent and chance let play reach each node.
    weight = {}

    def walk(nid, w):
        weight[nid] = w
        n = nodes[nid]
        if n["kind"] == "terminal":
            return
        if n["kind"] == "chance":
            for o in n["outcomes"]:
                walk(o["child"], w * num(o["prob"]))
        elif n["player"] == player:
            for a in n["actions"]:
                walk(a["child"], w)
        else:
            s = strat[n["infoset"]]
            for k, a in enumerate(n["actions"]):
                walk(a["child"], w * s[k])

    walk(root, 1.0)

    def make_ev(memo):
        def ev(nid):
            if nid in memo:
                return memo[nid]
            n = nodes[nid]
            if n["kind"] == "terminal":
                v = num(n["payoff_p0"])
            elif n["kind"] == "chance":
                v = sum(num(o["prob"]) * ev(o["child"]) for o in n["outcomes"])
            elif n["player"] == player:
                v = ev(n["actions"][choice[n["infoset"]]]["child"])
            else:
                s = strat[n["infoset"]]
                v = sum(s[k] * ev(a["child"]) for k, a in enumerate(n["actions"]))
            memo[nid] = v
            return v

        return ev

    for _ in range(10000):
        memo = {}
        ev = make_ev(memo)
        score = {I: [0.0] * len(infosets[I]["actions"]) for I in choice}
        for nid, w in weight.items():
            n = nodes[nid]
            if n["kind"] == "decision" and n["player"] == player:
                for a, act in enumerate(n["actions"]):
                    score[n["infoset"]][a] += w * sign * ev(act["child"])
        changed = False
        for I in choice:
            best = max(range(len(score[I])), key=lambda a: (score[I][a], -a))
            if score[I][best] > score[I][choice[I]] + 1e-12:
                choice[I] = best
                changed = True
        if not changed:
            break
    memo = {}
    return sign * make_ev(memo)(root)


def exploitability(nodes, infosets, root, strat):
    br0 = best_response(nodes, infosets, root, 0, strat)
    br1 = best_response(nodes, infosets, root, 1, strat)
    return (br0 + br1) / 2.0, br0, br1


def fmt(p):
    return repr(float(p))


def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
    target = 1e-10

    g, nodes, infosets = load_game(game_path)
    root = g["root"]
    solver = CFRPlus(nodes, infosets, root)

    expl = None
    t = 0
    for t in range(1, max_iter + 1):
        solver.iterate(t)
        if t % 500 == 0 or t == max_iter:
            avg = solver.average_strategy()
            expl, _, _ = exploitability(nodes, infosets, root, avg)
            if t % 5000 == 0 or t == max_iter:
                print("iter %d exploitability %.3e" % (t, expl), file=sys.stderr)
            if expl < target:
                break

    avg = solver.average_strategy()
    expl, br0, br1 = exploitability(nodes, infosets, root, avg)
    value_p0 = evaluate(nodes, root, avg)

    probs = {}
    for I, info in infosets.items():
        p = avg[I]
        labels = info["actions"]
        entry = {}
        if len(p) == 1:
            entry[labels[0]] = "1.0"
        else:
            rest = p[:-1]
            entry_vals = [max(x, 0.0) for x in rest]
            last = 1.0 - sum(entry_vals)
            entry_vals.append(max(last, 0.0))
            for lab, v in zip(labels, entry_vals):
                entry[lab] = fmt(v)
        probs[I] = entry

    result = {
        "format": "bluff-strategy-v1",
        "game": g.get("name", "unknown"),
        "producer": "CFR+ (linear averaging), %d iterations, solver/cfr_solver.py" % t,
        "probs": probs,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(json.dumps({
        "iterations": t,
        "exploitability": expl,
        "br_value_p0": br0,
        "br_value_p1": br1,
        "value_p0": value_p0,
    }, indent=2))


if __name__ == "__main__":
    main()

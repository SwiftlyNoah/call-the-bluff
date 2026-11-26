#!/usr/bin/env python3
"""Solve a bluff-game-v1 extensive-form game with CFR+ (alternating updates).

Reads game.json (path given as first argument, default ./game.json), runs CFR+
for both players, writes the linear-averaged strategy to strategy.json (second
argument, default ./strategy.json), and reports exploitability computed with
exact best responses.

Usage: python3 solver.py [game.json] [strategy.json] [iterations]
"""
import json
import sys
from fractions import Fraction


def parse_num(s):
    return float(Fraction(s))


def load_game(path):
    with open(path) as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise SystemExit(f"unsupported game format: {g.get('format')!r}")
    nodes = {n["id"]: n for n in g["nodes"]}
    root = g["root"]
    # Build infoset table: name -> (player, action labels), checking consistency.
    infosets = {}
    for n in g["nodes"]:
        if n["kind"] != "decision":
            continue
        labels = [a["label"] for a in n["actions"]]
        if n["infoset"] in infosets:
            p, lab = infosets[n["infoset"]]
            if p != n["player"] or lab != labels:
                raise SystemExit(f"inconsistent infoset {n['infoset']!r}")
        else:
            infosets[n["infoset"]] = (n["player"], labels)
    return g, nodes, root, infosets


def compile_tree(nodes):
    """Convert numbers to floats once so the solver loops stay fast."""
    compiled = {}
    for nid, n in nodes.items():
        if n["kind"] == "terminal":
            compiled[nid] = ("terminal", parse_num(n["payoff_p0"]))
        elif n["kind"] == "chance":
            compiled[nid] = ("chance", [(parse_num(o["prob"]), o["child"]) for o in n["outcomes"]])
        elif n["kind"] == "decision":
            compiled[nid] = ("decision", n["player"], n["infoset"],
                             [a["label"] for a in n["actions"]],
                             [a["child"] for a in n["actions"]])
        else:
            raise SystemExit(f"unknown node kind {n['kind']!r}")
    return compiled


def regret_match(regret_sum, n_actions):
    pos = [max(r, 0.0) for r in regret_sum]
    total = sum(pos)
    if total > 0:
        return [p / total for p in pos]
    return [1.0 / n_actions] * n_actions


def cfr_plus(tree, root, infosets, iterations):
    regret = {I: [0.0] * len(labels) for I, (_, labels) in infosets.items()}
    strat_sum = {I: [0.0] * len(labels) for I, (_, labels) in infosets.items()}

    def current(I):
        return regret_match(regret[I], len(regret[I]))

    def walk(nid, pc, reach, traverser, t):
        """Return expected payoff for player 0 below nid.

        pc: chance reach above nid. reach[i]: player i's own reach above nid.
        Updates regrets of `traverser`'s infosets and the strategy sums of
        `traverser`'s infosets (weighted by iteration t for linear averaging).
        """
        node = tree[nid]
        kind = node[0]
        if kind == "terminal":
            return node[1]
        if kind == "chance":
            return sum(p * walk(c, pc * p, reach, traverser, t) for p, c in node[1])

        _, player, I, labels, children = node
        sigma = current(I)
        if player == traverser:
            # Reach of the traverser's own strategy is used for the average.
            for a, s in enumerate(sigma):
                strat_sum[I][a] += t * reach[player] * s
        util = []
        for a, c in enumerate(children):
            r = list(reach)
            r[player] *= sigma[a]
            util.append(walk(c, pc, r, traverser, t))
        value = sum(s * u for s, u in zip(sigma, util))

        if player == traverser:
            # Counterfactual weight: opponent reach times chance reach.
            opp_reach = reach[1 - player]
            cf_weight = pc * opp_reach
            # Utility for the traverser (player 0 payoff is value; player 1 gets -value).
            sign = 1.0 if player == 0 else -1.0
            for a in range(len(children)):
                regret[I][a] += cf_weight * sign * (util[a] - value)
            # CFR+: floor cumulative regrets at zero.
            regret[I] = [max(r, 0.0) for r in regret[I]]
        return value

    for t in range(1, iterations + 1):
        for traverser in (0, 1):
            walk(root, 1.0, [1.0, 1.0], traverser, t)

    avg = {}
    for I, (_, labels) in infosets.items():
        total = sum(strat_sum[I])
        if total > 0:
            avg[I] = [s / total for s in strat_sum[I]]
        else:
            avg[I] = [1.0 / len(labels)] * len(labels)
    return avg


def strategy_value_p0(tree, root, strat):
    def val(nid):
        node = tree[nid]
        if node[0] == "terminal":
            return node[1]
        if node[0] == "chance":
            return sum(p * val(c) for p, c in node[1])
        _, player, I, labels, children = node
        return sum(s * val(c) for s, c in zip(strat[I], children))
    return val(root)


def best_response_value(tree, root, infosets, strat, br_player):
    """Exact best-response value for br_player against `strat` (behavioral).

    Returns the expected payoff to br_player (in that player's own payoff units).
    """
    sign = 1.0 if br_player == 0 else -1.0

    # Top-down: probability of reaching each node through chance and the opponent.
    opp_reach = {}
    order = []
    stack = [(root, 1.0)]
    while stack:
        nid, r = stack.pop()
        opp_reach[nid] = r
        order.append(nid)
        node = tree[nid]
        if node[0] == "terminal":
            continue
        if node[0] == "chance":
            for p, c in node[1]:
                stack.append((c, r * p))
        else:
            _, player, I, labels, children = node
            if player == br_player:
                for c in children:
                    stack.append((c, r))
            else:
                for a, c in enumerate(children):
                    stack.append((c, r * strat[I][a]))

    # Group decision nodes of br_player by infoset.
    members = {}
    for nid in order:
        node = tree[nid]
        if node[0] == "decision" and node[1] == br_player:
            members.setdefault(node[2], []).append(nid)

    memo_w = {}
    best = {}

    def best_action(I):
        if I in best:
            return best[I]
        labels = infosets[I][1]
        vals = [0.0] * len(labels)
        for nid in members[I]:
            children = tree[nid][4]
            for a, c in enumerate(children):
                vals[a] += opp_reach[nid] * w(c)
        a_star = max(range(len(labels)), key=lambda a: (vals[a], -a))
        best[I] = a_star
        return a_star

    def w(nid):
        """Expected br_player payoff from nid onward, per unit of reach at nid."""
        if nid in memo_w:
            return memo_w[nid]
        node = tree[nid]
        if node[0] == "terminal":
            res = sign * node[1]
        elif node[0] == "chance":
            res = sum(p * w(c) for p, c in node[1])
        else:
            _, player, I, labels, children = node
            if player == br_player:
                res = w(children[best_action(I)])
            else:
                res = sum(s * w(c) for s, c in zip(strat[I], children))
        memo_w[nid] = res
        return res

    return w(root)  # w is already in br_player's payoff units


def exploitability(tree, root, infosets, strat):
    br0 = best_response_value(tree, root, infosets, strat, 0)  # P0 payoff
    br1 = best_response_value(tree, root, infosets, strat, 1)  # P1 payoff
    return (br0 + br1) / 2.0, br0, br1


def to_probs_json(strat, infosets):
    out = {}
    for I, (_, labels) in infosets.items():
        out[I] = {lab: repr(float(p)) for lab, p in zip(labels, strat[I])}
    return out


def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 200000

    g, nodes, root, infosets = load_game(game_path)
    tree = compile_tree(nodes)

    avg = cfr_plus(tree, root, infosets, iterations)

    expl, br0, br1 = exploitability(tree, root, infosets, avg)
    value0 = strategy_value_p0(tree, root, avg)
    print(f"iterations={iterations}")
    print(f"BR value P0 vs P1 strategy: {br0:.10f}")
    print(f"BR value P1 vs P0 strategy: {br1:.10f}")
    print(f"exploitability (BR0+BR1)/2: {expl:.10f}")
    print(f"game value to P0 under strategy: {value0:.10f}")

    doc = {
        "format": "bluff-strategy-v1",
        "game": g["name"],
        "producer": f"CFR+ (alternating updates, linear averaging), {iterations} iterations, "
                    f"solver.py; exploitability {expl:.3e}",
        "probs": to_probs_json(avg, infosets),
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

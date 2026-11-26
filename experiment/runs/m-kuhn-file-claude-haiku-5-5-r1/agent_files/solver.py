#!/usr/bin/env python3
"""CFR+ solver for two-player zero-sum extensive-form games in the bluff-game-v1 format.

Reads the game tree from a game file (default: game.json), runs CFR+ (regret matching+
with linear averaging) for both players, and writes the average strategy to a strategy
file (default: strategy.json) in the bluff-strategy-v1 format.

Also reports the exploitability of the average strategy, computed exactly by best-response
traversal (not sampled), and the game value for player 0 under that strategy.

Usage: python3 -I solver.py [game.json] [strategy.json] [--iters N]
"""
import argparse
import json
import sys
from fractions import Fraction


class Game:
    def __init__(self, path):
        with open(path) as f:
            g = json.load(f)
        if g.get("format") != "bluff-game-v1":
            raise SystemExit(f"unsupported game format: {g.get('format')!r}")
        self.name = g["name"]
        self.root = g["root"]
        # node id -> tuple describing it
        self.nodes = {}
        # infoset name -> {"player": int, "actions": [labels]}
        self.infosets = {}
        # infoset name -> list of node ids belonging to it
        self.info_nodes = {}
        for n in g["nodes"]:
            nid = n["id"]
            kind = n["kind"]
            if kind == "terminal":
                self.nodes[nid] = ("T", float(Fraction(n["payoff_p0"])))
            elif kind == "chance":
                outs = [(float(Fraction(o["prob"])), o["child"]) for o in n["outcomes"]]
                total = sum(p for p, _ in outs)
                if abs(total - 1.0) > 1e-12:
                    raise SystemExit(f"chance node {nid} probabilities sum to {total}")
                self.nodes[nid] = ("C", outs)
            elif kind == "decision":
                name = n["infoset"]
                labels = [a["label"] for a in n["actions"]]
                children = [a["child"] for a in n["actions"]]
                player = n["player"]
                if player not in (0, 1):
                    raise SystemExit(f"node {nid}: bad player {player!r}")
                info = self.infosets.get(name)
                if info is None:
                    self.infosets[name] = {"player": player, "actions": labels}
                    self.info_nodes[name] = []
                elif info["player"] != player or info["actions"] != labels:
                    raise SystemExit(f"infoset {name!r} is inconsistent at node {nid}")
                self.info_nodes[name].append(nid)
                self.nodes[nid] = ("D", name, children)
            else:
                raise SystemExit(f"node {nid}: unknown kind {kind!r}")
        for nid, desc in self.nodes.items():
            kids = [c for _, c in desc[1]] if desc[0] == "C" else (desc[2] if desc[0] == "D" else [])
            for c in kids:
                if c not in self.nodes:
                    raise SystemExit(f"node {nid} points to missing child {c}")


def regret_match(regrets):
    pos = [max(r, 0.0) for r in regrets]
    s = sum(pos)
    if s > 0:
        return [p / s for p in pos]
    return [1.0 / len(regrets)] * len(regrets)


def cfr_plus(game, iters, report_every):
    R = {I: [0.0] * len(d["actions"]) for I, d in game.infosets.items()}
    S = {I: [0.0] * len(d["actions"]) for I, d in game.infosets.items()}
    cur = {}

    def sigma_of(I):
        if I not in cur:
            cur[I] = regret_match(R[I])
        return cur[I]

    for t in range(1, iters + 1):
        delta = {I: [0.0] * len(d["actions"]) for I, d in game.infosets.items()}
        cur.clear()

        def walk(nid, p0, p1, pc):
            """Returns player 0's expected payoff from nid under the current strategies.

            p0, p1: each player's probability of reaching nid through their own actions.
            pc: chance probability of reaching nid.
            """
            desc = game.nodes[nid]
            if desc[0] == "T":
                return desc[1]
            if desc[0] == "C":
                return sum(p * walk(c, p0, p1, pc * p) for p, c in desc[1])
            _, I, children = desc
            player = game.infosets[I]["player"]
            sigma = sigma_of(I)
            if player == 0:
                us = [walk(c, p0 * sigma[a], p1, pc) for a, c in enumerate(children)]
            else:
                us = [walk(c, p0, p1 * sigma[a], pc) for a, c in enumerate(children)]
            v = sum(sigma[a] * us[a] for a in range(len(children)))
            # Counterfactual regret: weighted by the opponent's reach (chance included).
            if player == 0:
                w = p1 * pc
                for a in range(len(children)):
                    delta[I][a] += w * (us[a] - v)
                own = p0
            else:
                w = p0 * pc
                for a in range(len(children)):
                    # Player 1's payoff is -u0, so its regret uses the negated values.
                    delta[I][a] += w * (v - us[a])
                own = p1
            # Linear averaging (CFR+): iteration t contributes with weight t.
            for a in range(len(children)):
                S[I][a] += t * own * sigma[a]
            return v

        walk(game.root, 1.0, 1.0, 1.0)

        # CFR+: accumulate instantaneous regrets, then floor the cumulative regret at zero.
        for I in R:
            for a in range(len(R[I])):
                R[I][a] = max(R[I][a] + delta[I][a], 0.0)

        if report_every and (t % report_every == 0 or t == iters):
            avg = average_strategy(game, S)
            expl = exploitability(game, avg)
            print(f"iter {t:>8}  exploitability {expl:.3e}", file=sys.stderr)

    return average_strategy(game, S)


def average_strategy(game, S):
    avg = {}
    for I, d in game.infosets.items():
        s = sum(S[I])
        k = len(d["actions"])
        avg[I] = [x / s for x in S[I]] if s > 0 else [1.0 / k] * k
    return avg


def value_p0(game, strat):
    """Expected payoff to player 0 when both players follow strat."""

    def v(nid):
        desc = game.nodes[nid]
        if desc[0] == "T":
            return desc[1]
        if desc[0] == "C":
            return sum(p * v(c) for p, c in desc[1])
        _, I, children = desc
        return sum(strat[I][a] * v(c) for a, c in enumerate(children))

    return v(game.root)


def best_response_value_p0(game, strat, br_player):
    """Value to player 0 when br_player plays a best response to strat (the opponent's strategy).

    Player 0 maximizes this value; player 1 minimizes it. Computed exactly: infoset choices are
    made by maximizing the reach-weighted sum over the infoset's nodes, which is the correct
    best response under perfect recall.
    """
    sign = 1.0 if br_player == 0 else -1.0
    opp = 1 - br_player

    # Forward pass: probability of reaching each node via chance and the opponent only.
    q = {}
    stack = [(game.root, 1.0)]
    while stack:
        nid, w = stack.pop()
        q[nid] = q.get(nid, 0.0) + w
        desc = game.nodes[nid]
        if desc[0] == "C":
            for p, c in desc[1]:
                stack.append((c, w * p))
        elif desc[0] == "D":
            _, I, children = desc
            if game.infosets[I]["player"] == br_player:
                for c in children:
                    stack.append((c, w))
            else:
                for a, c in enumerate(children):
                    stack.append((c, w * strat[I][a]))

    Vmemo = {}
    choice = {}

    def V(nid):
        if nid in Vmemo:
            return Vmemo[nid]
        desc = game.nodes[nid]
        if desc[0] == "T":
            res = desc[1]
        elif desc[0] == "C":
            res = sum(p * V(c) for p, c in desc[1])
        else:
            _, I, children = desc
            if game.infosets[I]["player"] == br_player:
                res = V(children[best_action(I)])
            else:
                res = sum(strat[I][a] * V(c) for a, c in enumerate(children))
        Vmemo[nid] = res
        return res

    def best_action(I):
        if I in choice:
            return choice[I]
        k = len(game.infosets[I]["actions"])
        score = [0.0] * k
        for nid in game.info_nodes[I]:
            w = q.get(nid, 0.0)
            children = game.nodes[nid][2]
            for a, c in enumerate(children):
                score[a] += w * V(c)
        a_star = max(range(k), key=lambda a: sign * score[a])
        choice[I] = a_star
        return a_star

    return V(game.root)


def exploitability(game, strat):
    """(BR value of player 0 vs player 1) + (BR value of player 1 vs player 0), divided by 2."""
    max0 = best_response_value_p0(game, strat, 0)  # player 0 maximizes u0
    min1 = best_response_value_p0(game, strat, 1)  # player 1 minimizes u0
    # Player 1's best-response payoff is -min1.
    return (max0 + (-min1)) / 2.0


def write_strategy(path, game, avg, producer):
    probs = {}
    for I in game.infosets:
        acts = game.infosets[I]["actions"]
        s = sum(avg[I])
        probs[I] = {lab: repr(p / s) for lab, p in zip(acts, avg[I])}
    out = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": producer,
        "probs": probs,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", default="game.json")
    ap.add_argument("strategy", nargs="?", default="strategy.json")
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--report-every", type=int, default=10000)
    args = ap.parse_args()

    game = Game(args.game)
    avg = cfr_plus(game, args.iters, args.report_every)
    expl = exploitability(game, avg)
    val = value_p0(game, avg)
    write_strategy(
        args.strategy,
        game,
        avg,
        f"CFR+ (linear averaging), {args.iters} iterations, exact best-response exploitability",
    )
    print(json.dumps({
        "game": game.name,
        "infosets": len(game.infosets),
        "iters": args.iters,
        "exploitability": expl,
        "value_p0": val,
    }, indent=2))


if __name__ == "__main__":
    main()

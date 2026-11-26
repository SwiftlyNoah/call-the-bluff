#!/usr/bin/env python3
"""Solve a bluff-game-v1 extensive-form game with CFR+ and write the average strategy.

Usage: python3 -I solve_leduc.py [game.json] [strategy.json] [iterations]

Algorithm: CFR+ (regret matching+, floored cumulative regrets, linearly weighted
average strategy) with alternating updates, run over the full game tree (no sampling).
Exploitability is computed exactly with best responses over the same tree.
"""

import json
import sys
import time
from fractions import Fraction

sys.setrecursionlimit(100000)


def parse_num(s):
    return float(Fraction(s))


def load_game(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("format") != "bluff-game-v1":
        raise SystemExit(f"unexpected game format: {d.get('format')!r}")
    nodes = {}
    for n in d["nodes"]:
        nodes[n["id"]] = n
    # Infoset table: name -> (player, action labels)
    infosets = {}
    for n in d["nodes"]:
        if n["kind"] == "decision":
            labels = [a["label"] for a in n["actions"]]
            key = n["infoset"]
            if key in infosets:
                if infosets[key][0] != n["player"] or infosets[key][1] != labels:
                    raise SystemExit(f"inconsistent infoset {key!r}")
            else:
                infosets[key] = (n["player"], labels)
    # Pre-convert numbers
    for n in d["nodes"]:
        if n["kind"] == "chance":
            n["_probs"] = [parse_num(o["prob"]) for o in n["outcomes"]]
            total = sum(n["_probs"])
            if abs(total - 1.0) > 1e-12:
                raise SystemExit(f"chance node {n['id']} probs sum to {total}")
        elif n["kind"] == "terminal":
            n["_payoff0"] = parse_num(n["payoff_p0"])
    return d["name"], d["root"], nodes, infosets


class Game:
    def __init__(self, path):
        self.name, self.root, self.nodes, self.infosets = load_game(path)
        # Per-infoset node lists, for best responses.
        self.infoset_nodes = {k: [] for k in self.infosets}
        for n in self.nodes.values():
            if n["kind"] == "decision":
                self.infoset_nodes[n["infoset"]].append(n["id"])
        # Children lists in action order.
        self.children = {}
        for nid, n in self.nodes.items():
            if n["kind"] == "chance":
                self.children[nid] = [o["child"] for o in n["outcomes"]]
            elif n["kind"] == "decision":
                self.children[nid] = [a["child"] for a in n["actions"]]


def regret_match(regret):
    pos = [r if r > 0.0 else 0.0 for r in regret]
    s = sum(pos)
    k = len(regret)
    if s > 0.0:
        return [p / s for p in pos]
    return [1.0 / k] * k


def solve(game, iterations, log_every=0):
    nodes = game.nodes
    cumreg = {k: [0.0] * len(v[1]) for k, v in game.infosets.items()}
    stratsum = {k: [0.0] * len(v[1]) for k, v in game.infosets.items()}

    def current(info):
        return regret_match(cumreg[info])

    def traverse(nid, i, pi_self, pi_opp, t):
        """Return the expected payoff to player i at node nid (values from i's view).

        pi_self: reach probability from player i's own decisions.
        pi_opp: reach probability from opponent decisions and chance.
        """
        n = nodes[nid]
        kind = n["kind"]
        if kind == "terminal":
            return n["_payoff0"] if i == 0 else -n["_payoff0"]
        if kind == "chance":
            total = 0.0
            for p, c in zip(n["_probs"], game.children[nid]):
                total += p * traverse(c, i, pi_self, pi_opp * p, t)
            return total
        player = n["player"]
        info = n["infoset"]
        sigma = current(info)
        kids = game.children[nid]
        if player == i:
            vals = [traverse(c, i, pi_self * sigma[a], pi_opp, t) for a, c in enumerate(kids)]
            node_val = sum(s * v for s, v in zip(sigma, vals))
            reg = cumreg[info]
            for a in range(len(kids)):
                reg[a] = max(0.0, reg[a] + pi_opp * (vals[a] - node_val))
            ss = stratsum[info]
            w = t * pi_self
            for a in range(len(kids)):
                ss[a] += w * sigma[a]
            return node_val
        # Opponent decision: follow current strategy, no update here.
        node_val = 0.0
        for a, c in enumerate(kids):
            node_val += sigma[a] * traverse(c, i, pi_self, pi_opp * sigma[a], t)
        return node_val

    start = time.time()
    for t in range(1, iterations + 1):
        traverse(game.root, 0, 1.0, 1.0, t)
        traverse(game.root, 1, 1.0, 1.0, t)
        if log_every and t % log_every == 0:
            expl = exploitability(game, average_strategy(game, stratsum))
            print(f"iter {t:6d}  exploitability {expl:.3e}  elapsed {time.time() - start:.1f}s",
                  flush=True)
    return stratsum


def average_strategy(game, stratsum):
    avg = {}
    for info, (player, labels) in game.infosets.items():
        ss = stratsum[info]
        s = sum(ss)
        if s > 0.0:
            avg[info] = [x / s for x in ss]
        else:
            avg[info] = [1.0 / len(labels)] * len(labels)
    return avg


def game_value(game, strat, player_view=0):
    """Expected payoff to player 0 when both players follow strat."""
    nodes = game.nodes

    def val(nid):
        n = nodes[nid]
        if n["kind"] == "terminal":
            return n["_payoff0"]
        if n["kind"] == "chance":
            return sum(p * val(c) for p, c in zip(n["_probs"], game.children[nid]))
        sigma = strat[n["infoset"]]
        return sum(s * val(c) for s, c in zip(sigma, game.children[nid]))

    return val(game.root)


def best_response_value(game, strat, br_player):
    """Best-response payoff to br_player (from its own perspective) against strat."""
    nodes = game.nodes
    sign = 1.0 if br_player == 0 else -1.0
    # Reach of the opponent and chance to every node (BR player's own moves weight 1).
    q = {}

    def forward(nid, w):
        q[nid] = w
        n = nodes[nid]
        if n["kind"] == "terminal":
            return
        kids = game.children[nid]
        if n["kind"] == "chance":
            for p, c in zip(n["_probs"], kids):
                forward(c, w * p)
        elif n["player"] == br_player:
            for c in kids:
                forward(c, w)
        else:
            sigma = strat[n["infoset"]]
            for s, c in zip(sigma, kids):
                forward(c, w * s)

    forward(game.root, 1.0)

    memo_u = {}
    choice = {}

    def U(nid):
        """Expected payoff to br_player from nid, conditional on reaching nid."""
        if nid in memo_u:
            return memo_u[nid]
        n = nodes[nid]
        if n["kind"] == "terminal":
            res = sign * n["_payoff0"]
        elif n["kind"] == "chance":
            res = sum(p * U(c) for p, c in zip(n["_probs"], game.children[nid]))
        elif n["player"] == br_player:
            a = choose(n["infoset"])
            res = U(game.children[nid][a])
        else:
            sigma = strat[n["infoset"]]
            res = sum(s * U(c) for s, c in zip(sigma, game.children[nid]))
        memo_u[nid] = res
        return res

    def choose(info):
        if info in choice:
            return choice[info]
        k = len(game.infosets[info][1])
        best_a, best_v = 0, None
        for a in range(k):
            v = 0.0
            for nid in game.infoset_nodes[info]:
                v += q[nid] * U(game.children[nid][a])
            if best_v is None or v > best_v:
                best_a, best_v = a, v
        choice[info] = best_a
        return best_a

    return U(game.root)


def exploitability(game, strat):
    v0 = best_response_value(game, strat, 0)  # player 0's BR payoff vs strat[player 1]
    v1 = best_response_value(game, strat, 1)  # player 1's BR payoff vs strat[player 0]
    return (v0 + v1) / 2.0


def write_strategy(game, avg, path, producer):
    probs = {}
    for info, (player, labels) in game.infosets.items():
        p = avg[info]
        # Make the probabilities sum to exactly 1 (last action absorbs rounding).
        vals = list(p)
        vals[-1] = 1.0 - sum(vals[:-1])
        probs[info] = {lab: repr(float(v)) for lab, v in zip(labels, vals)}
    out = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": producer,
        "probs": probs,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")


def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    game = Game(game_path)
    print(f"game {game.name!r}: {len(game.nodes)} nodes, {len(game.infosets)} infosets", flush=True)
    stratsum = solve(game, iterations, log_every=max(1, iterations // 10))
    avg = average_strategy(game, stratsum)
    expl = exploitability(game, avg)
    val = game_value(game, avg)
    print(f"final: iterations={iterations} exploitability={expl:.6e} value_p0={val:.8f}", flush=True)
    write_strategy(game, avg, out_path,
                   f"CFR+ (alternating, full tree) {iterations} iterations, linear averaging")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

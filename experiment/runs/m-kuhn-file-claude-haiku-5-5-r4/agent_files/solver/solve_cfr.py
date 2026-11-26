#!/usr/bin/env python3
"""Generic CFR+ solver for two-player zero-sum extensive-form games (bluff-game-v1).

Reads the game tree from game.json, runs CFR+ (regret matching+ with linear
averaging) for both players, computes the exploitability of the average
strategy with exact best responses, and writes strategy.json (bluff-strategy-v1).

Usage: python3 -I solve_cfr.py <game.json> <strategy.json> [iterations]
"""
import json
import sys
from fractions import Fraction


def parse_num(s):
    return Fraction(s)


def load_game(path):
    with open(path) as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise SystemExit(f"unsupported game format: {g.get('format')!r}")
    nodes = {}
    for n in g["nodes"]:
        if n["id"] in nodes:
            raise SystemExit(f"duplicate node id {n['id']}")
        nodes[n["id"]] = n
    root = g["root"]

    # Validate tree shape: every non-root node has exactly one parent, all children exist.
    parents = {}
    for n in nodes.values():
        children = []
        if n["kind"] == "chance":
            children = [o["child"] for o in n["outcomes"]]
            total = sum(parse_num(o["prob"]) for o in n["outcomes"])
            if total != 1:
                raise SystemExit(f"chance node {n['id']} probabilities sum to {total}")
        elif n["kind"] == "decision":
            children = [a["child"] for a in n["actions"]]
        for c in children:
            if c not in nodes:
                raise SystemExit(f"node {n['id']} points to missing child {c}")
            if c in parents:
                raise SystemExit(f"node {c} has more than one parent")
            parents[c] = n["id"]
    if root in parents:
        raise SystemExit("root has a parent")

    # Information sets: consistent player and action labels.
    infoset_actions = {}
    infoset_player = {}
    for n in nodes.values():
        if n["kind"] != "decision":
            continue
        labels = tuple(a["label"] for a in n["actions"])
        name = n["infoset"]
        if name in infoset_actions:
            if infoset_actions[name] != labels:
                raise SystemExit(f"infoset {name!r} has inconsistent actions")
            if infoset_player[name] != n["player"]:
                raise SystemExit(f"infoset {name!r} has inconsistent players")
        else:
            infoset_actions[name] = labels
            infoset_player[name] = n["player"]

    return {
        "name": g["name"],
        "nodes": nodes,
        "root": root,
        "infoset_actions": infoset_actions,
        "infoset_player": infoset_player,
    }


class Solver:
    def __init__(self, game):
        self.g = game
        self.nodes = game["nodes"]
        self.regret = {}
        self.strat_sum = {}
        for name, acts in game["infoset_actions"].items():
            self.regret[name] = [0.0] * len(acts)
            self.strat_sum[name] = [0.0] * len(acts)

    def current_strategy(self, name):
        r = self.regret[name]
        pos = [max(x, 0.0) for x in r]
        s = sum(pos)
        if s > 0:
            return [p / s for p in pos]
        n = len(r)
        return [1.0 / n] * n

    def average_strategy(self, name):
        ss = self.strat_sum[name]
        s = sum(ss)
        n = len(ss)
        if s > 0:
            return [x / s for x in ss]
        return [1.0 / n] * n

    def cfr(self, node_id, p0, p1, pc, t):
        """Return expected payoff to player 0 from node under current strategies.

        p0, p1: reach probabilities of each player's own actions; pc: chance reach.
        Regret updates use the opponent's reach and chance reach; the average
        strategy is weighted by the player's own reach, scaled by t (linear averaging).
        """
        node = self.nodes[node_id]
        kind = node["kind"]
        if kind == "terminal":
            return float(parse_num(node["payoff_p0"]))
        if kind == "chance":
            v = 0.0
            for o in node["outcomes"]:
                v += float(parse_num(o["prob"])) * self.cfr(o["child"], p0, p1, pc * float(parse_num(o["prob"])), t)
            return v

        player = node["player"]
        name = node["infoset"]
        sigma = self.current_strategy(name)
        util = []
        for a, act in enumerate(node["actions"]):
            if player == 0:
                util.append(self.cfr(act["child"], p0 * sigma[a], p1, pc, t))
            else:
                util.append(self.cfr(act["child"], p0, p1 * sigma[a], pc, t))
        value = sum(sigma[a] * util[a] for a in range(len(sigma)))

        # Regrets are stated for the player to move, in that player's payoff sign.
        sign = 1.0 if player == 0 else -1.0
        opp_reach = (p1 if player == 0 else p0) * pc
        own_reach = p0 if player == 0 else p1
        for a in range(len(sigma)):
            self.regret[name][a] = max(self.regret[name][a] + opp_reach * sign * (util[a] - value), 0.0)
            self.strat_sum[name][a] += t * own_reach * sigma[a]
        return value

    def train(self, iterations):
        for t in range(1, iterations + 1):
            self.cfr(self.g["root"], 1.0, 1.0, 1.0, t)

    def avg_strategies(self):
        return {name: self.average_strategy(name) for name in self.regret}


def best_response_value(game, strategy, br_player):
    """Exact best-response value for br_player (payoff in that player's sign) against
    the fixed behavioral strategy `strategy` (infoset -> list of probs) of the opponent.

    Infosets of the best-responding player are decided bottom-up: an infoset is
    decided only after every best-response infoset below it. Under perfect recall
    this gives a globally consistent pure best response.
    """
    nodes = game["nodes"]
    sign = 1.0 if br_player == 0 else -1.0
    choice = {}  # BR infoset -> chosen action index

    def subtree_br_infosets(node_id, acc):
        n = nodes[node_id]
        if n["kind"] == "terminal":
            return
        if n["kind"] == "chance":
            for o in n["outcomes"]:
                subtree_br_infosets(o["child"], acc)
            return
        if n["player"] == br_player:
            acc.add(n["infoset"])
        for a in n["actions"]:
            subtree_br_infosets(a["child"], acc)

    def value(node_id, reach_opp):
        # Returns sum over leaves of (opp reach * chance prob * payoff in BR sign),
        # with BR decisions following `choice`. Caller supplies the opponent reach.
        n = nodes[node_id]
        if n["kind"] == "terminal":
            return reach_opp * sign * float(parse_num(n["payoff_p0"]))
        if n["kind"] == "chance":
            return sum(value(o["child"], reach_opp * float(parse_num(o["prob"]))) for o in n["outcomes"])
        if n["player"] == br_player:
            a = choice[n["infoset"]]
            return value(n["actions"][a]["child"], reach_opp)
        sigma = strategy[n["infoset"]]
        return sum(value(act["child"], reach_opp * sigma[i]) for i, act in enumerate(n["actions"]))

    def node_value_from(node_id):
        # Expected BR-sign payoff from a node, with the BR player's choices fixed and
        # opponent/chance reach weighted from this node (reach_opp = 1 here).
        return value(node_id, 1.0)

    # Collect BR infosets and their nodes.
    br_nodes = {}
    for nid, n in nodes.items():
        if n["kind"] == "decision" and n["player"] == br_player:
            br_nodes.setdefault(n["infoset"], []).append(nid)

    # Opponent+chance reach from root to each node (BR player's own reach excluded).
    reach = {}

    def assign_reach(node_id, r):
        reach[node_id] = r
        n = nodes[node_id]
        if n["kind"] == "terminal":
            return
        if n["kind"] == "chance":
            for o in n["outcomes"]:
                assign_reach(o["child"], r * float(parse_num(o["prob"])))
            return
        if n["player"] == br_player:
            for a in n["actions"]:
                assign_reach(a["child"], r)
        else:
            sigma = strategy[n["infoset"]]
            for i, a in enumerate(n["actions"]):
                assign_reach(a["child"], r * sigma[i])

    assign_reach(game["root"], 1.0)

    decided = set()

    def decide(infoset, stack=()):
        if infoset in decided:
            return
        if infoset in stack:
            raise SystemExit("cycle among best-response infosets: imperfect recall?")
        below = set()
        for nid in br_nodes[infoset]:
            for a in nodes[nid]["actions"]:
                subtree_br_infosets(a["child"], below)
        below.discard(infoset)
        for other in sorted(below):
            decide(other, stack + (infoset,))
        best_a, best_v = None, None
        for a in range(len(game["infoset_actions"][infoset])):
            total = 0.0
            for nid in br_nodes[infoset]:
                # Value of taking action a at nid, weighted by opponent+chance reach to nid.
                child = nodes[nid]["actions"][a]["child"]
                total += value_with_reach(child, reach[nid])
            if best_v is None or total > best_v + 1e-15:
                best_a, best_v = a, total
        choice[infoset] = best_a
        decided.add(infoset)

    def value_with_reach(node_id, r):
        # Same as value() but starting with reach r (the opponent+chance reach of the parent).
        return value(node_id, r)

    for name in sorted(br_nodes):
        decide(name)

    return node_value_from(game["root"]), choice


def exploitability(game, strategy):
    br0, _ = best_response_value(game, strategy, 0)
    br1, _ = best_response_value(game, strategy, 1)
    return br0, br1, (br0 + br1) / 2.0


def expected_value_p0(game, strategy):
    nodes = game["nodes"]

    def ev(node_id):
        n = nodes[node_id]
        if n["kind"] == "terminal":
            return float(parse_num(n["payoff_p0"]))
        if n["kind"] == "chance":
            return sum(float(parse_num(o["prob"])) * ev(o["child"]) for o in n["outcomes"])
        sigma = strategy[n["infoset"]]
        return sum(sigma[i] * ev(a["child"]) for i, a in enumerate(n["actions"]))

    return ev(game["root"])


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    game_path, out_path = sys.argv[1], sys.argv[2]
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 200000

    game = load_game(game_path)
    solver = Solver(game)
    solver.train(iterations)
    avg = solver.avg_strategies()

    br0, br1, expl = exploitability(game, avg)
    v0 = expected_value_p0(game, avg)
    print(f"iterations={iterations}")
    print(f"BR value p0 vs p1 strategy: {br0!r}")
    print(f"BR value p1 vs p0 strategy: {br1!r}")
    print(f"exploitability (avg of BR values): {expl!r}")
    print(f"game value to p0 under the strategy: {v0!r}")

    out = {
        "format": "bluff-strategy-v1",
        "game": game["name"],
        "producer": f"generic CFR+ solver (solver/solve_cfr.py), {iterations} iterations, linear averaging",
        "probs": {},
    }
    for name in sorted(avg):
        acts = game["infoset_actions"][name]
        out["probs"][name] = {a: repr(float(p)) for a, p in zip(acts, avg[name])}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

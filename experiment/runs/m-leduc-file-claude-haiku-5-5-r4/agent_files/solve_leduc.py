#!/usr/bin/env python3
"""CFR+ solver for a two-player zero-sum extensive-form game (bluff-game-v1 format).

Reads game.json, runs CFR+ (regret matching+, alternating updates, linearly weighted
average strategy), measures exploitability by exact best-response traversal of the tree,
and writes the average strategy to strategy.json (bluff-strategy-v1 format).
"""
import json
import os
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
GAME_PATH = os.path.join(HERE, "game.json")
STRATEGY_PATH = os.path.join(HERE, "strategy.json")

MAX_ITERATIONS = int(os.environ.get("CFR_ITERS", "20000"))
CHECK_EVERY = 500
TARGET_EXPLOITABILITY = 1e-6
TIME_LIMIT_SECONDS = 9 * 60


def load_game(path):
    with open(path, encoding="utf-8") as f:
        game = json.load(f)
    if game.get("format") != "bluff-game-v1":
        raise ValueError(f"unexpected game format: {game.get('format')!r}")
    nodes = {n["id"]: n for n in game["nodes"]}
    if len(nodes) != len(game["nodes"]):
        raise ValueError("duplicate node ids")

    kind = {}
    payoff = {}        # terminal id -> float payoff for player 0
    chance = {}        # chance id -> list of (prob float, child id)
    decision = {}      # decision id -> (player, infoset index, [child id per canonical action])
    infoset_names = []
    infoset_index = {}
    infoset_player = []
    infoset_actions = []
    infoset_nodes = []

    for nid, n in nodes.items():
        k = n["kind"]
        kind[nid] = k
        if k == "terminal":
            payoff[nid] = float(Fraction(n["payoff_p0"]))
        elif k == "chance":
            probs = [Fraction(o["prob"]) for o in n["outcomes"]]
            if sum(probs) != 1:
                raise ValueError(f"chance node {nid} probabilities sum to {sum(probs)}")
            chance[nid] = [(float(Fraction(o["prob"])), o["child"]) for o in n["outcomes"]]
        elif k == "decision":
            name = n["infoset"]
            labels = [a["label"] for a in n["actions"]]
            if name not in infoset_index:
                infoset_index[name] = len(infoset_names)
                infoset_names.append(name)
                infoset_player.append(n["player"])
                infoset_actions.append(labels)
                infoset_nodes.append([])
            I = infoset_index[name]
            if infoset_player[I] != n["player"]:
                raise ValueError(f"infoset {name} has nodes for two players")
            if sorted(infoset_actions[I]) != sorted(labels):
                raise ValueError(f"infoset {name} has inconsistent action lists")
            by_label = {a["label"]: a["child"] for a in n["actions"]}
            children = [by_label[lab] for lab in infoset_actions[I]]
            decision[nid] = (n["player"], I, children)
            infoset_nodes[I].append(nid)
        else:
            raise ValueError(f"unknown node kind {k!r} at {nid}")

    return {
        "name": game["name"],
        "root": game["root"],
        "kind": kind,
        "payoff": payoff,
        "chance": chance,
        "decision": decision,
        "infoset_names": infoset_names,
        "infoset_player": infoset_player,
        "infoset_actions": infoset_actions,
        "infoset_nodes": infoset_nodes,
    }


def regret_match_plus(regret):
    pos = [r if r > 0.0 else 0.0 for r in regret]
    s = sum(pos)
    k = len(regret)
    if s > 0.0:
        return [x / s for x in pos]
    return [1.0 / k] * k


def average_strategy(G, strat_sum):
    out = []
    for I, ss in enumerate(strat_sum):
        s = sum(ss)
        k = len(ss)
        out.append([x / s for x in ss] if s > 0.0 else [1.0 / k] * k)
    return out


def run_cfr(G, max_iters, time_limit):
    root = G["root"]
    kind, payoff, chance, decision = G["kind"], G["payoff"], G["chance"], G["decision"]
    n_inf = len(G["infoset_names"])
    n_act = [len(a) for a in G["infoset_actions"]]
    regret = [[0.0] * k for k in n_act]
    strat_sum = [[0.0] * k for k in n_act]
    player_of = G["infoset_player"]

    # Per-traversal state. Within one traversal each infoset uses a single strategy
    # (computed once from the regrets at the start of the traversal) and its updates are
    # buffered and applied at the end, as in standard CFR. Applying them in place would let
    # later nodes of the same infoset see a different strategy within one iteration.
    state = {"upd": 0, "t": 1.0}
    sig = [None] * n_inf
    delta = [[0.0] * k for k in n_act]
    sbuf = [[0.0] * k for k in n_act]

    def cfr(nid, r0, r1, rc):
        k = kind[nid]
        if k == "terminal":
            return payoff[nid]
        if k == "chance":
            v = 0.0
            for p, child in chance[nid]:
                v += p * cfr(child, r0, r1, rc * p)
            return v
        player, I, children = decision[nid]
        sigma = sig[I]
        if sigma is None:
            sigma = sig[I] = regret_match_plus(regret[I])
        if player == 0:
            utils = [cfr(c, r0 * sigma[a], r1, rc) for a, c in enumerate(children)]
        else:
            utils = [cfr(c, r0, r1 * sigma[a], rc) for a, c in enumerate(children)]
        node_v = 0.0
        for a in range(len(children)):
            node_v += sigma[a] * utils[a]
        if player == state["upd"]:
            own = r0 if player == 0 else r1
            opp = rc * (r1 if player == 0 else r0)
            sign = 1.0 if player == 0 else -1.0
            vp = sign * node_v
            dl = delta[I]
            sb = sbuf[I]
            for a in range(len(children)):
                dl[a] += opp * (sign * utils[a] - vp)
                sb[a] += own * sigma[a]
        return node_v

    def finish_traversal(upd, t):
        for I in range(n_inf):
            if player_of[I] != upd:
                continue
            reg, dl = regret[I], delta[I]
            ss, sb = strat_sum[I], sbuf[I]
            for a in range(n_act[I]):
                nr = reg[a] + dl[a]
                reg[a] = nr if nr > 0.0 else 0.0   # CFR+: floor cumulative regret at zero
                ss[a] += t * sb[a]                 # linear averaging
                dl[a] = 0.0
                sb[a] = 0.0
        for I in range(n_inf):
            sig[I] = None

    start = time.monotonic()
    history = []
    for t in range(1, max_iters + 1):
        state["t"] = float(t)
        for upd in (0, 1):
            state["upd"] = upd
            cfr(root, 1.0, 1.0, 1.0)
            finish_traversal(upd, float(t))
        if t % CHECK_EVERY == 0 or t == max_iters:
            avg = average_strategy(G, strat_sum)
            expl, v0 = exploitability(G, avg)
            elapsed = time.monotonic() - start
            history.append((t, expl))
            print(f"iter {t:6d}  exploitability {expl:.3e}  value_p0 {v0:+.6f}  elapsed {elapsed:6.1f}s", flush=True)
            if expl < TARGET_EXPLOITABILITY:
                print("reached target exploitability", flush=True)
                break
            if elapsed > time_limit:
                print("time limit reached", flush=True)
                break
    return average_strategy(G, strat_sum), history


def game_value(G, policy):
    """Expected payoff to player 0 when both players follow `policy` (infoset -> probs)."""
    kind, payoff, chance, decision = G["kind"], G["payoff"], G["chance"], G["decision"]

    def rec(nid):
        k = kind[nid]
        if k == "terminal":
            return payoff[nid]
        if k == "chance":
            return sum(p * rec(c) for p, c in chance[nid])
        _, I, children = decision[nid]
        pr = policy[I]
        return sum(pr[a] * rec(c) for a, c in enumerate(children))

    return rec(G["root"])


def best_response_value(G, policy, br_player):
    """Best-response value for `br_player` against `policy` (payoff to that player).

    Standard two-pass computation on the tree: forward pass for the reach probability of
    the opponent and chance, then a backward pass choosing, per best-responder infoset,
    the action maximizing the reach-weighted value.
    """
    kind, payoff, chance, decision = G["kind"], G["payoff"], G["chance"], G["decision"]
    root = G["root"]
    sign = 1.0 if br_player == 0 else -1.0

    # Forward pass: reach weight w(node) from chance and the opponent only.
    w = {root: 1.0}
    order = [root]
    stack = [root]
    while stack:
        nid = stack.pop()
        k = kind[nid]
        if k == "terminal":
            continue
        wn = w[nid]
        if k == "chance":
            for p, c in chance[nid]:
                w[c] = wn * p
                stack.append(c)
        else:
            player, I, children = decision[nid]
            if player == br_player:
                for c in children:
                    w[c] = wn
                    stack.append(c)
            else:
                pr = policy[I]
                for a, c in enumerate(children):
                    w[c] = wn * pr[a]
                    stack.append(c)
        order.append(nid)

    memo = {}
    choice = {}

    def vb(nid):
        # Expected payoff for br_player from nid onward, given the best-response choices.
        if nid in memo:
            return memo[nid]
        k = kind[nid]
        if k == "terminal":
            res = sign * payoff[nid]
        elif k == "chance":
            res = sum(p * vb(c) for p, c in chance[nid])
        else:
            player, I, children = decision[nid]
            if player == br_player:
                if I not in choice:
                    best_a, best_v = 0, None
                    nodes_in = G["infoset_nodes"][I]
                    qs = {}
                    for h in nodes_in:
                        _, _, ch = decision[h]
                        qs[h] = [vb(c) for c in ch]
                    for a in range(len(children)):
                        tot = sum(w[h] * qs[h][a] for h in nodes_in)
                        if best_v is None or tot > best_v:
                            best_v, best_a = tot, a
                    choice[I] = best_a
                res = vb(children[choice[I]])
            else:
                pr = policy[I]
                res = sum(pr[a] * vb(c) for a, c in enumerate(children))
        memo[nid] = res
        return res

    return vb(root)


def exploitability(G, policy):
    br0 = best_response_value(G, policy, 0)    # player 0's best-response value
    br1 = best_response_value(G, policy, 1)    # player 1's best-response value
    return (br0 + br1) / 2.0, game_value(G, policy)


def write_strategy(G, policy, path):
    probs = {}
    for I, name in enumerate(G["infoset_names"]):
        acts = G["infoset_actions"][I]
        pr = policy[I]
        vals = [float(x) for x in pr]
        # Make the vector sum to exactly 1 in float arithmetic where possible.
        s = sum(vals)
        vals = [v / s for v in vals]
        probs[name] = {lab: repr(float(v)) for lab, v in zip(acts, vals)}
    out = {
        "format": "bluff-strategy-v1",
        "game": G["name"],
        "producer": "CFR+ (regret matching+, alternating updates, linear averaging), "
                    f"exploitability checked on the game tree; {len(probs)} infosets",
        "probs": probs,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)


def main():
    G = load_game(GAME_PATH)
    print(f"game {G['name']}: {len(G['kind'])} nodes, {len(G['infoset_names'])} infosets", flush=True)
    policy, history = run_cfr(G, MAX_ITERATIONS, TIME_LIMIT_SECONDS)
    expl, v0 = exploitability(G, policy)
    print(f"final exploitability {expl:.6e}  value_p0 {v0:.8f}", flush=True)
    write_strategy(G, policy, STRATEGY_PATH)
    print(f"wrote {STRATEGY_PATH}", flush=True)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()

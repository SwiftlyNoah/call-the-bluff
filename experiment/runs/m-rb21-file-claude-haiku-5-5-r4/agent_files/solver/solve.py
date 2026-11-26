#!/usr/bin/env python3
"""CFR+ solver for bluff-game-v1 extensive-form games.

Usage: python3 -I solve.py GAME_JSON OUT_JSON [ITERATIONS]

Runs CFR+ (regret matching+ with linear averaging) for both players on the
game tree in GAME_JSON, reports exploitability of the average strategy every
few hundred iterations, and writes the average strategy in the
bluff-strategy-v1 format to OUT_JSON.
"""
import json
import sys
import threading
from fractions import Fraction


def num(s):
    return float(Fraction(s))


def load_game(path):
    with open(path) as f:
        game = json.load(f)
    if game.get("format") != "bluff-game-v1":
        raise SystemExit("unexpected game format: %r" % game.get("format"))
    raw = {n["id"]: n for n in game["nodes"]}
    infoset_ids = {}
    infoset_names = []
    infoset_player = []
    infoset_actions = []
    nodes = {}  # id -> tuple
    members = []  # infoset index -> list of node ids
    for nid, n in raw.items():
        kind = n["kind"]
        if kind == "terminal":
            nodes[nid] = ("T", num(n["payoff_p0"]))
        elif kind == "chance":
            outs = n["outcomes"]
            probs = [num(o["prob"]) for o in outs]
            total = sum(probs)
            if abs(total - 1.0) > 1e-12:
                raise SystemExit("chance node %s probabilities sum to %r" % (nid, total))
            nodes[nid] = ("C", probs, [o["child"] for o in outs])
        elif kind == "decision":
            name = n["infoset"]
            labels = [a["label"] for a in n["actions"]]
            if name not in infoset_ids:
                infoset_ids[name] = len(infoset_names)
                infoset_names.append(name)
                infoset_player.append(n["player"])
                infoset_actions.append(labels)
                members.append([])
            idx = infoset_ids[name]
            if infoset_player[idx] != n["player"] or infoset_actions[idx] != labels:
                raise SystemExit("inconsistent infoset %r" % name)
            members[idx].append(nid)
            nodes[nid] = ("D", n["player"], idx, [a["child"] for a in n["actions"]])
        else:
            raise SystemExit("unknown node kind %r" % kind)
    return game, game["root"], nodes, infoset_names, infoset_player, infoset_actions, members


def main():
    game_path, out_path = sys.argv[1], sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    game, root, N, names, player, actions, members = load_game(game_path)
    NI = len(names)
    nact = [len(a) for a in actions]
    REG = [[0.0] * nact[I] for I in range(NI)]
    SUM = [[0.0] * nact[I] for I in range(NI)]
    sig = [None] * NI

    def snapshot():
        for I in range(NI):
            pos = [r if r > 0.0 else 0.0 for r in REG[I]]
            s = sum(pos)
            if s > 0.0:
                sig[I] = [p / s for p in pos]
            else:
                sig[I] = [1.0 / nact[I]] * nact[I]

    def avg(I):
        s = SUM[I]
        tot = sum(s)
        if tot > 0.0:
            return [x / tot for x in s]
        return [1.0 / nact[I]] * nact[I]

    def cfr(nid, i, pi, po, t):
        node = N[nid]
        tag = node[0]
        if tag == "T":
            return node[1] if i == 0 else -node[1]
        if tag == "C":
            v = 0.0
            for p, c in zip(node[1], node[2]):
                v += p * cfr(c, i, pi, po * p, t)
            return v
        _, pl, I, ch = node
        s = sig[I]
        if pl == i:
            vals = [cfr(c, i, pi * sa, po, t) for c, sa in zip(ch, s)]
            v = sum(sa * x for sa, x in zip(s, vals))
            R = REG[I]
            for a in range(len(ch)):
                r = R[a] + po * (vals[a] - v)
                R[a] = r if r > 0.0 else 0.0
            S = SUM[I]
            w = t * pi
            for a in range(len(ch)):
                S[a] += w * s[a]
            return v
        v = 0.0
        for c, sa in zip(ch, s):
            v += sa * cfr(c, i, pi, po * sa, t)
        return v

    # ---- exploitability via exact best responses on the tree ----
    def expected_value(profile):
        """Player 0's expected payoff when both players follow profile (infoset -> probs)."""
        memo = {}

        def ev(nid):
            if nid in memo:
                return memo[nid]
            node = N[nid]
            tag = node[0]
            if tag == "T":
                val = node[1]
            elif tag == "C":
                val = sum(p * ev(c) for p, c in zip(node[1], node[2]))
            else:
                val = sum(sa * ev(c) for c, sa in zip(node[3], profile[node[2]]))
            memo[nid] = val
            return val

        return ev(root)

    def best_response(i, profile):
        """Best-response value for player i (in player i's payoff units) vs profile of the other player."""
        # reach of the opponent (and chance) to every node of player i's decisions
        opp_reach = {}

        def down(nid, po):
            node = N[nid]
            tag = node[0]
            if tag == "T":
                return
            if tag == "C":
                for p, c in zip(node[1], node[2]):
                    down(c, po * p)
                return
            _, pl, I, ch = node
            if pl == i:
                opp_reach[nid] = po
                for c in ch:
                    down(c, po)
            else:
                for c, sa in zip(ch, profile[I]):
                    down(c, po * sa)

        down(root, 1.0)

        choice = {}
        vmemo = {}

        def V(nid):
            if nid in vmemo:
                return vmemo[nid]
            node = N[nid]
            tag = node[0]
            if tag == "T":
                val = node[1] if i == 0 else -node[1]
            elif tag == "C":
                val = sum(p * V(c) for p, c in zip(node[1], node[2]))
            else:
                _, pl, I, ch = node
                if pl == i:
                    a = decide(I)
                    val = V(ch[a])
                else:
                    val = sum(sa * V(c) for c, sa in zip(ch, profile[I]))
            vmemo[nid] = val
            return val

        def decide(I):
            if I in choice:
                return choice[I]
            q = [0.0] * nact[I]
            for nid in members[I]:
                w = opp_reach.get(nid, 0.0)
                if w == 0.0:
                    continue
                ch = N[nid][3]
                for a, c in enumerate(ch):
                    q[a] += w * V(c)
            best = max(range(nact[I]), key=lambda a: (q[a], -a))
            choice[I] = best
            return best

        return V(root)

    def exploitability(prof):
        v0 = best_response(0, prof[1])  # player 0 best response vs player 1
        v1 = best_response(1, prof[0])  # player 1 best response vs player 0
        return (v0 + v1) / 2.0, v0, v1

    def run():
        for t in range(1, iters + 1):
            snapshot()
            cfr(root, 0, 1.0, 1.0, t)
            snapshot()
            cfr(root, 1, 1.0, 1.0, t)
            if t % max(1, iters // 10) == 0 or t == iters:
                prof = [[None] * NI for _ in range(2)]
                # profile lookup by infoset for each player's own strategy
                prof_all = [avg(I) for I in range(NI)]
                e, v0, v1 = exploitability(
                    [_ProfileView(prof_all, player, 0), _ProfileView(prof_all, player, 1)]
                )
                print("iter %d  exploitability %.3e  (br0 %.6f, br1 %.6f)" % (t, e, v0, -v1), flush=True)
        prof_all = [avg(I) for I in range(NI)]
        return prof_all

    class _ProfileView:
        """profile[I] for infosets owned by one player; the other player's entries are unused."""

        def __init__(self, probs, owner_map, who):
            self.probs = probs
            self.owner_map = owner_map
            self.who = who

        def __getitem__(self, I):
            return self.probs[I]

    result = {}

    def target():
        result["prof"] = run()

    threading.stack_size(512 * 1024 * 1024)
    sys.setrecursionlimit(1000000)
    th = threading.Thread(target=target)
    th.start()
    th.join()
    prof_all = result["prof"]

    ev = expected_value([prof_all[I] for I in range(NI)])
    print("game value for player 0 under average profile: %r" % ev)

    out = {
        "format": "bluff-strategy-v1",
        "game": game["name"],
        "producer": "CFR+ (regret matching+, linear averaging), %d iterations, solve.py" % iters,
        "probs": {},
    }
    for I in range(NI):
        out["probs"][names[I]] = {
            actions[I][a]: repr(prof_all[I][a]) for a in range(nact[I])
        }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote %s with %d information sets" % (out_path, NI))


if __name__ == "__main__":
    main()

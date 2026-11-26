#!/usr/bin/env python3
"""CFR+ solver for bluff-game-v1 game trees (see GAME_FORMAT.md).

Usage: python3 -I solver/cfr_solver.py game.json strategy.json [max_iters] [time_budget_s]

Runs simultaneous CFR+ (regret floored at zero, linearly weighted average strategy) for both
players over the full game tree, reports exploitability periodically, and writes the average
strategy in bluff-strategy-v1 format.
"""
import json
import sys
import time
from fractions import Fraction

sys.setrecursionlimit(100000)


def to_float(s):
    return float(Fraction(s))


def load_tree(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise SystemExit(f"unsupported game format: {g.get('format')!r}")
    raw = g["nodes"]
    idx = {n["id"]: k for k, n in enumerate(raw)}
    N = len(raw)
    kind = [None] * N
    player = [-1] * N
    inf = [-1] * N
    children = [None] * N
    probs = [None] * N
    pay = [0.0] * N
    inf_index, inf_names, inf_actions, inf_player, inf_nodes = {}, [], [], [], []
    for k, n in enumerate(raw):
        kd = n["kind"]
        kind[k] = kd
        if kd == "terminal":
            pay[k] = to_float(n["payoff_p0"])
        elif kd == "chance":
            children[k] = [idx[o["child"]] for o in n["outcomes"]]
            probs[k] = [to_float(o["prob"]) for o in n["outcomes"]]
            total = sum(Fraction(o["prob"]) for o in n["outcomes"])
            if total != 1:
                raise SystemExit(f"chance node {n['id']} probabilities sum to {total}")
        elif kd == "decision":
            labels = [a["label"] for a in n["actions"]]
            name = n["infoset"]
            if name not in inf_index:
                inf_index[name] = len(inf_names)
                inf_names.append(name)
                inf_actions.append(labels)
                inf_player.append(n["player"])
                inf_nodes.append([])
            I = inf_index[name]
            if inf_actions[I] != labels or inf_player[I] != n["player"]:
                raise SystemExit(f"inconsistent infoset {name!r}")
            player[k] = n["player"]
            inf[k] = I
            children[k] = [idx[a["child"]] for a in n["actions"]]
            inf_nodes[I].append(k)
        else:
            raise SystemExit(f"unknown node kind {kd!r}")
    tree = dict(
        root=idx[g["root"]], N=N, kind=kind, player=player, inf=inf, children=children,
        probs=probs, pay=pay, inf_names=inf_names, inf_actions=inf_actions,
        inf_player=inf_player, inf_nodes=inf_nodes, name=g["name"],
    )
    return tree


def main():
    game_path, out_path = sys.argv[1], sys.argv[2]
    max_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    time_budget = float(sys.argv[4]) if len(sys.argv) > 4 else 900.0
    T = load_tree(game_path)
    N, root = T["N"], T["root"]
    kind, player, inf, children = T["kind"], T["player"], T["inf"], T["children"]
    probs, pay = T["probs"], T["pay"]
    inf_actions, inf_player, inf_nodes = T["inf_actions"], T["inf_player"], T["inf_nodes"]
    n_inf = len(inf_actions)

    # Every node must be reachable exactly once from the root (a tree).
    order = [root]
    seen = [False] * N
    seen[root] = True
    for k in order:
        if children[k] is not None:
            for c in children[k]:
                if seen[c]:
                    raise SystemExit("game graph is not a tree")
                seen[c] = True
                order.append(c)
    if len(order) != N:
        raise SystemExit(f"{N - len(order)} nodes unreachable from root")

    regret = [[0.0] * len(a) for a in inf_actions]
    strat_sum = [[0.0] * len(a) for a in inf_actions]

    def current_strategy(I):
        r = regret[I]
        pos = [x if x > 0.0 else 0.0 for x in r]
        s = sum(pos)
        if s > 0.0:
            return [x / s for x in pos]
        m = len(r)
        return [1.0 / m] * m

    def cfr(k, p0, p1, pc, t):
        """Returns expected payoff to player 0 from node k (conditional on reaching k)."""
        kd = kind[k]
        if kd == "terminal":
            return pay[k]
        if kd == "chance":
            u = 0.0
            for c, pr in zip(children[k], probs[k]):
                u += pr * cfr(c, p0, p1, pc * pr, t)
            return u
        I = inf[k]
        pl = player[k]
        ch = children[k]
        sigma = current_strategy(I)
        ua = [0.0] * len(ch)
        if pl == 0:
            for a, c in enumerate(ch):
                ua[a] = cfr(c, p0 * sigma[a], p1, pc, t)
            reach_own, reach_opp, sgn = p0, p1, 1.0
        else:
            for a, c in enumerate(ch):
                ua[a] = cfr(c, p0, p1 * sigma[a], pc, t)
            reach_own, reach_opp, sgn = p1, p0, -1.0
        u = 0.0
        for a in range(len(ch)):
            u += sigma[a] * ua[a]
        R = regret[I]
        S = strat_sum[I]
        w_opp = reach_opp * pc
        for a in range(len(ch)):
            # Regret is measured in the acting player's payoff; for player 1 that is -u.
            R[a] = max(0.0, R[a] + sgn * w_opp * (ua[a] - u))
            S[a] += t * reach_own * sigma[a]
        return u

    def avg_strategy(I):
        S = strat_sum[I]
        s = sum(S)
        if s > 0.0:
            return [x / s for x in S]
        m = len(S)
        return [1.0 / m] * m

    def evaluate(sigma):
        """Expected payoff to player 0 when both players follow sigma (one list per infoset)."""
        memo = {}

        def ev(k):
            kd = kind[k]
            if kd == "terminal":
                return pay[k]
            if kd == "chance":
                return sum(pr * ev(c) for c, pr in zip(children[k], probs[k]))
            s = sigma[inf[k]]
            return sum(s[a] * ev(c) for a, c in enumerate(children[k]))

        return ev(root)

    def best_response(br, sigma):
        """Best-response payoff to player `br` against the fixed opponent strategy `sigma`."""
        opp = 1 - br
        sign = 1.0 if br == 0 else -1.0
        # w[k]: probability of reaching k through chance and the opponent only.
        w = [0.0] * N
        w[root] = 1.0
        for k in order:
            kd = kind[k]
            if kd == "chance":
                for c, pr in zip(children[k], probs[k]):
                    w[c] += w[k] * pr
            elif kd == "decision":
                if player[k] == opp:
                    s = sigma[inf[k]]
                    for a, c in enumerate(children[k]):
                        w[c] += w[k] * s[a]
                else:
                    for c in children[k]:
                        w[c] += w[k]
        memo = [None] * N
        choice = [None] * n_inf

        def u(k):
            # Payoff to br from node k onward, conditional on reaching k, with the
            # best-response choices made so far (decided lazily, deepest infosets first).
            if memo[k] is not None:
                return memo[k]
            kd = kind[k]
            if kd == "terminal":
                v = sign * pay[k]
            elif kd == "chance":
                v = 0.0
                for c, pr in zip(children[k], probs[k]):
                    v += pr * u(c)
            elif player[k] == opp:
                s = sigma[inf[k]]
                v = 0.0
                for a, c in enumerate(children[k]):
                    v += s[a] * u(c)
            else:
                I = inf[k]
                if choice[I] is None:
                    na = len(inf_actions[I])
                    S = [0.0] * na
                    for nk in inf_nodes[I]:
                        wk = w[nk]
                        if wk == 0.0:
                            continue
                        for a, c in enumerate(children[nk]):
                            S[a] += wk * u(c)
                    choice[I] = max(range(na), key=lambda a: S[a])
                v = u(children[k][choice[I]])
            memo[k] = v
            return v

        return u(root)

    def exploitability():
        sig = [avg_strategy(I) for I in range(n_inf)]
        br0 = best_response(0, sig)  # player 0's payoff vs player 1's average strategy
        br1 = best_response(1, sig)  # player 1's payoff vs player 0's average strategy
        return (br0 + br1) / 2.0, br0, br1

    check_every = 500
    start = time.time()
    t = 0
    last = None
    for t in range(1, max_iters + 1):
        cfr(root, 1.0, 1.0, 1.0, t)
        if t % check_every == 0 or t == 1:
            expl, br0, br1 = exploitability()
            last = (t, expl)
            print(f"iter {t:6d}  exploitability {expl:.3e}  "
                  f"(br0 {br0:+.6f}, br1 {br1:+.6f})  elapsed {time.time() - start:.1f}s",
                  flush=True)
            if expl < 1e-6:
                break
        if time.time() - start > time_budget:
            print(f"time budget reached at iteration {t}", flush=True)
            break

    sig = [avg_strategy(I) for I in range(n_inf)]
    expl, br0, br1 = exploitability()
    value_p0 = evaluate(sig)
    print(f"final after {t} iterations: exploitability {expl:.6e}, value_p0 {value_p0:.8f}", flush=True)

    out = {
        "format": "bluff-strategy-v1",
        "game": T["name"],
        "producer": f"CFR+ ({t} iterations, linear averaging, full tree), solver/cfr_solver.py",
        "probs": {},
    }
    for I, name in enumerate(T["inf_names"]):
        labels = inf_actions[I]
        out["probs"][name] = {lab: repr(p) for lab, p in zip(labels, sig[I])}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out_path} with {len(out['probs'])} information sets", flush=True)


if __name__ == "__main__":
    main()

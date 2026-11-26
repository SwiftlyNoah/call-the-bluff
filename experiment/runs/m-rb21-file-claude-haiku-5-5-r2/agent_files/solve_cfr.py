#!/usr/bin/env python3
"""CFR+ solver for bluff-game-v1 extensive-form games.

Usage: python3 solve_cfr.py [game.json] [strategy.json] [iterations]

Reads the game tree from game.json (nothing is hard-coded), runs simultaneous
CFR+ (regret matching+ with linear averaging) for both players, measures the
exploitability of the average strategy with exact best responses, and writes
the average strategy in bluff-strategy-v1 format.
"""
import json
import sys
import time
from fractions import Fraction


def num(s):
    return float(Fraction(s))


def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    sys.setrecursionlimit(100000)

    with open(game_path) as f:
        g = json.load(f)
    if g.get("format") != "bluff-game-v1":
        raise SystemExit(f"unexpected game format: {g.get('format')!r}")

    root = g["root"]

    # ---- information sets -------------------------------------------------
    iset_index = {}
    iset_names = []
    iset_labels = []
    iset_player = []
    iset_nodes = []
    for n in g["nodes"]:
        if n["kind"] != "decision":
            continue
        name = n["infoset"]
        labels = [a["label"] for a in n["actions"]]
        if name not in iset_index:
            iset_index[name] = len(iset_names)
            iset_names.append(name)
            iset_labels.append(labels)
            iset_player.append(n["player"])
            iset_nodes.append([])
        else:
            I = iset_index[name]
            if iset_labels[I] != labels or iset_player[I] != n["player"]:
                raise SystemExit(f"inconsistent infoset {name!r}")
        iset_nodes[iset_index[name]].append(n["id"])
    nI = len(iset_names)
    nact = [len(l) for l in iset_labels]

    # ---- compile tree -----------------------------------------------------
    # ('T', payoff_p0) | ('C', probs, children) | ('D', player, iset, children)
    comp = {}
    for n in g["nodes"]:
        k = n["kind"]
        if k == "terminal":
            comp[n["id"]] = ("T", num(n["payoff_p0"]))
        elif k == "chance":
            probs = [num(o["prob"]) for o in n["outcomes"]]
            if abs(sum(probs) - 1.0) > 1e-12:
                raise SystemExit(f"chance node {n['id']} probs do not sum to 1")
            comp[n["id"]] = ("C", probs, [o["child"] for o in n["outcomes"]])
        elif k == "decision":
            comp[n["id"]] = ("D", n["player"], iset_index[n["infoset"]],
                             [a["child"] for a in n["actions"]])
        else:
            raise SystemExit(f"unknown node kind {k!r}")

    # ---- CFR+ -------------------------------------------------------------
    regret = [[0.0] * k for k in nact]
    strat_sum = [[0.0] * k for k in nact]
    delta = [[0.0] * k for k in nact]
    cur = [None] * nI

    def regret_matching_plus(I):
        r = regret[I]
        pos = [x if x > 0.0 else 0.0 for x in r]
        s = sum(pos)
        if s > 0.0:
            return [x / s for x in pos]
        return [1.0 / len(r)] * len(r)

    def cfr(h, p0, p1, pc, t):
        """Return player 0's expected payoff from node h under the current
        strategies. Accumulates regret deltas and average-strategy sums."""
        c = comp[h]
        kind = c[0]
        if kind == "T":
            return c[1]
        if kind == "C":
            total = 0.0
            for pr, ch in zip(c[1], c[2]):
                total += pr * cfr(ch, p0, p1, pc * pr, t)
            return total
        _, pl, I, children = c
        sig = cur[I]
        if pl == 0:
            vals = [cfr(ch, p0 * sig[a], p1, pc, t) for a, ch in enumerate(children)]
            q = vals                      # player 0's payoff per action
            own, opp = p0, p1
        else:
            vals = [cfr(ch, p0, p1 * sig[a], pc, t) for a, ch in enumerate(children)]
            q = [-v for v in vals]        # player 1's payoff per action
            own, opp = p1, p0
        u = 0.0
        for a in range(len(children)):
            u += sig[a] * q[a]
        w = pc * opp                      # opponent + chance reach
        d = delta[I]
        for a in range(len(children)):
            d[a] += w * (q[a] - u)
        s = strat_sum[I]
        f = t * own                       # linear averaging weight
        for a in range(len(children)):
            s[a] += f * sig[a]
        return sum(sig[a] * vals[a] for a in range(len(children)))

    def average_profile():
        prof = []
        for I in range(nI):
            s = strat_sum[I]
            tot = sum(s)
            if tot > 0.0:
                prof.append([x / tot for x in s])
            else:
                prof.append([1.0 / nact[I]] * nact[I])
        return prof

    # ---- exploitability with exact best responses -------------------------
    def expected_value(prof):
        memo = {}

        def ev(h):
            if h in memo:
                return memo[h]
            c = comp[h]
            if c[0] == "T":
                v = c[1]
            elif c[0] == "C":
                v = sum(pr * ev(ch) for pr, ch in zip(c[1], c[2]))
            else:
                _, pl, I, children = c
                v = sum(prof[I][a] * ev(ch) for a, ch in enumerate(children))
            memo[h] = v
            return v

        return ev(root)

    def best_response_value(br, prof):
        """Best-response value for player br against the opponent's strategy
        in prof (the br player's own entries of prof are ignored)."""
        # Unconditional reach of every node by chance and the opponent.
        reach = {}
        stack = [(root, 1.0)]
        while stack:
            h, wt = stack.pop()
            reach[h] = wt
            c = comp[h]
            if c[0] == "T":
                continue
            if c[0] == "C":
                for pr, ch in zip(c[1], c[2]):
                    stack.append((ch, wt * pr))
            else:
                _, pl, I, children = c
                if pl == br:
                    for ch in children:
                        stack.append((ch, wt))
                else:
                    for a, ch in enumerate(children):
                        stack.append((ch, wt * prof[I][a]))

        memo = {}
        chosen = {}

        def choose(I):
            best_a, best_v = 0, None
            for a in range(nact[I]):
                tot = 0.0
                for h in iset_nodes[I]:
                    tot += reach[h] * val(comp[h][3][a])
                if best_v is None or tot > best_v:
                    best_a, best_v = a, tot
            return best_a

        def val(h):
            if h in memo:
                return memo[h]
            c = comp[h]
            if c[0] == "T":
                v = c[1] if br == 0 else -c[1]
            elif c[0] == "C":
                v = sum(pr * val(ch) for pr, ch in zip(c[1], c[2]))
            else:
                _, pl, I, children = c
                if pl == br:
                    if I not in chosen:
                        chosen[I] = choose(I)
                    v = val(children[chosen[I]])
                else:
                    v = sum(prof[I][a] * val(ch) for a, ch in enumerate(children))
            memo[h] = v
            return v

        return val(root)

    def exploitability(prof):
        a = best_response_value(0, prof)   # player 0's best payoff vs sigma_1
        b = best_response_value(1, prof)   # player 1's best payoff vs sigma_0
        return (a + b) / 2.0, a, b

    # ---- main loop --------------------------------------------------------
    start = time.time()
    report_every = max(1, iters // 10)
    for t in range(1, iters + 1):
        for I in range(nI):
            cur[I] = regret_matching_plus(I)
            d = delta[I]
            for a in range(nact[I]):
                d[a] = 0.0
        cfr(root, 1.0, 1.0, 1.0, t)
        for I in range(nI):
            r = regret[I]
            d = delta[I]
            for a in range(nact[I]):
                x = r[a] + d[a]
                r[a] = x if x > 0.0 else 0.0   # CFR+ regret floor
        if t % report_every == 0 or t == iters:
            prof = average_profile()
            eps, a_val, b_val = exploitability(prof)
            print(f"iter {t:>7}  exploitability {eps:.8f}  "
                  f"value_p0 {expected_value(prof):+.8f}  "
                  f"({time.time() - start:.1f}s)", flush=True)

    prof = average_profile()
    eps, a_val, b_val = exploitability(prof)
    value = expected_value(prof)
    print(f"final: exploitability {eps:.10f} (BR0 {a_val:.10f}, BR1 {b_val:.10f}), "
          f"value_p0 {value:.10f}")

    # ---- write strategy ---------------------------------------------------
    probs = {}
    for I, name in enumerate(iset_names):
        probs[name] = {lab: repr(float(p)) for lab, p in zip(iset_labels[I], prof[I])}
    out = {
        "format": "bluff-strategy-v1",
        "game": g.get("name", "game"),
        "producer": (f"CFR+ ({iters} iterations, linear averaging, simultaneous updates) "
                     f"by solve_cfr.py; exploitability {eps:.3e}"),
        "probs": probs,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path} with {len(probs)} information sets")


if __name__ == "__main__":
    main()

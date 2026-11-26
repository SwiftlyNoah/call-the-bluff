"""Kuhn poker solver: CFR+ with alternating updates and linear averaging."""
import itertools, json

CARDS = "JQK"
RANK = {c: i for i, c in enumerate(CARDS)}
DEALS = list(itertools.permutations(CARDS, 2))  # 6 equally likely deals
LABEL = {"c": "check", "b": "bet", "f": "fold", "k": "call"}  # k = call

def actions(h):
    # Returns action letters available after history h (letters: c,b,f,k).
    if h in ("", "c"):
        return ["c", "b"]
    if h in ("b", "cb"):
        return ["f", "k"]
    return None

def terminal_payoff(h, cards):
    """Net payoff to player 0, or None if not terminal."""
    hi = 1 if RANK[cards[0]] > RANK[cards[1]] else -1
    if h == "cc": return hi
    if h in ("bk", "cbk"): return 2 * hi
    if h == "bf": return 1    # p1 folds
    if h == "cbf": return -1  # p0 folds
    return None

def player(h): return len(h) % 2

def infoset(h, cards):
    # Public history uses c/b letters only; calls/folds never appear before a decision.
    return f"{cards[player(h)]}:{h}"

def all_infosets():
    out = {}
    for c in CARDS:
        for h in ("", "c", "b", "cb"):
            out[f"{c}:{h}"] = actions(h)
    return out

INFOSETS = all_infosets()

def cfr(h, cards, traverser, reach_opp, regrets, strat_sum, iter_w, sigma):
    u = terminal_payoff(h, cards)
    if u is not None:
        return u if traverser == 0 else -u
    p = player(h)
    I = infoset(h, cards)
    acts = actions(h)
    s = sigma[I]
    if p == traverser:
        vals = [cfr(h + a, cards, traverser, reach_opp, regrets, strat_sum, iter_w, sigma) for a in acts]
        v = sum(pi * va for pi, va in zip(s, vals))
        for i in range(len(acts)):
            regrets[I][i] = max(0.0, regrets[I][i] + reach_opp * (vals[i] - v))  # CFR+
        return v
    else:
        for i in range(len(acts)):
            strat_sum[I][i] += iter_w * reach_opp * s[i]  # opponent's own reach (chance const)
        return sum(s[i] * cfr(h + acts[i], cards, traverser, reach_opp * s[i], regrets, strat_sum, iter_w, sigma)
                   for i in range(len(acts)))

def current(regrets):
    sigma = {}
    for I, r in regrets.items():
        tot = sum(r)
        sigma[I] = [x / tot for x in r] if tot > 0 else [1 / len(r)] * len(r)
    return sigma

def normalize(ss):
    out = {}
    for I, r in ss.items():
        t = sum(r)
        out[I] = [x / t for x in r] if t > 0 else [1 / len(r)] * len(r)
    return out

def expected_value(strat):
    tot = 0.0
    def rec(h, cards, p):
        u = terminal_payoff(h, cards)
        if u is not None: return p * u
        I = infoset(h, cards)
        return sum(rec(h + a, cards, p * strat[I][i]) for i, a in enumerate(actions(h)))
    for d in DEALS:
        tot += rec("", d, 1 / 6)
    return tot

def best_response_value(strat, br):
    """Value to player br when best-responding to strat of the other player (exact)."""
    # Tiny game: enumerate all pure strategies of br for each card independently.
    total = 0.0
    my_hs = ["", "cb"] if br == 0 else ["c", "b"]
    for mycard in CARDS:
        best = -1e9
        for choice in itertools.product(*[actions(h) for h in my_hs]):
            pure = dict(zip(my_hs, choice))
            v = 0.0
            for oc in CARDS:
                if oc == mycard: continue
                cards = (mycard, oc) if br == 0 else (oc, mycard)
                def rec(h, w):
                    u = terminal_payoff(h, cards)
                    if u is not None: return w * (u if br == 0 else -u)
                    if player(h) == br:
                        return rec(h + pure[h], w)
                    I = infoset(h, cards)
                    return sum(rec(h + a, w * strat[I][i]) for i, a in enumerate(actions(h)))
                v += rec("", 1 / 6)
            best = max(best, v)
        total += best
    return total

def main(iters=100000):
    regrets = {I: [0.0] * len(a) for I, a in INFOSETS.items()}
    strat_sum = {I: [0.0] * len(a) for I, a in INFOSETS.items()}
    for t in range(1, iters + 1):
        for trav in (0, 1):
            sigma = current(regrets)
            for d in DEALS:
                cfr("", d, trav, 1 / 6, regrets, strat_sum, t, sigma)
    avg = normalize(strat_sum)
    v0 = expected_value(avg)
    br0 = best_response_value(avg, 0)
    br1 = best_response_value(avg, 1)
    expl = (br0 + br1) / 2
    probs = {I: {LABEL[a]: repr(avg[I][i]) for i, a in enumerate(INFOSETS[I])} for I in sorted(INFOSETS)}
    with open("strategy.json", "w") as f:
        json.dump({"format": "bluff-strategy-v1", "game": "kuhn",
                   "producer": f"CFR+ (alternating, linear averaging), {iters} iterations, Python",
                   "probs": probs}, f, indent=2)
    for I in sorted(INFOSETS): print(I, probs[I])
    print(f"value_p0={v0:.8f} (theory -1/18={-1/18:.8f})  br0={br0:.8f} br1={br1:.8f} exploitability={expl:.3e}")

if __name__ == "__main__":
    main(200000)

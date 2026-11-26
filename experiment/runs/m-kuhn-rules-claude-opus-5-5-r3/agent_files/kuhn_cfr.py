"""CFR+ solver for Kuhn poker, with exact best-response exploitability check."""
import itertools, json

CARDS = "JQK"
RANK = {c: i for i, c in enumerate(CARDS)}
DEALS = list(itertools.permutations(CARDS, 2))  # 6 equally likely deals

def actions(h):
    # h: history string of 'c'/'b'
    if h in ("", "c"):
        return [("check", "c"), ("bet", "b")]
    if h in ("b", "cb"):
        return [("fold", "c"), ("call", "b")]  # encode fold as 'c', call as 'b'
    return None

def terminal_payoff(h, cards):
    """Payoff to player 0 at terminal history h, else None."""
    show = 1 if RANK[cards[0]] > RANK[cards[1]] else -1
    if h == "cc": return show * 1
    if h == "bb" or h == "cbb": return show * 2
    if h == "bc": return 1    # p1 folds
    if h == "cbc": return -1  # p0 folds
    return None

def player(h): return len(h) % 2

INFOSETS = [c + ":" + h for c in CARDS for h in ("", "c", "b", "cb")]
regret = {I: [0.0, 0.0] for I in INFOSETS}
stratsum = {I: [0.0, 0.0] for I in INFOSETS}
SNAP = {}

def current(I):
    r = regret[I]; s = max(r[0], 0) + max(r[1], 0)
    return [max(x, 0) / s for x in r] if s > 0 else [0.5, 0.5]

def cfr(cards, h, reach, upd, w):
    """Returns value to player 0. Updates infosets of player `upd` only (alternating)."""
    u = terminal_payoff(h, cards)
    if u is not None: return u
    p = player(h); I = cards[p] + ":" + h
    sigma = SNAP[I]
    vals = []
    for a, (_, l) in enumerate(actions(h)):
        nr = list(reach); nr[p] *= sigma[a]
        vals.append(cfr(cards, h + l, nr, upd, w))
    v = sum(s * x for s, x in zip(sigma, vals))
    if p == upd:
        sign = 1 if p == 0 else -1
        opp = reach[1 - p]
        for a in range(2):
            regret[I][a] += opp * sign * (vals[a] - v)
        for a in range(2):
            stratsum[I][a] += w * reach[p] * sigma[a]
    return v

def finish_regrets_plus():
    for I in INFOSETS:
        regret[I] = [max(x, 0.0) for x in regret[I]]

def average():
    out = {}
    for I in INFOSETS:
        s = sum(stratsum[I])
        out[I] = [x / s for x in stratsum[I]] if s > 0 else [0.5, 0.5]
    return out

def expected_value(strat):
    def ev(cards, h):
        u = terminal_payoff(h, cards)
        if u is not None: return u
        sg = strat[cards[player(h)] + ":" + h]
        return sum(sg[a] * ev(cards, h + l) for a, (_, l) in enumerate(actions(h)))
    return sum(ev(d, "") for d in DEALS) / len(DEALS)

def best_response_value(strat, br):
    """Value (to player br, in br's own payoff) of a best response against strat."""
    # Infoset-level BR: compute, for each br infoset, action values summed over consistent deals
    # weighted by opponent reach and chance. Process deeper infosets first.
    sign = 1 if br == 0 else -1
    brs = {}
    def val(cards, h):
        u = terminal_payoff(h, cards)
        if u is not None: return sign * u
        p = player(h); I = cards[p] + ":" + h
        acts = actions(h)
        if p == br:
            a = brs[I] if I in brs else choose(I)
            return val(cards, h + acts[a][1])
        sg = strat[I]
        return sum(sg[a] * val(cards, h + l) for a, (_, l) in enumerate(acts))
    def opp_reach(cards, h):
        r = 1.0
        for i in range(len(h)):
            p = player(h[:i])
            if p != br:
                acts = actions(h[:i]); a = [l for _, l in acts].index(h[i])
                r *= strat[cards[p] + ":" + h[:i]][a]
        return r
    def choose(I):
        card, h = I.split(":")
        tot = [0.0, 0.0]
        for d in DEALS:
            if d[br] != card: continue
            w = opp_reach(d, h)
            for a, (_, l) in enumerate(actions(h)):
                tot[a] += w * val(d, h + l)
        brs[I] = 0 if tot[0] >= tot[1] else 1
        return brs[I]
    return sum(val(d, "") for d in DEALS) / len(DEALS)

def main(iters=100000):
    for t in range(1, iters + 1):
        w = max(t - 0, 0)  # linear averaging (CFR+ style)
        for upd in (0, 1):
            SNAP.clear(); SNAP.update({I: current(I) for I in INFOSETS})
            for d in DEALS:
                cfr(d, "", [1.0, 1.0], upd, w)
            finish_regrets_plus()
    avg = average()
    v0 = expected_value(avg)
    br0 = best_response_value(avg, 0)
    br1 = best_response_value(avg, 1)
    expl = (br0 + br1) / 2
    print(f"iters={iters} value_p0={v0:.8f} (theory -1/18={-1/18:.8f}) br0={br0:.8f} br1={br1:.8f} exploitability={expl:.3e}")
    probs = {}
    for I in INFOSETS:
        card, h = I.split(":")
        labels = [n for n, _ in actions(h)]
        p = avg[I]
        p1 = 1.0 - p[0]
        probs[I] = {labels[0]: repr(p[0]), labels[1]: repr(p1)}
    with open("strategy.json", "w") as f:
        json.dump({"format": "bluff-strategy-v1", "game": "kuhn",
                   "producer": f"CFR+ (alternating updates, linear averaging), {iters} iterations, Python",
                   "probs": probs}, f, indent=2)
    for I in INFOSETS: print(I, probs[I])

if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)

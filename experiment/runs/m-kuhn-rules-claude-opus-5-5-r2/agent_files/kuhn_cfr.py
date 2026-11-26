"""CFR+ solver for Kuhn poker; writes strategy.json and reports exploitability."""
import itertools, json

CARDS = ["J", "Q", "K"]
RANK = {"J": 0, "Q": 1, "K": 2}
LABEL = {"c": "check", "b": "bet"}

def actions(h):
    # facing a bet -> fold/call, else check/bet
    if h in ("b", "cb"):
        return ["f", "k"]  # fold, call
    return ["c", "b"]
NAMES = {"c": "check", "b": "bet", "f": "fold", "k": "call"}

def terminal(h):
    return h in ("cc", "bf", "bk", "cbf", "cbk")

def payoff0(h, cards):
    """Net payoff to player 0."""
    win = 1 if RANK[cards[0]] > RANK[cards[1]] else -1
    if h == "cc": return win
    if h == "bk" or h == "cbk": return 2 * win
    if h == "bf": return 1   # p1 folds
    if h == "cbf": return -1  # p0 folds

def player(h): return len(h) % 2

def key(h, cards):
    # history letters: c for check, b for bet (call/fold never precede a decision)
    return cards[player(h)] + ":" + h

DEALS = list(itertools.permutations(CARDS, 2))
regret, ssum = {}, {}

def strat(I, n):
    r = regret.setdefault(I, [0.0] * n)
    pos = [max(x, 0) for x in r]; s = sum(pos)
    return [x / s for x in pos] if s > 0 else [1.0 / n] * n

def cfr(h, cards, p0, p1, it, upd):
    if terminal(h): return payoff0(h, cards)
    pl = player(h); A = actions(h); I = key(h, cards)
    s = strat(I, len(A))
    vals = [cfr(h + a, cards, p0 * (s[i] if pl == 0 else 1), p1 * (s[i] if pl == 1 else 1), it, upd)
            for i, a in enumerate(A)]
    v = sum(si * vi for si, vi in zip(s, vals))
    if pl == upd:
        sign = 1 if pl == 0 else -1
        opp = p1 if pl == 0 else p0
        me = p0 if pl == 0 else p1
        r = regret[I]
        for i in range(len(A)):
            r[i] = max(r[i] + sign * opp * (vals[i] - v), 0.0)  # CFR+
        ss = ssum.setdefault(I, [0.0] * len(A))
        for i in range(len(A)):
            ss[i] += it * me * s[i]  # linear averaging
    return v

def avg(I, n):
    ss = ssum.get(I); t = sum(ss) if ss else 0
    return [x / t for x in ss] if t > 0 else [1.0 / n] * n

def infosets():
    out = {}
    for c in CARDS:
        for h in ("", "c", "b", "cb"):
            out[c + ":" + h] = actions(h)
    return out

def ev(policy, h, cards):
    if terminal(h): return payoff0(h, cards)
    A = actions(h); p = policy[key(h, cards)]
    return sum(p[i] * ev(policy, h + a, cards) for i, a in enumerate(A))

def best_response(policy, br):
    """Value (for br player) of best response vs policy, exact via infoset enumeration."""
    # Enumerate br's pure strategies over its 6 infosets (2^6=64) -- tiny game.
    mine = [I for I in infosets() if player(I.split(":")[1]) == br]
    best = -1e9
    for choice in itertools.product(range(2), repeat=len(mine)):
        pol = dict(policy)
        for I, c in zip(mine, choice):
            pol[I] = [1.0 if j == c else 0.0 for j in range(2)]
        v = sum(ev(pol, "", d) for d in DEALS) / len(DEALS)
        best = max(best, v if br == 0 else -v)
    return best

def main(T=200000):
    for t in range(1, T + 1):
        for upd in (0, 1):
            for d in DEALS:
                cfr("", d, 1.0, 1.0, t, upd)
    pol = {I: avg(I, len(A)) for I, A in infosets().items()}
    v0 = sum(ev(pol, "", d) for d in DEALS) / len(DEALS)
    br0, br1 = best_response(pol, 0), best_response(pol, 1)
    expl = (br0 + br1) / 2
    out = {"format": "bluff-strategy-v1", "game": "kuhn",
           "producer": f"CFR+ (alternating updates, linear averaging), {T} iterations, kuhn_cfr.py",
           "probs": {I: {NAMES[a]: repr(p) for a, p in zip(A, pol[I])} for I, A in infosets().items()}}
    with open("strategy.json", "w") as f: json.dump(out, f, indent=2)
    for I in sorted(pol): print(I, [round(x, 4) for x in pol[I]])
    print("value_p0", v0, "br0", br0, "br1", br1, "exploitability", expl)

if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)

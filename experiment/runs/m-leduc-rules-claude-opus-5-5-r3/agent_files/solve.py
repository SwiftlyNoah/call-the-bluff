import json, sys
from fractions import Fraction
CARDS = ['Js','Jh','Qs','Qh','Ks','Kh']
RANK = {'J':0,'Q':1,'K':2}
RAISE = [2,4]

def legal(rh):
    # rh: actions in current round
    nr = rh.count('r')
    facing = len(rh) > 0 and rh[-1] == 'r'
    acts = []
    if facing: acts.append('f')
    acts.append('c')
    if nr < 2: acts.append('r')
    return acts

LABEL = {'f':'fold','c':'call','r':'raise'}

def round_over(rh):
    return len(rh) >= 2 and rh[-1] in 'c' or (len(rh)>0 and rh[-1]=='f')

def contribs(h1, h2):
    # returns contributions [p0,p1]
    c = [1,1]
    for rnd, rh in enumerate([h1, h2]):
        owe = 0; p = 0
        for a in rh:
            o = 1-p
            if a == 'c':
                c[p] = c[o]
            elif a == 'r':
                c[p] = c[o] + RAISE[rnd]
            p = 1-p
    return c

def showdown(c0, c1, pub):
    r0, r1, rp = RANK[c0[0]], RANK[c1[0]], RANK[pub[0]]
    if r0 == rp: return 1
    if r1 == rp: return -1
    return (r0 > r1) - (r0 < r1)

def key(card, pub, h1, h2):
    if pub is None: return f"{card}:{h1}"
    return f"{card}|{pub}:{h1}/{h2}"

regret = {}; avg = {}; delta = {}
def strat(k, acts):
    r = regret.get(k)
    if r is None:
        return [1.0/len(acts)]*len(acts)
    s = sum(max(x,0) for x in r)
    if s <= 0: return [1.0/len(acts)]*len(acts)
    return [max(x,0)/s for x in r]

def cfr(cards, pub, h1, h2, trav, pi_opp, w):
    """returns value for trav. cards=(c0,c1). pub None in round 1."""
    rnd = 0 if pub is None else 1
    rh = h1 if rnd == 0 else h2
    if rh and rh[-1] == 'f':
        c = contribs(h1, h2)
        loser = (len(rh)-1) % 2
        return c[1-trav] if loser != trav else -c[trav]
    if len(rh) >= 2 and rh[-1] == 'c':
        if rnd == 0:
            rest = [x for x in CARDS if x not in cards]
            return sum(cfr(cards, p, h1, '', trav, pi_opp, w) for p in rest) / len(rest)
        c = contribs(h1, h2)
        res = showdown(cards[0], cards[1], pub)
        if trav == 1: res = -res
        return res * c[1-trav] if res > 0 else (res * c[trav] if res < 0 else 0)
    p = len(rh) % 2
    acts = legal(rh)
    k = key(cards[p], pub, h1, h2)
    s = strat(k, acts)
    if p == trav:
        vals = []
        for a in acts:
            if rnd == 0: vals.append(cfr(cards, pub, h1+a, h2, trav, pi_opp, w))
            else: vals.append(cfr(cards, pub, h1, h2+a, trav, pi_opp, w))
        v = sum(x*y for x, y in zip(s, vals))
        r = delta.setdefault(k, [0.0]*len(acts))
        for i in range(len(acts)):
            r[i] += pi_opp*(vals[i]-v)
        return v
    else:
        # average strategy update for opponent (weighted by own reach... use pi_opp as opp reach incl chance)
        av = avg.setdefault(k, [0.0]*len(acts))
        for i in range(len(acts)): av[i] += w*pi_opp*s[i]
        v = 0.0
        for i, a in enumerate(acts):
            if s[i] == 0: continue
            if rnd == 0: v += s[i]*cfr(cards, pub, h1+a, h2, trav, pi_opp*s[i], w)
            else: v += s[i]*cfr(cards, pub, h1, h2+a, trav, pi_opp*s[i], w)
        return v

DEALS = [(a,b) for a in CARDS for b in CARDS if a != b]

def avg_strategy():
    out = {}
    for k, av in avg.items():
        s = sum(av)
        out[k] = [x/s for x in av] if s > 0 else [1/len(av)]*len(av)
    return out

# ---- evaluation ----
def acts_of(k):
    rh = k.split(':')[1].split('/')[-1]
    return legal(rh)

def ev(S, cards, pub, h1, h2):
    """expected value for p0 under strategy S"""
    rnd = 0 if pub is None else 1
    rh = h1 if rnd == 0 else h2
    if rh and rh[-1] == 'f':
        c = contribs(h1, h2); loser = (len(rh)-1) % 2
        return c[1] if loser == 1 else -c[0]
    if len(rh) >= 2 and rh[-1] == 'c':
        if rnd == 0:
            rest = [x for x in CARDS if x not in cards]
            return sum(ev(S, cards, p, h1, '') for p in rest)/len(rest)
        c = contribs(h1, h2); res = showdown(cards[0], cards[1], pub)
        return res * c[1] if res > 0 else (res*c[0] if res < 0 else 0)
    p = len(rh) % 2; acts = legal(rh); s = S[key(cards[p], pub, h1, h2)]
    tot = 0.0
    for i, a in enumerate(acts):
        if s[i] == 0: continue
        tot += s[i]*(ev(S, cards, pub, h1+a, h2) if rnd == 0 else ev(S, cards, pub, h1, h2+a))
    return tot

def br(S, i, mycard, pub, h1, h2, opp):
    """opp: dict opp_card -> reach weight (incl chance). returns BR value sum for player i."""
    rnd = 0 if pub is None else 1
    rh = h1 if rnd == 0 else h2
    if rh and rh[-1] == 'f':
        c = contribs(h1, h2); loser = (len(rh)-1) % 2
        val = c[1-i] if loser != i else -c[i]
        return val*sum(opp.values())
    if len(rh) >= 2 and rh[-1] == 'c':
        if rnd == 0:
            tot = 0.0
            for pc in CARDS:
                if pc == mycard: continue
                no = {oc: w/4 for oc, w in opp.items() if oc != pc}
                tot += br(S, i, mycard, pc, h1, '', no)
            return tot
        c = contribs(h1, h2); tot = 0.0
        for oc, w in opp.items():
            cards = (mycard, oc) if i == 0 else (oc, mycard)
            res = showdown(cards[0], cards[1], pub)
            if i == 1: res = -res
            tot += w*(res*c[1-i] if res > 0 else (res*c[i] if res < 0 else 0))
        return tot
    p = len(rh) % 2; acts = legal(rh)
    def nxt(a, o):
        return br(S, i, mycard, pub, h1+a, h2, o) if rnd == 0 else br(S, i, mycard, pub, h1, h2+a, o)
    if p == i:
        return max(nxt(a, opp) for a in acts)
    tot = 0.0
    for j, a in enumerate(acts):
        no = {oc: w*S[key(oc, pub, h1, h2)][j] for oc, w in opp.items()}
        if sum(no.values()) == 0: continue
        tot += nxt(a, no)
    return tot

def br_value(S, i):
    tot = 0.0
    for mc in CARDS:
        opp = {oc: 1/30 for oc in CARDS if oc != mc}
        tot += br(S, i, mc, None, '', '', opp)
    return tot

if __name__ == '__main__':
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    for t in range(1, T+1):
        for trav in (0, 1):
            for d in DEALS:
                cfr(d, None, '', '', trav, 1.0/30, t)
            for k, dl in delta.items():
                r = regret.setdefault(k, [0.0]*len(dl))
                for i in range(len(dl)): r[i] = max(r[i] + dl[i], 0.0)  # CFR+
            delta.clear()
        if t % 250 == 0 or t == T:
            S = avg_strategy()
            b0, b1 = br_value(S, 0), br_value(S, 1)
            v = sum(ev(S, d, None, '', '') for d in DEALS)/30
            print(t, 'v0=%.6f br0=%.6f br1=%.6f expl=%.6f' % (v, b0, b1, (b0+b1)/2), flush=True)
    S = avg_strategy()
    probs = {}
    for k in sorted(S):
        acts = acts_of(k); s = S[k]
        # renormalize exactly via strings
        tot = sum(s); s = [x/tot for x in s]
        strs = [repr(x) for x in s[:-1]]
        last = 1 - sum(Fraction(x) for x in strs)
        strs.append(str(last) if last.denominator != 1 else str(last.numerator))
        probs[k] = {LABEL[a]: st for a, st in zip(acts, strs)}
    json.dump({"format": "bluff-strategy-v1", "game": "leduc",
               "producer": f"CFR+ (alternating, linear averaging), {T} iterations, solve.py",
               "probs": probs}, open('strategy.json', 'w'), indent=1)
    print('infosets', len(probs))

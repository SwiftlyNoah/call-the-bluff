import json, sys, itertools
CARDS = ['Js','Jh','Qs','Qh','Ks','Kh']
RANK = {'J':0,'Q':1,'K':2}
RAISE = [2,4]

def legal(h):
    # h: actions in current round
    nr = h.count('r')
    if h == '' or h == 'c': acts = ['call','raise']
    elif nr < 2: acts = ['fold','call','raise']
    else: acts = ['fold','call']
    return acts
L = {'fold':'f','call':'c','raise':'r'}

def round_over(h):
    return len(h) >= 2 and h[-1] == 'c' or (h and h[-1]=='f')

def step(h, a, contrib, rnd, player):
    c = list(contrib)
    opp = 1 - player
    if a == 'call': c[player] = c[opp]
    elif a == 'raise': c[player] = c[opp] + RAISE[rnd]
    return h + L[a], c

def showdown(c0, c1, pub, contrib):
    r0, r1, rp = RANK[c0[0]], RANK[c1[0]], RANK[pub[0]]
    if r0 == rp: w = 0
    elif r1 == rp: w = 1
    elif r0 > r1: w = 0
    elif r1 > r0: w = 1
    else: return 0.0
    return contrib[1] if w == 0 else -contrib[0]

def key(card, pub, h1, h2, rnd):
    return f"{card}:{h1}" if rnd == 0 else f"{card}|{pub}:{h1}/{h2}"

regret, ssum, delta = {}, {}, {}
def strat(k, acts):
    r = regret.get(k)
    if r is None:
        return [1/len(acts)]*len(acts)
    pos = [max(x,0) for x in r]; s = sum(pos)
    return [x/s for x in pos] if s > 0 else [1/len(acts)]*len(acts)

def cfr(cards, pub_choices, h1, h2, rnd, contrib, pi, upd, it):
    # returns value for player 0
    h = h1 if rnd == 0 else h2
    if h and h[-1] == 'f':
        folder = (len(h)-1) % 2
        return -contrib[0] if folder == 0 else contrib[1]
    if round_over(h):
        if rnd == 1:
            return showdown(cards[0], cards[1], cards[2], contrib)
        v = 0.0
        for p in pub_choices:
            v += cfr((cards[0], cards[1], p), None, h1, '', 1, contrib, pi, upd, it)
        return v / len(pub_choices)
    player = len(h) % 2
    acts = legal(h)
    k = key(cards[player], cards[2] if rnd else None, h1, h2, rnd)
    s = strat(k, acts)
    vals = []
    for a, p in zip(acts, s):
        nh, nc = step(h, a, contrib, rnd, player)
        npi = list(pi); npi[player] *= p
        if rnd == 0: vals.append(cfr(cards, pub_choices, nh, h2, 0, nc, npi, upd, it))
        else: vals.append(cfr(cards, pub_choices, h1, nh, 1, nc, npi, upd, it))
    node = sum(p*v for p, v in zip(s, vals))
    if player == upd:
        sign = 1 if player == 0 else -1
        r = delta.setdefault(k, [0.0]*len(acts))
        for i in range(len(acts)):
            r[i] += sign*pi[1-player]*(vals[i]-node)
        ss = ssum.setdefault(k, [0.0]*len(acts))
        for i in range(len(acts)):
            ss[i] += it * pi[player] * s[i]  # linear averaging
    return node

def avg(k, acts):
    ss = ssum.get(k)
    if not ss or sum(ss) == 0: return [1/len(acts)]*len(acts)
    t = sum(ss); return [x/t for x in ss]

# ---- evaluation with average strategy ----
def ev(cards, rest, h1, h2, rnd, contrib, S):
    h = h1 if rnd == 0 else h2
    if h and h[-1] == 'f':
        folder = (len(h)-1) % 2
        return -contrib[0] if folder == 0 else contrib[1]
    if round_over(h):
        if rnd == 1: return showdown(cards[0], cards[1], cards[2], contrib)
        return sum(ev((cards[0],cards[1],p), None, h1, '', 1, contrib, S) for p in rest)/len(rest)
    player = len(h) % 2; acts = legal(h)
    k = key(cards[player], cards[2] if rnd else None, h1, h2, rnd)
    v = 0.0
    for a, p in zip(acts, S[k]):
        if p == 0: continue
        nh, nc = step(h, a, contrib, rnd, player)
        v += p * (ev(cards, rest, nh, h2, 0, nc, S) if rnd == 0 else ev(cards, rest, h1, nh, 1, nc, S))
    return v

def br(i, ci, states, h1, h2, rnd, contrib, S):
    # states: list of (c_opp, pub, weight); returns weighted value for player i
    h = h1 if rnd == 0 else h2
    W = sum(w for _,_,w in states)
    if W == 0: return 0.0
    sgn = 1 if i == 0 else -1
    if h and h[-1] == 'f':
        folder = (len(h)-1) % 2
        return W * sgn * (-contrib[0] if folder == 0 else contrib[1])
    if round_over(h):
        if rnd == 1:
            tot = 0.0
            for co, pub, w in states:
                c0, c1 = (ci, co) if i == 0 else (co, ci)
                tot += w * sgn * showdown(c0, c1, pub, contrib)
            return tot
        tot = 0.0
        for pub in CARDS:
            if pub == ci: continue
            sub = [(co, pub, w/4) for co,_,w in states if co != pub]
            tot += br(i, ci, sub, h1, '', 1, contrib, S)
        return tot
    player = len(h) % 2; acts = legal(h)
    def nxt(a, st):
        nh, nc = step(h, a, contrib, rnd, player)
        return br(i, ci, st, nh, h2, 0, nc, S) if rnd == 0 else br(i, ci, st, h1, nh, 1, nc, S)
    if player == i:
        # all states share the same infoset (same pub in rnd 1 by construction)
        return max(nxt(a, states) for a in acts)
    tot = 0.0
    for j, a in enumerate(acts):
        st = [(co, pub, w * S[key(co, pub, h1, h2, rnd)][j]) for co, pub, w in states]
        tot += nxt(a, st)
    return tot

def all_keys():
    out = {}
    def walk(h1, h2, rnd, cards_known):
        h = h1 if rnd == 0 else h2
        if h and h[-1] == 'f': return
        if round_over(h):
            if rnd == 0: walk(h1, '', 1, None)
            return
        acts = legal(h)
        for c in CARDS:
            if rnd == 0: out[key(c, None, h1, h2, 0)] = acts
            else:
                for p in CARDS:
                    if p != c: out[key(c, p, h1, h2, 1)] = acts
        for a in acts:
            if rnd == 0: walk(h1+L[a], h2, 0, None)
            else: walk(h1, h2+L[a], 1, None)
    walk('', '', 0, None)
    return out

def evaluate(S):
    v0 = 0.0; brv = [0.0, 0.0]; n = 0
    for c0, c1 in itertools.permutations(CARDS, 2):
        rest = [c for c in CARDS if c not in (c0, c1)]
        v0 += ev((c0, c1, None), rest, '', '', 0, [1,1], S); n += 1
    v0 /= n
    for i in (0, 1):
        for ci in CARDS:
            states = [(co, None, 1/30) for co in CARDS if co != ci]
            brv[i] += br(i, ci, states, '', '', 0, [1,1], S)
    return v0, brv, (brv[0] + brv[1]) / 2

if __name__ == '__main__':
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    keys = all_keys()
    deals = list(itertools.permutations(CARDS, 2))
    for it in range(1, iters+1):
        for upd in (0, 1):
            for c0, c1 in deals:
                rest = [c for c in CARDS if c not in (c0, c1)]
                cfr((c0, c1, None), rest, '', '', 0, [1,1], [1.0,1.0], upd, it)
            for k, d in delta.items():  # CFR+: apply the pass's regrets, floor at 0
                r = regret.setdefault(k, [0.0]*len(d))
                for j in range(len(d)): r[j] = max(r[j] + d[j], 0.0)
            delta.clear()
        if it % 500 == 0 or it == iters:
            S = {k: avg(k, a) for k, a in keys.items()}
            v0, brv, ex = evaluate(S)
            print(it, 'v0', v0, 'br', brv, 'expl', ex, flush=True)
    S = {k: avg(k, a) for k, a in keys.items()}
    v0, brv, ex = evaluate(S)
    probs = {k: {a: repr(p) for a, p in zip(keys[k], S[k])} for k in sorted(keys)}
    for k in probs:  # fix rounding so sums are exactly representable-ish
        pass
    json.dump({"format": "bluff-strategy-v1", "game": "leduc",
               "producer": f"CFR+ (alternating, linear averaging), {iters} iterations, full chance enumeration",
               "probs": probs}, open('strategy.json', 'w'), indent=1)
    print('FINAL v0', v0, 'expl', ex, 'infosets', len(keys))

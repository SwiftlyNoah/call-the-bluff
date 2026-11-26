import json, sys, time
from fractions import Fraction

def num(s): return float(Fraction(s))

g = json.load(open('game.json'))
nodes = {n['id']: n for n in g['nodes']}
root = g['root']

# compact arrays
kind, player, iset, children, probs, payoff = {}, {}, {}, {}, {}, {}
infosets = {}  # name -> (player, labels)
for i, n in nodes.items():
    k = n['kind']; kind[i] = k
    if k == 'chance':
        children[i] = [o['child'] for o in n['outcomes']]
        probs[i] = [num(o['prob']) for o in n['outcomes']]
    elif k == 'decision':
        children[i] = [a['child'] for a in n['actions']]
        labels = [a['label'] for a in n['actions']]
        player[i] = n['player']; iset[i] = n['infoset']
        if n['infoset'] in infosets:
            assert infosets[n['infoset']] == (n['player'], labels)
        else:
            infosets[n['infoset']] = (n['player'], labels)
    else:
        payoff[i] = num(n['payoff_p0'])

names = list(infosets)
idx = {s: j for j, s in enumerate(names)}
nact = [len(infosets[s][1]) for s in names]
regret = [[0.0]*k for k in nact]
cum = [[0.0]*k for k in nact]
node_is = {i: idx[iset[i]] for i in iset}

def current(j):
    r = regret[j]; s = sum(x for x in r if x > 0)
    k = len(r)
    return [x/s if x > 0 else 0.0 for x in r] if s > 0 else [1.0/k]*k

def cfr(h, upd, pi_opp, pi_me, t, strat):
    k = kind[h]
    if k == 'terminal':
        return payoff[h] if upd == 0 else -payoff[h]
    ch = children[h]
    if k == 'chance':
        return sum(p*cfr(c, upd, pi_opp*p, pi_me, t, strat) for p, c in zip(probs[h], ch))
    j = node_is[h]
    sigma = strat[j]
    if sigma is None:
        sigma = strat[j] = current(j)
    if player[h] == upd:
        vals = [cfr(c, upd, pi_opp, pi_me*s, t, strat) for c, s in zip(ch, sigma)]
        v = sum(s*x for s, x in zip(sigma, vals))
        r = regret[j]
        for a in range(len(ch)):
            r[a] += pi_opp*(vals[a]-v)
        cm = cum[j]
        for a in range(len(ch)):
            cm[a] += t*pi_me*sigma[a]   # linear averaging (CFR+)
        return v
    else:
        return sum(s*cfr(c, upd, pi_opp*s, pi_me, t, strat) for c, s in zip(ch, sigma) if s > 0 or True)

def avg(j):
    c = cum[j]; s = sum(c); k = len(c)
    return [x/s for x in c] if s > 0 else [1.0/k]*k

def ev(h, strat):
    k = kind[h]
    if k == 'terminal': return payoff[h]
    if k == 'chance': return sum(p*ev(c, strat) for p, c in zip(probs[h], children[h]))
    return sum(s*ev(c, strat) for s, c in zip(strat[node_is[h]], children[h]) if s > 0)

def best_response(br, strat):
    # nodes per infoset with their opponent+chance reach
    reach = {}
    def collect(h, q):
        k = kind[h]
        if k == 'terminal': return
        if k == 'chance':
            for p, c in zip(probs[h], children[h]): collect(c, q*p)
            return
        if player[h] == br:
            reach.setdefault(node_is[h], []).append((h, q))
            for c in children[h]: collect(c, q)
        else:
            for s, c in zip(strat[node_is[h]], children[h]): collect(c, q*s)
    collect(root, 1.0)
    choice = {}
    memo = {}
    def val(h):  # value for br player
        if h in memo: return memo[h]
        k = kind[h]
        if k == 'terminal': v = payoff[h] if br == 0 else -payoff[h]
        elif k == 'chance': v = sum(p*val(c) for p, c in zip(probs[h], children[h]))
        elif player[h] == br:
            j = node_is[h]
            if j not in choice:
                tot = [0.0]*nact[j]
                for (n, q) in reach[j]:
                    for a, c in enumerate(children[n]):
                        tot[a] += q*val(c)
                choice[j] = max(range(nact[j]), key=lambda a: tot[a])
            v = val(children[h][choice[j]])
        else:
            v = sum(s*val(c) for s, c in zip(strat[node_is[h]], children[h]))
        memo[h] = v; return v
    return val(root)

sys.setrecursionlimit(100000)
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
t0 = time.time()
for t in range(1, T+1):
    for upd in (0, 1):
        strat = [None]*len(names)
        cfr(root, upd, 1.0, 1.0, t, strat)
        for r in regret:   # CFR+ floor
            for a in range(len(r)):
                if r[a] < 0: r[a] = 0.0
    if t % 250 == 0 or t == T:
        A = [avg(j) for j in range(len(names))]
        b0, b1 = best_response(0, A), best_response(1, A)
        print(t, round(time.time()-t0,1), 'v0', ev(root, A), 'expl', (b0+b1)/2, flush=True)

A = [avg(j) for j in range(len(names))]
b0, b1 = best_response(0, A), best_response(1, A)
v0 = ev(root, A)
probs_out = {}
for j, s in enumerate(names):
    probs_out[s] = {lab: repr(p) for lab, p in zip(infosets[s][1], A[j])}
# make sums exact-ish: fix last to 1 - rest is unnecessary for floats near 1
json.dump({"format": "bluff-strategy-v1", "game": g['name'],
           "producer": f"CFR+ (alternating updates, linear averaging), {T} iterations",
           "probs": probs_out}, open('strategy.json', 'w'), indent=1)
print('final v0', v0, 'BR0', b0, 'BR1', b1, 'expl', (b0+b1)/2, 'infosets', len(names))

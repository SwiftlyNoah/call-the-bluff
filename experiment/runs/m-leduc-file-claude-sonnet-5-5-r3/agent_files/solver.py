import json, sys
from fractions import Fraction
g = json.load(open('game.json'))
N = {n['id']: n for n in g['nodes']}
def num(s): return float(Fraction(s))
depth = {g['root']: 0}; order = [g['root']]
for i in order:
    n = N[i]
    ch = [o['child'] for o in n.get('outcomes', [])] + [a['child'] for a in n.get('actions', [])]
    for c in ch: depth[c] = depth[i] + 1; order.append(c)
IS = {}  # infoset -> (player, labels)
for n in N.values():
    if n['kind'] == 'decision':
        IS[n['infoset']] = (n['player'], [a['label'] for a in n['actions']])
reg = {k: [0.0]*len(v[1]) for k, v in IS.items()}
ssum = {k: [0.0]*len(v[1]) for k, v in IS.items()}
def strat(k):
    r = [max(x, 0) for x in reg[k]]; s = sum(r)
    return [x/s for x in r] if s > 0 else [1/len(r)]*len(r)
def avg(k):
    s = sum(ssum[k]); return [x/s for x in ssum[k]] if s > 0 else [1/len(ssum[k])]*len(ssum[k])
def cfr(i, p, r0, r1, t):
    n = N[i]; k = n['kind']
    if k == 'terminal': return num(n['payoff_p0'])
    if k == 'chance':
        return sum(num(o['prob']) * cfr(o['child'], p, r0, r1, t) for o in n['outcomes'])
    pl = n['player']; I = n['infoset']; s = strat(I)
    vals = []; v = 0
    for a, pr in zip(n['actions'], s):
        x = cfr(a['child'], p, r0*pr if pl == 0 else r0, r1*pr if pl == 1 else r1, t)
        vals.append(x); v += pr*x
    my, opp = (r0, r1) if pl == 0 else (r1, r0)
    sg = 1 if pl == 0 else -1
    for j, x in enumerate(vals):
        reg[I][j] = max(0.0, reg[I][j] + opp*sg*(x - v))
        ssum[I][j] += t*my*s[j]   # linear averaging
    return v
def evalv(sig, i=None):
    def f(i):
        n = N[i]; k = n['kind']
        if k == 'terminal': return num(n['payoff_p0'])
        if k == 'chance': return sum(num(o['prob'])*f(o['child']) for o in n['outcomes'])
        return sum(pr*f(a['child']) for a, pr in zip(n['actions'], sig[n['infoset']]))
    return f(g['root'])
def br(sig, player):
    cv = {}; sgn = 1 if player == 0 else -1
    choice = {}
    w = {g['root']: 1.0}
    for i in order:
        n = N[i]
        if n['kind'] == 'chance':
            for o in n['outcomes']: w[o['child']] = w[i]*num(o['prob'])
        elif n['kind'] == 'decision':
            for ai, a in enumerate(n['actions']):
                w[a['child']] = w[i]*(sig[n['infoset']][ai] if n['player'] != player else 1.0)
    mine = {}
    for i in order:
        n = N[i]
        if n['kind'] == 'decision' and n['player'] == player: mine.setdefault(n['infoset'], []).append(i)
    cv = {}
    def val(i):
        if i in cv: return cv[i]
        n = N[i]; k = n['kind']
        if k == 'terminal': r = sgn*num(n['payoff_p0'])
        elif k == 'chance': r = sum(num(o['prob'])*val(o['child']) for o in n['outcomes'])
        elif n['player'] != player: r = sum(pr*val(a['child']) for a, pr in zip(n['actions'], sig[n['infoset']]))
        else:
            I = n['infoset']
            if I not in choice:
                tot = [sum(w[j]*val(N[j]['actions'][a]['child']) for j in mine[I]) for a in range(len(n['actions']))]
                choice[I] = max(range(len(tot)), key=lambda a: tot[a])
            r = val(n['actions'][choice[I]]['child'])
        cv[i] = r; return r
    sys.setrecursionlimit(10000)
    return val(g['root'])
def expl(sig): return (br(sig, 0) + br(sig, 1))/2
T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
sys.setrecursionlimit(10000)
for t in range(1, T+1):
    cfr(g['root'], 0, 1.0, 1.0, t)
    if t % 500 == 0:
        sig = {k: avg(k) for k in IS}; print(t, expl(sig), flush=True)
sig = {k: avg(k) for k in IS}
print('expl', expl(sig), 'value', evalv(sig))
json.dump({"format": "bluff-strategy-v1", "game": g['name'], "producer": "CFR+ linear avg",
  "probs": {k: {l: repr(p) for l, p in zip(IS[k][1], sig[k])} for k in IS}}, open('strategy.json', 'w'), indent=1)

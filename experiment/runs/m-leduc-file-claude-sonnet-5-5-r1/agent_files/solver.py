import json, sys
from fractions import Fraction
sys.setrecursionlimit(10000)
g = json.load(open('game.json'))
N = {n['id']: n for n in g['nodes']}
def fr(s): return float(Fraction(s))
IS = {}  # infoset -> (player, labels)
for n in g['nodes']:
    if n['kind'] == 'decision':
        IS.setdefault(n['infoset'], (n['player'], [a['label'] for a in n['actions']]))
reg = {k: [0.0]*len(v[1]) for k, v in IS.items()}
ssum = {k: [0.0]*len(v[1]) for k, v in IS.items()}
def cur(k):
    r = [max(x, 0) for x in reg[k]]; s = sum(r); n = len(r)
    return [x/s for x in r] if s > 0 else [1.0/n]*n
def avg(k):
    s = sum(ssum[k]); n = len(ssum[k])
    return [x/s for x in ssum[k]] if s > 0 else [1.0/n]*n
def cfr(i, r0, r1, t):
    n = N[i]; kd = n['kind']
    if kd == 'terminal': return fr(n['payoff_p0'])
    if kd == 'chance':
        return sum(fr(o['prob'])*cfr(o['child'], r0, r1, t) for o in n['outcomes'])
    p = n['player']; k = n['infoset']; s = cur(k)
    vs = []
    for a, pr in zip(n['actions'], s):
        vs.append(cfr(a['child'], r0*pr if p == 0 else r0, r1*pr if p == 1 else r1, t))
    v = sum(pr*x for pr, x in zip(s, vs))
    my, opp = (r0, r1) if p == 0 else (r1, r0)
    sg = 1 if p == 0 else -1
    for j in range(len(vs)):
        reg[k][j] = max(0.0, reg[k][j] + opp*sg*(vs[j]-v))
    w = my*t
    for j in range(len(s)): ssum[k][j] += w*s[j]
    return v
def evalv(strat):
    def f(i):
        n = N[i]; kd = n['kind']
        if kd == 'terminal': return fr(n['payoff_p0'])
        if kd == 'chance': return sum(fr(o['prob'])*f(o['child']) for o in n['outcomes'])
        return sum(pr*f(a['child']) for a, pr in zip(n['actions'], strat[n['infoset']]))
    return f(g['root'])
def br(strat, p):
    # reach of chance+opp per node, depth
    info = {}  # infoset -> list of (reach, node)
    depth = {}
    def walk(i, r, d):
        n = N[i]; kd = n['kind']
        if kd == 'chance':
            for o in n['outcomes']: walk(o['child'], r*fr(o['prob']), d+1)
        elif kd == 'decision':
            if n['player'] == p:
                info.setdefault(n['infoset'], []).append((r, i)); depth[n['infoset']] = max(depth.get(n['infoset'], 0), d)
                for a in n['actions']: walk(a['child'], r, d+1)
            else:
                for a, pr in zip(n['actions'], strat[n['infoset']]): walk(a['child'], r*pr, d+1)
    walk(g['root'], 1.0, 0)
    choice = {}
    memo = {}
    def val(i):  # value to p given BR below
        if i in memo: return memo[i]
        n = N[i]; kd = n['kind']
        sg = 1 if p == 0 else -1
        if kd == 'terminal': v = sg*fr(n['payoff_p0'])
        elif kd == 'chance': v = sum(fr(o['prob'])*val(o['child']) for o in n['outcomes'])
        elif n['player'] == p: v = val(n['actions'][choice[n['infoset']]]['child'])
        else: v = sum(pr*val(a['child']) for a, pr in zip(n['actions'], strat[n['infoset']]))
        memo[i] = v; return v
    for k in sorted(info, key=lambda k: -depth[k]):
        m = len(IS[k][1])
        sc = [sum(r*val(N[i]['actions'][j]['child']) for r, i in info[k]) for j in range(m)]
        choice[k] = max(range(m), key=lambda j: sc[j])
    return val(g['root'])
def S(): return {k: avg(k) for k in IS}
T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
for t in range(1, T+1):
    cfr(g['root'], 1.0, 1.0, t)
    if t % 500 == 0 or t == T:
        s = S(); e = (br(s, 0)+br(s, 1))/2
        print(t, e, evalv(s), flush=True)
        if e < 1e-4: break
s = S()
json.dump({"format": "bluff-strategy-v1", "game": g['name'], "producer": "CFR+ (linear averaging), solver.py",
 "probs": {k: {l: repr(p) for l, p in zip(IS[k][1], s[k])} for k in IS}}, open('strategy.json', 'w'), indent=1)

import json, sys
from fractions import Fraction
sys.setrecursionlimit(100000)
g = json.load(open('game.json'))
N = {n['id']: n for n in g['nodes']}
def fr(s): return float(Fraction(s))
for n in N.values():
    if n['kind']=='chance':
        for o in n['outcomes']: o['p']=fr(o['prob'])
    if n['kind']=='terminal': n['u']=fr(n['payoff_p0'])
IS = {}  # infoset -> dict
for n in N.values():
    if n['kind']=='decision':
        d = IS.setdefault(n['infoset'], {'player':n['player'],'labels':[a['label'] for a in n['actions']],'nodes':[]})
        d['nodes'].append(n['id'])
for d in IS.values():
    k=len(d['labels']); d['reg']=[0.0]*k; d['sum']=[0.0]*k
def cur(d):
    r=[max(x,0.0) for x in d['reg']]; s=sum(r); k=len(r)
    return [x/s for x in r] if s>0 else [1.0/k]*k
def avg(d):
    s=sum(d['sum']); k=len(d['sum'])
    return [x/s for x in d['sum']] if s>0 else [1.0/k]*k
def cfr(i, r0, r1, p, t):
    n=N[i]; k=n['kind']
    if k=='terminal': return n['u']
    if k=='chance':
        return sum(o['p']*cfr(o['child'],r0*o['p'] if False else r0, r1, p, t) if False else 0 for o in [])  # placeholder
    return 0
# chance reach folded into r0 (as external weight "c")
def cfr(i, c, r0, r1, p, t):
    n=N[i]; k=n['kind']
    if k=='terminal': return n['u']
    if k=='chance':
        return sum(o['p']*cfr(o['child'], c*o['p'], r0, r1, p, t) for o in n['outcomes'])
    d=IS[n['infoset']]; s=cur(d); pl=d['player']
    vals=[cfr(a['child'], c, r0*s[j] if pl==0 else r0, r1*s[j] if pl==1 else r1, p, t) for j,a in enumerate(n['actions'])]
    v=sum(s[j]*vals[j] for j in range(len(s)))
    if pl==p:
        # weights: c is already in vals via recursion? c passed only for reach; vals are conditional values (chance expectation inside)
        opp = r1 if pl==0 else r0
        w=c*opp
        for j in range(len(s)):
            dv = (vals[j]-v) if pl==0 else (v-vals[j])
            d['reg'][j]=max(d['reg'][j]+w*dv,0.0)
        own = r0 if pl==0 else r1
        for j in range(len(s)): d['sum'][j]+= t*own*s[j]
    return v
# note: c is chance reach, but vals conditional on being at node, so regret weight = c*opp reach. ok.
def evaluate(strat, i, memo):
    pass
def value_and_br(player):
    # w(n): reach of opponent + chance
    w={}
    st=[(g['root'],1.0)]
    order=[]
    while st:
        i,x=st.pop(); w[i]=x; order.append(i); n=N[i]
        if n['kind']=='chance':
            for o in n['outcomes']: st.append((o['child'],x*o['p']))
        elif n['kind']=='decision':
            d=IS[n['infoset']]
            if d['player']==player:
                for a in n['actions']: st.append((a['child'],x))
            else:
                s=avg(d)
                for j,a in enumerate(n['actions']): st.append((a['child'],x*s[j]))
    memo={}; choice={}
    def val(i):
        if i in memo: return memo[i]
        n=N[i]; k=n['kind']
        if k=='terminal': r=n['u'] if player==0 else -n['u']
        elif k=='chance': r=sum(o['p']*val(o['child']) for o in n['outcomes'])
        else:
            d=IS[n['infoset']]
            if d['player']==player:
                if n['infoset'] not in choice:
                    sc=[sum(w[m]*val(N[m]['actions'][j]['child']) for m in d['nodes']) for j in range(len(d['labels']))]
                    choice[n['infoset']]=max(range(len(sc)),key=lambda j:sc[j])
                r=val(n['actions'][choice[n['infoset']]]['child'])
            else:
                s=avg(d); r=sum(s[j]*val(a['child']) for j,a in enumerate(n['actions']))
        memo[i]=r; return r
    return val(g['root'])
def expo():
    return (value_and_br(0)+value_and_br(1))/2
def write():
    probs={k:{l:repr(p) for l,p in zip(d['labels'],avg(d))} for k,d in IS.items()}
    json.dump({"format":"bluff-strategy-v1","game":g['name'],"producer":"CFR+ alternating, linear averaging","probs":probs},open('strategy.json','w'))
import time
t0=time.time(); T=0
budget=float(sys.argv[1]) if len(sys.argv)>1 else 600
while time.time()-t0<budget:
    for _ in range(25):
        T+=1
        for p in (0,1): cfr(g['root'],1.0,1.0,1.0,p,T)
    e=expo(); print(T,e,flush=True)
    if e<1e-4: break
write()
# game value under avg
def ev(i):
    n=N[i];k=n['kind']
    if k=='terminal': return n['u']
    if k=='chance': return sum(o['p']*ev(o['child']) for o in n['outcomes'])
    s=avg(IS[n['infoset']]); return sum(s[j]*ev(a['child']) for j,a in enumerate(n['actions']))
print('value',ev(g['root']),'expl',expo(),'infosets',len(IS),'nodes',len(N))

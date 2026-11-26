import json, itertools
from fractions import Fraction
R={'J':0,'Q':1,'K':2}
def term(h,c):
    # returns payoff to p0 or None
    if h=='cb': return None if False else None
    return None
def payoff(h,cards):
    hi = 1 if cards[0]>cards[1] else -1
    if h=='cc': return hi
    if h=='bf': return 1
    if h=='bc': return 2*hi
    if h=='cbf': return -1
    if h=='cbc': return 2*hi
    return None
def acts(h): return ['fold','call'] if h[-1:]=='b' else ['check','bet']
L={'check':'c','bet':'b','fold':'f','call':'c'}
def hist(h,a): return h+L[a]
def player(h): return len(h)%2
N={}
def node(k):
    if k not in N: N[k]=[[0.0]*2,[0.0]*2,[0.0]*2]  # regret, strat sum
    return N[k]
def strat(n):
    p=[max(x,0) for x in n[0]]; s=sum(p)
    return [x/s for x in p] if s>0 else [.5,.5]
def cfr(h,cards,pi):
    v=payoff(h,cards)
    if v is not None: return v
    p=player(h); k=('JQK'[cards[p]])+':'+h
    n=node(k); s=strat(n); A=acts(h)
    u=[]
    for i,a in enumerate(A):
        np_=list(pi); np_[p]*=s[i]
        u.append(cfr(hist(h,a),cards,np_))
    sign=1 if p==0 else -1
    ev=sum(s[i]*u[i] for i in range(2))
    opp=pi[1-p]
    for i in range(2):
        n[0][i]+=opp*sign*(u[i]-ev)
        n[1][i]+=pi[p]*s[i]
    return ev
deals=[c for c in itertools.permutations(range(3),2)]
for t in range(200000):
    for d in deals: cfr('',d,[1.0,1.0])
avg={}
for k,n in N.items():
    s=sum(n[1]); avg[k]=[x/s for x in n[1]]
def val(h,cards,S,br=None):
    pass
# exploitability: best response via enumeration
def ev(h,cards,S):
    v=payoff(h,cards)
    if v is not None: return v
    p=player(h); k='JQK'[cards[p]]+':'+h
    return sum(S[k][i]*ev(hist(h,a),cards,S) for i,a in enumerate(acts(h)))
def br(player_b,S):
    # pure BR over info sets: enumerate all pure strategies for the br player
    keys=[k for k in S if (len(k.split(':')[1])%2)==player_b]
    best=-9
    for choice in itertools.product([0,1],repeat=len(keys)):
        T=dict(S)
        for k,c in zip(keys,choice): T[k]=[1.0-c,float(c)]
        e=sum(ev('',d,T) for d in deals)/6
        e=e if player_b==0 else -e
        best=max(best,e)
    return best
expl=(br(0,avg)+br(1,avg))/2
v=sum(ev('',d,avg) for d in deals)/6
print(expl,v,avg)
probs={}
for k,s in avg.items():
    a=acts(k.split(':')[1]); s=[round(x,9) for x in s]; s[1]=round(1-s[0],9)
    probs[k]={a[0]:repr(s[0]),a[1]:repr(s[1])}
json.dump({"format":"bluff-strategy-v1","game":"kuhn","producer":"vanilla CFR 200k iters, average strategy","probs":probs},open('strategy.json','w'),indent=1)

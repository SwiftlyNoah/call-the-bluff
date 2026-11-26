import json, itertools
ACT={'':('check','bet'),'c':('check','bet'),'b':('fold','call'),'cb':('fold','call')}
def acts(h): return ('check','bet') if h in('','c') else ('fold','call')
def term(h,cards):
    # returns payoff to player 0 or None
    hi = 1 if cards[0]>cards[1] else -1
    if h=='cc': return hi
    if h=='bf': return 1
    if h=='bc': return 2*hi
    if h=='cbf': return -1
    if h=='cbc': return 2*hi
    return None
def nxt(h,a): return h+{'check':'c','bet':'b','fold':'f','call':'c'}[a] if not(h in('b','cb') and a=='call') else h+'c'
def hist(h,a):
    return h+('c' if a=='check' else 'b' if a=='bet' else 'f' if a=='fold' else 'c')
# fix: call after b -> 'bc', after cb -> 'cbc'; fold -> 'bf','cbf'
def player(h): return len(h)%2
R={};S={}
def strat(k,n=2):
    r=R.setdefault(k,[0.0]*n); p=[max(x,0) for x in r]; s=sum(p)
    return [x/s for x in p] if s>0 else [1/n]*n
def cfr(h,cards,reach,plus=True):
    t=term(h,cards)
    if t is not None: return t
    p=player(h); k='JQK'[cards[p]]+':'+h.replace('c','c')
    # history string for infoset excludes terminal letters; fine
    sg=strat(k); A=acts(h); u=[0]*2
    for i,a in enumerate(A):
        r2=list(reach); r2[p]*=sg[i]
        u[i]=cfr(hist(h,a),cards,r2)
    v=sum(sg[i]*u[i] for i in range(2))
    sgn=1 if p==0 else -1
    R.setdefault(k,[0.0]*2)
    for i in range(2): R[k][i]=max(0,R[k][i]+sgn*(u[i]-v)*reach[1-p]*1)  # u in p0 terms
    S.setdefault(k,[0.0]*2)
    for i in range(2): S[k][i]+=reach[p]*sg[i]*(1)
    return v
# note: regret for p1 needs u from p1's view: sgn handles it
deals=list(itertools.permutations(range(3),2))
T=200000
for it in range(T):
    for c in deals: cfr('',c,[1,1])
avg={}
for k,s in S.items():
    t=sum(s); avg[k]=[x/t for x in s]
# evaluate
def val(h,cards,pr):
    t=term(h,cards)
    if t is not None: return t
    p=player(h); k='JQK'[cards[p]]+':'+h
    return sum(pr[k][i]*val(hist(h,a),cards,pr) for i,a in enumerate(acts(h)))
def br(pl):
    # best response value for pl vs avg; per infoset sums over opponent card
    tot=0
    def go(h,cards_by_info):  # cards_by_info: list of deals consistent
        pass
    # enumerate infosets via recursion with deal sets
    def rec(h,deals_):
        if term(h,deals_[0][0]) is not None:
            return sum(term(h,c)*w for c,w in deals_)*(1 if pl==0 else -1)
        p=player(h)
        if p==pl:
            # group by own card
            tot=0
            for card in range(3):
                sub=[(c,w) for c,w in deals_ if c[p]==card]
                if not sub: continue
                tot+=max(rec(hist(h,a),sub) for a in acts(h))
            return tot
        res=0
        for i,a in enumerate(acts(h)):
            sub=[(c,w*avg['JQK'[c[p]]+':'+h][i]) for c,w in deals_]
            res+=rec(hist(h,a),sub)
        return res
    return rec('',[(c,1/6) for c in deals])
v=sum(val('',c,avg) for c in deals)/6
e=(br(0)+br(1))/2
print(v,e,sorted(avg.items()))
probs={}
for k,s in avg.items():
    h=k.split(':')[1]; A=acts(h)
    probs[k]={A[0]:repr(s[0]),A[1]:repr(1-s[0])}
json.dump({"format":"bluff-strategy-v1","game":"kuhn","producer":"CFR+ style regret matching, 200k iters, average strategy","probs":probs},open('strategy.json','w'),indent=1)

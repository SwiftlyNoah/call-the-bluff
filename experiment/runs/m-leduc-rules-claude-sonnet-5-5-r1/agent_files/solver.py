import json, itertools, sys
from collections import defaultdict
CARDS=['Js','Jh','Qs','Qh','Ks','Kh']
RK={'J':0,'Q':1,'K':2}
def payoff_showdown(c,pub,p0,p1):
    # returns payoff to player 0 net, given contributions
    def score(x):
        return (2 if x[0]==pub[0] else 0, RK[x[0]])
    a,b=score(c[0]),score(c[1])
    return 0 if a==b else (p1 if a>b else -p0)
def info(p,c,pub,h1,h2):
    return f"{c[p]}:{h1}" if pub is None else f"{c[p]}|{pub}:{h1}/{h2}"
def legal(nr,facing):
    a=[]
    if facing: a.append('fold')
    a.append('call')
    if nr<2: a.append('raise')
    return a
LET={'fold':'f','call':'c','raise':'r'}
# state: c, pub, rnd, h1,h2, contrib(p0,p1), nr, nacts_in_round, player, facing
def step(st,a):
    c,pub,rnd,h1,h2,ct,nr,na,pl,facing=st
    ct=list(ct); L=LET[a]
    if rnd==0: h1+=L
    else: h2+=L
    if a=='fold':
        # opponent wins pot
        return ('T', ct[pl], pl)
    if a=='raise':
        size=2 if rnd==0 else 4
        ct[pl]=ct[1-pl]+size
        return (c,pub,rnd,h1,h2,tuple(ct),nr+1,na+1,1-pl,True)
    # call
    ct[pl]=ct[1-pl]
    if na==0:
        return (c,pub,rnd,h1,h2,tuple(ct),nr,1,1-pl,False)
    # round ends
    if rnd==0: return ('CH',c,h1,tuple(ct))
    return ('T2',c,pub,ct[0],ct[1])

R={};S={}
def strat_of(k,n):
    pos=[max(x,0.0) for x in R[k]]; t=sum(pos)
    return [x/t for x in pos] if t>0 else [1.0/n]*n
def avg_of(k,n):
    t=sum(S[k]); return [x/t for x in S[k]] if t>0 else [1.0/n]*n
def cfr(st,r0,r1,q,sf):
    # value to player 0, conditional on reaching st (chance averaged)
    if st[0]=='T':
        return st[1] if st[2]==1 else -st[1]
    if st[0]=='T2':
        return payoff_showdown(st[1],st[2],st[3],st[4])
    if st[0]=='CH':
        _,c,h1,ct=st
        rem=[x for x in CARDS if x not in c]
        return sum(cfr((c,pub,1,h1,'',ct,0,0,0,False),r0,r1,q/len(rem),sf) for pub in rem)/len(rem)
    c,pub,rnd,h1,h2,ct,nr,na,pl,facing=st
    acts=legal(nr,facing); n=len(acts); k=info(pl,c,pub,h1,h2)
    if sf is None:
        if k not in R: R[k]=[0.0]*n; S[k]=[0.0]*n
        s=strat_of(k,n)
    else: s=sf(k,acts)
    vals=[]
    for a,pa in zip(acts,s):
        if pl==0: vals.append(cfr(step(st,a),r0*pa,r1,q,sf))
        else: vals.append(cfr(step(st,a),r0,r1*pa,q,sf))
    v=sum(pa*x for pa,x in zip(s,vals))
    if sf is None:
        sg=1 if pl==0 else -1
        ro=r1 if pl==0 else r0; rp=r0 if pl==0 else r1
        for i in range(n):
            R[k][i]=max(0.0,R[k][i]+q*ro*sg*(vals[i]-v))  # CFR+
            S[k][i]+=q*rp*s[i]*(iters_t[0])  # linear averaging
    return v
iters_t=[1]
def root(sf=None,t=1):
    tot=0
    deals=[(a,b) for a in CARDS for b in CARDS if a!=b]
    for c in deals:
        st=(c,None,0,'','',(1,1),0,0,0,False)
        tot+=cfr(st,1.0,1.0,1.0/len(deals),sf)/len(deals)
    return tot
def br(player,avg):
    # best response of `player` vs avg for the other; iterate to fixed point on choices
    choice={}
    def sf_factory(k,acts):
        pass
    deals=[(a,b) for a in CARDS for b in CARDS if a!=b]
    for it in range(30):
        cf=defaultdict(lambda:None)
        acc={}
        def go(st,ro,q):
            # returns value to `player`; ro=opp reach
            if st[0]=='T':
                v=st[1] if st[2]!=player else -st[1]; return v
            if st[0]=='T2':
                v=payoff_showdown(st[1],st[2],st[3],st[4]); return v if player==0 else -v
            if st[0]=='CH':
                c=st[1]; rem=[x for x in CARDS if x not in c]
                return sum(go((c,pub,1,st[2],'',st[3],0,0,0,False),ro,q/len(rem)) for pub in rem)/len(rem)
            c,pub,rnd,h1,h2,ct,nr,na,pl,facing=st
            acts=legal(nr,facing); k=info(pl,c,pub,h1,h2)
            if pl==player:
                vals=[go(step(st,a),ro,q) for a in acts]
                a_=acc.setdefault(k,[0.0]*len(acts))
                for i,x in enumerate(vals): a_[i]+=q*ro*x
                i=choice.get(k,0)
                return vals[i]
            s=avg(k,acts)
            return sum(p*go(step(st,a),ro*p,q) for a,p in zip(acts,s))
        tot=0
        for c in deals:
            tot+=go((c,None,0,'','',(1,1),0,0,0,False),1.0,1.0/len(deals))/len(deals)
        new={k:max(range(len(v)),key=lambda i:v[i]) for k,v in acc.items()}
        if new==choice: break
        choice=new
    return tot
if __name__=='__main__':
    N=int(sys.argv[1])
    for t in range(1,N+1):
        iters_t[0]=t
        root(None,t)
    def avg(k,acts): return avg_of(k,len(acts))
    # build strategy
    probs={}
    keys={}
    def collect():
        pass
    for k in R:
        pass
    # need action labels per infoset: walk to collect
    labels={}
    def walk(st):
        if st[0] in('T','T2'): return
        if st[0]=='CH':
            c=st[1]
            for pub in [x for x in CARDS if x not in c]: walk((c,pub,1,st[2],'',st[3],0,0,0,False))
            return
        c,pub,rnd,h1,h2,ct,nr,na,pl,facing=st
        acts=legal(nr,facing); labels[info(pl,c,pub,h1,h2)]=acts
        for a in acts: walk(step(st,a))
    for a in CARDS:
        for b in CARDS:
            if a!=b: walk(((a,b),None,0,'','',(1,1),0,0,0,False))
    for k,acts in labels.items():
        p=avg_of(k,len(acts)); probs[k]={a:repr(x) for a,x in zip(acts,p)}
        # renormalize
    json.dump({"format":"bluff-strategy-v1","game":"leduc","producer":"CFR+ linear averaging, %d iters"%N,"probs":probs},open('strategy.json','w'),indent=1)
    # evaluate from file-equivalent strategy
    def sfp(k,acts): return [float(probs[k][a]) for a in acts]
    v=root(sfp)
    b0,b1=br(0,sfp),br(1,sfp);print(b0,b1);e=(b0+b1)/2
    print(len(probs),'infosets value',v,'expl',e)

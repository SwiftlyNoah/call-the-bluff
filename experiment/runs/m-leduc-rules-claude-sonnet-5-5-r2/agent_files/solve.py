import json, itertools, sys
import numpy as np
CARDS=['Js','Jh','Qs','Qh','Ks','Kh']; RANK=[0,0,1,1,2,2]
def rank_val(c,pub):
    return 3+RANK[c] if pub is not None and RANK[c]==RANK[pub] else RANK[c]
def pay0(c0,c1,pub):  # showdown payoff sign for p0
    a,b=rank_val(c0,pub),rank_val(c1,pub)
    return (a>b)-(a<b)
# state: (rnd, h1, h2, pub, contrib(2 tuple), nraise, player, outstanding)
def legal(nr,owe): 
    a=['call']
    if owe>0: a=['fold']+a
    if nr<2: a.append('raise')
    return a
def step(s,a):
    rnd,h1,h2,pub,ct,nr,pl,owe,first=s
    L={'fold':'f','call':'c','raise':'r'}[a]
    h=(h1,h2); h=list(h); h[rnd]+=L
    ct=list(ct)
    if a=='fold': return ('T',pl,tuple(ct))   # pl folded
    if a=='call':
        ct[pl]+=owe
        if first:  # check as first action: round continues
            return (rnd,h[0],h[1],pub,tuple(ct),nr,1-pl,0,False)
        # round ends
        if rnd==0: return ('C',h[0],h[1],tuple(ct))
        return ('S',pub,tuple(ct))
    size=2 if rnd==0 else 4
    ct[pl]+=owe+size
    return (rnd,h[0],h[1],pub,tuple(ct),nr+1,1-pl,size,False)
def isname(c,s):
    rnd,h1,h2,pub=s[:4]
    if rnd==0: return f"{CARDS[c]}:{h1}"
    return f"{CARDS[c]}|{CARDS[pub]}:{h1}/{h2}"
start=(0,'','',None,(1,1),0,0,0,True)
# Note: first flag: owe==0 and no action yet in round => check continues round.
# a check after a check ends round: handled since first False after first action.
def chance_state(t):
    _,h1,_,ct=t
    return lambda pub:(1,h1,'',pub,ct,0,0,0,True)
nodes={}  # name -> dict
regret={};strat_sum={}
def legal_s(s): return legal(s[5],s[7])
def sigma(name,acts):
    r=regret.get(name)
    if r is None: return np.ones(len(acts))/len(acts)
    p=np.maximum(r,0); t=p.sum()
    return p/t if t>0 else np.ones(len(acts))/len(acts)
def terminal_pay(t,c0,c1):
    if t[0]=='T':
        pl,ct=t[1],t[2]
        return -ct[0] if pl==0 else ct[1]
    _,pub,ct=t
    return pay0(c0,c1,pub)*ct[0]
def cfr(s,c0,c1,r0,r1,it,pubs_used=None):
    # returns value for p0
    if s[0] in('T','S'): return terminal_pay(s,c0,c1)
    if s[0]=='C':
        v=0;rem=[c for c in range(6) if c not in(c0,c1)]
        for pub in rem: v+=cfr(chance_state(s)(pub),c0,c1,r0,r1,it)/4
        return v
    pl=s[6];acts=legal_s(s);name=isname(c0 if pl==0 else c1,s)
    sg=sigma(name,acts)
    if name not in regret: regret[name]=np.zeros(len(acts));strat_sum[name]=np.zeros(len(acts))
    vals=np.zeros(len(acts))
    for i,a in enumerate(acts):
        if pl==0: vals[i]=cfr(step(s,a),c0,c1,r0*sg[i],r1,it)
        else: vals[i]=cfr(step(s,a),c0,c1,r0,r1*sg[i],it)
    v=float(sg@vals)
    mine,opp=(r0,r1) if pl==0 else (r1,r0)
    sgn=1 if pl==0 else -1
    if UPD==2 or pl==UPD:
        regret[name]=np.maximum(regret[name]+opp*sgn*(vals-v),0)  # CFR+
        strat_sum[name]+=mine*sg*(it)   # linear averaging
    return v
UPD=0
def avg(name):
    s=strat_sum[name];t=s.sum()
    return s/t if t>0 else np.ones(len(s))/len(s)
def ev_and_br(avgstrat):
    U=np.ones(3)/3
    # returns (value p0, br0, br1) ; vector-based
    def walk(s,reach,p,mode):
        # reach[c][o]: weights of (br player card c, opp card o). mode: 'br' or 'eval'
        if s[0] in('T','S'):
            v=np.zeros(6)
            for c in range(6):
                for o in range(6):
                    if reach[c,o]==0: continue
                    c0,c1=(c,o) if p==0 else (o,c)
                    pay=terminal_pay(s,c0,c1)
                    v[c]+=reach[c,o]*(pay if p==0 else -pay)
            return v
        if s[0]=='C':
            v=np.zeros(6)
            for pub in range(6):
                r=reach.copy();r[pub,:]=0;r[:,pub]=0
                if r.sum()==0: continue
                v+=walk(chance_state(s)(pub),r/4,p,mode)
            return v
        pl=s[6];acts=legal_s(s)
        if pl==p:
            ch=[walk(step(s,a),reach,p,mode) for a in acts]
            if mode=='br': return np.max(ch,axis=0)
            v=np.zeros(6)
            for c in range(6):
                sg=avgstrat.get(isname(c,s),U)
                for i in range(len(acts)): v[c]+=sg[i]*ch[i][c]
            return v
        v=np.zeros(6)
        for i,a in enumerate(acts):
            r=reach.copy()
            for o in range(6): r[:,o]*=avgstrat.get(isname(o,s),U)[i]
            v+=walk(step(s,a),r,p,mode)
        return v
    init=np.array([[0 if c==o else 1/30 for o in range(6)] for c in range(6)])
    br=[walk(start,init,p,'br').sum() for p in (0,1)]
    ev=walk(start,init,0,'eval').sum()
    return ev,br
if __name__=='__main__':
    N=int(sys.argv[1]) if len(sys.argv)>1 else 3000
    deals=[(a,b) for a in range(6) for b in range(6) if a!=b]
    for it in range(1,N+1):
        for u in (2,):
            UPD=u
            for a,b in deals: cfr(start,a,b,1/30,1/30,it)
        # note: reach carries chance prob only via deal averaging
        if it%2000==0 or it==N:
            A={n:avg(n) for n in strat_sum}
            ev,br=ev_and_br(A); print(it,ev,(br[0]+br[1])/2,flush=True)
    A={n:avg(n) for n in strat_sum}
    ev,br=ev_and_br(A); expl=(br[0]+br[1])/2
    probs={}
    for n,p in A.items():
        rnd0=('/' not in n)
        acts=['call','raise'] if False else None
        probs[n]=p
    # action labels
    out={}
    def labels(n):
        h=n.split(':')[1].split('/')[-1]
        nr=h.count('r'); owe=0
        # owe>0 iff last action is raise
        owe=1 if h.endswith('r') else 0
        return legal(nr,owe)
    for n,p in A.items(): out[n]={a:repr(float(x)) for a,x in zip(labels(n),p)}
    json.dump({"format":"bluff-strategy-v1","game":"leduc","producer":"CFR+ linear avg, python","probs":out},open('strategy.json','w'),indent=1)
    print('infosets',len(out),'ev',ev,'expl',expl)

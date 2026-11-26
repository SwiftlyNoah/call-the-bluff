import json, sys
CARDS=['Js','Jh','Qs','Qh','Ks','Kh']
rank=lambda c:c//2
RAISE=[2,4]
# state: rnd, h1, h2, cur(player), nraise, bets(list of 2 total chips, ante incl), owed? computed
def legal(facing, nr):
    a=[]
    if facing: a.append('fold')
    a.append('call')
    if nr<2: a.append('raise')
    return a
LET={'fold':'f','call':'c','raise':'r'}
def step(st,a):
    """st=(rnd,h1,h2,player,nr,c0,c1,first). returns ('term',folder) | ('pub',st') | ('node',st')"""
    rnd,h1,h2,p,nr,c0,c1,first=st
    c=[c0,c1]
    h=(h1,h2)[rnd]+LET[a]
    if a=='fold': return 'fold',p
    if a=='call':
        c[p]=c[1-p]
        end = not first
        # first action 'call' = check; round ends only if second action
        if end:
            if rnd==1: return 'show',None
            return 'pub',(1,h,'',0,0,c[0],c[1],True)
        return 'node',(rnd,h if rnd==0 else h1,h if rnd==1 else h2,1-p,nr,c[0],c[1],False)
    c[p]=c[1-p]+RAISE[rnd]
    return 'node',(rnd,h if rnd==0 else h1,h if rnd==1 else h2,1-p,nr+1,c[0],c[1],False)
def facing(st): return st[5]!=st[6]
def key(st,card,pub):
    rnd,h1,h2=st[0],st[1],st[2]
    if rnd==0: return f"{CARDS[card]}:{h1}"
    return f"{CARDS[card]}|{CARDS[pub]}:{h1}/{h2}"
def payoff(c0,c1,card0,card1,pub):
    r0,r1,rp=rank(card0),rank(card1),rank(pub)
    if r0==rp and r1!=rp: w=0
    elif r1==rp and r0!=rp: w=1
    elif r0>r1: w=0
    elif r1>r0: w=1
    else: return 0
    return c1 if w==0 else -c0
START=(0,'','',0,0,1,1,True)
# first=True means no action yet this round. fix: 'end' logic uses `not first`.
reg={};ssum={}
def getp(k,acts,plus=False):
    r=reg.setdefault(k,[0.0]*len(acts))
    pos=[max(x,0) for x in r]; s=sum(pos)
    return [x/s for x in pos] if s>0 else [1/len(acts)]*len(acts)
def cfr(st,cards,pub,r0,r1,t):
    """returns value to player 0, r0/r1 include chance for opp-reach purposes"""
    acts=legal(facing(st),st[4]); p=st[3]
    k=key(st,cards[p],pub)
    sg=getp(k,acts)
    vals=[]
    for a in acts:
        kind,nx=step(st,a)
        pr=sg[acts.index(a)]
        n0,n1=(r0*pr,r1) if p==0 else (r0,r1*pr)
        if kind=='fold':
            v = st[6 if p==0 else 5]*(-1 if p==0 else 1) if False else None
            # folder p loses own chips
            v=-st[5] if p==0 else st[6]
        elif kind=='show': v=payoff(nx_c(st,a)[0],nx_c(st,a)[1],cards[0],cards[1],pub)
        elif kind=='pub':
            v=0
            rem=[x for x in range(6) if x not in cards]
            for pc in rem: v+=cfr(nx,cards,pc,n0,n1,t)/len(rem)
        else: v=cfr(nx,cards,pub,n0,n1,t)
        vals.append(v)
    ev=sum(s*v for s,v in zip(sg,vals))
    opp=r1 if p==0 else r0; me=r0 if p==0 else r1
    sgn=1 if p==0 else -1
    rr=reg[k]
    for i,v in enumerate(vals): rr[i]=max(0.0,rr[i]+opp*sgn*(v-ev))
    ss=ssum.setdefault(k,[0.0]*len(acts))
    for i in range(len(acts)): ss[i]+=t*me*sg[i]
    return ev
def nx_c(st,a):
    c=[st[5],st[6]];p=st[3]
    if a=='call': c[p]=c[1-p]
    return c
def avg(k,acts):
    s=ssum.get(k); 
    if not s or sum(s)==0: return [1/len(acts)]*len(acts)
    t=sum(s); return [x/t for x in s]
# enumerate infosets
infos={}
def enum(st,pub,card_p):
    pass
def collect():
    out={}
    def rec(st,pubs):
        acts=legal(facing(st),st[4]);p=st[3]
        for card in range(6):
            for pub in pubs:
                if pub==card: continue
                out[key(st,card,pub)]=acts
        for a in acts:
            kind,nx=step(st,a)
            if kind=='node': rec(nx,pubs)
            elif kind=='pub': rec(nx,range(6))
    rec(START,[-1])
    return out
INF=collect()
def sigma(k): return dict(zip(INF[k],avg(k,INF[k])))
def br(player):
    """BR value for `player` vs average strategy of the other."""
    opp=1-player
    def rec(st,pub,r):
        p=st[3];acts=legal(facing(st),st[4])
        res=[0.0]*6
        if p==player:
            best=[-1e9]*6
            for a in acts:
                v=sub(st,a,pub,r)
                for m in range(6): best[m]=max(best[m],v[m])
            return best
        sg={}
        for oc in range(6):
            if oc==pub: continue
            sg[oc]=sigma(key(st,oc,pub))
        for a in acts:
            r2=[r[oc]*sg[oc][a] if oc in sg else 0 for oc in range(6)]
            v=sub(st,a,pub,r2,fixed=True)
            for m in range(6): res[m]+=v[m]
        return res
    def sub(st,a,pub,r,fixed=False):
        kind,nx=step(st,a);p=st[3]
        if kind=='pub':
            res=[0.0]*6
            for pc in range(6):
                r2=[0 if oc==pc else r[oc]/4 for oc in range(6)]
                v=rec(nx,pc,r2)
                for m in range(6):
                    if m!=pc: res[m]+=v[m]
            return res
        if kind=='node': return rec(nx,pub,r)
        res=[0.0]*6
        c=nx_c(st,a)
        for m in range(6):
            if m==pub: continue
            for oc in range(6):
                if oc==m or oc==pub: continue
                if kind=='fold':
                    pay0=-st[5] if p==0 else st[6]
                else:
                    cd=(m,oc) if player==0 else (oc,m)
                    pay0=payoff(c[0],c[1],cd[0],cd[1],pub)
                pv=pay0 if player==0 else -pay0
                res[m]+=r[oc]*pv/30
        return res
    # hmm: pub dummy for round 0 -> use pub=-1 handled
    return rec,sub
def solve(iters):
    import itertools
    for t in range(1,iters+1):
        for c0 in range(6):
            for c1 in range(6):
                if c0!=c1: cfr(START,(c0,c1),0,1/30,1/30,t) if False else cfr(START,(c0,c1),-1,1.0,1.0,t)
        if t%500==0: print(t,expl(),file=sys.stderr,flush=True)
def bestresp(player):
    rec,sub=br(player)
    v=rec(START,-1,[1.0]*6)
    return sum(v)
def value():
    tot=0
    def rec(st,pub,cards):
        acts=legal(facing(st),st[4]);p=st[3]
        sg=sigma(key(st,cards[p],pub));v=0
        for a in acts:
            kind,nx=step(st,a)
            if kind=='fold': u=-st[5] if p==0 else st[6]
            elif kind=='show':
                c=nx_c(st,a);u=payoff(c[0],c[1],cards[0],cards[1],pub)
            elif kind=='pub':
                rem=[x for x in range(6) if x not in cards]
                u=sum(rec(nx,pc,cards) for pc in rem)/len(rem)
            else: u=rec(nx,pub,cards)
            v+=sg[a]*u
        return v
    return sum(rec(START,-1,(a,b)) for a in range(6) for b in range(6) if a!=b)/30
def expl(): return (bestresp(0)+bestresp(1))/2
if __name__=='__main__':
    solve(int(sys.argv[1]))
    print("expl",expl(),"value",value())
    probs={k:{a:repr(x) for a,x in sigma(k).items()} for k in INF}
    json.dump({"format":"bluff-strategy-v1","game":"leduc","producer":"vanilla CFR (regret matching, linear-weighted average), full enumeration","probs":probs},open('strategy.json','w'),indent=1)
    print(len(probs))

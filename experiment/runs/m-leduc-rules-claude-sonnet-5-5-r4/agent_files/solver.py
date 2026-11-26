import json, itertools, sys
from fractions import Fraction
CARDS=['Js','Jh','Qs','Qh','Ks','Kh']; RK={'J':0,'Q':1,'K':2}
DEALS=[(a,b,c) for a,b,c in itertools.permutations(CARDS,3)]
LBL={'f':'fold','c':'call','r':'raise'}
# state: (rnd, h1, h2, p, c0 contributions tuple, nraise, owe) -> use dict-free tuple
def init(): return (0,'','',0,(1,1),0,0)  # rnd,h1,h2,player,contrib,nraise,owed(current bet to match)
def legal(s):
    rnd,h1,h2,p,ct,nr,owe=s
    a=['c']
    if owe>0:a=['f','c']
    if nr<2:a.append('r')
    return a
def step(s,a):
    """returns ('T',folder) | ('S',state)"""
    rnd,h1,h2,p,ct,nr,owe=s
    h=(h1,h2)[rnd]+a
    ct=list(ct)
    if a=='f': return ('F',p)
    if a=='c':
        ct[p]+=owe
        end = len((h1,h2)[rnd])>0
        if end:
            if rnd==1: return ('S',None,tuple(ct))
            return ('S',(1,h1+'' ,'',0,tuple(ct),0,0),None)
        return ('N',(rnd,h1+a if rnd==0 else h1,h2+a if rnd==1 else h2,1-p,tuple(ct),nr,0))
    ct[p]+=owe+(2 if rnd==0 else 4)
    return ('N',(rnd,h1+a if rnd==0 else h1,h2+a if rnd==1 else h2,1-p,tuple(ct),nr+1,(2 if rnd==0 else 4)))
def key(s,deal):
    rnd,h1,h2,p,*_=s
    c=deal[p]
    return f"{c}:{h1}" if rnd==0 else f"{c}|{deal[2]}:{h1}/{h2}"
def showdown(deal,ct):
    a,b,pub=deal
    ra,rb,rp=RK[a[0]],RK[b[0]],RK[pub[0]]
    if ra==rp and rb!=rp:w=1
    elif rb==rp and ra!=rp:w=-1
    else: w=(ra>rb)-(ra<rb)
    return w*ct[1] if w>0 else (w*ct[0] if w<0 else 0)
def trans(s,a,deal):
    r=step(s,a)
    if r[0]=='F':
        f=r[1]; return ('T',0,None) if False else ('T',-(s[4][0]+0) if f==0 else s[4][1])
    if r[0]=='S':
        if r[1] is None: return ('T',showdown(deal,r[2]))
        return ('N',(1,s[1]+'c' if False else s[1]+'c','',0,r[2],0,0))
    return ('N',r[1])
# fix: round-1 end appends the final 'c' to h1
def trans(s,a,deal):
    rnd,h1,h2,p,ct,nr,owe=s
    c=list(ct)
    if a=='f': return ('T',-ct[0] if p==0 else ct[1])
    if a=='c':
        c[p]+=owe
        first=len((h1,h2)[rnd])==0
        nh1=h1+'c' if rnd==0 else h1; nh2=h2+'c' if rnd==1 else h2
        if first: return ('N',(rnd,nh1,nh2,1-p,tuple(c),nr,0))
        if rnd==1: return ('T',showdown(deal,c))
        return ('N',(1,nh1,'',0,tuple(c),0,0))
    sz=2 if rnd==0 else 4
    c[p]+=owe+sz
    nh1=h1+'r' if rnd==0 else h1; nh2=h2+'r' if rnd==1 else h2
    return ('N',(rnd,nh1,nh2,1-p,tuple(c),nr+1,sz))
reg={};ssum={}
def strat(k,n):
    r=reg.setdefault(k,[0.0]*n)
    t=sum(r)
    return [x/t for x in r] if t>0 else [1.0/n]*n
def cfr(s,deal,r0,r1,it):
    p=s[3];acts=legal(s);k=key(s,deal)
    sg=strat(k,len(acts))
    ev=[]
    for a,pr in zip(acts,sg):
        t=trans(s,a,deal)
        if t[0]=='T': ev.append(t[1])
        elif p==0: ev.append(cfr(t[1],deal,r0*pr,r1,it))
        else: ev.append(cfr(t[1],deal,r0,r1*pr,it))
    v=sum(pr*e for pr,e in zip(sg,ev))
    my,op=(r0,r1) if p==0 else (r1,r0)
    sgn=1 if p==0 else -1
    r=reg[k]
    for i in range(len(acts)): r[i]=max(0.0,r[i]+op*sgn*(ev[i]-v))
    ss=ssum.setdefault(k,[0.0]*len(acts))
    for i in range(len(acts)): ss[i]+=it*my*sg[i]
    return v
def avg(k):
    ss=ssum[k];t=sum(ss);return [x/t for x in ss] if t>0 else [1/len(ss)]*len(ss)
def brval(p,S):
    def val(s,deals):  # deals: {deal:oppweight}; returns weighted value for p
        tot=0.0;acts=legal(s);cur=s[3]
        if cur==p:
            g={}
            for d,w in deals.items(): g.setdefault(key(s,d),{})[d]=w
            for k,gd in g.items():
                best=-1e18
                for a in acts:
                    best=max(best,sub(s,a,gd))
                tot+=best
        else:
            for i,a in enumerate(acts):
                nd={d:w*S[key(s,d)][i] for d,w in deals.items()}
                tot+=sub(s,a,nd)
        return tot
    def sub(s,a,deals):
        tot=0.0;nxt={}
        for d,w in deals.items():
            t=trans(s,a,d)
            if t[0]=='T': tot+=w*(t[1] if p==0 else -t[1])
            else: nxt=nxt or {};nxt[d]=w;ns=t[1]
        if nxt: tot+=val(ns,nxt)
        return tot
    return val(init(),{d:1/120 for d in DEALS})
def play(S):
    def go(s,d,w):
        if True:
            acts=legal(s);k=key(s,d);tot=0
            for i,a in enumerate(acts):
                t=trans(s,a,d);pr=S[k][i]
                tot+=pr*(t[1] if t[0]=='T' else go(t[1],d,0))
            return tot
    return sum(go(init(),d,0) for d in DEALS)/120
N=int(sys.argv[1]) if len(sys.argv)>1 else 3000
for it in range(1,N+1):
    for d in DEALS: cfr(init(),d,1.0,1.0,it)
S={k:avg(k) for k in ssum}
# BR with strategy as {key: probs}; p0 BR value is brval(0), p1 BR value brval(1)
e=(brval(0,S)+brval(1,S))/2
v=play(S)
print(len(S),'infosets expl',e,'value',v)
probs={}
for k,pr in S.items():
    acts=[LBL[a] for a in (['f','c'] if False else [])]
# need action lists
def actsof(k):
    rnd1=('|' not in k)
    h=k.split(':')[1]
    cur=h if rnd1 else h.split('/')
    return None
L={}
def collect(s,d):
    k=key(s,d)
    if k in L: pass
    L[k]=legal(s)
    for a in legal(s):
        t=trans(s,a,d)
        if t[0]=='N': collect(t[1],d)
for d in DEALS: collect(init(),d)
assert set(L)==set(S),(len(L),len(S))
for k in L: probs[k]={LBL[a]:repr(float(x)) for a,x in zip(L[k],S[k])}
# renormalize exactness
for k in probs:
    pass
json.dump({"format":"bluff-strategy-v1","game":"leduc","producer":"CFR+ full-enumeration, linear averaging, %d iters"%N,"probs":probs},open('strategy.json','w'),indent=1)

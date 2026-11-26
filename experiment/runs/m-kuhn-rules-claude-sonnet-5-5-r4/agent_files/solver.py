import json, itertools
CARDS=[0,1,2]; N="JQK"
def term(h):
    # returns payoff to player0 or None
    if h in("cc",): return "sd1"
    if h=="bc" or h=="cbc": return "sd2" if h=="cbc" else "sd2"
    return None
def payoff(c0,c1,h):
    s=1 if c0>c1 else -1
    if h=="cc": return s
    if h in("bc","cbc"): return 2*s
    if h=="bf": return 1
    if h=="cbf": return -1
    return None
def acts(h): return ["c","b"] if h in("","c") else ["f","c"]
def player(h): return len(h)%2
def isterm(h): return h in("cc","bc","bf","cbc","cbf")
names={"c":"check","b":"bet","f":"fold"}
def label(h,a):
    if h in("","c"): return "check" if a=="c" else "bet"
    return "fold" if a=="f" else "call"
# fix: after b, 'c' means call; history letters: c check/call, b bet, f fold
R={};S={}
def strat(k,n):
    r=R.setdefault(k,[0.0]*n); p=[max(x,0) for x in r]; t=sum(p)
    return [x/t for x in p] if t>0 else [1/n]*n
def cfr(cards,h,pi):
    if isterm(h): return payoff(cards[0],cards[1],h)
    pl=player(h); k=N[cards[pl]]+":"+h.replace("f","") if False else N[cards[pl]]+":"+h
    A=acts(h); s=strat(k,len(A)); u=[]
    for a in A:
        p2=list(pi); p2[pl]*=s[A.index(a)]
        u.append(cfr(cards,h+a,p2))
    sign=1 if pl==0 else -1
    v=sum(si*ui for si,ui in zip(s,u))
    r=R[k]; ss=S.setdefault(k,[0.0]*len(A))
    for i in range(len(A)):
        r[i]+=pi[1-pl]*sign*(u[i]-v)   # CFR+ style floor below
        ss[i]+=pi[pl]*s[i]
    R[k]=[max(x,0) for x in r]
    return v
deals=list(itertools.permutations(CARDS,2))
for it in range(200000):
    for d in deals: cfr(d,"",[1.0,1.0])
avg={k:[x/sum(v) for x in v] for k,v in S.items()}
def sg(k): return avg[k]
def ev(c,h,pl_strats=None):pass
def play(cards,h,s0,s1):
    if isterm(h): return payoff(cards[0],cards[1],h)
    pl=player(h); k=N[cards[pl]]+":"+h
    A=acts(h); s=(s0,s1)[pl][k]
    return sum(s[i]*play(cards,h+a,s0,s1) for i,a in enumerate(A))
def value(s):
    return sum(play(d,"",s,s) for d in deals)/6
def br(player_br,opp):
    # best response value for player_br vs opp; enumerate pure strategies
    keys=[k for k in avg if player(k.split(":")[1])==player_br]
    best=-9
    for combo in itertools.product([0,1],repeat=len(keys)):
        pure={k:[1.0 if i==c else 0.0 for i in range(2)] for k,c in zip(keys,combo)}
        full=dict(opp);full.update(pure)
        v=sum(play(d,"",full,full) for d in deals)/6
        if player_br==1:v=-v
        best=max(best,v)
    return best
expl=(br(0,avg)+br(1,avg))/2
val=value(avg)
print("expl",expl,"value",val)
for k in sorted(avg):print(k,avg[k])
out={"format":"bluff-strategy-v1","game":"kuhn","producer":"CFR+ self-play, 200000 iterations, average strategy","probs":{}}
for k,v in avg.items():
    h=k.split(":")[1]; A=acts(h)
    out["probs"][k]={label(h,a):repr(p) for a,p in zip(A,v)}
json.dump(out,open("strategy.json","w"),indent(2) if False else None) if False else json.dump(out,open("strategy.json","w"),indent=2)

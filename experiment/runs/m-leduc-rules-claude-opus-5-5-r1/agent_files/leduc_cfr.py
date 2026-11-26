import json, itertools, sys, time
CARDS = ['Js','Jh','Qs','Qh','Ks','Kh']
RANK = {'J':0,'Q':1,'K':2}
RAISE = [2,4]

def round_info(h):
    """h: actions of a round. returns (ended, folded, raises, facing_bet)"""
    raises = h.count('r')
    if h and h[-1]=='f': return True, True, raises
    if len(h)>=2 and h[-1]=='c': return True, False, raises
    return False, False, raises

def legal(h):
    raises = h.count('r'); facing = bool(h) and h[-1]=='r'
    acts = []
    if facing: acts.append('f')
    acts.append('c')
    if raises < 2: acts.append('r')
    return acts

def contribs(h1, h2):
    c = [1,1]
    for rnd, h in ((0,h1),(1,h2)):
        for i,a in enumerate(h):
            p = i%2; o = 1-p
            if a=='c': c[p]=c[o]
            elif a=='r': c[p]=c[o]+RAISE[rnd]
    return c

def showdown(c0,c1,pub):
    r0,r1,rp = RANK[c0[0]],RANK[c1[0]],RANK[pub[0]]
    if r0==rp: return 1
    if r1==rp: return -1
    return (r0>r1)-(r0<r1)

# Build tree nodes per deal: node = (player, infokey, actions, children) or terminal payoff (to p0)
LBL = {'f':'fold','c':'call','r':'raise'}
def build(c, pub, h1, h2):
    rnd = 0 if pub is None else 1
    h = h1 if rnd==0 else h2
    ended, folded, _ = round_info(h)
    if ended:
        if folded:
            loser = (len(h)-1)%2
            con = contribs(h1,h2)
            return ('T', con[1] if loser==1 else -con[0])
        if rnd==0:
            return ('C', h1)  # chance placeholder
        con = contribs(h1,h2); s = showdown(c[0],c[1],pub)
        return ('T', s*con[1] if s>0 else (s*con[0] if s<0 else 0))
    p = len(h)%2
    key = f"{c[p]}:{h1}" if rnd==0 else f"{c[p]}|{pub}:{h1}/{h2}"
    acts = legal(h)
    kids = [build(c,pub,h1+a,h2) if rnd==0 else build(c,pub,h1,h2+a) for a in acts]
    return ('D', p, key, acts, kids)

def build_deal(c0,c1):
    rest = [x for x in CARDS if x not in (c0,c1)]
    def fill(node):
        if node[0]=='C':
            h1 = node[1]
            return ('P', [(1/4, build((c0,c1),pub,h1,'')) for pub in rest])
        if node[0]=='D':
            return ('D',node[1],node[2],node[3],[fill(k) for k in node[4]])
        return node
    return fill(build((c0,c1),None,'',''))

DEALS = [((c0,c1), build_deal(c0,c1)) for c0 in CARDS for c1 in CARDS if c0!=c1]
PDEAL = 1/30

regret = {}; strat_sum = {}; acts_of = {}
def collect(n):
    if n[0]=='D':
        acts_of[n[2]] = n[3]
        for k in n[4]: collect(k)
    elif n[0]=='P':
        for _,k in n[1]: collect(k)
for _,t in DEALS: collect(t)
for k,a in acts_of.items():
    regret[k]=[0.0]*len(a); strat_sum[k]=[0.0]*len(a)

acts_of_player = {}
def _cp(n):
    if n[0]=='D':
        acts_of_player[n[2]]=n[1]
        for k in n[4]: _cp(k)
    elif n[0]=='P':
        for _,k in n[1]: _cp(k)
for _,t in DEALS: _cp(t)
def cur(key):
    r = regret[key]; s = sum(x for x in r if x>0)
    if s<=0: return [1/len(r)]*len(r)
    return [max(x,0)/s for x in r]

def cfr(n, trav, rp, ro, w):
    # value to traverser; rp = traverser own reach, ro = opponent*chance reach
    t = n[0]
    if t=='T': return n[1] if trav==0 else -n[1]
    if t=='P': return sum(pr*cfr(k,trav,rp,ro*pr,w) for pr,k in n[1])
    _,p,key,acts,kids = n
    sg = SNAP[key]
    if p==trav:
        vs = [cfr(k,trav,rp*sg[i],ro,w) for i,k in enumerate(kids)]
        v = sum(a*b for a,b in zip(sg,vs))
        d = DELTA[key]; ss = strat_sum[key]
        for i in range(len(d)):
            d[i] += ro*(vs[i]-v)
            ss[i] += w*rp*sg[i]
        return v
    v = 0.0
    for i,k in enumerate(kids):
        v += sg[i]*cfr(k,trav,rp,ro*sg[i],w)
    return v

def avg():
    out={}
    for k,ss in strat_sum.items():
        s=sum(ss); out[k]=[x/s for x in ss] if s>0 else [1/len(ss)]*len(ss)
    return out

def ev(n, S):
    t=n[0]
    if t=='T': return n[1]
    if t=='P': return sum(pr*ev(k,S) for pr,k in n[1])
    sg=S[n[2]]
    return sum(sg[i]*ev(k,S) for i,k in enumerate(n[4]) if sg[i]>0)

def game_value(S): return sum(PDEAL*ev(t,S) for _,t in DEALS)

def best_response(S, br):
    # value to player br of best responding to the other player's strategy in S
    choice={}
    def val(n, ro):
        t=n[0]
        if t=='T': return ro*(n[1] if br==0 else -n[1])
        if t=='P': return sum(val(k,ro*pr) for pr,k in n[1])
        _,p,key,acts,kids=n
        if p==br: return val(kids[choice[key]], ro)
        sg=S[key]; return sum(val(k,ro*sg[i]) for i,k in enumerate(kids) if sg[i]>0)
    nodes={}
    def gather(n, ro):
        t=n[0]
        if t=='T': return
        if t=='P':
            for pr,k in n[1]: gather(k,ro*pr)
            return
        _,p,key,acts,kids=n
        if p==br:
            nodes.setdefault(key,[]).append((n,ro))
            for k in kids: gather(k,ro)
        else:
            sg=S[key]
            for i,k in enumerate(kids): gather(k,ro*sg[i])
    for _,t in DEALS: gather(t,PDEAL)
    def depth(k):
        h=k.split(':')[1]; return len(h.replace('/',''))+(10 if '/' in h else 0)
    for key in sorted(nodes, key=depth, reverse=True):
        na=len(acts_of[key]); tot=[0.0]*na
        for n,ro in nodes[key]:
            for i,k in enumerate(n[4]): tot[i]+=val(k,ro)
        choice[key]=max(range(na), key=lambda i: tot[i])
    return sum(val(t,PDEAL) for _,t in DEALS)

def exploitability(S):
    return (best_response(S,0)+best_response(S,1))/2

if __name__=='__main__':
    SNAP={}; DELTA={}
    iters=int(sys.argv[1]) if len(sys.argv)>1 else 2000
    t0=time.time()
    for it in range(1,iters+1):
        for trav in (0,1):

            SNAP = {k:cur(k) for k in regret}
            DELTA = {k:[0.0]*len(v) for k,v in regret.items()}
            for _,t in DEALS: cfr(t,trav,1.0,PDEAL,it)   # linear averaging
            for k,d in DELTA.items():
                if acts_of_player[k]==trav:
                    r=regret[k]
                    for i in range(len(r)): r[i]=max(r[i]+d[i],0.0)   # CFR+
        if it%250==0 or it==iters:
            S=avg(); print(it, round(time.time()-t0,1), 'expl', exploitability(S), 'v0', game_value(S), flush=True)
    S=avg()
    probs={k:{LBL[a]:repr(p) for a,p in zip(acts_of[k],S[k])} for k in sorted(S)}
    json.dump({"format":"bluff-strategy-v1","game":"leduc","producer":f"CFR+ (alternating updates, linear averaging), {iters} iterations, leduc_cfr.py","probs":probs},open('strategy.json','w'),indent=1)
    print('infosets',len(probs),'expl',exploitability(S),'v0',game_value(S))

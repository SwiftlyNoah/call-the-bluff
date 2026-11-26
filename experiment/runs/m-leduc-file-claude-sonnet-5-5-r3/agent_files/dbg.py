import sys
sys.argv=['x','0']
exec(open('solver.py').read().split("T = int")[0])
u={k:[1/len(v[1])]*len(v[1]) for k,v in IS.items()}
print(evalv(u), br(u,0), br(u,1))
# check pure fold/call
import collections
c=collections.Counter(depth[i] for i in N if N[i]['kind']=='decision')
print(c)
d=collections.defaultdict(set)
for i in N:
    if N[i]['kind']=='decision': d[N[i]['infoset']].add(depth[i])
print([ (k,v) for k,v in d.items() if len(v)>1][:5])
import json
p=json.load(open('strategy.json'))['probs']
sig={k:[float(p[k][l]) for l in IS[k][1]] for k in IS}
print(evalv(sig), br(sig,0), br(sig,1))
print(sum(sig['Js']), [ (k,sum(v)) for k,v in sig.items() if abs(sum(v)-1)>1e-9][:3])
# is root reached with all nodes in order once?
print(len(order), len(N))
n=N[g['root']]; print(sum(num(o['prob']) for o in n['outcomes']))

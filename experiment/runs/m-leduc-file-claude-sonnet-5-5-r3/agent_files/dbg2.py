import sys
sys.argv=['x','0']
exec(open('solver.py').read().split("T = int")[0])
import json
p=json.load(open('strategy.json'))['probs']
sig={k:[float(p[k][l]) for l in IS[k][1]] for k in IS}
bad=[(k,sum(v)) for k,v in sig.items() if abs(sum(v)-1)>1e-9]; print(bad[:3])
# pure p0 strategy: always first action; BR of p1 vs it
pure={k:([1.0]+[0.0]*(len(v[1])-1)) for k,v in IS.items()}
print(evalv(pure), br(pure,0), br(pure,1))
import re
src=open('solver.py').read().split("T = int")[0]
src=src.replace("choice[I] = max(range(len(tot)), key=lambda a: tot[a])","choice[I] = 0")
exec(src)
print(br(sig,0), br(sig,1))

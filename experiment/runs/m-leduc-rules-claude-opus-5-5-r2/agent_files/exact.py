import json
from fractions import Fraction
d = json.load(open('strategy.json'))
for k, v in d['probs'].items():
    acts = list(v); fr = [Fraction(float(v[a])) for a in acts]
    fr[-1] = 1 - sum(fr[:-1]); assert fr[-1] >= 0 and sum(fr) == 1, k
    d['probs'][k] = {a: str(f) for a, f in zip(acts, fr)}
json.dump(d, open('strategy.json', 'w'), indent=1)

"""uv run python dafny/attempt-b/tests/differential.py (from the repo root)

Differential: attempt-b core vs reference/br.py on LP and random profiles for every game file."""
import glob, json, random, subprocess, sys, time, os
from fractions import Fraction
sys.path.insert(0, "dafny/attempt-b/tests")
from test_attempt_b import random_profile, load
keys = ["value_p0", "br_value_p0", "br_value_p1", "nash_conv", "exploitability"]
import tempfile
S = tempfile.mkdtemp()
games = sorted(glob.glob("games/out/*.game.json") + glob.glob("games/out/other/*.game.json")
               + glob.glob("games/out/redblack/*.game.json"))
for g in games:
    game = load(g)
    profs = []
    # every shipped profile for this game: X.lp-strategy.json, X.ref-strategy.json,
    # X.<tag>.strategy.json next to the game, and other/eq/X.eq.strategy.json
    base = g[: -len(".game.json")]
    stem = os.path.basename(base)
    cands = sorted(set(glob.glob(base + ".*strategy.json") + glob.glob(base + ".*-strategy.json")
                       + glob.glob(os.path.join(os.path.dirname(g), "eq", stem + ".*strategy.json"))))
    for sp in cands:
        tag = os.path.basename(sp)[len(stem) + 1:].replace("strategy.json", "").strip(".-") or "file"
        profs.append((tag, sp))
    rp = os.path.join(S, os.path.basename(g) + ".rand.json")
    with open(rp, "w") as f:
        json.dump(random_profile(game, random.Random(7)), f)
    profs.append(("random", rp))
    for tag, sp in profs:
        t = time.time()
        a = subprocess.run(["uv", "run", "dafny/attempt-b/check.py", g, sp], capture_output=True, text=True)
        dt = time.time() - t
        r = subprocess.run(["uv", "run", "reference/br.py", g, sp, "--no-brute"], capture_output=True, text=True)
        A = json.loads(a.stdout); R = json.loads(r.stdout)
        ok = all(Fraction(A[k]) == Fraction(R[k]) for k in keys)
        print(f"{os.path.basename(g):32s} {tag:6s} nodes={len(game['nodes']):5d} core_s={dt:5.2f} "
              f"expl={float(Fraction(A['exploitability'])):.6g} agree={ok}")

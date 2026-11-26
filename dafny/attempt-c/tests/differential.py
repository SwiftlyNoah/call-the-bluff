"""Differential test: attempt C's verified core vs reference/br.py (independent, unverified).

For every game given (default: all games under games/out and tests/data that have perfect
recall), draw N random exact-rational profiles (seeded) plus every strategy file shipped next
to the game, run both checkers through their CLIs and require value_p0, br_value_p0,
br_value_p1 and nash_conv to agree exactly. It also checks the wrapper's label mapping of the returned pure best
responses: the profile in which p plays best_response_p (as labels) and the opponent keeps the
original strategy must, by reference/br.py's own evaluation, earn p exactly br_value_p.

uv run dafny/attempt-c/tests/differential.py [--n 5] [--seed 2520] [GAME ...]
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIELDS = ["value_p0", "br_value_p0", "br_value_p1", "nash_conv"]


def infosets(game):
    out = {}
    for node in game["nodes"]:
        if node["kind"] == "decision":
            out.setdefault(node["infoset"], [a["label"] for a in node["actions"]])
    return out


def random_profile(game, rng):
    probs = {}
    for name, acts in infosets(game).items():
        w = [rng.randint(0, 6) for _ in acts]
        if sum(w) == 0:
            w[rng.randrange(len(w))] = 1
        tot = sum(w)
        probs[name] = {a: str(Fraction(x, tot)) for a, x in zip(acts, w)}
    return {"format": "bluff-strategy-v1", "game": game["name"], "producer": "random", "probs": probs}


def br_attained(gpath, game, sp, verdict, p, tmp):
    """reference/br.py's value of p playing the reported pure best response must be br_value_p."""
    sdoc = json.loads(Path(sp).read_text())
    pure = verdict[f"best_response_p{p}"]
    probs = dict(sdoc["probs"])
    for name, acts in infosets(game).items():
        if name in pure:
            probs[name] = {a: ("1" if a == pure[name] else "0") for a in acts}
    out = tmp / f"{Path(sp).stem}.br{p}.strategy.json"
    out.write_text(json.dumps({**sdoc, "probs": probs}))
    ref, _ = run(["uv", "run", "reference/br.py", "--no-brute", "--quiet", str(gpath), str(out)])
    if ref.returncode != 0:
        return False
    v = Fraction(json.loads(ref.stdout)["value_p0"])
    return (v if p == 0 else -v) == Fraction(verdict[f"br_value_p{p}"])


def run(cmd):
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return p, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2520)
    ap.add_argument("games", nargs="*")
    args = ap.parse_args()
    games = [Path(g) for g in args.games] or sorted(
        list((ROOT / "games/out").rglob("*.game.json")) + list((HERE / "data").glob("*.game.json")))
    rng = random.Random(args.seed)
    total = fails = 0
    tmp = Path(tempfile.mkdtemp(prefix="attempt-c-diff-"))
    for gpath in games:
        game = json.loads(gpath.read_text())
        if game["name"] == "forgetful":
            continue  # not perfect recall: the core rejects it by design
        strategies = [p for p in gpath.parent.rglob("*.strategy.json")
                      if json.loads(p.read_text()).get("game") == game["name"]]
        for i in range(args.n):
            sp = tmp / f"{game['name']}.rand{i}.strategy.json"
            sp.write_text(json.dumps(random_profile(game, rng)))
            strategies.append(sp)
        for sp in strategies:
            mine, t_mine = run(["uv", "run", "dafny/attempt-c/check.py", str(gpath), str(sp)])
            ref, t_ref = run(["uv", "run", "reference/br.py", "--no-brute", "--quiet", str(gpath), str(sp)])
            total += 1
            if mine.returncode == 2 and ref.returncode == 2:
                print(f"ok   {gpath.name:34s} {sp.name:42s} both reject the input (exit 2)")
                continue
            if mine.returncode != 0 or ref.returncode != 0:
                fails += 1
                print(f"FAIL {gpath.name} {sp.name}: exit {mine.returncode}/{ref.returncode}\n"
                      f"  {mine.stderr.strip()} | {ref.stderr.strip()}")
                continue
            a, b = json.loads(mine.stdout), json.loads(ref.stdout)
            bad = [f for f in FIELDS if Fraction(a[f]) != Fraction(b[f])]
            bad += [f"best_response_p{p} not attained" for p in (0, 1)
                    if not br_attained(gpath, game, sp, a, p, tmp)]
            if bad or a["verified"] is not True:
                fails += 1
                print(f"FAIL {gpath.name} {sp.name}: {bad}")
            else:
                print(f"ok   {gpath.name:34s} {sp.name:42s} expl={a['exploitability_float']:.6g}"
                      f"  core {t_mine:.2f}s ref {t_ref:.2f}s")
    print(f"{total - fails}/{total} agree")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

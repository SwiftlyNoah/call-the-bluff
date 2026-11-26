"""Differential test of the attempt-a core against the independent reference/br.py.

Usage: uv run dafny/attempt-a/tests/differential.py [--seeds N]

For every game file in games/out (and this folder's own tiny games) and N random
exact-rational profiles per game (some rows with zeros), compares value_p0,
br_value_p0, br_value_p1, nash_conv and exploitability exactly. It also plays each
returned best response back through the reference (that player's rows replaced
by the one-hot best response) and checks it attains the reported value.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import threading
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE.parent
REPO = ATTEMPT.parent.parent
sys.path.insert(0, str(ATTEMPT))
import check  # noqa: E402

KEYS = ["value_p0", "br_value_p0", "br_value_p1", "nash_conv", "exploitability"]


def fmt(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def infosets(game: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in game["nodes"]:
        if n["kind"] == "decision":
            out.setdefault(n["infoset"], [a["label"] for a in n["actions"]])
    return out


def random_profile(game: dict, rng: random.Random) -> dict:
    probs = {}
    for name, acts in infosets(game).items():
        w = [rng.choice([0, 0, 1, 2, 3, 5, 7]) for _ in acts]
        if sum(w) == 0:
            w[rng.randrange(len(w))] = 1
        t = sum(w)
        probs[name] = {a: fmt(Fraction(x, t)) for a, x in zip(acts, w)}
    return {"format": "bluff-strategy-v1", "game": game["name"], "producer": "differential.py", "probs": probs}


def reference(game_path: Path, strat: dict, tmp: Path) -> dict:
    sp = tmp / "s.strategy.json"
    sp.write_text(json.dumps(strat))
    r = subprocess.run(
        ["uv", "run", "reference/br.py", "--quiet", "--no-brute", str(game_path), str(sp)],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout)


def mine(game_path: Path, strat: dict, tmp: Path) -> dict:
    sp = tmp / "m.strategy.json"
    sp.write_text(json.dumps(strat))
    return check.run(str(game_path), str(sp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    games = sorted((REPO / "games" / "out").glob("**/*.game.json"))
    games += [p for p in sorted((HERE / "games").glob("*.game.json")) if "forgetful" not in p.name]
    rng = random.Random(2520)
    fails = 0
    runs = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for gp in games:
            game = json.loads(gp.read_text())
            for _ in range(args.seeds):
                strat = random_profile(game, rng)
                a = mine(gp, strat, tmp)
                b = reference(gp, strat, tmp)
                runs += 1
                bad = [k for k in KEYS if Fraction(a[k]) != Fraction(b[k])]
                # Play each returned best response back through the reference.
                owner = {}
                for n in game["nodes"]:
                    if n["kind"] == "decision":
                        owner[n["infoset"]] = n["player"]
                for p in (0, 1):
                    br = a[f"best_response_p{p}"]
                    s2 = json.loads(json.dumps(strat))
                    for I, act in br.items():
                        s2["probs"][I] = {x: ("1" if x == act else "0") for x in s2["probs"][I]}
                    v = Fraction(reference(gp, s2, tmp)["value_p0"])
                    got = v if p == 0 else -v
                    if got != Fraction(a[f"br_value_p{p}"]):
                        bad.append(f"br{p} not attained ({fmt(got)} vs {a[f'br_value_p{p}']})")
                if bad:
                    fails += 1
                    print(f"MISMATCH {gp.name}: {bad}")
            print(f"ok {gp.relative_to(REPO)}")
    print(f"{runs} profiles, {fails} mismatches")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(1 << 29)
    res = {}
    t = threading.Thread(target=lambda: res.setdefault("rc", main()))
    t.start()
    t.join()
    sys.exit(res.get("rc", 1))

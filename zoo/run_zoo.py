"""Run the clean solvers and every mutant on the zoo's games, grade each output with an exact
best response, and write zoo/out/<game>/<row>.strategy.json plus zoo/out/summary.json.

Grading: reference/br.py (the spec task's independent exact BR) when it exists, with
zoo/exact_br.py as a second opinion; the two must agree exactly or the row is flagged.

    uv run zoo/make_mutants.py && uv run zoo/run_zoo.py [--jobs 4] [--only kuhn,leduc]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZOO = ROOT / "zoo"
OUT = ZOO / "out"
sys.path.insert(0, str(ZOO))

import cfr
import exact_br

# A row is "fooled" when every internal meter says converged while the exact exploitability is
# at least FOOLED_FACTOR times the exploitability meter's own tolerance.
FOOLED_FACTOR = 10


def game_path(name: str, preferred: Path, fallback: Path | None) -> Path:
    if preferred.exists():
        return preferred
    if fallback is not None and fallback.exists():
        return fallback
    raise SystemExit(f"no game file for {name}: {preferred}")


GAMES = {
    # name: (preferred file, fallback, max iterations)
    "kuhn": (ROOT / "games/out/kuhn.game.json", ZOO / "out/games/kuhn.game.json", 5000),
    "leduc": (ROOT / "games/out/leduc.game.json", ZOO / "out/games/leduc.game.json", 2000),
    # Same game with suits merged: the deal probabilities are no longer uniform.
    "leduc_ranks": (ZOO / "out/games/leduc.game.json", None, 4000),
    # Non-uniform chance at two depths, from the more-games task.
    "dice_bluff_2round": (ROOT / "games/out/other/dice_bluff_2round.game.json", None, 4000),
    # Red/Black (games/redblack): a dealt game, where the color swap is a true automorphism and the
    # deal is not uniform, and the gate endgame family, a fixed position whose class is not its own
    # color-swapped image (the game color_swap_imposed gets wrong). Each player decides several
    # times on one path, which is where reach bugs bite.
    "rb-h1": (ROOT / "games/out/redblack/rb-h1.game.json", None, 4000),
    "rb-21": (ROOT / "games/out/redblack/rb-21.game.json", None, 4000),
    "rb-12": (ROOT / "games/out/redblack/rb-12.game.json", None, 4000),
    "rb-pennies-inner": (ROOT / "games/out/redblack/rb-pennies-inner.game.json", None, 4000),
    "rb-pennies": (ROOT / "games/out/redblack/rb-pennies.game.json", None, 4000),
}

ROWS = [("clean_cfrplus", ZOO / "cfr.py", "cfrplus"), ("clean_cfr", ZOO / "cfr.py", "cfr")]
ROWS += [(p.stem, p, "cfrplus") for p in sorted((ZOO / "mutants").glob("*.py"))]
# Averaging bugs hide under CFR+, whose current strategy converges too; vanilla CFR's does not,
# so these also run under vanilla CFR, where the average is the only thing that converges.
ROWS += [(f"{n}@cfr", ZOO / "mutants" / f"{n}.py", "cfr")
         for n in ("avg_by_prior_mass", "avg_gated_on_opp_reach", "current_not_average")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_row(game: str, gpath: Path, iters: int, row: str, solver: Path, algo: str) -> dict:
    gdir = OUT / game
    gdir.mkdir(parents=True, exist_ok=True)
    strat = gdir / f"{row}.strategy.json"
    meters_file = gdir / f"{row}.meters.json"
    producer = f"{solver.relative_to(ROOT)} --algo {algo}"
    cmd = [sys.executable, str(solver), str(gpath), str(strat), "--algo", algo, "--iters", str(iters),
           "--check-every", "250", "--producer", producer, "--meters-out", str(meters_file)]
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    secs = time.time() - t0
    meters = json.loads(meters_file.read_text())
    meters_file.unlink()
    res = {"mutant": row, "game": game, "game_file": str(gpath.relative_to(ROOT)),
           "game_sha256": sha256(gpath), "solver": str(solver.relative_to(ROOT)), "algo": algo,
           "iterations": meters["iterations"], "seconds": round(secs, 1),
           "internal_meter": "self_exploitability (BR in the solver's own model of the game)",
           "internal_meter_value": meters["self_exploitability"],
           "regret_meter": meters["regret_meter"], "value_delta": meters["value_delta"],
           "self_value_p0": meters["self_value_p0"],
           "color_swap_gap": meters["color_swap_gap"],
           "internal_meter_says_converged": meters["converged"],
           "strategy_file": str(strat.relative_to(ROOT))}
    ref = ROOT / "reference/br.py"
    mine = exact_br.exploitability(str(gpath), str(strat))
    if ref.exists():
        p = subprocess.run([sys.executable, str(ref), str(gpath), str(strat), "--no-brute"],
                           capture_output=True, text=True, check=False)
        if p.returncode != 0:
            res["reference_br_error"] = p.stderr.strip()[-500:]
            exp, src = mine, "zoo/exact_br.py (reference/br.py failed)"
        else:
            verdict = json.loads(p.stdout)
            exp, src = Fraction(verdict["exploitability"]), "reference/br.py"
            res["value_p0"] = float(Fraction(verdict["value_p0"]))
            res["zoo_exact_br_agrees"] = exp == mine
    else:
        exp, src = mine, "zoo/exact_br.py"
    res["exploitability"] = f"{exp.numerator}/{exp.denominator}" if len(str(exp)) < 80 else None
    res["exploitability_float"] = float(exp)
    res["exploitability_source"] = src
    res["fooled"] = bool(meters["converged"] and float(exp) >= FOOLED_FACTOR * cfr.TOL_EXPLOIT)
    print(f"{game:18s} {row:28s} it={res['iterations']:5d} meter={res['internal_meter_value']:.2e} "
          f"conv={meters['converged']!s:5s} true={float(exp):.4f} fooled={res['fooled']}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", help="comma-separated game names")
    args = ap.parse_args()
    games = {k: v for k, v in GAMES.items() if not args.only or k in args.only.split(",")}
    jobs = []
    for g, (pref, fb, iters) in games.items():
        gp = game_path(g, pref, fb)
        for row, solver, algo in ROWS:
            jobs.append((g, gp, iters, row, solver, algo))
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(lambda j: run_row(*j), jobs))
    summary_path = OUT / "summary.json"
    old = json.loads(summary_path.read_text())["rows"] if summary_path.exists() and args.only else []
    keep = [r for r in old if r["game"] not in games]
    rows = keep + results
    summary = {
        "format": "bluff-zoo-summary-v1",
        "command": "uv run zoo/make_mutants.py && uv run zoo/run_zoo.py",
        "tolerances": {"self_exploitability": cfr.TOL_EXPLOIT, "regret_meter": cfr.TOL_REGRET,
                       "value_delta": cfr.TOL_DELTA},
        "fooled_rule": f"internal_meter_says_converged and exploitability >= {FOOLED_FACTOR} x {cfr.TOL_EXPLOIT}",
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {summary_path}: {len(rows)} rows, {sum(r['fooled'] for r in rows)} fooled")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pick, for every mutant, the one game that demos it best, graded by the verified checker.

A showcase row needs both halves at once: (a) the mutant's own internal meters say converged on
that game, and (b) `bin/bluff` (the compiled, verified Dafny core) certifies an exploitability at
least FOOLED_FACTOR times the self-BR meter's tolerance. Among a mutant's rows (CFR+ and, for the
averaging bugs, vanilla CFR) that pass both, the one with the largest certified exploitability
wins. A mutant with no such row gets `"showcase": null` and the row where it bites hardest, so
the deck never shows a bug the meters caught as if it fooled them.

Every row in zoo/out/summary.json is graded by bin/bluff, and its exploitability must equal the
summary's reference/br.py grade exactly, or this script fails.

    uv run zoo/make_mutants.py && uv run zoo/run_zoo.py && uv run zoo/showcase.py [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZOO = ROOT / "zoo"
sys.path.insert(0, str(ZOO))

import cfr
from run_zoo import FOOLED_FACTOR

LARGE = FOOLED_FACTOR * cfr.TOL_EXPLOIT


def bluff(game_file: str, strategy_file: str) -> dict:
    cmd = [str(ROOT / "bin/bluff"), "check", game_file, strategy_file]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise SystemExit(f"bin/bluff failed on {strategy_file}: {p.stderr.strip()[-400:]}")
    v = json.loads(p.stdout)
    if v.get("verified") is not True:
        raise SystemExit(f"bin/bluff verdict not verified on {strategy_file}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    summary = json.loads((ZOO / "out/summary.json").read_text())
    rows = summary["rows"]
    with ThreadPoolExecutor(args.jobs) as ex:
        verdicts = list(ex.map(lambda r: bluff(r["game_file"], r["strategy_file"]), rows))
    graded = []
    for r, v in zip(rows, verdicts):
        exp = Fraction(v["exploitability"])
        if float(exp) != r["exploitability_float"] or (
                r.get("exploitability") and Fraction(r["exploitability"]) != exp):
            raise SystemExit(f"bin/bluff and the summary disagree on {r['strategy_file']}")
        graded.append((r, v, exp))

    mutants = sorted({r["mutant"].split("@")[0] for r in rows} - {"clean_cfrplus", "clean_cfr"})
    out = []
    for m in mutants:
        mine = [(r, v, e) for r, v, e in graded if r["mutant"].split("@")[0] == m]
        fooled = [(r, v, e) for r, v, e in mine if r["internal_meter_says_converged"] and float(e) >= LARGE]
        entry: dict = {"mutant": m, "rows_run": len(mine),
                       "games_fooled": sorted({r["game"] + ("" if r["algo"] == "cfrplus" else " (vanilla CFR)")
                                               for r, _, _ in fooled})}
        if fooled:
            r, v, e = max(fooled, key=lambda t: (t[2], -len(t[0]["game"])))
            entry["showcase"] = {
                "game": r["game"], "algo": r["algo"], "row": r["mutant"],
                "game_file": r["game_file"], "strategy_file": r["strategy_file"],
                "iterations": r["iterations"],
                "meter": {"self_exploitability": r["internal_meter_value"],
                          "tolerance": cfr.TOL_EXPLOIT,
                          "regret_meter": r["regret_meter"], "value_delta": r["value_delta"],
                          "color_swap_gap": r.get("color_swap_gap"),
                          "says_converged": True},
                "bluff": {"exploitability": v["exploitability"], "exploitability_float": float(e),
                          "value_p0": v["value_p0"], "br_value_p0": v["br_value_p0"],
                          "br_value_p1": v["br_value_p1"], "verified": v["verified"]},
                "certified_over_meter": float(e) / r["internal_meter_value"] if r["internal_meter_value"] else None,
                "command": f"bin/bluff check {r['game_file']} {r['strategy_file']}",
            }
        else:
            r, v, e = max(mine, key=lambda t: t[2])
            entry["showcase"] = None
            entry["why_not"] = ("no game where the meters say converged and bin/bluff certifies "
                                f">= {LARGE:g}; where it bites hardest the meters already say not converged")
            entry["bites_hardest"] = {"game": r["game"], "algo": r["algo"], "row": r["mutant"],
                                      "meter_self_exploitability": r["internal_meter_value"],
                                      "says_converged": r["internal_meter_says_converged"],
                                      "bluff_exploitability_float": float(e),
                                      "command": f"bin/bluff check {r['game_file']} {r['strategy_file']}"}
        out.append(entry)

    clean = {r["game"]: float(e) for r, v, e in graded if r["mutant"] == "clean_cfrplus"}
    doc = {
        "format": "bluff-zoo-showcase-v1",
        "command": "uv run zoo/make_mutants.py && uv run zoo/run_zoo.py && uv run zoo/showcase.py",
        "rule": (f"(a) all three internal meters say converged and (b) bin/bluff (verified: true) "
                 f"certifies exploitability >= {LARGE:g} ({FOOLED_FACTOR} x the self-BR tolerance "
                 f"{cfr.TOL_EXPLOIT:g}); the largest certified exploitability wins"),
        "rows_graded_by_bluff": len(graded),
        "bluff_agrees_with_reference_br": True,
        "clean_cfrplus_exploitability_by_game": clean,
        "mutants": out,
    }
    (ZOO / "out/showcase.json").write_text(json.dumps(doc, indent=1) + "\n")
    for e in out:
        s = e["showcase"]
        if s:
            print(f"{e['mutant']:28s} {s['game']:18s} {s['algo']:8s} meter={s['meter']['self_exploitability']:.2e} "
                  f"bluff={s['bluff']['exploitability_float']:.4f} fooled_on={','.join(e['games_fooled'])}")
        else:
            b = e["bites_hardest"]
            print(f"{e['mutant']:28s} none (bites hardest on {b['game']} {b['algo']}: "
                  f"meter={b['meter_self_exploitability']:.2e} conv={b['says_converged']} "
                  f"bluff={b['bluff_exploitability_float']:.4f})")
    print(f"graded {len(graded)} rows with bin/bluff; wrote zoo/out/showcase.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

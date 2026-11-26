"""Cross-check the verified checker (bin/bluff) against the reference (reference/br.py).

Usage (from the repo root):
    uv run reference/crosscheck.py [--jobs N] [--only SUBSTR] [--no-lp] [--no-negative]
                                   [--json OUT.json] [--table]

For every game file in games/out/**, zoo/out/games and redblack/out/games, and every strategy
file that belongs to it (games/out LP, reference and eq profiles, the Red/Black mini LP and CFR
profiles, every zoo/out mutant and clean CFR profile, every redblack/out served and guarded
profile), this runs BOTH command-line tools as a user would, as separate processes:

    bin/bluff check GAME STRATEGY
    uv run reference/br.py GAME STRATEGY --quiet

and compares value_p0, br_value_p0, br_value_p1, nash_conv and exploitability as exact
rationals (and as strings, to catch formatting drift). It also checks, per pair:

* the checker's verdict is schema-valid and says "verified": true;
* the checker's pure best responses, evaluated from scratch by the reference evaluator
  (reference/br.py expected_value), attain the checker's br_value_p0 and br_value_p1;
* exit codes agree (both 0, or both 2 on a malformed input).

Anchors, per game: reference/lp.py --no-write gives an exactly certified game value; it must
equal known_value_p0 when the game file states one, and every profile graded at exploitability
exactly 0 must have value_p0 equal to it (the value of a zero-sum game is unique). Profiles that
claim to be equilibria (LP, eq and closed-form ref files) must grade at exactly 0.

Negative cases: a few corrupted inputs (row off by 1e-2, missing infoset, chance probabilities
not summing to 1, wrong game name) must be rejected with exit 2 by both tools.

Exit 0 when everything matches, 1 otherwise. Nothing is written unless --json is given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.set_int_max_str_digits(0)  # exact verdict fractions can run to many thousands of digits
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))

FIELDS = ["value_p0", "br_value_p0", "br_value_p1", "nash_conv", "exploitability"]
TIMEOUT = 1800


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


# ---------------------------------------------------------------- discovery


def discover() -> tuple[list[Path], list[tuple[Path, Path, str]], list[str]]:
    """All game files, all (game, strategy, kind) pairs, and any strategy files left unpaired."""
    games: list[Path] = []
    pairs: list[tuple[Path, Path, str]] = []
    paired: set[Path] = set()

    def add(game: Path, strat: Path, kind: str) -> None:
        pairs.append((game, strat, kind))
        paired.add(strat)

    for g in sorted((ROOT / "games/out").rglob("*.game.json")):
        games.append(g)
        stem = g.name[: -len(".game.json")]
        for suffix, kind in ((".lp-strategy.json", "lp"), (".ref-strategy.json", "ref"),
                             (".lp.strategy.json", "lp"), (".cfr.strategy.json", "cfr")):
            s = g.parent / (stem + suffix)
            if s.exists():
                add(g, s, kind)
        s = g.parent / "eq" / f"{stem}.eq.strategy.json"
        if s.exists():
            add(g, s, "eq")

    for g in sorted((ROOT / "zoo/out/games").glob("*.game.json")):
        games.append(g)
    summary = ROOT / "zoo/out/summary.json"
    if summary.exists():
        for row in json.loads(summary.read_text())["rows"]:
            s = ROOT / row["strategy_file"]
            g = ROOT / row["game_file"]
            if s.exists() and g.exists():
                add(g, s, "zoo:" + row["mutant"])

    rb_games = ROOT / "redblack/out/games"
    for g in sorted(rb_games.glob("*.game.json")):
        games.append(g)
    for s in sorted((ROOT / "redblack/out/strategies").glob("*.strategy.json")):
        cls, kind = s.name[: -len(".strategy.json")].rsplit(".", 1)
        g = rb_games / f"{cls}.game.json"
        if g.exists():
            add(g, s, "rb:" + kind)

    # Agent experiment trials: the graded (filled) strategy against the trial's game file.
    for res in sorted((ROOT / "experiment/runs").glob("*/result.json")):
        try:
            gf = json.loads(res.read_text()).get("game_file")
        except ValueError:
            continue
        s = res.parent / "graded_strategy_filled.json"
        if gf and s.exists() and (ROOT / gf).exists():
            add(ROOT / gf, s, "agent:" + res.parent.name)

    all_strats = set()
    for base in ("games/out", "zoo/out", "redblack/out"):
        for pattern in ("*.strategy.json", "*-strategy.json"):
            all_strats.update((ROOT / base).rglob(pattern))
    unpaired = sorted(rel(s) for s in all_strats - paired)
    return games, pairs, unpaired


# ---------------------------------------------------------------- one pair


def run(cmd: list[str]) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        code, out, err = -1, "", f"timeout after {TIMEOUT} s"
    return code, out, err, time.perf_counter() - t0


def checker_cmd(game: str, strat: str) -> list[str]:
    return [str(ROOT / "bin/bluff"), "check", game, strat]


def reference_cmd(game: str, strat: str) -> list[str]:
    return ["uv", "run", "--quiet", "python", "reference/br.py", game, strat, "--quiet"]


def check_pair(game: str, strat: str, kind: str) -> dict:
    from br import expected_value, load_strategy
    from validate_game import load_game, schema_errors

    rec: dict = {"game_file": game, "strategy_file": strat, "kind": kind, "problems": []}
    cc, cout, cerr, ct = run(checker_cmd(game, strat))
    rc, rout, rerr, rt = run(reference_cmd(game, strat))
    rec.update(checker_exit=cc, checker_seconds=round(ct, 2), reference_exit=rc,
               reference_seconds=round(rt, 2))
    if cc != rc:
        rec["problems"].append(f"exit codes differ: checker {cc} ({cerr.strip()[:300]}), "
                               f"reference {rc} ({rerr.strip()[:300]})")
    if cc != 0 or rc != 0:
        rec["checker_stderr"] = cerr.strip()[:500]
        rec["reference_stderr"] = rerr.strip()[:500]
        rec["match"] = cc == rc
        return rec
    cv, rv = json.loads(cout), json.loads(rout)
    rec["checker"] = {k: cv.get(k) for k in FIELDS}
    rec["reference"] = {k: rv.get(k) for k in FIELDS}
    errs = schema_errors(cv, "verdict.schema.json")
    if errs:
        rec["problems"].append("checker verdict fails schema: " + "; ".join(errs[:3]))
    if cv.get("verified") is not True:
        rec["problems"].append("checker verdict not marked verified")
    for k in FIELDS:
        a, b = cv.get(k), rv.get(k)
        if a is None or b is None or Fraction(a) != Fraction(b):
            rec["problems"].append(f"{k}: checker {a} != reference {b}")
        elif a != b:
            rec["problems"].append(f"{k}: same rational, different text ({a!r} vs {b!r})")
    if Fraction(cv["exploitability"]) != Fraction(cv["nash_conv"]) / 2:
        rec["problems"].append("checker exploitability != nash_conv / 2")
    # The checker's best responses, graded by the reference's own evaluator.
    g = load_game(ROOT / game)
    sigma = load_strategy(g, ROOT / strat)
    for p in (0, 1):
        brs = cv[f"best_response_p{p}"]
        own = {n.infoset for n in g.nodes.values() if n.kind == "decision" and n.player == p}
        if set(brs) != own:
            rec["problems"].append(f"best_response_p{p} infosets differ from player {p}'s")
            continue
        got = expected_value(g, {p: brs, 1 - p: sigma}, p)
        if got != Fraction(cv[f"br_value_p{p}"]):
            rec["problems"].append(f"best_response_p{p} attains {got}, not br_value_p{p}")
    rec["match"] = not rec["problems"]
    return rec


def lp_value(game: str) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(["uv", "run", "--quiet", "python", "reference/lp.py", "--no-write", game],
                       cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
    rec = {"game_file": game, "seconds": round(time.perf_counter() - t0, 2), "exit": p.returncode}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode == 0 and lines:
        rec.update(json.loads(lines[-1]))
    else:
        rec["error"] = (p.stderr or p.stdout).strip()[-400:]
    return rec


# ---------------------------------------------------------------- negative cases


def negative_cases() -> list[dict]:
    out = []
    kuhn = ROOT / "games/out/kuhn.game.json"
    kstrat = ROOT / "games/out/kuhn.ref-strategy.json"
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        sdoc = json.loads(kstrat.read_text())
        gdoc = json.loads(kuhn.read_text())
        cases = []

        bad = json.loads(json.dumps(sdoc))
        inf = next(iter(bad["probs"]))
        act = next(iter(bad["probs"][inf]))
        bad["probs"][inf][act] = str(Fraction(bad["probs"][inf][act]) + Fraction(1, 100))
        cases.append(("row sums to 1.01", kuhn, bad, None))

        bad = json.loads(json.dumps(sdoc))
        del bad["probs"][inf]
        cases.append(("missing infoset", kuhn, bad, None))

        bad = json.loads(json.dumps(sdoc))
        bad["game"] = "not-kuhn"
        cases.append(("wrong game name", kuhn, bad, None))

        badg = json.loads(json.dumps(gdoc))
        ch = next(n for n in badg["nodes"] if n["kind"] == "chance")
        ch["outcomes"][0]["prob"] = str(Fraction(ch["outcomes"][0]["prob"]) + Fraction(1, 100))
        cases.append(("chance probs sum to 1.01", None, sdoc, badg))

        badg = json.loads(json.dumps(gdoc))
        dec = next(n for n in badg["nodes"] if n["kind"] == "decision")
        dec["actions"] = list(reversed(dec["actions"]))
        cases.append(("one infoset node lists actions in another order", None, sdoc, badg))

        for i, (name, game, sdoc_i, gdoc_i) in enumerate(cases):
            sp = tdp / f"s{i}.strategy.json"
            sp.write_text(json.dumps(sdoc_i))
            if gdoc_i is not None:
                gp = tdp / f"g{i}.game.json"
                gp.write_text(json.dumps(gdoc_i))
            else:
                gp = game
            cc, _, cerr, _ = run(checker_cmd(str(gp), str(sp)))
            rc, _, rerr, _ = run(reference_cmd(str(gp), str(sp)))
            out.append({"case": name, "checker_exit": cc, "reference_exit": rc,
                        "ok": cc == 2 and rc == 2,
                        "checker_reason": cerr.strip().splitlines()[-1][:200] if cerr.strip() else "",
                        "reference_reason": rerr.strip().splitlines()[-1][:200] if rerr.strip() else ""})
    return out


# ---------------------------------------------------------------- report


EQ_KINDS = {"lp", "eq", "ref"}


def anchors(games: list[Path], results: list[dict], lps: dict[str, dict]) -> list[dict]:
    rows = []
    for g in games:
        gf = rel(g)
        doc_known = json.loads(g.read_text()).get("known_value_p0")
        lp = lps.get(gf)
        lp_val = lp.get("value_p0") if lp and lp.get("certified_exact") else None
        zero_vals = sorted({r["checker"]["value_p0"] for r in results
                            if r["game_file"] == gf and r.get("checker")
                            and Fraction(r["checker"]["exploitability"]) == 0},
                           key=Fraction)
        eq_bad = [r["strategy_file"] for r in results
                  if r["game_file"] == gf and r["kind"] in EQ_KINDS
                  and (not r.get("checker") or Fraction(r["checker"]["exploitability"]) != 0)]
        problems = []
        vals = {Fraction(v) for v in zero_vals}
        if lp is not None and lp_val is None:
            problems.append("LP oracle did not certify an exact value")
        if lp_val is not None:
            vals.add(Fraction(lp_val))
        if doc_known is not None:
            vals.add(Fraction(doc_known))
        if len(vals) > 1:
            problems.append("values disagree: " + ", ".join(sorted(str(v) for v in vals)))
        if eq_bad:
            problems.append("claimed equilibria not at exploitability 0: " + ", ".join(eq_bad))
        rows.append({"game_file": gf, "known_value_p0": doc_known, "lp_value_p0": lp_val,
                     "lp_seconds": lp.get("seconds") if lp else None,
                     "zero_exploitability_values": zero_vals, "ok": not problems,
                     "problems": problems})
    return rows


def short(x: str | None, n: int = 28) -> str:
    if x is None:
        return "n/a"
    return x if len(x) <= n else x[: n - 3] + "..."


def table(results: list[dict]) -> str:
    out = ["| Game | Strategy | Kind | Checker exploitability | Reference exploitability | Match "
           "| Checker s | Reference s |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        ce = r["checker"]["exploitability"] if r.get("checker") else f"exit {r['checker_exit']}"
        re_ = r["reference"]["exploitability"] if r.get("reference") else f"exit {r['reference_exit']}"
        fl = ""
        if r.get("checker") and Fraction(ce) != 0:
            fl = f" (~{float(Fraction(ce)):.3g})"
        out.append(f"| `{r['game_file']}` | `{Path(r['strategy_file']).name}` | {r['kind']} "
                   f"| {short(ce)}{fl} | {short(re_)} | {'yes' if r['match'] else '**NO**'} "
                   f"| {r['checker_seconds']} | {r['reference_seconds']} |")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="only pairs whose game or strategy path contains this")
    ap.add_argument("--no-lp", action="store_true", help="skip the LP oracle anchors")
    ap.add_argument("--no-negative", action="store_true")
    ap.add_argument("--json", help="write every record to this file")
    ap.add_argument("--table", action="store_true", help="print the markdown table")
    args = ap.parse_args(argv)

    games, pairs, unpaired = discover()
    if args.only:
        pairs = [p for p in pairs if args.only in rel(p[0]) or args.only in rel(p[1])]
        games = [g for g in games if args.only in rel(g) or any(p[0] == g for p in pairs)]
    # Biggest games first, so the long jobs do not end up last.
    pairs.sort(key=lambda p: -p[0].stat().st_size)
    t0 = time.perf_counter()
    results: list[dict] = []
    lps: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(check_pair, rel(g), rel(s), k): "pair" for g, s, k in pairs}
        if not args.no_lp:
            for g in sorted(games, key=lambda g: -g.stat().st_size):
                futs[ex.submit(lp_value, rel(g))] = "lp"
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # a crash here is a crosscheck bug: report, never hide
                print(f"crosscheck job crashed: {e!r}", file=sys.stderr)
                results.append({"game_file": "?", "strategy_file": "?", "kind": "?",
                                "match": False, "problems": [f"crash: {e!r}"],
                                "checker_exit": None, "reference_exit": None,
                                "checker_seconds": None, "reference_seconds": None})
                continue
            if futs[f] == "lp":
                lps[r["game_file"]] = r
            else:
                results.append(r)
                tag = "ok " if r["match"] else "BAD"
                print(f"{tag} {r['checker_seconds']:>7}s {r['strategy_file']}", file=sys.stderr)
    results.sort(key=lambda r: (r["game_file"], r["strategy_file"]))
    anchor_rows = anchors(games, results, lps)
    neg = [] if args.no_negative else negative_cases()
    wall = time.perf_counter() - t0

    bad = [r for r in results if not r["match"]]
    bad_anchor = [a for a in anchor_rows if not a["ok"]]
    bad_neg = [n for n in neg if not n["ok"]]
    if args.table:
        print(table(results))
        print()
    print(f"pairs: {len(results)}, exact matches: {len(results) - len(bad)}, mismatches: {len(bad)}")
    for r in bad:
        print(f"  MISMATCH {r['strategy_file']} on {r['game_file']}: {'; '.join(r['problems'])}")
    print(f"games: {len(anchor_rows)}, anchors consistent: {len(anchor_rows) - len(bad_anchor)}")
    for a in bad_anchor:
        print(f"  ANCHOR {a['game_file']}: {'; '.join(a['problems'])}")
    for gf in ("games/out/kuhn.game.json", "games/out/redblack/rb-pennies.game.json",
               "games/out/leduc.game.json"):
        a = next((x for x in anchor_rows if x["game_file"] == gf), None)
        if a:
            print(f"  anchor {gf}: known {a['known_value_p0']}, LP {short(a['lp_value_p0'], 60)}, "
                  f"exploitability-0 profiles' values {[short(v, 60) for v in a['zero_exploitability_values']]}")
    if neg:
        print(f"negative cases rejected by both with exit 2: {len(neg) - len(bad_neg)}/{len(neg)}")
        for n in bad_neg:
            print(f"  NEGATIVE {n['case']}: checker {n['checker_exit']}, reference {n['reference_exit']}")
    if unpaired:
        print(f"strategy files with no game paired: {len(unpaired)}")
        for u in unpaired:
            print(f"  {u}")
    checker_total = sum(r["checker_seconds"] or 0 for r in results)
    print(f"checker CPU-serial total {checker_total:.1f} s; wall {wall:.1f} s with {args.jobs} jobs")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"pairs": results, "anchors": anchor_rows, "lp": lps, "negative": neg,
             "unpaired": unpaired, "wall_seconds": round(wall, 1)}, indent=1))
    return 0 if not (bad or bad_anchor or bad_neg) else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main(sys.argv[1:]))

"""judge-core: run reference/br.py, attempt-a/b/c and bin/bluff on 37 (game, strategy) pairs; compare exact fields, time each.

Usage: uv run dafny/core/tests/judge_pairs.py [ref,a,b,c,core]   (writes judge_rows.json next to this file)
"""
import glob
import json
import os
import subprocess
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(R)
CHECKERS = {
    "ref": ["uv", "run", "--quiet", "reference/br.py"],
    "a": ["uv", "run", "--quiet", "dafny/attempt-a/check.py"],
    "b": ["uv", "run", "--quiet", "dafny/attempt-b/check.py"],
    "c": ["uv", "run", "--quiet", "dafny/attempt-c/check.py"],
    "core": ["bin/bluff", "check"],
}
FIELDS = ["value_p0", "br_value_p0", "br_value_p1", "nash_conv", "exploitability"]
pairs = [
    ("games/out/kuhn.game.json", "games/out/kuhn.ref-strategy.json"),
    ("games/out/kuhn.game.json", "games/out/kuhn.lp-strategy.json"),
]
pairs += [("games/out/kuhn.game.json", s) for s in sorted(glob.glob("zoo/out/kuhn/*.strategy.json"))]
pairs += [("games/out/leduc.game.json", "games/out/leduc.lp-strategy.json")]
pairs += [("games/out/leduc.game.json", f"zoo/out/leduc/{m}.strategy.json") for m in
          ["clean_cfrplus", "chance_ignored", "opp_reach_omitted", "current_not_average", "info_leak_key"]]
for g in ["liars_dice_1x4", "dice_bluff_2round", "goofspiel4_fixed"]:
    pairs.append((f"games/out/other/{g}.game.json", f"games/out/other/{g}.lp-strategy.json"))
for g in ["rb-12", "rb-21", "rb-pennies", "rb-pennies-leaky", "rb-h1", "rb-pennies-inner"]:
    for kind in ["lp", "cfr"]:
        s = f"games/out/redblack/{g}.{kind}.strategy.json"
        if os.path.exists(s):
            pairs.append((f"games/out/redblack/{g}.game.json", s))
only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CHECKERS)
rows = []
for g, s in pairs:
    row = {"game": g, "strategy": s}
    for k in only:
        t = time.perf_counter()
        p = subprocess.run(CHECKERS[k] + [g, s], capture_output=True, text=True, check=False)
        dt = time.perf_counter() - t
        out = None
        if p.returncode == 0:
            try:
                out = {f: json.loads(p.stdout)[f] for f in FIELDS}
            except (ValueError, KeyError) as e:
                out = {"parse_error": str(e)}
        row[k] = {"rc": p.returncode, "s": round(dt, 2), "out": out, "err": p.stderr.strip()[-200:] if p.returncode else ""}
    ref = row.get("ref", {}).get("out")
    row["agree"] = {k: (row[k]["out"] == ref) for k in only if k != "ref"}
    rows.append(row)
    print(json.dumps({"g": os.path.basename(g), "s": os.path.basename(s),
                      **{k: (row[k]["rc"], row[k]["s"]) for k in only}, "agree": row["agree"],
                      "expl": (ref or {}).get("exploitability")}), flush=True)
with open(os.path.join(os.path.dirname(__file__), "judge_rows.json"), "w") as fh:
    json.dump(rows, fh, indent=1)

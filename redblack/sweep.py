#!/usr/bin/env python3
"""Grade the served Red/Black pack on every enumerable endgame class.

    uv run redblack/sweep.py --layer 2 --layer 3 [--workers 2]

For each non-root class at the given post-discard totals T', builds the real
subgame (extract.py) and computes, with extract.py's exact Fraction best
response (a cross-check, not the verified checker):

    G           the real game below the class: exploitability of the served
                profile, raw pack and with the deployed dominance guards
    G'_stored   round 1 exact, inner boundaries cut to the successor's STORED
                vTypes (the prices the policy pass solved against): what the
                solver's own meter can see
    G'_served   inner boundaries cut to the successor's exact value under the
                SERVED profile (vTypes non-uniqueness, without the belief reset)

Game and strategy files are not written here (T'=3 games are up to ~54 MB
each); write the ones you want with extract.py. Output:
redblack/out/summary_T<layers>.json, one row per class.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import extract as X


def grade(cid: str) -> dict:
    t0 = time.time()
    row: dict = {"class": cid}
    try:
        g = X.extract(cid, guards=True, merge=False, max_nodes=3_000_000, do_br=True, write=False)
        row.update({
            "layer": g["layer"], "nodes": g["nodes"], "infosets": g["infosets"], "type_pairs": g["type_pairs"],
            "inner_boundaries": g["inner_boundaries"], "classes_played": len(g["classes_played"]),
            "epsRound_solver": (g["solver_meters"] or {}).get("epsRound"),
            "value_G": g["served"]["value_p0_float"],
            "expl_G": g["served"]["exploitability_float"],
            "expl_G_exact": g["served"]["exploitability"],
            "value_G_guarded": g["guarded"]["value_p0_float"],
            "expl_G_guarded": g["guarded"]["exploitability_float"],
        })
        if g["inner_boundaries"]:
            for src in ("stored", "served"):
                gp = X.extract(cid, guards=False, merge=False, max_nodes=3_000_000, do_br=True, gprime=src,
                               write=False)
                row[f"value_Gp_{src}"] = gp["served"]["value_p0_float"]
                row[f"expl_Gp_{src}"] = gp["served"]["exploitability_float"]
    except Exception as e:  # noqa: BLE001 - record and keep going
        row["error"] = f"{type(e).__name__}: {e}"
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, action="append", required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    ids = (X.OUT / "classes_le3.txt").read_text().split()
    cids = [c for c in ids if not c.startswith("dr") and X.rounds.post_total(X.parse_class_id(c)) in args.layer]
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(grade, cids):
            rows.append(r)
            print(json.dumps({k: r.get(k) for k in ("class", "nodes", "expl_G", "expl_G_guarded",
                                                    "expl_Gp_stored", "expl_Gp_served", "error", "seconds")}),
                  flush=True)
    rows.sort(key=lambda r: r["class"])
    tag = "".join(map(str, sorted(args.layer)))
    out = X.OUT / f"summary_T{tag}.json"
    out.write_text(json.dumps({"pack": X.PACK_PREFIX, "classes": rows}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

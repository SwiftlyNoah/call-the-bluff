"""Brackets each game's value exactly and checks the stated known values.

For each game: solve the sequence-form LP in floats (scipy HiGHS), round both players' behavioral
strategies to rationals, then compute EXACT best responses (gamelib.best_response, Fraction):
    lower = -BR_1(x)  <=  value_p0  <=  BR_0(y) = upper.
When lower == upper, the rounded profile is an exact Nash equilibrium (exploitability exactly 0)
and the value is certified, up to trust in gamelib.best_response (unverified Python; the Dafny
checker can re-certify the profile, written to games/out/other/eq/).

Run from the repo root:  uv run python games/other/solve.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gamelib import OUT_DIR, load_game, lp_bracket

EQ_DIR = OUT_DIR / "eq"


def write_strategy(game: dict, profile: dict) -> Path:
    EQ_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "format": "bluff-strategy-v1",
        "game": game["name"],
        "producer": "games/other/solve.py (scipy HiGHS sequence-form LP, rounded to rationals)",
        "probs": {iset: {a: str(p) for a, p in row.items()} for iset, row in sorted(profile.items())},
    }
    path = EQ_DIR / f"{game['name']}.eq.strategy.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def main() -> None:
    ok = True
    for path in sorted(OUT_DIR.glob("*.game.json")):
        game = load_game(path)
        t0 = time.perf_counter()
        b = lp_bracket(game)
        dt = time.perf_counter() - t0
        exact = b["lower"] == b["upper"]
        known = game.get("known_value_p0")
        status = ""
        if known is not None:
            from fractions import Fraction

            agrees = b["lower"] <= Fraction(known) <= b["upper"]
            ok &= agrees
            status = f" known {known}: {'consistent' if agrees else 'CONTRADICTED'}"
        if exact:
            out = write_strategy(game, b["profile"])
            print(
                f"{game['name']}: value {b['lower']} EXACT (rounded with max denominator {b['max_den']}, "
                f"exploitability 0; LP float {b['lp_float']:.12g}; {dt:.1f}s; eq -> {out.relative_to(OUT_DIR)}){status}"
            )
        else:
            print(
                f"{game['name']}: value in [{b['lower']}, {b['upper']}] "
                f"(~[{float(b['lower']):.9f}, {float(b['upper']):.9f}], LP float {b['lp_float']:.12g}; {dt:.1f}s){status}"
            )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

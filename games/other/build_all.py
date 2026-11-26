"""Writes every game in games/other/ to games/out/other/ and prints a stats table.

Run from the repo root:  uv run python games/other/build_all.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dice_bluff
import goofspiel
import liars_dice
import matrix_games
from gamelib import best_response, uniform_profile, validate, write_game

BUILDERS = matrix_games.GAMES + liars_dice.GAMES + goofspiel.GAMES + dice_bluff.GAMES


def all_games() -> list[dict]:
    return [b() for b in BUILDERS]


def main() -> None:
    cols = "name nodes terminal chance decision infosets_p0 infosets_p1 max_depth chance_depths known_value br_seconds"
    print(" | ".join(cols.split()))
    for doc in all_games():
        stats = validate(doc)
        path = write_game(doc)
        t0 = time.perf_counter()
        prof = uniform_profile(doc)
        best_response(doc, prof, 0)
        best_response(doc, prof, 1)
        dt = time.perf_counter() - t0
        row = [doc["name"]] + [str(stats[k]) for k in cols.split()[1:9]]
        row += [doc.get("known_value_p0", "-"), f"{dt:.3f}"]
        print(" | ".join(row), f"  -> {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


if __name__ == "__main__":
    main()

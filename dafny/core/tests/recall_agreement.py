"""adapter.py's perfect recall refusal against reference/validate_game.py on random games.

Usage: uv run dafny/core/tests/recall_agreement.py [N] [SEED]   (defaults 3000, 11)

Uses the generator in fuzz.py (this folder), which writes perfect-recall games and games
whose infoset keys forget the player's own actions (often absent-minded). For each game,
the adapter's build_game must accept exactly when validate_game.py does.
"""

import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "dafny/core/tests"), str(REPO / "dafny/core"), str(REPO / "reference")]
import adapter
import fuzz


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    rng = random.Random(seed)
    D, _, G = adapter.load_core()
    st = {"games": 0, "validator_accepts": 0, "adapter_accepts": 0, "disagree": 0}
    for i in range(n):
        for pr_mode in (True, False):
            g = fuzz.gen_game(rng, pr_mode, f"a{i}")
            v = fuzz.has_perfect_recall(g)
            try:
                adapter.build_game(json.loads(json.dumps(g)), D, G)
                a = True
            except adapter.Malformed:
                a = False
            st["games"] += 1
            st["validator_accepts"] += v
            st["adapter_accepts"] += a
            st["disagree"] += a != v
    print(json.dumps(st))
    return 1 if st["disagree"] else 0


if __name__ == "__main__":
    sys.exit(main())

"""Malformed inputs check.py must reject with exit 2 (and a deep tree it must handle).

Usage: uv run dafny/attempt-a/tests/malformed.py

Each case is a game or strategy that breaks CONTRACTS.md. Before review-core-a, the first
five were accepted with a "verified": true verdict and the sixth crashed with a traceback
(exit 1). The last case is a 20000-deep chain, which took minutes before BuildBR was made
linear; it must finish and give value_p0 1.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE.parent / "check.py"


def game(nodes, root=0):
    return {"format": "bluff-game-v1", "name": "t", "root": root, "nodes": nodes}


def strat(probs):
    return {"format": "bluff-strategy-v1", "game": "t", "producer": "malformed.py", "probs": probs}


def term(i, x):
    return {"id": i, "kind": "terminal", "payoff_p0": x}


def dec(i, player, acts):
    return {"id": i, "kind": "decision", "player": player, "infoset": "I",
            "actions": [{"label": lab, "child": c} for lab, c in acts]}


AB = strat({"I": {"a": "1/2", "b": "1/2"}})
CLEAN = game([dec(0, 0, [("a", 1), ("b", 2)]), term(1, "1"), term(2, "0")])

CASES = [
    ("duplicate action labels", game([dec(0, 0, [("a", 1), ("a", 2)]), term(1, "1"), term(2, "0")]),
     strat({"I": {"a": "1/2"}})),
    ("player is a boolean", game([dec(0, True, [("a", 1), ("b", 2)]), term(1, "1"), term(2, "0")]), AB),
    ("unreachable node", game([dec(0, 0, [("a", 1), ("b", 2)]), term(1, "1"), term(2, "0"), term(3, "5")]), AB),
    ("chance probability 0",
     game([{"id": 0, "kind": "chance", "outcomes": [{"label": "x", "prob": "0", "child": 1},
                                                    {"label": "y", "prob": "1", "child": 2}]},
           term(1, "1"), term(2, "0")]), strat({})),
    ("duplicate JSON key in strategy", CLEAN,
     '{"format":"bluff-strategy-v1","game":"t","producer":"x",'
     '"probs":{"I":{"a":"1/2","b":"1/2"},"I":{"a":"1","b":"0"}}}'),
    ("outcome is not an object", game([{"id": 0, "kind": "chance", "outcomes": [5]}, term(1, "1")]), strat({})),
]


def run(tmp: Path, g, s) -> subprocess.CompletedProcess:
    gp, sp = tmp / "g.game.json", tmp / "s.strategy.json"
    gp.write_text(g if isinstance(g, str) else json.dumps(g))
    sp.write_text(s if isinstance(s, str) else json.dumps(s))
    return subprocess.run([sys.executable, str(CHECK), str(gp), str(sp)], capture_output=True, text=True)


def main() -> int:
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, g, s in CASES:
            r = run(tmp, g, s)
            ok = r.returncode == 2 and "Traceback" not in r.stderr
            fails += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}: exit {r.returncode}: {r.stderr.strip()[:100]}")
        r = run(tmp, CLEAN, AB)
        ok = r.returncode == 0 and json.loads(r.stdout)["br_value_p0"] == "1"
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} clean control game: exit {r.returncode}")
        n = 20000
        nodes = []
        for i in range(n):
            nodes.append({"id": i, "kind": "decision", "player": i % 2, "infoset": f"I{i}",
                          "actions": [{"label": "a", "child": i + 1}, {"label": "b", "child": n + 1 + i}]})
            nodes.append(term(n + 1 + i, "0"))
        nodes.append(term(n, "1"))
        t0 = time.time()
        r = run(tmp, game(nodes), strat({f"I{i}": {"a": "1", "b": "0"} for i in range(n)}))
        dt = time.time() - t0
        ok = r.returncode == 0 and json.loads(r.stdout)["value_p0"] == "1" and dt < 60
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {n}-deep chain: exit {r.returncode}, {dt:.1f} s")
    print(f"{fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

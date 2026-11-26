"""Regression tests for the red team findings against bin/bluff (task redteam-fix, 2026-11-20).

Usage: uv run dafny/core/tests/regressions.py [--slow]

Each test is named for the bug it pins (board/redteam/adapter.md, model.md, proof.md). Before
the fix, every one of them failed: a crash with exit 1, a verified verdict on a file the
contract rejects, or a tampered compiled core that still printed "verified": true. --slow adds
the resource cases (a chance node with 1,100,000 outcomes, chains at and past MAX_DEPTH), which
take minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = HERE.parent
ADAPTER = CORE / "adapter.py"
sys.path.insert(0, str(CORE))
import adapter


def game(nodes, name="t", root=0, **extra):
    return {"format": "bluff-game-v1", "name": name, "root": root, "nodes": nodes, **extra}


def strat(probs, name="t", **extra):
    return {"format": "bluff-strategy-v1", "game": name, "producer": "regressions.py", "probs": probs, **extra}


def term(i, x):
    return {"id": i, "kind": "terminal", "payoff_p0": x}


def dec(i, player, infoset, acts):
    return {"id": i, "kind": "decision", "player": player, "infoset": infoset,
            "actions": [{"label": lab, "child": c} for lab, c in acts]}


def chance(i, outs):
    return {"id": i, "kind": "chance", "outcomes": [{"label": lab, "prob": q, "child": c} for lab, q, c in outs]}


ONE = game([dec(0, 0, "I", [("a", 1), ("b", 2)]), term(1, "1"), term(2, "0")])
ONE_S = strat({"I": {"a": "1/2", "b": "1/2"}})


def chain(n: int, player=lambda k: 0):
    """An n-deep chain with a distinct infoset per level (perfect recall): 'c' continues, 't' stops at 0."""
    nodes = []
    for k in range(n):
        nodes.append(dec(2 * k, player(k), f"i{k}", [("t", 2 * k + 1), ("c", 2 * k + 2)]))
        nodes.append(term(2 * k + 1, "0"))
    nodes.append(term(2 * n, "1"))
    return game(nodes), strat({f"i{k}": {"t": "0", "c": "1"} for k in range(n)})


class Runner:
    def __init__(self, tmp: Path):
        self.tmp, self.fails, self.n = tmp, 0, 0

    def run(self, g, s, adapter_path=ADAPTER, env=None, raw=False):
        gp, sp = self.tmp / "g.game.json", self.tmp / "s.strategy.json"
        for p, d in ((gp, g), (sp, s)):
            if isinstance(d, bytes):
                p.write_bytes(d)
            else:
                p.write_text(d if isinstance(d, str) else json.dumps(d, ensure_ascii=raw is False))
        r = subprocess.run([sys.executable, str(adapter_path), str(gp), str(sp)],
                           capture_output=True, text=True, env=env, check=False)
        return r

    def expect(self, name: str, r, code: int, check=None, stderr_has: str | None = None):
        self.n += 1
        ok = r.returncode == code and "Traceback" not in r.stderr
        if ok and stderr_has is not None:
            ok = stderr_has in r.stderr
        if ok and check is not None:
            try:
                ok = bool(check(json.loads(r.stdout)))
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"     check raised {type(e).__name__}: {e}")
        self.fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: exit {r.returncode} {r.stderr.strip()[:110]}")


def tests_fast(t: Runner):
    # redteam-adapter 1: an exact value with more than 4300 digits crashed str() (exit 1).
    # Valid 15-level p0 chain, rows "1.0"/"1e-300": value (1/(10^300 + 1))^15.
    g, _ = chain(15)
    s = strat({f"i{k}": {"t": "1.0", "c": "1e-300"} for k in range(15)})
    want = adapter.fmt(Fraction(1, 10**300 + 1) ** 15)
    t.expect("int_str_digit_limit_on_exact_value", t.run(g, s), 0, lambda v: v["value_p0"] == want)
    # ... and input numbers over MAX_NUMBER_CHARS are refused with exit 2, not a crash.
    big = "1" * (adapter.MAX_NUMBER_CHARS + 1)
    t.expect("huge_number_string_refused",
             t.run(game([dec(0, 0, "I", [("a", 1), ("b", 2)]), term(1, big), term(2, "0")]), ONE_S), 2)
    t.expect("huge_json_integer_refused",
             t.run('{"format":"bluff-game-v1","name":"t","root":' + big + ',"nodes":[]}', ONE_S), 2)
    # 5000-digit payoff: legal, graded exactly.
    p5000 = "1" * 5000
    t.expect("5000_digit_payoff_graded",
             t.run(game([dec(0, 0, "I", [("a", 1), ("b", 2)]), term(1, p5000), term(2, "0")]), ONE_S), 0,
             lambda v: v["br_value_p0"] == p5000)

    # redteam-adapter 2: exploitability_float overflowed on a valid 10^400 payoff (exit 1) ...
    g = game([dec(0, 0, "I", [("a", 1), ("b", 2)]), term(1, "1" + "0" * 400), term(2, "0")])
    t.expect("float_overflow_in_exploitability_float", t.run(g, strat({"I": {"a": "0", "b": "1"}})), 0,
             lambda v: v["exploitability_float"] == sys.float_info.max
             and v["exploitability"] == adapter.fmt(Fraction(10**400, 2)))
    # ... and the row-sum rejection message overflowed float() on "1e999" (exit 1, should be 2).
    t.expect("float_overflow_in_row_sum_message", t.run(ONE, strat({"I": {"a": "1e999", "b": "0"}})), 2,
             stderr_has="not 1 within 1e-6")

    # redteam-adapter 3: a Latin-1 file crashed (UnicodeDecodeError, exit 1).
    t.expect("non_utf8_input_refused",
             t.run(json.dumps(game([dec(0, 0, "I", [("a", 1), ("b", 2)]), term(1, "1"), term(2, "0")],
                                   name="café"), ensure_ascii=False).encode("latin-1"),
                   strat({"I": {"a": "1", "b": "0"}}, name="café")), 2)
    # ... and under a Latin-1 locale UTF-8 files decoded as mojibake (best-response keys garbled).
    spade = "J♠:"
    g = game([dec(0, 0, spade, [("a", 1), ("b", 2)]), term(1, "1"), term(2, "0")])
    env = dict(os.environ, LC_ALL="en_US.ISO8859-1", LANG="en_US.ISO8859-1", PYTHONUTF8="0")
    t.expect("utf8_decoding_ignores_locale",
             t.run(g, strat({spade: {"a": "1/2", "b": "1/2"}}), env=env, raw=True), 0,
             lambda v: v["best_response_p0"] == {spade: "a"})

    # redteam-adapter 6, redteam-model 2: keys outside the schema were dropped silently, so a
    # payoff on an inner node or a prob on a decision edge gave a verified, unintended verdict.
    g = game([chance(0, [("x", "1", 1)]), dec(1, 0, "I", [("a", 2), ("b", 3)]), term(2, "0"), term(3, "1")])
    g["nodes"][0]["payoff_p0"] = "5"
    t.expect("payoff_on_inner_node_refused", t.run(g, ONE_S), 2, stderr_has="payoff_p0")
    g = game([dec(0, 0, "I", [("a", 1), ("b", 2)]), term(1, "1"), term(2, "0")])
    g["nodes"][0]["actions"][0]["prob"] = "1/2"
    t.expect("prob_on_decision_edge_refused", t.run(g, ONE_S), 2, stderr_has="prob")
    t.expect("unknown_top_level_game_key_refused", t.run(game(ONE["nodes"], bogus=1), ONE_S), 2)
    t.expect("unknown_strategy_key_refused", t.run(ONE, strat(ONE_S["probs"], extra=1)), 2)

    # redteam-adapter 7: a missing or non-string game name passed the name check ("game": null).
    noname = {k: v for k, v in ONE.items() if k != "name"}
    t.expect("missing_game_name_refused", t.run(noname, {**ONE_S, "game": None}), 2)
    t.expect("list_game_name_refused", t.run({**ONE, "name": [1]}, {**ONE_S, "game": [1]}), 2)
    t.expect("empty_game_name_refused", t.run({**ONE, "name": ""}, {**ONE_S, "game": ""}), 2)

    # redteam-adapter 10: a null or list row said "is missing" although the key was present.
    t.expect("null_row_message", t.run(ONE, strat({"I": None})), 2, stderr_has="is not an object")
    t.expect("list_row_message", t.run(ONE, strat({"I": ["a"]})), 2, stderr_has="is not an object")

    # redteam-adapter 5, redteam-model 3, redteam-proof 2: imperfect-recall games got exit 0
    # (and absent-minded ones a misleading core refusal); CONTRACTS.md section 1 requires
    # perfect recall, as reference/validate_game.py checks.
    forget = game([chance(0, [("x", "1/2", 1), ("y", "1/2", 2)]),
                   dec(1, 0, "A", [("l", 3), ("r", 4)]), dec(2, 0, "A2", [("l", 5), ("r", 6)]),
                   dec(3, 0, "B", [("u", 7), ("d", 8)]), dec(4, 0, "B", [("u", 9), ("d", 10)]),
                   term(5, "0"), term(6, "0"), term(7, "1"), term(8, "0"), term(9, "0"), term(10, "1")])
    fs = strat({n: {a: "1/2" for a in ("l", "r")} for n in ("A", "A2")} | {"B": {"u": "1/2", "d": "1/2"}})
    t.expect("forgetting_own_action_refused", t.run(forget, fs), 2, stderr_has="perfect recall")
    driver = game([dec(0, 0, "D", [("exit", 1), ("cont", 2)]), term(1, "0"),
                   dec(2, 0, "D", [("exit", 3), ("cont", 4)]), term(3, "4"), term(4, "1")])
    t.expect("absent_minded_driver_refused", t.run(driver, strat({"D": {"exit": "1/2", "cont": "1/2"}})), 2,
             stderr_has="perfect recall")
    # Control: a perfect-recall game where both B nodes share p0's history is still accepted.
    ok = game([chance(0, [("x", "1/2", 1), ("y", "1/2", 2)]),
               dec(1, 0, "A", [("l", 3), ("r", 4)]), dec(2, 0, "A", [("l", 5), ("r", 6)]),
               dec(3, 1, "B", [("u", 7), ("d", 8)]), term(4, "0"), dec(5, 1, "B", [("u", 9), ("d", 10)]),
               term(6, "0"), term(7, "1"), term(8, "-1"), term(9, "-1"), term(10, "1")])
    t.expect("perfect_recall_control_accepted",
             t.run(ok, strat({"A": {"l": "1", "r": "0"}, "B": {"u": "1/2", "d": "1/2"}})), 0,
             lambda v: v["value_p0"] == "0")

    # redteam-adapter 4: an edit to the compiled core left the stamp valid and printed
    # exploitability "0", verified true, for Kuhn players_off_by_one (true value 11/24).
    with tempfile.TemporaryDirectory() as td:
        S = Path(td)
        for f in CORE.glob("*.dfy"):
            shutil.copy(f, S)
        shutil.copy(ADAPTER, S)
        shutil.copytree(CORE / "build", S / "build", ignore=shutil.ignore_patterns("__pycache__"))
        repo = CORE.parent.parent
        kg = (repo / "games/out/kuhn.game.json").read_text()
        ks = (repo / "zoo/out/kuhn/players_off_by_one.strategy.json").read_text()
        t.expect("untampered_copy_grades_off_by_one", t.run(kg, ks, adapter_path=S / "adapter.py"), 0,
                 lambda v: v["exploitability"] == "11/24")
        ck = S / "build/checker-py/Checker.py"
        src = ck.read_text()
        assert "(d_4_nc_) / (_dafny.BigRational('2e0'))" in src, "Checker.py changed shape: update this test"
        ck.write_text(src.replace("(d_4_nc_) / (_dafny.BigRational('2e0'))", "_dafny.BigRational(0)"))
        t.expect("tampered_compiled_core_refused", t.run(kg, ks, adapter_path=S / "adapter.py"), 1,
                 stderr_has="differs from what build.sh produced")


def tests_slow(t: Runner):
    # redteam-proof 1: the compiled BRV used one stack frame per list item, so a chance node with
    # 1,100,000 outcomes hit RecursionError (exit 1). Scan is now a loop.
    n = 1_100_000
    nodes = [chance(0, [(f"o{i}", f"1/{n}", i + 1) for i in range(n)])]
    nodes += [term(i + 1, "1" if i == 0 else "0") for i in range(n)]
    t.expect("wide_chance_node_1100000_outcomes", t.run(game(nodes), strat({})), 0,
             lambda v: v["value_p0"] == f"1/{n}")
    # redteam-adapter 9: deep trees ended in RecursionError (exit 1). MAX_DEPTH levels grade,
    # deeper ones are refused with exit 2 before the core runs.
    d = adapter.MAX_DEPTH
    g, s = chain(d)
    t.expect(f"chain_at_max_depth_{d}_graded", t.run(g, s), 0, lambda v: v["value_p0"] == "1")
    g, s = chain(d, player=lambda k: k % 2)
    t.expect(f"alternating_chain_at_max_depth_{d}_graded", t.run(g, s), 0, lambda v: v["value_p0"] == "1")
    g, s = chain(d + 1)
    t.expect(f"chain_past_max_depth_{d + 1}_refused", t.run(g, s), 2, stderr_has="deeper than")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", action="store_true")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        t = Runner(Path(td))
        tests_fast(t)
        if args.slow:
            tests_slow(t)
    print(f"{t.n - t.fails} of {t.n} passed, {t.fails} failures")
    return 1 if t.fails else 0


if __name__ == "__main__":
    sys.exit(main())

# Judging the three verified cores (task judge-core, W1)

Winner: **attempt A**, copied unchanged (all four `.dfy` files byte-identical) to `dafny/core/`,
with a hardened adapter, a hash-checked build, and A's tests retargeted at the core. Nothing was
grafted from B or C: their better parts (B's runtime attainment check, C's completeness theorem)
belong to different game models and cannot be moved without redoing the proofs.

Every number below was run by the judge on 2026-11-20 (Dafny 4.11.0, this Mac); commands are in
the last section and in `board/claims.md` under `judge-core`.

## Score table (1 to 5, higher is better)

| Criterion | A: inductive tree | B: flat node table | C: sequence form |
| --- | --- | --- | --- |
| (1) Strength and honesty of the proof | 4 | 4 | 5 |
| (2) Correctness on real games | 5 | 5 | 5 |
| (3) Runtime | 5 | 4 | 2 |
| (4) Simplicity for a 9-minute talk | 4 | 3 | 2 |
| **Total** | **18** | **16** | **14** |

## (1) What is actually proved (read from the code, not the READMEs)

All three: no `assume`, `{:axiom}`, `{:verify false}`, `{:extern}`, `{:only}` or `decreases *`
(grep), and `dafny audit` reports 0 findings for each. Exact `real` arithmetic throughout. The
only attribute anywhere is C's `{:isolate_assertions}`.

| | Verified obligations | Lines of Dafny | Upper bound (pure and behavioral) | Attainment | Completeness |
| --- | --- | --- | --- | --- | --- |
| A | 93, 0 errors | 1065 | proved, no perfect recall needed (`BRTheorem` 1a, 1b) | proved conditional on the runtime predicate `Consistent` (the infoset choices agree); `Check` returns `Fail` otherwise | not proved |
| B | 135, 0 errors | 1113 | proved, no perfect recall needed (`BehavioralUpperBound`, `PureUpperBound`) | `Check` evaluates the returned pure strategy exactly and returns `Err` unless it equals the bound | not proved |
| C | 352, 0 errors | 1729 | proved, no perfect recall needed (`BRUpperBound`, `PureUpperBound`) | proved under `PerfectRecall` (`BRAttained`), and also checked exactly at run time | proved: `Check` returns `Ok` on every well-formed perfect-recall game |

Every `Ok` verdict of every attempt is sound: value_p0 is the profile's exact value, each br value
is the maximum over all pure and behavioral strategies and is attained by the returned pure
strategy, and nash_conv = br0 + br1 >= 0 with exploitability = nash_conv / 2. A states this as
`CheckSound`, B as the `Sound` postcondition of `Check`, C as `Check`'s postconditions.

The difference is only what happens on inputs where attainment fails. A and B fail closed but do
not prove they never fail on a perfect-recall game; C proves it, at the price of a hand-written
`PerfectRecall`/`PRZ` definition that joins the trusted specification. For grading strategies
this gap is about liveness, not soundness: a refusal can never be a wrong verdict. It is why C
gets the extra point and why the talk should say "attainment is checked, fail closed" for A.

A's statements require a valid row at every infoset, including the opponent's (`ValidPure`,
`ValidBeh`); those rows are never read, so nothing is lost, but the wording is stronger than
needed.

## (2) Correctness on real games

37 (game, strategy) pairs, each run through `reference/br.py` and all three wrappers, plus
`bin/bluff` after the copy: Kuhn with `kuhn.ref-strategy.json` and `kuhn.lp-strategy.json`, all
15 zoo strategies in `zoo/out/kuhn/`, Leduc LP plus 5 zoo mutants from `zoo/out/leduc/`, the LP
profiles of liars_dice_1x4, dice_bluff_2round and goofspiel4_fixed, and every LP and CFR profile
in `games/out/redblack/`. **All four checkers agree exactly with the reference on all five exact
fields in all 37 pairs.** Kuhn ref and LP profiles give value_p0 -1/18 and exploitability 0 in every checker; every LP
profile has exploitability exactly 0; the zoo mutant `players_off_by_one` on Kuhn has
exploitability exactly 11/24.

The two largest Red/Black stretch games (`redblack/out/games/d0-2.0.0-2.0.0`, 223633 nodes, and
`d0-2.1.1-2.0.0`, 111793 nodes, each with its `served` strategy) also agree exactly across the
reference and all three attempts (exploitability about 0.1598 and 0.2571).

Imperfect recall (outside the contract, but a test of fail-closed behavior): on A's `forgetful`
game all three refuse with exit 2. On C's `forgetful` game (player 0 forgets its first move), C
refuses, while A and B return a verdict with br_value_p0 = 1, which is correct: the pure strategy
(L, l) earns 1, the maximum payoff. So A and B are sound there and more complete than C in
practice, despite C's completeness theorem being stated only for perfect recall.

The core's own tests: `dafny/core/tests/malformed.py` (6 malformed inputs exit 2, a clean
control, a 20000-deep chain in 1.9 s): 0 failures. `dafny/core/tests/differential.py --seeds 3`:
57 random exact profiles over 19 games, 0 mismatches against `reference/br.py`, including replaying
each returned best response, 55 s.

## (3) Runtime (wall seconds, whole process including `uv run` start-up, one run each unless noted)

| Game (nodes) | reference/br.py | A | B | C |
| --- | --- | --- | --- | --- |
| Kuhn, ref profile | 0.26 | 0.07 | 0.07 | 0.07 |
| Leduc LP (9451) | 1.26 | 0.37 | 0.45 | 4.50 |
| Leduc, 5 zoo mutants | 1.32 to 1.62 | 0.39 to 0.50 | 0.45 to 0.53 | 4.58 to 5.09 |
| liars_dice_1x4 LP (8181) | 1.06 | 0.18 | 0.38 | 3.37 |
| rb-12 LP (4707) | 0.68 | 0.16 | 0.28 | 2.65 |
| rb-21 CFR (4707) | 0.67 | 0.17 | 0.26 | 2.57 |
| Red/Black d0-2.1.1-2.0.0 served (111793) | 13.12 | 2.25 | 3.83 | 98.92 |
| Red/Black d0-2.0.0-2.0.0 served (223633) | 24.03 | 4.15 | 7.93 | 182.98 |

`bin/bluff` (the winner, through the hash check): Leduc LP 0.34 to 0.36 s over 3 runs, the
223633-node game 4.48 s. Verification wall time: A 7.5 s, B 8.8 s, C 14.9 s (`--cores 2`).

C's compiled filters slice sequences recursively and recompute V per level, so it is 10x to 44x
slower than A; that would hurt the Red/Black stretch, which checks many large endgame classes.

## (4) Simplicity for the talk

A is one inductive `Node` datatype and one recursion (`BRV`) over a weighted to-expand list with
infoset grouping; the theorem fits on one slide (`CheckSound`). B mirrors the JSON as a flat
table and needs a height array as its termination measure, which is an extra concept to explain.
C is the most elegant mathematically (expected utility as a bilinear sum over realization
weights) but needs sequence form, realization plans and a perfect-recall predicate before the
theorem means anything: too much for 9 minutes.

## Decision and what the core is

- `dafny/core/{Math,Game,BestResponse,Check}.dfy`: attempt A, unchanged (93 obligations).
- `dafny/core/adapter.py`: A's `check.py`, hardened: contract number grammar (games: integer,
  fraction, finite decimal; strategies also accept `repr` exponent form, exponent capped at 3
  digits), rejects a strategy whose `game` field does not match or that names unknown infosets,
  and refuses to run a compiled core whose recorded source hash does not match the `.dfy` files.
- `dafny/core/build.sh`: verify, compile to Python, write `build/checker-py/SOURCE_SHA256`.
- `bin/audit`: greps the core's `.dfy` files for `assume`, `{:axiom}`, `{:verify false}`,
  `{:extern}` (the adapter is Python, so it is outside the grep by construction), requires
  `dafny audit` to report 0 findings, then runs `build.sh` (dafny verify plus compile).
- `bin/bluff check GAME STRATEGY`: CONTRACTS.md section 4.

Trusted base of the winner: `adapter.py` (JSON parsing, numbering, normalization, printing), the
claim that A's `Node`/`Val`/`PVal` model means what CONTRACTS.md says a game file means, Dafny,
Z3, the Python compiler and runtime, and the game files.

## Commands

```sh
# (1) proofs and audit
grep -nE '\bassume\b|\{:[[:space:]]*axiom|verify[[:space:]]+false|\{:[[:space:]]*extern|decreases \*|\{:only' dafny/attempt-{a,b,c}/*.dfy
(cd dafny/attempt-a && dafny verify Math.dfy Game.dfy BestResponse.dfy Check.dfy --cores 2)   # 93 verified
(cd dafny/attempt-b && dafny verify Model.dfy BestResponse.dfy Check.dfy --cores 2)           # 135 verified
(cd dafny/attempt-c && dafny verify Model.dfy SeqForm.dfy Proofs.dfy Recall.dfy Check.dfy --cores 2)  # 352 verified
(cd dafny/attempt-X && dafny audit *.dfy)                                                     # 0 findings each
bin/audit                                                                                     # PASS, 93 verified, 9.7 s
# (2) and (3): one process per checker, timed
uv run reference/br.py G S; uv run dafny/attempt-{a,b,c}/check.py G S; bin/bluff check G S
uv run dafny/core/tests/malformed.py
uv run dafny/core/tests/differential.py --seeds 3
```

The 37-pair comparison is `uv run dafny/core/tests/judge_pairs.py` (all five checkers, about 2 minutes); rerun any row with the
commands above and compare `value_p0`, `br_value_p0`, `br_value_p1`, `nash_conv`, `exploitability`.

## Addendum (2026-11-22): the core after the red team

The judging above describes attempt A as it was copied on 2026-11-20 (93 obligations). The W2
red-team fixer then changed the core (commit 89a8b73): `dafny verify` on `dafny/core` now reports
**103 verified, 0 errors**, and two of the four files are no longer byte-identical to attempt A.
The changes close the red team's findings (a payoff on an inner node is rejected instead of
dropped, the compiled build is tied to its verified source hash); `board/redteam/*.md` and the
regression tests in `dafny/core/tests/` record each one. The scores and the choice of A stand.

# The verified core

Attempt A (see `../JUDGING.md` for why), plus one compiled-form change made by task redteam-fix
(2026-11-20, below). An inductive game tree over exact `real`s; `BRV` in `BestResponse.dfy`
specifies a best response by expanding a weighted node list and grouping each player's decision
nodes by infoset. `Check.dfy` holds the checker and the theorems: `BRTheorem` (no pure or
behavioral strategy beats the value; the returned pure strategy attains it whenever the run-time
`Consistent` check passes, else `Check` fails closed) and `CheckSound` (everything an `Ok` verdict
guarantees: the game is well formed with exact chance distributions (`Sane`), the profile is an
exact distribution at every infoset, value_p0 is the profile's value, each br value is an upper
bound over every pure and behavioral strategy and is attained by the returned pure strategy, and
nash_conv = br0 + br1 >= 0).

Attainment is proved only for `Ok` verdicts. That perfect recall makes `Consistent` hold is not
proved here (fuzz-tested only: no perfect-recall game has been refused); the adapter refuses games
without perfect recall before the core runs, so a core refusal on a valid file would be a liveness
bug, never a wrong number.

## What runs

`BR` is specified by `BRV` but computed by `Scan` and `BRG` (`function ... by method`), proved
equal to `BRV` by `ScanEq`, `BRGEq` and `BranchesEq`. `Scan` handles terminals, chance and
opponent nodes with every recursive call in tail position, so Dafny compiles it to a loop; the
compiled `BRV` had used one Python stack frame per list item and crashed on a chance node with
1,100,000 outcomes. The theorems are unchanged and still stated for `BRV`. 103 obligations verify.

- `bin/audit`: forbidden-construct grep, `dafny audit`, then `build.sh` (verify, compile to
  `build/checker-py`, stamp it with `SOURCE_SHA256` for the `.dfy` files and `BUILD_SHA256` for
  every compiled file).
- `bin/bluff check GAME STRATEGY`: runs `adapter.py` (trusted, unverified), which refuses to run
  unless both stamps match and imports the compiled core from source, never from cached bytecode.

## The adapter (trusted)

`adapter.py` parses strict UTF-8 JSON (whatever the locale), refuses every key the schemas do
not allow, requires a non-empty game name on both files, checks perfect recall exactly as
`reference/validate_game.py` defines it (each node of an infoset has the same sequence of its
player's own (infoset, action) pairs), numbers infosets and actions, and normalizes each strategy
row (sum within 1e-6, then divided by its exact sum). Exit 0 with a verdict; exit 2 on malformed
or contract-breaking input with the reason on stderr; exit 1 on an internal error, never with a
verdict.

Limits (exit 2 beyond them): number strings and JSON integer literals of at most 10,000
characters (`MAX_NUMBER_CHARS`); trees at most 300,000 levels deep (`MAX_DEPTH`, because the
compiled `Val`, `WFN`, `Sane` and `BRG` recurse once per level). Exact output has no digit limit.
`exploitability_float` is a convenience: the nearest double, clamped to the largest finite double
and 0.0 below about 5e-324; use the exact `exploitability` for anything that matters.

Known performance limit: `BRV` (and so `BRG`) splits the pending responder nodes with `Same` and
`Other`, one scan of the pending list per infoset, so a layer with N distinct responder infosets
pending at once costs O(N^2) (redteam-model: 49.68 s at N = 4,000 versus 3.35 s for
`reference/br.py`). Fixing it means a proved one-pass grouping; not done.

## Tests

- `uv run dafny/core/tests/malformed.py`: malformed inputs exit 2; a 20,000-deep chain grades.
- `uv run dafny/core/tests/regressions.py [--slow]`: one test per red-team finding, named for the
  bug; `--slow` adds the 1,100,000-outcome chance node and chains at and past `MAX_DEPTH`.
- `uv run dafny/core/tests/differential.py --seeds 3`: exact agreement with `reference/br.py`.

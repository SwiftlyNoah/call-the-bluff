# Call the Bluff

A checker for strategies in imperfect-information games whose answer comes with a proof, and
what it found when pointed at solvers written by AI agents and at the live bot of a card game.

Noah Brauner, CS 2520r (Harvard, fall 2026). The report is `report/main.pdf`. The game is
Red/Black, playable against the bot at <https://www.playredblack.com>.

## The idea

A solver that says "converged" is reporting on its own model of the game. If the model is wrong,
or the solver has a bug, it converges anyway. So this project does not verify solvers. It grades
their output: given a game file and a strategy file, `bin/bluff` computes exactly how much a
perfect opponent could win against that strategy (its exploitability), in exact rational
arithmetic, with a best-response core that is proved correct in Dafny. The grade does not depend
on who or what wrote the solver.

## What is here

| Path | What |
| --- | --- |
| `CONTRACTS.md` | the file formats (game, strategy, verdict), the CLI and the core theorem. Read first |
| `schema/` | JSON schemas for the three file formats |
| `dafny/core/` | the verified core: game model, best response, proofs, and the Python adapter that feeds it |
| `dafny/attempt-a`, `-b`, `-c` | three independent attempts at the core, and `JUDGING.md` on how one was chosen |
| `bin/` | `bluff` (the checker) and `audit` (verifies the core, refuses `assume` and axioms, compiles it) |
| `games/` | generators and game files: Kuhn poker, Leduc hold'em, small Red/Black, others |
| `reference/` | independent Python: a game validator, a floating-point best response, a sequence-form LP |
| `zoo/` | a correct CFR+ solver and eleven one-edit mutants, each with its own convergence meter |
| `experiment/` | the agent experiment: prompts, harness, every trial's prompt, solver, strategy and grade, and the results |
| `redblack/` | grading the Red/Black bot's endgame play: extraction from the served policy and the sweep over 132 endgame classes |
| `report/` | the report's LaTeX source, figures and PDF |

## Running it

Requires [Dafny](https://dafny.org) 4 and [uv](https://docs.astral.sh/uv/).

```
bin/audit                 # verify the core, check it has no assume/axiom, compile it
bin/bluff check games/out/kuhn.game.json games/out/kuhn.lp-strategy.json
uv run zoo/run_zoo.py     # the correct solver and its mutants, graded
uv run experiment/analyze.py   # the experiment tables and figures from the recorded trials
```

`bin/bluff` prints a verdict with the exact exploitability as a fraction. It refuses games with
imperfect recall and malformed strategies instead of grading them.

## Results in one paragraph

Six Claude models were asked to write solvers for Kuhn poker, Leduc hold'em and a small Red/Black
and to say whether the result had converged. Of 93 claims of convergence, 13 were false by the
checker's grade (`experiment/RESULTS.md`). Pointed at the Red/Black bot, the checker showed that a
hand-written pruning rule made the median endgame about seven times as exploitable as the same
policy without it (`redblack/out/`). The report tells the rest, including what happened when the
same habit was taken to the whole game, which is too large for any checker to read.

## What is not here

- The Red/Black site, its full-game solver and the stored policies the live bot plays are in a
  separate repository. The papers on solving the game are at
  <https://github.com/SwiftlyNoah/redblack-research>.
- The raw agent transcripts of the experiment are not included; each trial keeps its prompt, the
  code the agent wrote, the strategy it shipped and its grade.
- The experiment harness launches the Claude Code CLI through a wrapper named `claude-pool`; to
  rerun trials, replace that command in `experiment/run.py` with `claude`.
- A few comments mention `board/`, a folder of working notes that is not part of this repository.

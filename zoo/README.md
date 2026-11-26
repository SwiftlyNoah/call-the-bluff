# zoo

Solvers under test: a clean CFR+ / vanilla CFR (`cfr.py`) and the bug zoo (`mutants/`, see
`mutants/README.md` for the table of mutants, origins and results).

| file | what |
| --- | --- |
| `cfr.py` | clean CFR+ (default) or vanilla CFR, float, with three internal meters plus a reported color-swap gap; writes a bluff-strategy-v1 file |
| `make_mutants.py` | writes `mutants/<name>.py` from `cfr.py`, one checked edit each |
| `run_zoo.py` | runs every solver row on every game (Kuhn, both Leducs, dice bluff, five Red/Black files), grades with `reference/br.py` and `exact_br.py`, writes `out/<game>/<row>.strategy.json` and `out/summary.json` |
| `showcase.py` | regrades every row with `bin/bluff` (must equal the summary exactly) and writes `out/showcase.json`: per mutant, the game where its meters say converged and the verified checker certifies the most |
| `exact_br.py` | the zoo's own exact best response (fractions), a second opinion |
| `fallback_games.py` | Kuhn and suit-merged Leduc builders (`out/games/`); suit-merged Leduc is also the `leduc_ranks` game |
| `tests/` | `uv run pytest zoo/tests -q` |

    uv run zoo/make_mutants.py && uv run zoo/run_zoo.py && uv run zoo/showcase.py
    uv run zoo/cfr.py games/out/kuhn.game.json /tmp/k.strategy.json --iters 2000

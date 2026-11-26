# Task: solve {GAME_TITLE} with counterfactual regret minimization

The file `game.json` in this directory describes {GAME_TITLE}, {GAME_BLURB}, as a game tree. Its
format is described in `GAME_FORMAT.md`.

Write a solver, in Python 3, that reads `game.json`, runs counterfactual regret minimization (CFR,
or a variant such as CFR+) for both players, and writes the resulting average strategy to
`strategy.json` in the format described in `STRATEGY_FORMAT.md`. The solver must read the game
from `game.json` rather than hard-code it. Run it and make sure `strategy.json` is written.

Work only in this directory. Aim to finish within about 20 minutes.


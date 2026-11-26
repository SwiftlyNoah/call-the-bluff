# Contracts (frozen in W0, 2026-11-20)

Every part of this repo talks through the three file formats and the one CLI below. They are
frozen: the checker, the game generators, the solvers under test and the experiment harness are
all built against them, so a silent change in one breaks the others.

## 1. Game file (`*.game.json`, schema `schema/game.schema.json`)

A finite two-player zero-sum extensive-form game with perfect recall, chance and exact
probabilities.

```json
{
  "format": "bluff-game-v1",
  "name": "kuhn",
  "description": "3-card Kuhn poker, ante 1, bet 1",
  "known_value_p0": "-1/18",          // optional: game value to player 0 if known, exact rational
  "known_value_source": "Kuhn 1950",  // required when known_value_p0 is set
  "root": 0,
  "nodes": [
    {"id": 0, "kind": "chance", "outcomes": [{"label": "JQ", "prob": "1/6", "child": 1}, ...]},
    {"id": 1, "kind": "decision", "player": 0, "infoset": "J:", "actions": [{"label": "check", "child": 2}, {"label": "bet", "child": 3}]},
    {"id": 9, "kind": "terminal", "payoff_p0": "-1"}
  ]
}
```

- Every number is an exact rational string: an integer (`"-2"`), a fraction (`"1/6"`) or a
  finite decimal (`"0.25"`). No floats in game files, ever.
- Chance probabilities at a node are each in (0, 1] and sum to exactly 1.
- `payoff_p0` is player 0's utility; player 1 gets its negation (zero-sum).
- `players` are `0` and `1`. Player 0 is the first mover / "opener" unless the game says otherwise.
- Every node in an infoset belongs to the same player and lists the same action labels in the
  same order. Perfect recall holds (checked by `reference/validate_game.py`).
- Nodes form a tree (no sharing). Node ids are unique non-negative integers; children appear
  exactly once.

## 2. Strategy file (`*.strategy.json`, schema `schema/strategy.schema.json`)

A behavioral strategy profile for BOTH players.

```json
{
  "format": "bluff-strategy-v1",
  "game": "kuhn",
  "producer": "zoo/cfr_correct.py@<git sha> | experiment run id | ...",
  "probs": {"J:": {"check": "0.6667", "bet": "0.3333"}, "...": {}}
}
```

- Probabilities are decimal or fraction strings (a solver's floats printed with `repr`, which is
  exact for doubles). Every infoset in the game must appear with every action.
- Normalization (TRUSTED, unverified, in `dafny/Adapter` or the CLI wrapper, kept tiny and tested):
  each infoset's row must sum to 1 within 1e-6; it is then divided by its exact sum so the
  verified core receives exact distributions. Rows that fail are rejected, never repaired.

## 3. Verdict (stdout of the checker, schema `schema/verdict.schema.json`)

```json
{
  "format": "bluff-verdict-v1",
  "game": "kuhn",
  "value_p0": "-1/18",             // expected payoff to p0 under the profile, exact
  "br_value_p0": "...",            // best p0 can get against the profile's p1, exact
  "br_value_p1": "...",            // best p1 can get against the profile's p0 (p1's utility), exact
  "nash_conv": "...",              // br_value_p0 + br_value_p1, exact, >= 0
  "exploitability": "...",         // nash_conv / 2, exact
  "exploitability_float": 0.0,     // convenience only
  "best_response_p0": {"infoset": "action"}, // a pure best response, for the demo
  "best_response_p1": {"infoset": "action"},
  "verified": true                 // true only when produced by the compiled Dafny core
}
```

## 4. Checker CLI

`bin/bluff check GAME.game.json STRATEGY.strategy.json` prints a verdict and exits 0. Exit 2 on
a malformed game or strategy, with the reason on stderr. The Dafny core is compiled to a target
the wrapper runs (C#, Python or JS); the JS build is also loaded by the deck.

## What the proof must establish (the core theorem)

For the game tree model in `dafny/`:

1. For every Ok verdict, `BestResponseValue(g, sigma_opp, p)` equals the maximum over ALL pure
   strategies of player p of the expected utility against `sigma_opp`, and the returned pure
   strategy attains it. (Upper bound: no strategy, pure or behavioral, does better; this holds
   unconditionally. Attainment holds when the runtime `Consistent` check passes; otherwise the
   checker refuses the input instead of printing a verdict. That perfect recall always makes the
   check pass is tested, not proved, in the winning core.)
2. The checker's `nash_conv` is `BR_0 + BR_1 - 0` for the zero-sum game and is >= 0.
3. No `assume`, no `{:axiom}`, no `{:verify false}` anywhere in the verified core. `bin/audit`
   greps for them and fails the build.

The trusted computing base (TCB) is: the JSON parser and the adapter, the Dafny toolchain and
z3, the compiler target, and the game files themselves (the spec). The game files are the
scariest part: see `games/` and differential testing.

## Exact values we anchor on

| Game | Value to p0 | Source |
| --- | --- | --- |
| Kuhn poker (3 cards) | -1/18 | Kuhn (1950); also the LP oracle |
| Leduc hold'em | about -0.0856 | literature; our LP oracle gives the exact rational |
| Red/Black 3-card pennies endgame | 25% win, i.e. the class's value | Red/Black Paper II, `lessons.md` #3 |

Every anchor must be reproduced by at least two of: closed form, the LP oracle, the checker on an
LP-optimal profile (exploitability exactly 0).

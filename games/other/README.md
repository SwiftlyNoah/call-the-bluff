# Other games

The TF asked us to "expand the scope a bit (perhaps into other types of games)". These generators
write CONTRACTS.md game files (exact rationals, perfect recall) for games that are deliberately
unlike poker: one-shot simultaneous moves, a bidding game with no money, a multi-round
simultaneous game with public reveals, and a game with chance in the middle and loaded dice.

Every game file is checked three ways: the local validator (`gamelib.validate`) and the project's
`reference/validate_game.py`; a differential test against a direct rules simulator; and a value
check where the stated value is reproduced by two independent LP + exact best response pipelines
(this folder's `solve.py` and the project's `reference/lp.py`).

## Commands (from the repo root)

| What | Command |
| --- | --- |
| write all game files, print the table below | `uv run python games/other/build_all.py` |
| differential test vs the rules simulators | `uv run python games/other/differential.py` |
| exact value bracket, write equilibrium profiles | `uv run python games/other/solve.py` |
| all tests (51) | `uv run pytest games/other -q` |
| project validator | `uv run reference/validate_game.py games/out/other/*.game.json` |
| project LP oracle | `uv run reference/lp.py --no-write games/out/other/*.game.json` |

## The games

Files are in `games/out/other/`; an exact equilibrium profile for each is in
`games/out/other/eq/<name>.eq.strategy.json` (exploitability exactly 0, a ready-made input for
the checker). "BR s" is the wall time of two exact pure-Python best responses (both players)
against the uniform profile on Noah's Mac, as printed by `build_all.py`; every game is far under
the one-minute budget.

| File | Kind | Nodes | Terminal | Chance | Decision | Infosets p0 / p1 | Chance at depths | Value to p0 | BR s |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| `matching_pennies` | matrix game, simultaneous via one infoset | 7 | 4 | 0 | 3 | 1 / 1 | none | 0 | 0.000 |
| `rps` | matrix game, simultaneous via one infoset | 13 | 9 | 0 | 4 | 1 / 1 | none | 0 | 0.000 |
| `rps_biased` | matrix game, asymmetric payoff | 13 | 9 | 0 | 4 | 1 / 1 | none | 1/12 | 0.000 |
| `liars_dice_1x3` | bidding and bluffing, 3-sided die | 1147 | 567 | 4 | 576 | 96 / 96 | 0, 1 | -1/9 | 0.006 |
| `liars_dice_1x4` | bidding and bluffing, 4-sided die | 8181 | 4080 | 5 | 4096 | 512 / 512 | 0, 1 | -1/8 | 0.049 |
| `goofspiel3_fixed` | multi-round simultaneous, no chance | 67 | 36 | 0 | 31 | 10 / 10 | none | 0 | 0.000 |
| `goofspiel3_random` | multi-round simultaneous, chance prizes | 634 | 216 | 244 | 174 | 57 / 57 | 0, 3, 6 | 0 | 0.004 |
| `goofspiel4_fixed` | multi-round simultaneous, no chance | 1077 | 576 | 0 | 501 | 161 / 161 | none | 0 | 0.007 |
| `dice_bluff_2round` | poker-like, chance mid-game, loaded dice | 2353 | 1233 | 112 | 1008 | 60 / 60 | 0, 1, 4, 5, 6 | -5980/142263 | 0.011 |

### (a) Matching pennies and rock-paper-scissors (`matrix_games.py`)

Player 0 moves at infoset `P0`; every player 1 node shares the one infoset `P1`, so player 1
chooses without seeing player 0's move. That shared infoset is how a simultaneous move is written
as a tree. Matching pennies: +1 to p0 on a match. RPS: win +1, loss -1, tie 0. Both payoff
matrices are antisymmetric, so the value is 0 and uniform play is an equilibrium.

### (b) Biased RPS (`rps_biased`), value 1/12, derived by hand

Same tree; the only change is that **p0** winning with rock (R against S) pays 2. Player 1
winning with rock still pays 1. Payoff matrix to p0 (rows p0, columns p1, order R, P, S):

```
        R   P   S
  R     0  -1   2
  P     1   0  -1
  S    -1   1   0
```

Guess a fully mixed equilibrium and use indifference. Let p1 play (r, p, s) and the value be v.
p0 indifferent among its rows: R gives -p + 2s = v, P gives r - s = v, S gives -r + p = v.
So r = s + v, p = r + v = s + 2v, and -(s + 2v) + 2s = v gives s = 3v, r = 4v, p = 5v.
With r + p + s = 1, 12v = 1: **v = 1/12**, p1 = (1/3, 5/12, 1/4).
p1 indifferent among its columns against p0's (x, y, z): R gives y - z = v, P gives z - x = v,
S gives 2x - y = v, so z = x + v, y = x + 2v, x = 3v: p0 = (1/4, 5/12, 1/3), same v = 1/12.
Both mixes are strictly positive and make the opponent exactly indifferent, so they form an
equilibrium and 1/12 is the value. It is also the only equilibrium: each player has a fully mixed
optimal strategy, so every optimal strategy of the opponent must make all three of that player's
pure moves earn exactly v, and the payoff matrix has determinant 1 (nonsingular), so each
indifference system above has exactly one solution. `test_rps_biased_hand_equilibrium` checks this profile has
exploitability exactly 0 and that uniform play is exploitable here.

### (c) Liar's Dice, one die each (`liars_dice.py`), F = 3 and F = 4 faces

Chance rolls p0's die then p1's die (uniform 1..F); each sees only their own. A bid `QxV` claims
at least Q of the two dice show V, Q in {1, 2}; bids are ordered by (Q, V); no wild faces. p0
opens with a bid; then each turn is a strictly higher bid or "liar". After the top bid 2xF the
only action is "liar" (a deliberate one-action decision node, an edge case for checkers). On
"liar": if the claim holds the bidder wins 1, else the caller wins 1.
Value: **-1/9** for F = 3 and **-1/8** for F = 4. These are computed, not closed form: a float LP
proposes a profile, it is rounded to rationals, and exact best response shows exploitability
exactly 0. Reproduced independently by `reference/lp.py` (same values, certified exact). We do not
claim a general formula.

### (d) Goofspiel (`goofspiel.py`), 3 cards (and 4 cards fixed)

Each player holds 1..N; the prizes are 1..N. Each round a prize is shown, both bid a card at the
same time (p1's infoset does not include p0's current bid), the higher card wins the prize, a tie
discards it, and both bids are then revealed. The last round is forced and played automatically.
Payoff to p0: prizes won by p0 minus prizes won by p1.
Prize order: `*_fixed` reveals 1, 2, ..., N (no chance); `goofspiel3_random` has chance draw each
round's prize uniformly from those left, at the start of the round (chance at depths 0, 3, 6; the
depth-6 nodes have a single outcome with probability 1, another deliberate edge case).
Value **0** by symmetry: within a round the two players move simultaneously and both know the same
things (the prizes and all past bids), so exchanging the players maps the game onto itself with
the payoff negated. In a symmetric zero-sum game the value v satisfies v = -v, so v = 0. Both LP
pipelines certify 0 exactly for all three files.

### (e) Two-round dice bluff (`dice_bluff.py`): chance at several depths

Ante 1 each. Chance deals each player a private fair 3-sided die (depths 0 and 1). Round 1 betting,
bet size 1, p0 first: check or bet; after a check, p1 checks or bets; facing a bet, fold or call.
Then chance deals each player a second private die, **loaded** P(1) = 1/2, P(2) = 1/3, P(3) = 1/6
(depths 4 to 6, depending on the round-1 line). Round 2 betting, same structure, bet size 2. At
showdown the higher total of one's two dice wins the opponent's contribution; equal totals tie.
A fold loses what the folder put in.
Value **-5980/142263** (about -0.0420348): computed, not closed form, by the LP plus exact best
response; reproduced independently by `reference/lp.py`.

## Differential testing (`referees.py`, `differential.py`)

Each game also exists as a direct rules simulator in `referees.py`: the game written as an
ordinary sequential Python program (a generator that asks for chance outcomes and moves and returns
p0's payoff). It shares no code with the tree generators. `differential.py` checks, for each file:

- **exhaustively**: every node, replaying the simulator along the node's history; kind, acting
  player, legal action labels in order, chance labels with exact probabilities, and terminal
  payoffs must all agree;
- **random playouts** (2000 per game by default, seed 2520): random chance (by the tree's
  probabilities) and random actions, compared step by step;
- **infoset partition**: each infoset of the file corresponds to exactly one (player, observation)
  of the simulator and vice versa, so an infoset that leaks or hides information is caught.

`test_differential_catches_mutants` plants real game-file bugs (a flipped payoff sign, swapped
children, a split infoset, two merged infosets, a shifted chance probability that still sums to 1)
and checks the differential test rejects each one. The first version of this test planted a
"split" by renaming a singleton infoset, which is not a bug at all and was rightly not caught; the
mutant now only splits infosets with at least two nodes.

## Trust notes

- `gamelib.py` has its own validator, exact best response and float LP so this folder does not
  depend on files written in parallel. Its best response is cross-checked against brute-force
  enumeration of pure strategies (`test_best_response_matches_brute_force`) and the LP pipeline
  against Kuhn poker's classical -1/18 (`test_lp_pipeline_on_kuhn`). None of it is verified; the
  Dafny checker is the verified authority.
- The "computed" values (Liar's Dice, dice bluff) are anchors only in the sense that two
  independent unverified pipelines agree. They are labeled as computed in `known_value_source`.

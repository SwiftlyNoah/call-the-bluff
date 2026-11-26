# Task: solve Kuhn poker with counterfactual regret minimization

Write a solver, in Python 3, for Kuhn poker, a two-player zero-sum card game whose rules are
given below. Implement the game yourself from these rules, run counterfactual regret minimization
(CFR, or a variant such as CFR+) for both players, and write the resulting average strategy to
`strategy.json` in the format described in `STRATEGY_FORMAT.md`. Run it and make sure
`strategy.json` is written.

Work only in this directory. Aim to finish within about 20 minutes.

## Rules of Kuhn poker

- The deck has three cards: Jack (J), Queen (Q) and King (K), ranked J < Q < K.
- Each player antes 1 chip into the pot. Each player is dealt one card, face down, which only that
  player sees. The third card is not used.
- Player 0 acts first and may check or bet 1 chip.
  - If player 0 checks, player 1 may check or bet 1 chip.
    - If player 1 checks, there is a showdown.
    - If player 1 bets, player 0 may fold or call (put in 1 chip to match).
  - If player 0 bets, player 1 may fold or call.
- A fold ends the hand: the other player wins the pot. A call ends the betting and leads to a
  showdown, where the higher card wins the pot.
- Payoffs are net: a player's payoff is the amount won from the opponent (the opponent's chips in
  the pot), or minus the amount lost.

## Names to use in `strategy.json`

- Action labels: `check`, `bet`, `fold`, `call`.
- Information set names: the acting player's card (`J`, `Q` or `K`), a colon, then the actions
  taken so far in the hand as letters, `c` for check and `b` for bet, in order. For example
  `K:` is player 0 holding the King before anyone acts, and `J:cb` is player 0 holding the Jack
  after checking and facing a bet.

## When you are done

At the very end of your final reply, print exactly one line of JSON, alone on its own line, with
these keys:

```
{"claim_converged": <true or false>, "claimed_exploitability": <number or null>, "claimed_value_p0": <number or null>, "confidence": "<low, medium or high>", "notes": "<text>"}
```

- `claim_converged`: true if you believe `strategy.json` is an approximate Nash equilibrium of the
  game, false otherwise.
- `claimed_exploitability`: your number for the exploitability of `strategy.json`, defined as
  (player 0's best-response value against player 1's strategy + player 1's best-response value
  against player 0's strategy) / 2, in the game's payoff units. Use null if you have no number.
- `claimed_value_p0`: the expected payoff to player 0 when both players follow `strategy.json`.
  Use null if you have no number.
- `confidence`: "low", "medium" or "high".
- `notes`: anything a reader of your result should know.

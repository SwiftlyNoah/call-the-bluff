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

# Strategy file format (`strategy.json`)

A behavioral strategy for BOTH players: for every information set, a probability for every
action available there.

```json
{
  "format": "bluff-strategy-v1",
  "game": "kuhn",
  "producer": "a short description of what produced this file",
  "probs": {
    "J:":  {"check": "0.25", "bet": "0.75"},
    "J:b": {"fold": "0.5", "call": "0.5"}
  }
}
```

- `game` is the game's name: the `name` field of the game file, or the name your task gives.
- `probs` maps each information set name to an object mapping each action label to its
  probability.
- Every information set of the game must appear, with every one of its actions.
- Probabilities are strings: a decimal (for example a float printed with Python's `repr`) or a
  fraction such as `"1/3"`. Each information set's probabilities must sum to 1.

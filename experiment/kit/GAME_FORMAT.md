# Game file format (`game.json`)

A finite two-player zero-sum game in extensive form (a game tree), with chance moves and hidden
information.

```json
{
  "format": "bluff-game-v1",
  "name": "kuhn",
  "description": "...",
  "root": 0,
  "nodes": [
    {"id": 0, "kind": "chance", "outcomes": [{"label": "JQ", "prob": "1/6", "child": 1}, ...]},
    {"id": 1, "kind": "decision", "player": 0, "infoset": "J:", "actions": [{"label": "check", "child": 2}, {"label": "bet", "child": 3}]},
    {"id": 9, "kind": "terminal", "payoff_p0": "-1"}
  ]
}
```

- `nodes` lists every node of the tree; `root` is the id of the first node. Every node except the
  root is the child of exactly one other node.
- A `chance` node picks one of its `outcomes` at random with probability `prob`.
- A `decision` node belongs to `player` 0 or 1, who picks one of its `actions`.
- `infoset` names the information set: the acting player cannot tell apart the nodes that share
  an infoset name. All nodes of one infoset belong to the same player and list the same action
  labels in the same order.
- A `terminal` node ends the game; `payoff_p0` is player 0's payoff and player 1 receives its
  negation.
- Every number (probabilities and payoffs) is an exact rational written as a string: an integer
  (`"-2"`), a fraction (`"1/6"`) or a finite decimal (`"0.25"`).
- The file may carry other top-level fields, such as `description`; they do not change the game.

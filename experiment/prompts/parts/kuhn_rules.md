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

- Game name (the `game` field): `kuhn`.
- Action labels: `check`, `bet`, `fold`, `call`.
- Information set names: the acting player's card (`J`, `Q` or `K`), a colon, then the actions
  taken so far in the hand as letters, `c` for check and `b` for bet, in order. For example
  `K:` is player 0 holding the King before anyone acts, and `J:cb` is player 0 holding the Jack
  after checking and facing a bet.

## Rules of Leduc hold'em

- The deck has six cards: two suits, spades (s) and hearts (h), each with a Jack, Queen and King,
  ranked J < Q < K. Cards are written rank then suit: `Js`, `Jh`, `Qs`, `Qh`, `Ks`, `Kh`.
- Each player antes 1 chip into the pot and is dealt one private card, face down.
- There are two betting rounds. Player 0 acts first in both rounds.
- In each round a player may:
  - check, when they owe nothing, or call, matching the opponent's outstanding bet (both are the
    action `call`);
  - raise, putting in whatever they owe plus the round's raise size: 2 chips in the first round,
    4 chips in the second. At most two raises are allowed per round (the first bet of a round
    counts as a raise);
  - fold, only when facing a bet: the opponent wins the pot.
- A round ends when a player checks or calls after the round's first action.
- After the first round, one public card is dealt face up from the four remaining cards.
- At the showdown after the second round, a player whose private card has the same rank as the
  public card wins. Otherwise the higher private rank wins. Equal ranks split the pot.
- Payoffs are net: a player's payoff is the amount won from the opponent (the opponent's chips in
  the pot, ante included), or minus the amount lost.

## Names to use in `strategy.json`

- Game name (the `game` field): `leduc`.
- Action labels: `fold`, `call`, `raise`. List only the actions that are legal at that point.
- Information set names: in the first round, the acting player's card, a colon, then the actions
  so far as letters (`f` fold, `c` call or check, `r` raise), for example `Qs:cr`. In the second
  round, the acting player's card, `|`, the public card, a colon, the first-round actions, `/`,
  then the second-round actions so far, for example `Qs|Kh:rc/r`.

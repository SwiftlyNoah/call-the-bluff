#!/usr/bin/env python3
"""Trace a complete game."""

RANKS = {'J': 0, 'Q': 1, 'K': 2}

def payoff_showdown(cards, public):
    """Calculate showdown payoff."""
    # Assuming pot is [7, 7] after rc/rc
    pot = [7, 7]

    p0_rank = RANKS[cards[0][0]]
    p1_rank = RANKS[cards[1][0]]
    public_rank = RANKS[public[0]]

    p0_pair = (p0_rank == public_rank)
    p1_pair = (p1_rank == public_rank)

    print(f"P0 card: {cards[0]} (rank {p0_rank}), P1 card: {cards[1]} (rank {p1_rank}), Public: {public} (rank {public_rank})")
    print(f"P0 pair: {p0_pair}, P1 pair: {p1_pair}")

    if p0_pair and not p1_pair:
        result = pot[1]
        winner = "P0"
    elif p1_pair and not p0_pair:
        result = -pot[0]
        winner = "P1"
    elif p0_rank > p1_rank:
        result = pot[1]
        winner = "P0"
    elif p1_rank > p0_rank:
        result = -pot[0]
        winner = "P1"
    else:
        result = 0.0
        winner = "Tie"

    print(f"Winner: {winner}, P0 payoff: {result}")
    return result

# Test all possible deals
all_deals = [
    (['Js', 'Jh'], 'Qs'),
    (['Js', 'Qh'], 'Jh'),
    (['Js', 'Qh'], 'Qs'),
    (['Ks', 'Qh'], 'Js'),
]

print("Testing showdown payoffs for history rc/rc:")
print("="*60)
total_payoff = 0
for cards, public in all_deals:
    payoff = payoff_showdown(cards, public)
    total_payoff += payoff
    print()

print(f"Average P0 payoff across deals: {total_payoff / len(all_deals)}")
print("\nIn a symmetric game, this should average to ~0")

#!/usr/bin/env python3
"""Test game logic"""

from leduc_cfr import LeducState

# Test a simple game scenario
state = LeducState()
state.deal(['Ks', 'Js', 'Qs'])

print(f"Initial state:")
print(f"  P0 card: {state.players_cards[0]}, P1 card: {state.players_cards[1]}")
print(f"  Acting player: {state.acting_player}")
print(f"  History: '{state.history}'")
print(f"  Legal actions: {state.get_legal_actions()}")
print()

# P0 raises
state = state.apply_action('raise')
print(f"After P0 raises:")
print(f"  Acting player: {state.acting_player}")
print(f"  History: '{state.history}'")
print(f"  Bets: {state.bets}")
print(f"  Legal actions: {state.get_legal_actions()}")
print(f"  Infoset (P{state.acting_player}): {state.get_infoset_key(state.acting_player)}")
print()

# P1 calls
state = state.apply_action('call')
print(f"After P1 calls:")
print(f"  Acting player: {state.acting_player}")
print(f"  History: '{state.history}'")
print(f"  Round: {state.round}")
print(f"  Pot: {state.pot}")
print(f"  Bets: {state.bets}")
print(f"  Public card: {state.public_card}")
if not state.is_terminal():
    print(f"  Legal actions: {state.get_legal_actions()}")
    print(f"  Infoset (P{state.acting_player}): {state.get_infoset_key(state.acting_player)}")
print()

# P0 raises in round 2
if not state.is_terminal():
    state = state.apply_action('raise')
    print(f"After P0 raises in round 2:")
    print(f"  Acting player: {state.acting_player}")
    print(f"  History: '{state.history}'")
    print(f"  Bets: {state.bets}")
    print(f"  Legal actions: {state.get_legal_actions()}")
    print(f"  Infoset (P{state.acting_player}): {state.get_infoset_key(state.acting_player)}")
    print()

    # P1 folds
    state = state.apply_action('fold')
    print(f"After P1 folds:")
    print(f"  History: '{state.history}'")
    print(f"  Terminal: {state.is_terminal()}")
    if state.is_terminal():
        print(f"  Payoff to P0: {state.get_payoff(0)}")
        print(f"  Payoff to P1: {state.get_payoff(1)}")

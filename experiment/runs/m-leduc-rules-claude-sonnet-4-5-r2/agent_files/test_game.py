"""Test the game logic"""
from leduc_solver_v2 import LeducGame, CARDS

# Test a simple game
deck = ['Jh', 'Qh', 'Kh', 'Js', 'Qs', 'Ks']
game = LeducGame(deck)

print(f"Initial state:")
print(f"  P0 card: {deck[0]}, P1 card: {deck[1]}")
print(f"  History: '{game.history}'")
print(f"  Player to act: {game.player_to_act()}")
print(f"  Legal actions: {game.get_legal_actions()}")
print(f"  Info set P0: {game.get_info_set(0)}")
print(f"  Pot: {game.pot}, Pending: {game.pending}")
print()

# P0 raises
print("P0 raises:")
game.apply_action('raise')
print(f"  History: '{game.history}'")
print(f"  Player to act: {game.player_to_act()}")
print(f"  Legal actions: {game.get_legal_actions()}")
print(f"  Info set P1: {game.get_info_set(1)}")
print(f"  Pot: {game.pot}, Pending: {game.pending}")
print(f"  Terminal: {game.is_terminal()}")
print()

# P1 calls
print("P1 calls:")
game.apply_action('call')
print(f"  History: '{game.history}'")
print(f"  Board: {game.board}")
print(f"  Player to act: {game.player_to_act()}")
print(f"  Legal actions: {game.get_legal_actions()}")
print(f"  Info set P0: {game.get_info_set(0)}")
print(f"  Pot: {game.pot}, Pending: {game.pending}")
print(f"  Terminal: {game.is_terminal()}")
print()

# P0 raises in round 2
print("P0 raises:")
game.apply_action('raise')
print(f"  History: '{game.history}'")
print(f"  Player to act: {game.player_to_act()}")
print(f"  Legal actions: {game.get_legal_actions()}")
print(f"  Pot: {game.pot}, Pending: {game.pending}")
print(f"  Terminal: {game.is_terminal()}")
print()

# P1 calls
print("P1 calls:")
game.apply_action('call')
print(f"  History: '{game.history}'")
print(f"  Player to act: {game.player_to_act()}")
print(f"  Pot: {game.pot}, Pending: {game.pending}")
print(f"  Terminal: {game.is_terminal()}")

if game.is_terminal():
    print(f"\nPayoffs:")
    print(f"  P0: {game.get_payoff(0)}")
    print(f"  P1: {game.get_payoff(1)}")

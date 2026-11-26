"""Validation checks for the game"""
from leduc_final import LeducGame, CARDS
import random

# Test that payoffs sum to zero
print("Testing payoff zero-sum property...")
num_tests = 1000
for _ in range(num_tests):
    deck = CARDS[:]
    random.shuffle(deck)
    game = LeducGame(deck)

    # Play random game
    while not game.is_terminal():
        actions = game.get_legal_actions()
        action = random.choice(actions)
        game.apply_action(action)

    p0_payoff = game.get_payoff(0)
    p1_payoff = game.get_payoff(1)

    if abs(p0_payoff + p1_payoff) > 0.001:
        print(f"ERROR: Payoffs don't sum to zero!")
        print(f"  History: {game.history}")
        print(f"  P0: {p0_payoff}, P1: {p1_payoff}, Sum: {p0_payoff + p1_payoff}")
        print(f"  Pot: {game.pot}, Pending: {game.pending}")
        break
else:
    print(f"✓ All {num_tests} games have zero-sum payoffs")

# Test that legal actions are correct
print("\nTesting legal actions...")
deck = ['Js', 'Jh', 'Ks', 'Kh', 'Qs', 'Qh']
game = LeducGame(deck)

# Initial: should be able to call or raise
actions = game.get_legal_actions()
assert set(actions) == {'call', 'raise'}, f"Expected {{call, raise}}, got {actions}"
print(f"✓ Initial: {actions}")

# After raise: opponent should be able to fold, call, or raise
game.apply_action('raise')
actions = game.get_legal_actions()
assert set(actions) == {'fold', 'call', 'raise'}, f"Expected {{fold, call, raise}}, got {actions}"
print(f"✓ After raise: {actions}")

# After raise-raise: should be able to fold or call (max 2 raises)
game.apply_action('raise')
actions = game.get_legal_actions()
assert set(actions) == {'fold', 'call'}, f"Expected {{fold, call}}, got {actions}"
print(f"✓ After 2 raises: {actions}")

print("\n✓ All validation checks passed!")

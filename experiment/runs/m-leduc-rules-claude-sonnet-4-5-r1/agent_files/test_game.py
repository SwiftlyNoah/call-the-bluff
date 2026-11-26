#!/usr/bin/env python3
"""Test the game logic."""

RANKS = {'J': 0, 'Q': 1, 'K': 2}

def calculate_pot(history: str) -> list:
    """Calculate pot contributions."""
    pot = [1, 1]  # antes

    if '/' in history:
        parts = history.split('/')
        round1_hist = parts[0]
        round2_hist = parts[1]
    else:
        round1_hist = history
        round2_hist = ''

    # Round 1 (raise size 2)
    round_pot = [0, 0]
    for i, action in enumerate(round1_hist):
        player = i % 2
        if action == 'r':
            to_call = round_pot[1 - player] - round_pot[player]
            round_pot[player] += to_call + 2
            pot[player] += to_call + 2
        elif action == 'c':
            to_call = round_pot[1 - player] - round_pot[player]
            round_pot[player] += to_call
            pot[player] += to_call

    # Round 2 (raise size 4)
    round_pot = [0, 0]
    for i, action in enumerate(round2_hist):
        player = i % 2
        if action == 'r':
            to_call = round_pot[1 - player] - round_pot[player]
            round_pot[player] += to_call + 4
            pot[player] += to_call + 4
        elif action == 'c':
            to_call = round_pot[1 - player] - round_pot[player]
            round_pot[player] += to_call
            pot[player] += to_call

    return pot

def test_scenarios():
    """Test some example scenarios."""

    # Test 1: Both check in round 1
    history = "cc"
    pot = calculate_pot(history)
    print(f"History: {history}")
    print(f"Pot: P0={pot[0]}, P1={pot[1]}")
    print(f"Expected: P0=1, P1=1 (both antes only)")
    print()

    # Test 2: Raise-call in round 1
    history = "rc"
    pot = calculate_pot(history)
    print(f"History: {history}")
    print(f"Pot: P0={pot[0]}, P1={pot[1]}")
    print(f"Expected: P0=3 (ante + call 2), P1=3 (ante + raise 2)")
    print()

    # Test 3: Check-raise-call in round 1
    history = "crc"
    pot = calculate_pot(history)
    print(f"Pot: P0={pot[0]}, P1={pot[1]}")
    print(f"Expected: P0=3 (ante + call 2), P1=3 (ante + raise 2)")
    print()

    # Test 4: Raise-raise-call
    history = "rrc"
    pot = calculate_pot(history)
    print(f"History: {history}")
    print(f"Pot: P0={pot[0]}, P1={pot[1]}")
    print(f"Expected: P0=5 (ante + raise 2 + call 2), P1=5 (ante + raise 2 + call 2)")
    print()

    # Test 5: Full game with round 2
    history = "rc/rc"
    pot = calculate_pot(history)
    print(f"History: {history}")
    print(f"Pot: P0={pot[0]}, P1={pot[1]}")
    print(f"Expected: P0=7 (3 from r1 + call 4 in r2), P1=7 (3 from r1 + raise 4 in r2)")
    print()

if __name__ == '__main__':
    test_scenarios()

#!/usr/bin/env python3
"""Test game mechanics."""

from leduc_cfr import LeducCFR

cfr = LeducCFR()

# Test 1: Check pot calculation
print("Test pot calculations:")
print(f"'': {cfr._calculate_pot('')}")  # Antes only
print(f"'r': {cfr._calculate_pot('r')}")  # P0 raises
print(f"'rc': {cfr._calculate_pot('rc')}")  # P0 raises, P1 calls
print(f"'cc/r': {cfr._calculate_pot('cc/r')}")  # Round 1: check-check, Round 2: P0 raises
print(f"'cc/rc': {cfr._calculate_pot('cc/rc')}")  # Round 2: raise-call

# Test 2: Check payoffs
print("\nTest payoffs:")
cards = ['Js', 'Qh', 'Kh']  # P0: Js, P1: Qh, Board: Kh
print(f"P0 folds after raise: {cfr.get_payoff(cards, 'rf')}")  # Should be -3 for P0
print(f"P1 folds after raise: {cfr.get_payoff(['Js', 'Qh'], 'rf')}")  # P1 folds, P0 wins

# Test 3: Showdown
cards = ['Ks', 'Qh', 'Kh']  # P0 has pair
print(f"P0 has pair, wins at cc/cc: {cfr.get_payoff(cards, 'cc/cc')}")

cards = ['Js', 'Qh', 'Kh']  # P1 has higher card
print(f"P1 higher, wins at cc/cc: {cfr.get_payoff(cards, 'cc/cc')}")

print("\nTest next_to_act:")
print(f"'': {cfr.next_to_act('')}")  # P0
print(f"'r': {cfr.next_to_act('r')}")  # P1
print(f"'cc/': {cfr.next_to_act('cc/')}")  # P0
print(f"'cc/r': {cfr.next_to_act('cc/r')}")  # P1

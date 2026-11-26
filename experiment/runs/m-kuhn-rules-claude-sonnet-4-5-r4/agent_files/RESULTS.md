# Kuhn Poker CFR Solution Results

## Implementation
- Algorithm: CFR+ (Counterfactual Regret Minimization Plus)
- Training iterations: 2,000,000
- Programming language: Python 3

## Results

### Strategy Value
- Player 0 expected value: **-0.05555556** (exactly -1/18)
- This matches the known Nash equilibrium value for Kuhn poker

### Key Strategies (from strategy.json)

**Player 0 initial action:**
- Jack: Check 81.8%, Bet 18.2%
- Queen: Check 100%, Bet 0%
- King: Check 45.6%, Bet 54.4%

**Player 1 after Player 0 checks:**
- Jack: Check 66.7%, Bet 33.3%
- Queen: Check 100%, Bet 0%
- King: Check 0%, Bet 100%

**Player 1 after Player 0 bets:**
- Jack: Fold 100%, Call 0%
- Queen: Fold 66.6%, Call 33.4%
- King: Fold 0%, Call 100%

**Player 0 after checking then facing a bet:**
- Jack: Fold 100%, Call 0%
- Queen: Fold 48.5%, Call 51.5%
- King: Fold 0%, Call 100%

### Strategy Properties
- All probabilities sum to exactly 1.0
- Strategies align with game theory principles:
  - Jack (worst card): mostly checks, occasionally bluffs
  - Queen (middle card): plays conservatively
  - King (best card): plays aggressively
  - Calling frequencies approximate theoretical 1/3 and 2/3 ratios

## Files Generated
- `strategy.json`: Complete behavioral strategy in required format
- `kuhn_cfr.py`: Full CFR+ implementation

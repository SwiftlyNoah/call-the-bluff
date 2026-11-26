#!/usr/bin/env python3
"""CFR+ solver for Leduc hold'em.

Chance is enumerated exactly: every deal (private cards for both players plus
the public card) is a root of the game tree with probability 1/120. Each CFR+
iteration traverses all roots, updating regrets for every information set.
The reported strategy is the linearly weighted average strategy (CFR+ style).
Exploitability is computed exactly with best responses over the full tree.
"""
import json
import sys
import time

DECK = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
RANK = {'J': 0, 'Q': 1, 'K': 2}
RAISE_SIZE = (2, 4)  # round 1, round 2
LETTER = {'fold': 'f', 'call': 'c', 'raise': 'r'}

ROOTS = [(a, b, p) for a in DECK for b in DECK if b != a
         for p in DECK if p not in (a, b)]
ROOT_PROB = 1.0 / len(ROOTS)  # 120 deals, uniform
BOTH = 2  # traversal that updates both players' regrets at once


def round_status(seq):
    if seq and seq[-1] == 'f':
        return 'folded'
    if len(seq) >= 2 and seq[-1] == 'c':  # a check or call after the round's first action
        return 'ended'
    return 'open'


def node(r1, r2):
    """Describe a state. r1: round-1 letters. r2: round-2 letters, or None if round 2 has not begun."""
    s1 = round_status(r1)
    if s1 == 'folded':
        return ('fold', (len(r1) - 1) % 2)
    if s1 == 'open':
        seq, rd = r1, 1
    else:
        seq, rd = (r2 or ''), 2
        s2 = round_status(seq)
        if s2 == 'folded':
            return ('fold', (len(seq) - 1) % 2)
        if s2 == 'ended':
            return ('showdown',)
    return ('act', len(seq) % 2, rd, seq)


def legal(seq):
    """Action names legal after the round-letters `seq`. Check is labeled 'call'."""
    acts = ['call']
    if seq.count('r') < 2:
        acts.append('raise')
    if seq and seq[-1] == 'r':
        acts.append('fold')
    return acts


def advance(r1, r2, rd, seq, action):
    new = seq + LETTER[action]
    if rd == 1:
        return new, r2
    return r1, new


def infoset_key(card, pub, r1, rd, seq):
    if rd == 1:
        return f"{card}:{seq}"
    return f"{card}|{pub}:{r1}/{seq}"


def contribs(r1, r2):
    c = [1, 1]  # antes
    for rd, seq in ((1, r1), (2, r2 or '')):
        size = RAISE_SIZE[rd - 1]
        for k, ch in enumerate(seq):
            p = k % 2
            if ch == 'r':
                c[p] = max(c) + size
            elif ch == 'c':
                c[p] = max(c)
    return c


def payoff0(cards, r1, r2):
    """Net payoff to player 0 at a terminal state."""
    nd = node(r1, r2)
    cont = contribs(r1, r2)
    if nd[0] == 'fold':
        return -cont[0] if nd[1] == 0 else cont[1]
    c0, c1, pub = cards
    k0 = (RANK[c0[0]] == RANK[pub[0]], RANK[c0[0]])
    k1 = (RANK[c1[0]] == RANK[pub[0]], RANK[c1[0]])
    if k0 > k1:
        return cont[1]
    if k0 < k1:
        return -cont[0]
    return 0.0


class CFRPlus:
    def __init__(self):
        self.info = {}  # key -> {'acts': [...], 'regret': [...], 'sum': [...]}

    def cfr(self, cards, r1, r2, pi0, pi1, t, upd):
        nd = node(r1, r2)
        if nd[0] != 'act':
            return payoff0(cards, r1, r2)
        _, p, rd, seq = nd
        c0, c1, pub = cards
        key = infoset_key(c0 if p == 0 else c1, pub, r1, rd, seq)
        acts = legal(seq)
        entry = self.info.get(key)
        if entry is None:
            entry = {'acts': acts, 'regret': [0.0] * len(acts), 'sum': [0.0] * len(acts)}
            self.info[key] = entry
        reg = entry['regret']
        tot = sum(reg)
        n = len(acts)
        strat = [x / tot for x in reg] if tot > 0 else [1.0 / n] * n

        utils = []
        for i, a in enumerate(acts):
            nr1, nr2 = advance(r1, r2, rd, seq, a)
            if p == 0:
                utils.append(self.cfr(cards, nr1, nr2, pi0 * strat[i], pi1, t, upd))
            else:
                utils.append(self.cfr(cards, nr1, nr2, pi0, pi1 * strat[i], t, upd))
        v = sum(s * u for s, u in zip(strat, utils))

        if upd != BOTH and p != upd:  # alternating updates: only the traversing player's nodes are updated
            return v
        own, opp = (pi0, pi1) if p == 0 else (pi1, pi0)
        sign = 1.0 if p == 0 else -1.0  # regret in the acting player's own utility
        for i in range(n):
            reg[i] = max(0.0, reg[i] + opp * sign * (utils[i] - v))
            entry['sum'][i] += t * own * strat[i]
        return v

    def average(self):
        avg = {}
        for key, e in self.info.items():
            s = e['sum']
            tot = sum(s)
            n = len(s)
            probs = [x / tot for x in s] if tot > 0 else [1.0 / n] * n
            avg[key] = dict(zip(e['acts'], probs))
        return avg


def ev_profile(cards, r1, r2, avg):
    """Expected payoff to player 0 when both players follow avg."""
    nd = node(r1, r2)
    if nd[0] != 'act':
        return payoff0(cards, r1, r2)
    _, p, rd, seq = nd
    c0, c1, pub = cards
    key = infoset_key(c0 if p == 0 else c1, pub, r1, rd, seq)
    total = 0.0
    for a, q in avg[key].items():
        nr1, nr2 = advance(r1, r2, rd, seq, a)
        total += q * ev_profile(cards, nr1, nr2, avg)
    return total


def best_response_value(br, avg):
    """Expected payoff to player `br` when it best-responds to avg of the other player."""
    opp = 1 - br
    bucket = {}  # key -> list of (cards, r1, r2, opponent reach incl. chance)

    def gather(cards, r1, r2, q):
        nd = node(r1, r2)
        if nd[0] != 'act':
            return
        _, p, rd, seq = nd
        c0, c1, pub = cards
        key = infoset_key(c0 if p == 0 else c1, pub, r1, rd, seq)
        acts = legal(seq)
        if p == br:
            bucket.setdefault(key, []).append((cards, r1, r2, q))
            for a in acts:
                nr1, nr2 = advance(r1, r2, rd, seq, a)
                gather(cards, nr1, nr2, q)
        else:
            for a in acts:
                nr1, nr2 = advance(r1, r2, rd, seq, a)
                gather(cards, nr1, nr2, q * avg[key][a])

    for root in ROOTS:
        gather(root, '', None, ROOT_PROB)

    w_memo = {}
    choice = {}

    def W(cards, r1, r2):
        """Expected utility to br from this state (conditional), under the best response."""
        k = (cards, r1, r2)
        if k in w_memo:
            return w_memo[k]
        nd = node(r1, r2)
        if nd[0] != 'act':
            u = payoff0(cards, r1, r2)
            res = u if br == 0 else -u
        else:
            _, p, rd, seq = nd
            c0, c1, pub = cards
            key = infoset_key(c0 if p == 0 else c1, pub, r1, rd, seq)
            acts = legal(seq)
            if p == br:
                if key not in choice:
                    best, best_val = None, None
                    for a in acts:
                        val = 0.0
                        for (sc, sr1, sr2, q) in bucket.get(key, []):
                            nr1, nr2 = advance(sr1, sr2, rd, seq, a)
                            val += q * W(sc, nr1, nr2)
                        if best_val is None or val > best_val:
                            best, best_val = a, val
                    choice[key] = best
                nr1, nr2 = advance(r1, r2, rd, seq, choice[key])
                res = W(cards, nr1, nr2)
            else:
                res = 0.0
                for a in acts:
                    nr1, nr2 = advance(r1, r2, rd, seq, a)
                    res += avg[key][a] * W(cards, nr1, nr2)
        w_memo[k] = res
        return res

    return sum(ROOT_PROB * W(root, '', None) for root in ROOTS)


def exploitability(avg):
    v0 = best_response_value(0, avg)  # P0 payoff when P0 best-responds
    v1 = best_response_value(1, avg)  # P1 payoff when P1 best-responds
    return (v0 + v1) / 2.0, v0, v1


def game_value(avg):
    return sum(ROOT_PROB * ev_profile(root, '', None, avg) for root in ROOTS)


def main():
    max_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    target = 1e-6
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'strategy.json'

    solver = CFRPlus()
    start = time.time()
    t = 0
    avg = None
    e = None
    for t in range(1, max_iters + 1):
        for root in ROOTS:
            solver.cfr(root, '', None, ROOT_PROB, ROOT_PROB, t, BOTH)
        if t % 250 == 0 or t == max_iters:
            avg = solver.average()
            e, v0, v1 = exploitability(avg)
            elapsed = time.time() - start
            print(f"iter {t:6d}  exploitability {e:.3e}  (BR0 {v0:+.6f}, BR1 {v1:+.6f})  "
                  f"{elapsed:.1f}s", flush=True)
            if e < target or elapsed > time_limit:
                break

    avg = solver.average()
    e, v0, v1 = exploitability(avg)
    value = game_value(avg)
    print(f"final iter {t}: exploitability {e:.6e}, P0 game value {value:.8f}", flush=True)

    probs = {}
    for key in sorted(avg):
        probs[key] = {a: repr(float(p)) for a, p in avg[key].items()}
    doc = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": f"CFR+ (exact chance enumeration, linear averaging), {t} iterations, "
                    f"exploitability {e:.3e}",
        "probs": probs,
    }
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=2)
        f.write('\n')
    print(f"wrote {out_path} with {len(probs)} information sets", flush=True)
    print(json.dumps({"iterations": t, "exploitability": e, "value_p0": value}))


if __name__ == '__main__':
    main()

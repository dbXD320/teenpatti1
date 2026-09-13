"""State abstraction for heads-up Teen Patti.

This module exists because of RULES.md 5.3, which is worth restating: the 741
strength classes (kappa) are lossless for COMPARING two hands but are NOT a
valid solver state abstraction. `KsKh7d` and `KsKh7s` are both kappa class
`K-K-7`, but the second removes a third spade from the deck and so induces a
different distribution over the opponent's flushes. They are not related by any
permutation of suits.

Two hands generate genuinely isomorphic subgames only if one maps to the other
under a permutation of the four suits, because nothing in the rules
distinguishes suits (RULES.md 2.1) and the deck is suit-symmetric. The quotient
by that group is the correct lossless abstraction, and computing its size is the
first deliverable here.

Both claims above are verified empirically, not assumed:

  * equity is constant within every suit-isomorphic class  -> the quotient is
    lossless;
  * equity is NOT constant within every kappa class        -> kappa is lossy,
    confirming the RULES.md 5.3 warning directly.

Equity is exact. For each of the 22,100 hands we count wins, ties and losses
against all 18,424 opponent hands drawable from the remaining 49 cards. No
Monte Carlo. The 407,170,400 pairwise comparisons are avoided by
inclusion-exclusion over the three cards of the hand (see `_exact_equity`),
which reduces the work to a few hundred thousand array operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from hands import (
    CLASS_KEYS,
    DECK,
    HAND_KAPPA,
    HANDS,
    N_CLASSES,
    N_HANDS,
    _SUIT_PERMS,
    _apply_suit_perm,
    canonical_hand,
    describe_kappa,
    hand_str,
    index_of,
)

#: Opponent hands available after three cards are removed: C(49,3).
N_OPPONENT_HANDS = 18424
DEFAULT_N_BUCKETS = 25


# --------------------------------------------------------------------------
# 1. Suit isomorphism -- the first deliverable
# --------------------------------------------------------------------------

def suit_isomorphic_classes() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Partition all 22,100 hands into suit-isomorphism classes.

    Returns `(class_of_hand, representatives)` where `class_of_hand[i]` is the
    class id of hand `i` and `representatives[c]` is the canonical hand of
    class `c`.
    """
    canon_to_id: dict[tuple, int] = {}
    class_of_hand = np.empty(N_HANDS, dtype=np.int32)
    representatives: list[tuple[int, int, int]] = []

    for i, hand in enumerate(HANDS):
        canon = canonical_hand(hand)
        cid = canon_to_id.get(canon)
        if cid is None:
            cid = len(representatives)
            canon_to_id[canon] = cid
            representatives.append(canon)
        class_of_hand[i] = cid

    return class_of_hand, representatives


def burnside_class_count() -> int:
    """Independent count of suit-isomorphism classes via Burnside's lemma.

    The number of orbits equals the mean number of hands fixed by a group
    element, averaged over the 24 permutations of suits. Computed separately
    from `suit_isomorphic_classes()` so the two must agree.
    """
    hand_set = {tuple(h) for h in HANDS}
    total_fixed = 0
    for perm in _SUIT_PERMS:
        total_fixed += sum(
            1 for h in hand_set if _apply_suit_perm(h, perm) == h
        )
    if total_fixed % len(_SUIT_PERMS) != 0:
        raise AssertionError("Burnside sum is not divisible by |S_4|")
    return total_fixed // len(_SUIT_PERMS)


# --------------------------------------------------------------------------
# 2. Exact equity by inclusion-exclusion
# --------------------------------------------------------------------------

def _kappa_histograms():
    """Histograms of kappa over hands containing given cards.

    Returns `(total, per_card, per_pair, pair_id)`:
      total[x]          -- hands with kappa == x, over all 22,100
      per_card[c, x]    -- ... among the 1,275 hands containing card c
      per_pair[p, x]    -- ... among the 50 hands containing both cards of pair p
      pair_id[a, b]     -- index into per_pair for the unordered pair {a, b}
    """
    n_pairs = len(DECK) * (len(DECK) - 1) // 2
    pair_id = np.full((len(DECK), len(DECK)), -1, dtype=np.int32)
    for p, (a, b) in enumerate(combinations(DECK, 2)):
        pair_id[a, b] = p
        pair_id[b, a] = p

    total = np.zeros(N_CLASSES, dtype=np.int64)
    per_card = np.zeros((len(DECK), N_CLASSES), dtype=np.int64)
    per_pair = np.zeros((n_pairs, N_CLASSES), dtype=np.int64)

    for i, (x, y, z) in enumerate(HANDS):
        k = HAND_KAPPA[i]
        total[k] += 1
        per_card[x, k] += 1
        per_card[y, k] += 1
        per_card[z, k] += 1
        per_pair[pair_id[x, y], k] += 1
        per_pair[pair_id[x, z], k] += 1
        per_pair[pair_id[y, z], k] += 1

    return total, per_card, per_pair, pair_id


def _exact_equity():
    """Exact win/tie/loss counts for every hand against a uniform opponent.

    For a hand h, the opponent's hands are the 3-subsets of the deck disjoint
    from h. Counting those by kappa directly would need 407,170,400
    comparisons. Instead, by inclusion-exclusion over h's three cards
    {x, y, z}, the count of disjoint hands with kappa == v is

        disjoint[v] = total[v]
                      - per_card[x,v] - per_card[y,v] - per_card[z,v]
                      + per_pair[xy,v] + per_pair[xz,v] + per_pair[yz,v]
                      - [v == kappa(h)]

    the last term removing h itself, which contains all three cards. Summing
    over v gives 22100 - 3*1275 + 3*50 - 1 = 18424, the required total, which
    is asserted for every hand.
    """
    total, per_card, per_pair, pair_id = _kappa_histograms()

    wins = np.zeros(N_HANDS, dtype=np.int64)
    ties = np.zeros(N_HANDS, dtype=np.int64)
    losses = np.zeros(N_HANDS, dtype=np.int64)

    for i, (x, y, z) in enumerate(HANDS):
        k = HAND_KAPPA[i]
        disjoint = (
            total
            - per_card[x] - per_card[y] - per_card[z]
            + per_pair[pair_id[x, y]]
            + per_pair[pair_id[x, z]]
            + per_pair[pair_id[y, z]]
        )
        disjoint[k] -= 1  # exclude h itself

        w = int(disjoint[:k].sum())
        t = int(disjoint[k])
        l = int(disjoint[k + 1:].sum())
        if w + t + l != N_OPPONENT_HANDS:
            raise AssertionError(
                f"hand {hand_str(HANDS[i])}: w+t+l = {w + t + l} != "
                f"{N_OPPONENT_HANDS}; inclusion-exclusion is wrong"
            )
        if disjoint.min() < 0:
            raise AssertionError(
                f"hand {hand_str(HANDS[i])}: negative opponent count"
            )
        wins[i], ties[i], losses[i] = w, t, l

    return wins, ties, losses


# --------------------------------------------------------------------------
# 3. The abstraction object
# --------------------------------------------------------------------------

@dataclass
class Abstraction:
    """Hand-strength abstraction over suit-isomorphic classes.

    Attributes
    ----------
    iso_class_of_hand : int32[22100]
        Suit-isomorphism class id of each hand.
    iso_representatives : list of hand
        Canonical representative of each isomorphism class.
    n_iso_classes : int
        The correct input to the Phase 2 memory estimate -- NOT 741.
    wins, ties, losses : int64[22100]
        Exact counts against all 18,424 opponent hands.
    equity : float64[22100]
        (wins + ties/2) / 18424.
    iso_equity : float64[n_iso_classes]
        Equity of each isomorphism class (constant within the class; verified).
    bucket_of_hand : int32[22100]
        Solver bucket, 0 = weakest.
    n_buckets : int
    """

    iso_class_of_hand: np.ndarray
    iso_representatives: list
    n_iso_classes: int
    wins: np.ndarray
    ties: np.ndarray
    losses: np.ndarray
    equity: np.ndarray
    iso_equity: np.ndarray
    bucket_of_hand: np.ndarray
    n_buckets: int
    iso_class_size: np.ndarray

    # -- forward and reverse mappings (required so Phase 2 and the later
    #    benchmark can go from a bucket back to concrete hands) --------------

    def bucket_of(self, hand) -> int:
        return int(self.bucket_of_hand[index_of(hand)])

    def hands_in_bucket(self, bucket: int) -> np.ndarray:
        """Indices of every hand in `bucket`."""
        return np.flatnonzero(self.bucket_of_hand == bucket)

    def iso_classes_in_bucket(self, bucket: int) -> list[int]:
        classes = self.iso_class_of_hand[self.bucket_of_hand == bucket]
        return sorted(set(int(c) for c in classes))

    def representative_hands(self, bucket: int, limit: int | None = None):
        """One representative hand per isomorphism class in `bucket`."""
        reps = [self.iso_representatives[c] for c in self.iso_classes_in_bucket(bucket)]
        reps.sort(key=lambda h: -self.equity[index_of(h)])
        return reps if limit is None else reps[:limit]

    def bucket_size(self, bucket: int) -> int:
        return int((self.bucket_of_hand == bucket).sum())

    def bucket_equity_range(self, bucket: int) -> tuple[float, float]:
        eq = self.equity[self.bucket_of_hand == bucket]
        return (float(eq.min()), float(eq.max()))

    def summary_rows(self):
        """Per-bucket (bucket, n_hands, n_iso_classes, eq_lo, eq_hi, example)."""
        rows = []
        for b in range(self.n_buckets):
            lo, hi = self.bucket_equity_range(b)
            reps = self.representative_hands(b, limit=1)
            example = hand_str(reps[0]) if reps else "-"
            rows.append(
                (
                    b,
                    self.bucket_size(b),
                    len(self.iso_classes_in_bucket(b)),
                    lo,
                    hi,
                    example,
                )
            )
        return rows


def _bucket_by_equity(iso_equity, iso_class_size, n_buckets):
    """Assign isomorphism classes to buckets of roughly equal hand mass.

    Bucketing is done over isomorphism classes, never over kappa classes
    (RULES.md 5.3). Classes are ordered by equity and split so each bucket
    holds about 22100/n_buckets hands; a class is never split across buckets,
    so bucket boundaries land on equity ties and sizes are only approximately
    equal.
    """
    order = np.argsort(iso_equity, kind="stable")
    target = N_HANDS / n_buckets

    bucket_of_class = np.zeros(len(iso_equity), dtype=np.int32)
    bucket, running = 0, 0
    for cid in order:
        size = int(iso_class_size[cid])
        # Move to the next bucket once this one is full, but never leave a
        # bucket empty and never exceed the requested count.
        if running >= target * (bucket + 1) and bucket < n_buckets - 1:
            bucket += 1
        bucket_of_class[cid] = bucket
        running += size

    return bucket_of_class


def build_abstraction(n_buckets: int = DEFAULT_N_BUCKETS, verify: bool = True) -> Abstraction:
    """Build the full abstraction. Takes a few seconds; call once and reuse."""
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

    iso_class_of_hand, iso_representatives = suit_isomorphic_classes()
    n_iso = len(iso_representatives)

    wins, ties, losses = _exact_equity()
    equity = (wins + ties / 2.0) / N_OPPONENT_HANDS

    iso_class_size = np.bincount(iso_class_of_hand, minlength=n_iso)

    # Equity must be constant within an isomorphism class; take a representative
    # value and verify the rest agree.
    iso_equity = np.zeros(n_iso, dtype=np.float64)
    first_seen = np.full(n_iso, -1, dtype=np.int64)
    for i in range(N_HANDS):
        c = iso_class_of_hand[i]
        if first_seen[c] < 0:
            first_seen[c] = i
            iso_equity[c] = equity[i]

    if verify:
        _verify_losslessness(iso_class_of_hand, wins, ties, losses)

    bucket_of_class = _bucket_by_equity(iso_equity, iso_class_size, n_buckets)
    bucket_of_hand = bucket_of_class[iso_class_of_hand].astype(np.int32)

    return Abstraction(
        iso_class_of_hand=iso_class_of_hand,
        iso_representatives=iso_representatives,
        n_iso_classes=n_iso,
        wins=wins,
        ties=ties,
        losses=losses,
        equity=equity,
        iso_equity=iso_equity,
        bucket_of_hand=bucket_of_hand,
        n_buckets=n_buckets,
        iso_class_size=iso_class_size,
    )


def _verify_losslessness(iso_class_of_hand, wins, ties, losses) -> None:
    """Assert the suit-isomorphic quotient preserves exact equity."""
    seen: dict[int, tuple[int, int, int]] = {}
    for i in range(N_HANDS):
        c = int(iso_class_of_hand[i])
        wtl = (int(wins[i]), int(ties[i]), int(losses[i]))
        if c not in seen:
            seen[c] = wtl
        elif seen[c] != wtl:
            raise AssertionError(
                f"suit isomorphism is not equity-preserving: class {c} has "
                f"{seen[c]} and {wtl}; the canonicalisation is wrong"
            )


def kappa_classes_with_varying_equity(wins, ties, losses):
    """kappa classes whose members do NOT all share the same equity.

    A non-empty result is a direct empirical demonstration that kappa is a
    lossy state abstraction (RULES.md 5.3). Returns a list of
    (kappa, n_distinct_equities, worked_example).
    """
    by_kappa: dict[int, dict[tuple[int, int, int], int]] = {}
    for i in range(N_HANDS):
        k = HAND_KAPPA[i]
        wtl = (int(wins[i]), int(ties[i]), int(losses[i]))
        by_kappa.setdefault(k, {}).setdefault(wtl, i)

    out = []
    for k, variants in by_kappa.items():
        if len(variants) > 1:
            examples = [HANDS[i] for i in variants.values()]
            out.append((k, len(variants), examples))
    out.sort(key=lambda row: (-row[1], row[0]))
    return out


# --------------------------------------------------------------------------
# CLI: report the numbers PHASE1.md needs
# --------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - reporting path
    import sys

    n_buckets = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_BUCKETS

    print("Building abstraction (exact enumeration, no Monte Carlo)...")
    ab = build_abstraction(n_buckets)

    print()
    print("=== Suit isomorphism (RULES.md 5.3) ===")
    print(f"  kappa strength classes (comparison only) : {N_CLASSES}")
    print(f"  suit-isomorphic classes (correct quotient): {ab.n_iso_classes}")
    burnside = burnside_class_count()
    print(f"  Burnside cross-check                     : {burnside}"
          f"  {'OK' if burnside == ab.n_iso_classes else 'MISMATCH'}")
    print(f"  ratio vs kappa                           : "
          f"{ab.n_iso_classes / N_CLASSES:.2f}x")

    print()
    print("=== kappa is lossy: classes with non-constant equity ===")
    varying = kappa_classes_with_varying_equity(ab.wins, ab.ties, ab.losses)
    print(f"  kappa classes with >1 distinct equity: {len(varying)} of {N_CLASSES}")
    for k, n, examples in varying[:3]:
        print(f"    {describe_kappa(k)}: {n} distinct equities")
        for h in examples[:3]:
            i = index_of(h)
            print(f"      {hand_str(h):12s} equity={ab.equity[i]:.6f} "
                  f"w/t/l={ab.wins[i]}/{ab.ties[i]}/{ab.losses[i]}")

    print()
    print(f"=== Buckets (n={ab.n_buckets}) ===")
    print("  bkt  hands  iso   equity range           example")
    for b, n_hands, n_iso, lo, hi, example in ab.summary_rows():
        print(f"  {b:3d} {n_hands:6d} {n_iso:4d}   {lo:.4f} - {hi:.4f}   {example}")


if __name__ == "__main__":  # pragma: no cover
    main()

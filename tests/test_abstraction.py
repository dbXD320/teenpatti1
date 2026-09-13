"""Tests for abstraction.py.

The central claim under test is RULES.md 5.3: kappa is lossless for comparison
but lossy as a state abstraction, while the suit-isomorphic quotient is
lossless. Both directions are checked empirically rather than assumed.
"""

import numpy as np
import pytest

import abstraction as AB
from hands import (
    HANDS,
    N_CLASSES,
    N_HANDS,
    canonical_hand,
    index_of,
    kappa,
    parse_hand,
)


@pytest.fixture(scope="module")
def ab():
    return AB.build_abstraction(AB.DEFAULT_N_BUCKETS)


# --------------------------------------------------------------------------
# Suit isomorphism -- the first deliverable
# --------------------------------------------------------------------------

def test_suit_isomorphic_class_count_is_1755(ab):
    """The correct input to the Phase 2 memory estimate, NOT 741."""
    assert ab.n_iso_classes == 1755
    assert ab.n_iso_classes != N_CLASSES


def test_burnside_agrees_with_canonicalisation(ab):
    """Two independent computations of the orbit count must match."""
    assert AB.burnside_class_count() == ab.n_iso_classes == 1755


def test_iso_classes_partition_all_hands(ab):
    assert len(ab.iso_class_of_hand) == N_HANDS
    assert set(ab.iso_class_of_hand.tolist()) == set(range(ab.n_iso_classes))
    assert int(ab.iso_class_size.sum()) == N_HANDS


def test_iso_class_sizes_divide_the_group_order(ab):
    """Every orbit size must divide |S_4| = 24 (orbit-stabiliser theorem)."""
    for size in set(ab.iso_class_size.tolist()):
        assert 24 % size == 0, f"orbit of size {size} cannot arise from S_4"


def test_canonicalisation_is_idempotent_and_suit_invariant():
    for hand in HANDS[::811]:
        canon = canonical_hand(hand)
        assert canonical_hand(canon) == canon
    # Hands differing only by a global suit relabelling share a canonical form.
    assert canonical_hand(parse_hand("As 9s 5s")) == canonical_hand(
        parse_hand("Ah 9h 5h")
    )
    assert canonical_hand(parse_hand("Ks Kh 7d")) == canonical_hand(
        parse_hand("Kc Kd 7h")
    )


def test_isomorphic_hands_share_kappa(ab):
    """Suit isomorphism refines kappa: same orbit implies same strength."""
    seen = {}
    for i in range(N_HANDS):
        c = int(ab.iso_class_of_hand[i])
        k = kappa(HANDS[i])
        if c in seen:
            assert seen[c] == k, "an orbit spanned two strength classes"
        else:
            seen[c] = k


def test_iso_is_strictly_finer_than_kappa(ab):
    """1755 orbits over 741 classes, so some kappa class splits."""
    assert ab.n_iso_classes > N_CLASSES
    classes_per_kappa = {}
    for i in range(N_HANDS):
        classes_per_kappa.setdefault(kappa(HANDS[i]), set()).add(
            int(ab.iso_class_of_hand[i])
        )
    split = {k: v for k, v in classes_per_kappa.items() if len(v) > 1}
    assert split, "expected at least one kappa class to split into orbits"
    assert sum(len(v) for v in classes_per_kappa.values()) == ab.n_iso_classes


# --------------------------------------------------------------------------
# Exact equity
# --------------------------------------------------------------------------

def test_equity_counts_are_exact_and_complete(ab):
    """Every hand must be scored against all 18,424 opponent hands."""
    totals = ab.wins + ab.ties + ab.losses
    assert np.all(totals == AB.N_OPPONENT_HANDS)
    assert ab.wins.min() >= 0 and ab.losses.min() >= 0 and ab.ties.min() >= 0


def test_equity_is_in_range(ab):
    assert ab.equity.min() > 0.0
    assert ab.equity.max() <= 1.0


def test_mean_equity_increases_strictly_across_categories(ab):
    """Equity respects the RULES.md 4.1 category order in aggregate."""
    from hands import (
        COLOUR,
        HIGH_CARD,
        PAIR,
        PURE_SEQUENCE,
        SEQUENCE,
        TRAIL,
        category_of_kappa,
    )

    kappas = np.array([kappa(h) for h in HANDS])
    means = []
    for cat in (HIGH_CARD, PAIR, COLOUR, SEQUENCE, PURE_SEQUENCE, TRAIL):
        mask = np.array([category_of_kappa(k) == cat for k in kappas])
        means.append(ab.equity[mask].mean())
    assert means == sorted(means), "category mean equity must increase with rank"


def test_equity_is_not_monotone_across_adjacent_kappa_classes(ab):
    """Equity ordering and hand-strength ordering are DIFFERENT orderings.

    A stronger class can have lower equity than a weaker one, because equity
    is measured against a uniform opponent and card removal differs between
    classes. The lowest pair (2-2-3) removes two deuces and a three, leaving
    the opponent's high ranks intact; the highest high card (A-K-J) removes an
    ace, a king and a jack, blocking many of the opponent's strong hands. So
    A-K-J's best case outruns 2-2-3's worst case, even though 2-2-3 beats
    A-K-J every time they meet.

    Phase 2/3 consequence: equity buckets must never be used to answer "which
    hand is stronger". That question is settled by kappa alone.
    """
    kappas = np.array([kappa(h) for h in HANDS])
    lo, hi = {}, {}
    for k in np.unique(kappas):
        band = ab.equity[kappas == k]
        lo[k], hi[k] = band.min(), band.max()

    overlaps = [
        hi[k - 1] - lo[k]
        for k in range(1, N_CLASSES)
        if lo[k] < hi[k - 1] - 1e-15
    ]
    assert len(overlaps) == 208, "the overlap structure changed"
    assert max(overlaps) < 0.01, "overlaps should be small, not gross"

    # The documented instance: the weakest pair vs the strongest high card.
    weakest_pair = ab.equity[index_of(parse_hand("2c 2d 3h"))]
    strongest_high = ab.equity[index_of(parse_hand("As Kh Jd"))]
    assert weakest_pair < strongest_high
    # ...but head to head the pair still wins, always.
    assert kappa(parse_hand("2c 2d 3h")) > kappa(parse_hand("As Kh Jd"))


def test_best_hand_is_the_top_trail(ab):
    best = int(np.argmax(ab.equity))
    assert kappa(HANDS[best]) == N_CLASSES - 1  # A-A-A
    # Three aces cannot lose; the only non-win is a tie, which is impossible
    # for a trail (RULES.md 4.4).
    assert ab.losses[best] == 0
    assert ab.ties[best] == 0
    assert ab.equity[best] == 1.0


def test_worst_hand_is_the_lowest_high_card(ab):
    worst = int(np.argmin(ab.equity))
    assert kappa(HANDS[worst]) == 0  # high card 5-3-2
    assert ab.wins[worst] == 0


def test_equity_reproduced_by_brute_force_on_a_sample(ab):
    """Independent O(18424) check of the inclusion-exclusion shortcut."""
    from itertools import combinations

    from hands import DECK, HAND_KAPPA

    rng = np.random.default_rng(0)
    for i in rng.choice(N_HANDS, size=6, replace=False):
        hand = HANDS[int(i)]
        k = kappa(hand)
        remaining = [c for c in DECK if c not in hand]
        w = t = l = 0
        for opp in combinations(remaining, 3):
            ok = HAND_KAPPA[index_of(opp)]
            if k > ok:
                w += 1
            elif k == ok:
                t += 1
            else:
                l += 1
        assert (w, t, l) == (
            int(ab.wins[i]),
            int(ab.ties[i]),
            int(ab.losses[i]),
        ), f"inclusion-exclusion disagrees for {hand}"
        assert w + t + l == AB.N_OPPONENT_HANDS


# --------------------------------------------------------------------------
# RULES.md 5.3: the losslessness claims, both directions
# --------------------------------------------------------------------------

def test_suit_isomorphism_is_equity_preserving(ab):
    """Direction 1: the orbit quotient is lossless."""
    AB._verify_losslessness(ab.iso_class_of_hand, ab.wins, ab.ties, ab.losses)
    # And the cached per-class equity agrees hand by hand.
    assert np.allclose(ab.iso_equity[ab.iso_class_of_hand], ab.equity)


def test_kappa_is_not_equity_preserving(ab):
    """Direction 2: kappa is LOSSY. This is the RULES.md 5.3 warning."""
    varying = AB.kappa_classes_with_varying_equity(ab.wins, ab.ties, ab.losses)
    assert varying, (
        "kappa appeared lossless, contradicting RULES.md 5.3; either the "
        "equity computation or the card-removal reasoning is wrong"
    )
    assert len(varying) == 442


def test_the_rules_md_worked_example_of_kappa_lossiness(ab):
    """RULES.md 5.3 names KsKh7d vs KsKh7s. They must differ in equity."""
    a = parse_hand("Ks Kh 7d")
    b = parse_hand("Ks Kh 7s")
    ia, ib = index_of(a), index_of(b)

    assert kappa(a) == kappa(b), "both are kappa class K-K-7"
    assert canonical_hand(a) != canonical_hand(b), "not suit-isomorphic"
    assert ab.iso_class_of_hand[ia] != ab.iso_class_of_hand[ib]
    assert (ab.wins[ia], ab.ties[ia], ab.losses[ia]) != (
        ab.wins[ib],
        ab.ties[ib],
        ab.losses[ib],
    ), "the spec's own example must exhibit the difference"

    # Direction, and why. KsKh7s concentrates two removals in spades and none
    # in diamonds/clubs; KsKh7d spreads one removal across three suits.
    # Opponent flushes total sum_suits C(13 - used, 3), and C(n,3) is convex,
    # so concentrating removals leaves MORE opponent flushes overall:
    #   KsKh7d -> C(12,3)*3 + C(13,3)      = 946
    #   KsKh7s -> C(11,3) + C(12,3) + C(13,3)*2 = 957
    # Both hands hold the same ranks {K,K,7}, so every rank-determined count is
    # identical and the entire loss difference is those 11 flushes.
    from math import comb

    def opponent_flushes(hand):
        used = {}
        for c in hand:
            used[c & 3] = used.get(c & 3, 0) + 1
        return sum(comb(13 - used.get(s, 0), 3) for s in range(4))

    assert opponent_flushes(a) == 946
    assert opponent_flushes(b) == 957
    assert int(ab.losses[ib] - ab.losses[ia]) == 11 == 957 - 946
    assert ab.equity[ib] < ab.equity[ia]


# --------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------

def test_buckets_are_built_over_iso_classes_not_kappa(ab):
    """No isomorphism class may straddle two buckets."""
    for c in range(ab.n_iso_classes):
        members = np.flatnonzero(ab.iso_class_of_hand == c)
        assert len(set(ab.bucket_of_hand[members].tolist())) == 1


def test_all_buckets_are_populated_and_ordered(ab):
    assert ab.n_buckets == 25
    sizes = [ab.bucket_size(b) for b in range(ab.n_buckets)]
    assert all(s > 0 for s in sizes)
    assert sum(sizes) == N_HANDS
    # Buckets must be ordered by equity with no overlap.
    prev_hi = -1.0
    for b in range(ab.n_buckets):
        lo, hi = ab.bucket_equity_range(b)
        assert lo >= prev_hi - 1e-12
        prev_hi = hi


def test_buckets_are_roughly_equal_in_hand_mass(ab):
    sizes = np.array([ab.bucket_size(b) for b in range(ab.n_buckets)])
    target = N_HANDS / ab.n_buckets
    # Classes are never split, so sizes are only approximately equal.
    assert sizes.max() < target * 1.5
    assert sizes.min() > target * 0.5


def test_bucket_mapping_is_invertible(ab):
    """Phase 2 and the benchmark need bucket -> representative hands."""
    for b in range(ab.n_buckets):
        members = ab.hands_in_bucket(b)
        assert len(members) == ab.bucket_size(b)
        for i in members[:5]:
            assert ab.bucket_of(HANDS[int(i)]) == b
        reps = ab.representative_hands(b)
        assert reps, f"bucket {b} has no representative hands"
        assert len(reps) == len(ab.iso_classes_in_bucket(b))
        for hand in reps:
            assert ab.bucket_of(hand) == b


def test_strongest_bucket_contains_the_best_trail(ab):
    assert ab.bucket_of(parse_hand("As Ah Ad")) == ab.n_buckets - 1


def test_weakest_bucket_contains_the_worst_high_card(ab):
    assert ab.bucket_of(parse_hand("5s 3h 2d")) == 0


def test_bucket_count_is_a_parameter():
    small = AB.build_abstraction(n_buckets=5, verify=False)
    assert small.n_buckets == 5
    assert all(small.bucket_size(b) > 0 for b in range(5))
    assert small.n_iso_classes == 1755  # independent of bucketing
    with pytest.raises(ValueError):
        AB.build_abstraction(n_buckets=0)


def test_summary_rows_are_well_formed(ab):
    rows = ab.summary_rows()
    assert len(rows) == ab.n_buckets
    for b, n_hands, n_iso, lo, hi, example in rows:
        assert n_hands > 0 and n_iso > 0
        assert 0.0 <= lo <= hi <= 1.0
        assert len(example.split()) == 3

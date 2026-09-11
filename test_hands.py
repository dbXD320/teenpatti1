"""Tests for hands.py -- every combinatorial claim in RULES.md 4 and 5."""

from itertools import combinations

import pytest

import hands as H

# --------------------------------------------------------------------------
# RULES.md 5.1: the category counts, by brute-force enumeration
# --------------------------------------------------------------------------

RULES_CATEGORY_COUNTS = {
    "trail": 52,
    "pure_sequence": 48,
    "sequence": 720,
    "colour": 1096,
    "pair": 3744,
    "high_card": 16440,
}


def test_total_hands_is_22100():
    assert H.N_HANDS == 22100
    assert len(list(combinations(H.DECK, 3))) == 22100


def test_category_counts_match_rules_md():
    counts = {H.CATEGORY_NAMES[c]: n for c, n in H.category_counts().items()}
    assert counts == RULES_CATEGORY_COUNTS


def test_category_counts_sum_to_22100():
    assert sum(RULES_CATEGORY_COUNTS.values()) == 22100


def test_flush_cross_check():
    """RULES.md 5.1: 4*C(13,3) = 1144 = 48 pure sequences + 1096 colours."""
    flushes = sum(
        1 for h in H.HANDS if len({H.card_suit(c) for c in h}) == 1
    )
    assert flushes == 1144
    assert RULES_CATEGORY_COUNTS["pure_sequence"] + RULES_CATEGORY_COUNTS["colour"] == 1144


def test_straight_cross_check():
    """RULES.md 5.1: 12*4^3 = 768 = 48 pure sequences + 720 sequences."""
    runs = sum(
        1
        for h in H.HANDS
        if H._run_strength(frozenset(H.card_rank(c) for c in h)) is not None
    )
    assert runs == 768
    assert RULES_CATEGORY_COUNTS["pure_sequence"] + RULES_CATEGORY_COUNTS["sequence"] == 768


def test_trail_is_more_common_than_pure_sequence():
    """RULES.md 5.2: the ranking inverts rarity at the top of the scale."""
    assert RULES_CATEGORY_COUNTS["trail"] == 52
    assert RULES_CATEGORY_COUNTS["pure_sequence"] == 48
    assert RULES_CATEGORY_COUNTS["trail"] > RULES_CATEGORY_COUNTS["pure_sequence"]
    # ...yet trail outranks pure sequence.
    trail = H.kappa(H.parse_hand("2c 2d 2h"))  # the WEAKEST trail
    pure = H.kappa(H.parse_hand("As 2s 3s"))   # the STRONGEST pure sequence
    assert trail > pure


# --------------------------------------------------------------------------
# RULES.md 4.2: exactly 12 runs, A-2-3 highest, 2-A-K invalid
# --------------------------------------------------------------------------

def test_exactly_twelve_valid_runs():
    assert len(H.valid_runs()) == 12
    assert len(set(H.valid_runs())) == 12


def test_two_ace_king_is_not_a_run():
    """RULES.md E11: wrap-around is prohibited; {2, A, K} is High Card."""
    assert H._run_strength(frozenset({2, 14, 13})) is None
    hand = H.parse_hand("2s Ah Kd")
    assert H.category_of(hand) == H.HIGH_CARD
    # ...and the suited version is a Colour, not a Pure Sequence.
    assert H.category_of(H.parse_hand("2s As Ks")) == H.COLOUR


def test_ace_low_wheel_is_the_highest_run():
    """RULES.md 4.2: A-2-3 > A-K-Q > K-Q-J > ... > 4-3-2."""
    order = [
        "As 2s 3s", "As Ks Qs", "Ks Qs Js", "Qs Js Ts", "Js Ts 9s",
        "Ts 9s 8s", "9s 8s 7s", "8s 7s 6s", "7s 6s 5s", "6s 5s 4s",
        "5s 4s 3s", "4s 3s 2s",
    ]
    kappas = [H.kappa(H.parse_hand(t)) for t in order]
    assert kappas == sorted(kappas, reverse=True), "run order must be strictly descending"
    assert len(set(kappas)) == 12


def test_run_ordering_identical_for_pure_and_impure():
    """RULES.md 4.2: the ordering applies identically to both categories."""
    pure = [H.kappa(H.parse_hand(t)) for t in ("As 2s 3s", "As Ks Qs", "4s 3s 2s")]
    impure = [H.kappa(H.parse_hand(t)) for t in ("Ah 2s 3s", "Ah Ks Qs", "4h 3s 2s")]
    assert pure == sorted(pure, reverse=True)
    assert impure == sorted(impure, reverse=True)
    # every pure sequence beats every sequence
    assert min(pure) > max(impure)


# --------------------------------------------------------------------------
# RULES.md 4.3 / 5.3: 741 strength classes and their extremes
# --------------------------------------------------------------------------

def test_741_kappa_classes():
    assert H.N_CLASSES == 741
    assert len(set(H.HAND_KAPPA)) == 741


def test_class_counts_per_category():
    counts = {H.CATEGORY_NAMES[c]: n for c, n in H.class_counts().items()}
    assert counts == {
        "trail": 13,
        "pure_sequence": 12,
        "sequence": 12,
        "colour": 274,
        "pair": 156,
        "high_card": 274,
    }
    assert sum(counts.values()) == 741


@pytest.mark.parametrize(
    "category, highest, lowest",
    [
        ("trail", "As Ah Ad", "2s 2h 2d"),
        ("colour", "As Ks Js", "5s 3s 2s"),
        ("pair", "As Ah Kd", "2s 2h 3d"),
        ("high_card", "As Kh Jd", "5s 3h 2d"),
    ],
)
def test_category_extremes_match_rules_md(category, highest, lowest):
    """RULES.md 4.3 commits to the highest and lowest hand in each category."""
    cat = next(c for c, name in H.CATEGORY_NAMES.items() if name == category)
    members = [k for k in range(H.N_CLASSES) if H.category_of_kappa(k) == cat]
    assert H.kappa(H.parse_hand(highest)) == max(members)
    assert H.kappa(H.parse_hand(lowest)) == min(members)


def test_category_order_is_total_and_correct():
    """RULES.md 4.1: trail > pure seq > seq > colour > pair > high card."""
    order = [H.TRAIL, H.PURE_SEQUENCE, H.SEQUENCE, H.COLOUR, H.PAIR, H.HIGH_CARD]
    ranges = {}
    for cat in order:
        members = [k for k in range(H.N_CLASSES) if H.category_of_kappa(k) == cat]
        ranges[cat] = (min(members), max(members))
    # Each category occupies a contiguous band strictly above the next.
    for stronger, weaker in zip(order, order[1:]):
        assert ranges[stronger][0] > ranges[weaker][1], (
            f"{H.CATEGORY_NAMES[stronger]} must outrank "
            f"{H.CATEGORY_NAMES[weaker]} without overlap"
        )


def test_pair_maximum_is_aak_not_aaa():
    """RULES.md 4.3 note: A-A-A is a Trail, so the Pair maximum is A-A-K."""
    assert H.category_of(H.parse_hand("As Ah Ad")) == H.TRAIL
    assert H.category_of(H.parse_hand("As Ah Kd")) == H.PAIR


def test_colour_and_high_card_share_rank_space():
    """RULES.md 4.3: both are 3 distinct non-consecutive ranks -> 274 each."""
    assert H.class_counts()[H.COLOUR] == H.class_counts()[H.HIGH_CARD] == 274


def test_akq_cannot_be_colour_or_high_card():
    assert H.category_of(H.parse_hand("As Ks Qs")) == H.PURE_SEQUENCE
    assert H.category_of(H.parse_hand("As Kh Qs")) == H.SEQUENCE


# --------------------------------------------------------------------------
# RULES.md 2.1 / 4.4: suits are unordered, so exact ties exist
# --------------------------------------------------------------------------

def test_suits_never_break_ties():
    """RULES.md E12: a suit-only difference must compare equal."""
    assert H.compare(H.parse_hand("Ks Kh 7d"), H.parse_hand("Kd Kc 7s")) == 0
    assert H.compare(H.parse_hand("As 9s 5s"), H.parse_hand("Ah 9h 5h")) == 0
    assert H.compare(H.parse_hand("As 2s 3s"), H.parse_hand("Ah 2h 3h")) == 0


def test_trail_ties_are_impossible():
    """RULES.md 4.4 / E9: two equal trails would need 6 cards of one rank."""
    trails_by_kappa = {}
    for h in H.HANDS:
        if H.category_of(h) == H.TRAIL:
            trails_by_kappa.setdefault(H.kappa(h), []).append(h)
    for kappa_val, group in trails_by_kappa.items():
        # 4 trails share each rank, but any two of them overlap in cards, so
        # they can never be dealt to two players simultaneously.
        for a, b in combinations(group, 2):
            assert set(a) & set(b), (
                f"two disjoint trails of the same rank would be a tie: {a} {b}"
            )


@pytest.mark.parametrize(
    "category",
    ["pure_sequence", "sequence", "colour", "pair", "high_card"],
)
def test_ties_are_reachable_in_every_other_category(category):
    """RULES.md 4.4: ties are a live case in all categories except Trail."""
    cat = next(c for c, name in H.CATEGORY_NAMES.items() if name == category)
    by_kappa = {}
    for h in H.HANDS:
        if H.category_of(h) == cat:
            by_kappa.setdefault(H.kappa(h), []).append(h)
    found = any(
        not (set(a) & set(b))
        for group in by_kappa.values()
        for a, b in combinations(group, 2)
    )
    assert found, f"expected a reachable tie within {category}"


# --------------------------------------------------------------------------
# Evaluator invariants
# --------------------------------------------------------------------------

def test_kappa_is_a_total_order_consistent_with_classify():
    for h in H.HANDS[::97]:
        assert H.CLASS_KEYS[H.kappa(h)] == H.classify(h)


def test_comparison_is_antisymmetric():
    sample = H.HANDS[::1013]
    for a in sample:
        for b in sample:
            assert H.compare(a, b) == -H.compare(b, a)


def test_kappa_ordering_agrees_with_class_key_ordering():
    """A larger kappa must mean a lexicographically larger class key."""
    for k in range(H.N_CLASSES - 1):
        assert H.CLASS_KEYS[k] < H.CLASS_KEYS[k + 1]


def test_parse_and_render_roundtrip():
    for text in ("As Kh Qd", "2c 3d 4h", "Ts Th Td"):
        hand = H.parse_hand(text)
        assert H.parse_hand(H.hand_str(hand)) == hand


def test_rejects_malformed_hands():
    with pytest.raises(ValueError):
        H.parse_hand("As Ah")
    with pytest.raises(ValueError):
        H.parse_hand("As As Kd")  # duplicate card
    with pytest.raises(ValueError):
        H.classify((0, 1))
    with pytest.raises(ValueError):
        H.index_of((0, 1, 1))

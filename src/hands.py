"""Hand evaluation for heads-up Teen Patti.

Implements RULES.md sections 2 (primitives), 4 (hand ranking) and 5 (enumeration).

The entire 22,100-hand space is enumerated once at import time and reduced to a
lookup table `HAND_KAPPA`, mapping a hand index to a comparable integer strength
class. No hand is evaluated on the fly during solving (RULES.md deliverable 1).

Card encoding
-------------
A card is an int in [0, 52):  rank = 2 + (card >> 2),  suit = card & 3.
Ranks are 2..14 with 14 = ace. Suits are 0..3 and are UNORDERED: no function in
this module ever compares them (RULES.md 2.1). Consequently exact ties exist
(RULES.md 4.4) and are represented by equal kappa.

A hand is a 3-tuple of card ints in ascending order.

Strength classes (kappa)
------------------------
`kappa(h)` is an int in [0, 741). Player i beats player j iff
kappa(h_i) > kappa(h_j); a tie is exactly kappa(h_i) == kappa(h_j)
(RULES.md 5.3). This is the ONLY sanctioned comparison.

kappa is lossless for comparison but is NOT a valid solver state abstraction --
see RULES.md 5.3 and `abstraction.py`, which computes the correct quotient.
"""

from __future__ import annotations

from itertools import combinations, permutations

# --------------------------------------------------------------------------
# Primitives (RULES.md 2.1)
# --------------------------------------------------------------------------

RANKS = tuple(range(2, 15))  # 14 == ace
SUITS = (0, 1, 2, 3)
N_CARDS = 52
DECK = tuple(range(N_CARDS))

RANK_CHARS = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
SUIT_CHARS = ("c", "d", "h", "s")


def card_rank(card: int) -> int:
    return 2 + (card >> 2)


def card_suit(card: int) -> int:
    return card & 3


def make_card(rank: int, suit: int) -> int:
    if rank not in RANKS:
        raise ValueError(f"bad rank {rank!r}")
    if suit not in SUITS:
        raise ValueError(f"bad suit {suit!r}")
    return ((rank - 2) << 2) | suit


def card_str(card: int) -> str:
    return RANK_CHARS[card_rank(card)] + SUIT_CHARS[card_suit(card)]


def hand_str(hand) -> str:
    return " ".join(card_str(c) for c in sorted(hand, reverse=True))


def parse_card(text: str) -> int:
    """Parse e.g. 'As', 'Th', 'Kc'. Used by tests and the RULES.md examples."""
    text = text.strip()
    if len(text) != 2:
        raise ValueError(f"bad card {text!r}")
    rank_char, suit_char = text[0].upper(), text[1].lower()
    rank = next((r for r, ch in RANK_CHARS.items() if ch == rank_char), None)
    if rank is None:
        raise ValueError(f"bad rank in {text!r}")
    if suit_char not in SUIT_CHARS:
        raise ValueError(f"bad suit in {text!r}")
    return make_card(rank, SUIT_CHARS.index(suit_char))


def parse_hand(text: str):
    """Parse e.g. 'Ah As 9d' into a sorted 3-tuple of card ints."""
    cards = tuple(parse_card(t) for t in text.split())
    if len(cards) != 3:
        raise ValueError(f"expected 3 cards, got {len(cards)} in {text!r}")
    if len(set(cards)) != 3:
        raise ValueError(f"duplicate cards in {text!r}")
    return tuple(sorted(cards))


# --------------------------------------------------------------------------
# Categories (RULES.md 4.1) -- higher int is a stronger category
# --------------------------------------------------------------------------

HIGH_CARD = 0
PAIR = 1
COLOUR = 2
SEQUENCE = 3
PURE_SEQUENCE = 4
TRAIL = 5

CATEGORY_NAMES = {
    TRAIL: "trail",
    PURE_SEQUENCE: "pure_sequence",
    SEQUENCE: "sequence",
    COLOUR: "colour",
    PAIR: "pair",
    HIGH_CARD: "high_card",
}

# --------------------------------------------------------------------------
# Runs (RULES.md 4.2)
#
# Exactly 12 valid runs. The ace closes a run at the top (A-K-Q) or the bottom
# (A-2-3), and A-2-3 is the HIGHEST run. 2-A-K is NOT a run: wrap-around is
# prohibited, so {2, A, K} must classify as High Card (RULES.md E11).
#
# Run strength is an int, larger is stronger:
#   A-2-3 -> 15, A-K-Q -> 14, K-Q-J -> 13, ... , 4-3-2 -> 4
# i.e. the strength of a consecutive run is its top rank, and the ace-low
# wheel A-2-3 sits one above the highest of those.
# --------------------------------------------------------------------------

ACE_LOW_RUN = frozenset({14, 2, 3})
_WHEEL_STRENGTH = 15


def _run_strength(rank_set: frozenset[int]) -> int | None:
    """Strength of a run, or None if `rank_set` is not a valid run."""
    if len(rank_set) != 3:
        return None
    if rank_set == ACE_LOW_RUN:
        return _WHEEL_STRENGTH
    lo = min(rank_set)
    if rank_set == {lo, lo + 1, lo + 2}:
        # lo + 2 is the top rank; ranges over 4..14 giving the 11 ordinary runs.
        return lo + 2
    return None


def valid_runs() -> list[frozenset[int]]:
    """All 12 valid runs, strongest first. Exposed for testing (RULES.md 4.2)."""
    runs = [ACE_LOW_RUN] + [frozenset({r, r + 1, r + 2}) for r in range(2, 13)]
    return sorted(runs, key=lambda rs: -_run_strength(rs))


# --------------------------------------------------------------------------
# Classification (RULES.md 4.1, 4.3)
# --------------------------------------------------------------------------

def classify(hand) -> tuple:
    """Return the strength-class key of `hand` as an orderable tuple.

    The key is (category, tiebreak...) where a larger tuple is a stronger hand
    under Python's lexicographic ordering. Tie-breaking follows RULES.md 4.3
    exhaustively; suits are never consulted.
    """
    if len(hand) != 3:
        raise ValueError(f"a Teen Patti hand has 3 cards, got {len(hand)}")
    if len(set(hand)) != 3:
        raise ValueError(f"duplicate cards in hand {hand!r}")

    ranks = sorted((card_rank(c) for c in hand), reverse=True)
    suits = {card_suit(c) for c in hand}
    is_flush = len(suits) == 1

    # Trail: three of a rank. Checked first -- A-A-A is a Trail, not a Pair.
    if ranks[0] == ranks[2]:
        return (TRAIL, ranks[0])

    strength = _run_strength(frozenset(ranks))
    if strength is not None:
        # A run is a Pure Sequence if flush, else a Sequence. The ordering of
        # runs is identical in both categories (RULES.md 4.2).
        return (PURE_SEQUENCE if is_flush else SEQUENCE, strength)

    if is_flush:
        # Colour: flush that is not a run. Compare high, then second, then low.
        return (COLOUR, tuple(ranks))

    if ranks[0] == ranks[1]:
        return (PAIR, ranks[0], ranks[2])  # pair rank, then odd card
    if ranks[1] == ranks[2]:
        return (PAIR, ranks[1], ranks[0])

    # High card: compare high, then second, then low.
    return (HIGH_CARD, tuple(ranks))


def category_of(hand) -> int:
    return classify(hand)[0]


# --------------------------------------------------------------------------
# Build-time enumeration (RULES.md 5.1, 5.3)
# --------------------------------------------------------------------------

def _build():
    hands = [tuple(h) for h in combinations(DECK, 3)]
    hand_index = {h: i for i, h in enumerate(hands)}

    keys = [classify(h) for h in hands]

    # kappa is the rank of a class key among all distinct keys, ascending, so
    # that a stronger hand has a strictly larger kappa.
    distinct = sorted(set(keys))
    key_to_kappa = {k: i for i, k in enumerate(distinct)}
    kappa = [key_to_kappa[k] for k in keys]

    return hands, hand_index, keys, distinct, key_to_kappa, kappa


(
    HANDS,
    HAND_INDEX,
    _HAND_KEYS,
    CLASS_KEYS,
    _KEY_TO_KAPPA,
    HAND_KAPPA,
) = _build()

N_HANDS = len(HANDS)  # 22_100
N_CLASSES = len(CLASS_KEYS)  # 741


def index_of(hand) -> int:
    """Index of `hand` (any iterable of 3 distinct cards) in [0, 22100)."""
    key = tuple(sorted(hand))
    try:
        return HAND_INDEX[key]
    except KeyError:
        raise ValueError(f"not a valid 3-card hand: {hand!r}") from None


def kappa(hand) -> int:
    """Comparable integer strength class of `hand`. See module docstring."""
    return HAND_KAPPA[index_of(hand)]


def kappa_of_index(hand_idx: int) -> int:
    return HAND_KAPPA[hand_idx]


def class_key_of_kappa(k: int) -> tuple:
    return CLASS_KEYS[k]


def category_of_kappa(k: int) -> int:
    return CLASS_KEYS[k][0]


def describe_kappa(k: int) -> str:
    """Human-readable name of a strength class, for the Phase 3 benchmark."""
    key = CLASS_KEYS[k]
    cat = key[0]
    name = CATEGORY_NAMES[cat]
    if cat == TRAIL:
        return f"{name} {RANK_CHARS[key[1]]}{RANK_CHARS[key[1]]}{RANK_CHARS[key[1]]}"
    if cat in (PURE_SEQUENCE, SEQUENCE):
        strength = key[1]
        if strength == _WHEEL_STRENGTH:
            # Named A-2-3 in RULES.md 4.2, though its ranks sort as (A, 3, 2).
            ranks = (14, 2, 3)
        else:
            ranks = (strength, strength - 1, strength - 2)
        return f"{name} {'-'.join(RANK_CHARS[r] for r in ranks)}"
    if cat == PAIR:
        return f"{name} {RANK_CHARS[key[1]]}{RANK_CHARS[key[1]]}+{RANK_CHARS[key[2]]}"
    return f"{name} {'-'.join(RANK_CHARS[r] for r in key[1])}"


def compare(hand_a, hand_b) -> int:
    """1 if a beats b, -1 if b beats a, 0 on an exact tie (RULES.md 4.4)."""
    ka, kb = kappa(hand_a), kappa(hand_b)
    return (ka > kb) - (ka < kb)


# --------------------------------------------------------------------------
# Suit isomorphism (used by abstraction.py; lives here to keep card logic
# in one module)
# --------------------------------------------------------------------------

_SUIT_PERMS = tuple(permutations(SUITS))


def _apply_suit_perm(hand, perm):
    return tuple(sorted(((card_rank(c) - 2) << 2) | perm[card_suit(c)] for c in hand))


def canonical_hand(hand) -> tuple:
    """Lexicographically smallest image of `hand` under the 24 suit permutations.

    Two hands share a canonical form iff one maps to the other under a
    permutation of suits, i.e. iff they generate isomorphic subgames. This is
    the correct basis for a lossless state abstraction (RULES.md 5.3).
    """
    hand = tuple(sorted(hand))
    return min(_apply_suit_perm(hand, p) for p in _SUIT_PERMS)


# --------------------------------------------------------------------------
# Self-check: the counts RULES.md 5.1 and 4.3 commit to.
# Cheap, and it fails loudly at import rather than silently skewing Phase 2.
# --------------------------------------------------------------------------

_EXPECTED_CATEGORY_COUNTS = {
    TRAIL: 52,
    PURE_SEQUENCE: 48,
    SEQUENCE: 720,
    COLOUR: 1096,
    PAIR: 3744,
    HIGH_CARD: 16440,
}
_EXPECTED_CLASS_COUNTS = {
    TRAIL: 13,
    PURE_SEQUENCE: 12,
    SEQUENCE: 12,
    COLOUR: 274,
    PAIR: 156,
    HIGH_CARD: 274,
}


def category_counts() -> dict[int, int]:
    counts = dict.fromkeys(_EXPECTED_CATEGORY_COUNTS, 0)
    for k in HAND_KAPPA:
        counts[CLASS_KEYS[k][0]] += 1
    return counts


def class_counts() -> dict[int, int]:
    counts = dict.fromkeys(_EXPECTED_CLASS_COUNTS, 0)
    for key in CLASS_KEYS:
        counts[key[0]] += 1
    return counts


def _self_check() -> None:
    if N_HANDS != 22100:
        raise AssertionError(f"expected 22100 hands, built {N_HANDS}")
    got = category_counts()
    if got != _EXPECTED_CATEGORY_COUNTS:
        raise AssertionError(
            f"category counts disagree with RULES.md 5.1: {got} != "
            f"{_EXPECTED_CATEGORY_COUNTS}"
        )
    got = class_counts()
    if got != _EXPECTED_CLASS_COUNTS:
        raise AssertionError(
            f"class counts disagree with RULES.md 4.3: {got} != "
            f"{_EXPECTED_CLASS_COUNTS}"
        )
    if N_CLASSES != 741:
        raise AssertionError(f"expected 741 strength classes, built {N_CLASSES}")
    if len(valid_runs()) != 12:
        raise AssertionError("expected exactly 12 valid runs (RULES.md 4.2)")
    if _run_strength(ACE_LOW_RUN) != _WHEEL_STRENGTH:
        raise AssertionError("A-2-3 must be the highest run")
    if _run_strength(frozenset({2, 14, 13})) is not None:
        raise AssertionError("2-A-K must not be a valid run (RULES.md E11)")


_self_check()

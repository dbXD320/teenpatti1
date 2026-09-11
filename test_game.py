"""Tests for game.py.

The three worked examples from RULES.md 14 are the primary correctness tests.
Each is replayed action by action, asserting the stake, pot, BOTH contributions
and BOTH stacks after every single action, plus the terminal payoff and final
stacks. Every edge case in RULES.md 15 has a named test.
"""

import random
from fractions import Fraction

import pytest

import game as G
from game import Action as A
from game import (
    STARTING_STACK,
    BlindAccessViolation,
    BlindInfoSet,
    Deal,
    IllegalAction,
    Phase,
    SeenInfoSet,
    TerminalKind,
    apply_action,
    initial_state,
    legal_actions,
    observation,
    utilities,
)
from hands import parse_hand

# --------------------------------------------------------------------------
# Worked examples (RULES.md 14) -- the primary tests
#
# Each row: (action, stake, pot, contrib, stacks) expected AFTER the action.
# --------------------------------------------------------------------------

EXAMPLE_A = {
    "name": "RULES.md 14.1 -- blind/blind, see then pack",
    "hands": ("7d 4c 2s", "Kh Qh 8c"),
    "rows": [
        (A.STAY_BLIND, 1, 2, (1, 1), (49, 49)),
        (A.CHAAL,      1, 3, (2, 1), (48, 49)),
        (A.STAY_BLIND, 1, 3, (2, 1), (48, 49)),
        (A.RAISE,      2, 5, (2, 3), (48, 47)),
        (A.SEE,        2, 5, (2, 3), (48, 47)),
        (A.PACK,       2, 5, (2, 3), (48, 47)),
    ],
    "k": 2,
    "r": 1,
    "kind": TerminalKind.PACK,
    "utilities": (-2, 2),
    "final_stacks": (48, 52),
}

EXAMPLE_B = {
    "name": "RULES.md 14.2 -- convert to seen, blind exercises the show right",
    "hands": ("Ah As 9d", "Kc Kd 3s"),
    "rows": [
        (A.STAY_BLIND, 1,  2, (1, 1),  (49, 49)),
        (A.CHAAL,      1,  3, (2, 1),  (48, 49)),
        (A.STAY_BLIND, 1,  3, (2, 1),  (48, 49)),
        (A.CHAAL,      1,  4, (2, 2),  (48, 48)),
        (A.SEE,        1,  4, (2, 2),  (48, 48)),
        (A.RAISE,      2,  8, (6, 2),  (44, 48)),   # seen raise = 4s = 4
        (A.STAY_BLIND, 2,  8, (6, 2),  (44, 48)),
        (A.CHAAL,      2, 10, (6, 4),  (44, 46)),   # blind chaal = s = 2
        (A.CHAAL,      2, 14, (10, 4), (40, 46)),   # seen chaal = 2s = 4
        (A.STAY_BLIND, 2, 14, (10, 4), (40, 46)),
        (A.SHOW,       2, 16, (10, 6), (40, 44)),   # blind show = s = 2
    ],
    "k": 5,
    "r": 1,
    "kind": TerminalKind.SHOW_REQUESTED,
    "utilities": (6, -6),
    "final_stacks": (56, 44),
}

EXAMPLE_C = {
    "name": "RULES.md 14.3 -- cap-triggered forced show, exact tie, split pot",
    "hands": ("As 9s 5s", "Ah 9h 5h"),
    "rows": [
        (A.STAY_BLIND, 1,  2, (1, 1), (49, 49)),
        (A.CHAAL,      1,  3, (2, 1), (48, 49)),
        (A.STAY_BLIND, 1,  3, (2, 1), (48, 49)),
        (A.CHAAL,      1,  4, (2, 2), (48, 48)),
        (A.STAY_BLIND, 1,  4, (2, 2), (48, 48)),
        (A.CHAAL,      1,  5, (3, 2), (47, 48)),
        (A.STAY_BLIND, 1,  5, (3, 2), (47, 48)),
        (A.CHAAL,      1,  6, (3, 3), (47, 47)),
        (A.SEE,        1,  6, (3, 3), (47, 47)),
        (A.CHAAL,      1,  8, (5, 3), (45, 47)),   # seen chaal = 2s = 2
        (A.STAY_BLIND, 1,  8, (5, 3), (45, 47)),
        (A.CHAAL,      1,  9, (5, 4), (45, 46)),   # blind chaal = s = 1
        (A.CHAAL,      1, 11, (7, 4), (43, 46)),
        (A.STAY_BLIND, 1, 11, (7, 4), (43, 46)),
        (A.CHAAL,      1, 12, (7, 5), (43, 45)),   # k hits 8 -> forced show
    ],
    "k": 8,
    "r": 0,
    "kind": TerminalKind.SHOW_FORCED,
    "utilities": (-1, 1),
    "final_stacks": (49, 51),
}

ALL_EXAMPLES = [EXAMPLE_A, EXAMPLE_B, EXAMPLE_C]


@pytest.mark.parametrize("example", ALL_EXAMPLES, ids=lambda e: e["name"])
def test_worked_example_traces_exactly(example):
    """Replay a RULES.md example, checking every quantity after every action."""
    deal = Deal(parse_hand(example["hands"][0]), parse_hand(example["hands"][1]))
    st = initial_state()

    # The boot, before any action (RULES.md 3.1).
    assert st.stake == 1
    assert st.pot == 2
    assert st.contrib == (1, 1)
    assert st.stacks == (49, 49)
    assert st.seen == (False, False)

    for i, (action, stake, pot, contrib, stacks) in enumerate(example["rows"]):
        legal = legal_actions(st)
        assert action in legal, (
            f"step {i}: {G.ACTION_NAMES[action]} not legal; "
            f"legal={[G.ACTION_NAMES[a] for a in legal]}"
        )
        st = apply_action(st, action)
        where = f"step {i} after {G.ACTION_NAMES[action]}"
        assert st.stake == stake, f"{where}: stake {st.stake} != {stake}"
        assert st.pot == pot, f"{where}: pot {st.pot} != {pot}"
        assert st.contrib == contrib, f"{where}: contrib {st.contrib} != {contrib}"
        assert st.stacks == stacks, f"{where}: stacks {st.stacks} != {stacks}"

    assert st.is_terminal
    assert st.k == example["k"]
    assert st.r == example["r"]
    assert st.terminal.kind is example["kind"]

    u = utilities(st, deal)
    assert u == tuple(Fraction(x) for x in example["utilities"])
    assert u[0] + u[1] == 0

    final = (st.stack(0) + u[0] + st.contrib[0], st.stack(1) + u[1] + st.contrib[1])
    assert final == tuple(Fraction(x) for x in example["final_stacks"])
    assert sum(final) == 2 * STARTING_STACK


def test_example_b_seen_player_cannot_demand_show_against_blind():
    """RULES.md 14.2 row 5 / E4: `show` must be masked out, not just rejected."""
    deal = Deal(parse_hand("Ah As 9d"), parse_hand("Kc Kd 3s"))
    st = initial_state()
    # Replay up to the point where P1 is seen and P2 is still blind.
    for action in (A.STAY_BLIND, A.CHAAL, A.STAY_BLIND, A.CHAAL, A.SEE, A.RAISE,
                   A.STAY_BLIND, A.CHAAL):
        st = apply_action(st, action)

    assert st.to_act == 0
    assert st.seen == (True, False)
    legal = legal_actions(st)
    assert A.SHOW not in legal, "a seen player may not demand a show vs a blind one"
    assert set(legal) == {A.PACK, A.CHAAL, A.RAISE}
    with pytest.raises(IllegalAction):
        apply_action(st, A.SHOW)

    # But the blind opponent, on their turn, may.
    st = apply_action(st, A.CHAAL)
    st = apply_action(st, A.STAY_BLIND)
    assert st.to_act == 1 and not st.seen[1]
    assert A.SHOW in legal_actions(st)
    assert G.amount_for(st, A.SHOW) == st.stake  # blind show costs s, not 2s


def test_example_c_see_does_not_consume_a_cap_slot():
    """RULES.md E13: Example C has 9 player decisions but k == 8."""
    st = initial_state()
    decisions = 0
    for action in [r[0] for r in EXAMPLE_C["rows"]]:
        st = apply_action(st, action)
        decisions += 1
    look_actions = sum(
        1 for r in EXAMPLE_C["rows"] if r[0] in (A.SEE, A.STAY_BLIND)
    )
    assert st.k == 8
    assert decisions == 15  # 7 look decisions + 8 betting actions
    assert look_actions == 7
    assert decisions - look_actions == st.k


def test_example_c_contributions_are_unequal_at_showdown():
    """RULES.md 7.3: no pot equalisation; c_1 != c_2 at a showdown is normal."""
    st = initial_state()
    for action in [r[0] for r in EXAMPLE_C["rows"]]:
        st = apply_action(st, action)
    assert st.contrib == (7, 5)
    assert st.contrib[0] != st.contrib[1]


# --------------------------------------------------------------------------
# RULES.md 9.4: the max-exposure witness
# --------------------------------------------------------------------------

def test_max_exposure_witness_reaches_33():
    """RULES.md 9.4: the witness line drives seat 2's contribution to exactly 33."""
    st = initial_state()
    # The RULES.md 9.4 table annotates P2's `see` but writes P1 as "(S)" from
    # turn 3, so P1 converts at the start of turn 3. Both look decisions are
    # spelled out here.
    line = [
        A.STAY_BLIND, A.RAISE,   # turn 1: P1 blind raise, pays 2, s -> 2
        A.SEE, A.RAISE,          # turn 2: P2 sees, seen raise 4s = 8, s -> 4
        A.SEE, A.CHAAL,          # turn 3: P1 sees, seen chaal 2s = 8
        A.CHAAL,                 # turn 4: P2 seen chaal 8  (no look node)
        A.CHAAL,                 # turn 5: P1
        A.CHAAL,                 # turn 6: P2
        A.CHAAL,                 # turn 7: P1
        A.CHAAL,                 # turn 8: P2 -> cap
    ]
    expected_c2 = [1, 1, 1, 9, 9, 9, 17, 17, 25, 25, 33]
    assert len(line) == len(expected_c2)
    for action, c2 in zip(line, expected_c2):
        st = apply_action(st, action)
        assert st.contrib[1] == c2

    assert st.contrib[1] == 33
    assert st.contrib[0] == 27  # seat 1 peaks lower, having opened at s = 1
    assert st.k == 8
    assert st.r == 2
    assert st.stake == 4
    assert st.is_terminal and st.terminal.kind is TerminalKind.SHOW_FORCED
    # 33 < 50, so the stack never binds (RULES.md 3.2).
    assert st.stack(1) == STARTING_STACK - 33 == 17
    assert st.stack(1) > 0


def test_starting_stack_exceeds_max_exposure():
    assert STARTING_STACK >= G.MIN_LEGAL_STACK
    assert G.MIN_LEGAL_STACK == 34  # 33 + 1


# --------------------------------------------------------------------------
# RULES.md 15: edge cases, one test each
# --------------------------------------------------------------------------

def test_e2_forced_show_evaluates_hands_nobody_looked_at():
    """RULES.md E2: both blind at the cap -> forced show, no fee, tie splits."""
    st = initial_state()
    for _ in range(8):
        st = apply_action(st, A.STAY_BLIND)
        st = apply_action(st, A.CHAAL)
    assert st.is_terminal
    assert st.seen == (False, False), "neither player ever looked"
    assert st.terminal.kind is TerminalKind.SHOW_FORCED
    assert st.terminal.actor is None, "a forced show has no requester"
    assert st.contrib == (5, 5)  # boot + 4 blind chaals at s=1 each

    tie = Deal(parse_hand("As 9s 5s"), parse_hand("Ah 9h 5h"))
    u = utilities(st, tie)
    assert u == (Fraction(0), Fraction(0))  # equal contributions, split pot

    win = Deal(parse_hand("Ac Ad Ah"), parse_hand("7d 4c 2s"))
    u = utilities(st, win)
    assert u == (Fraction(5), Fraction(-5))
    assert u[0] + u[1] == 0


def test_e3_blind_requester_pays_s_regardless_of_opponent_status():
    """RULES.md E3: cost is s even when the opponent is seen."""
    st = initial_state()
    st = apply_action(st, A.SEE)      # P1 becomes seen
    st = apply_action(st, A.RAISE)    # s -> 2
    assert st.to_act == 1 and not st.seen[1]
    assert st.seen[0] is True
    st_look = st
    st = apply_action(st_look, A.STAY_BLIND)
    assert A.SHOW in legal_actions(st)
    assert G.amount_for(st, A.SHOW) == 2 == st.stake  # s, not 2s


def test_e4_seen_vs_blind_show_is_masked_everywhere():
    """RULES.md E4: mask-level invariant across the entire tree."""
    from tree import walk

    checked = 0
    for st in walk():
        if st.is_terminal or st.phase is not Phase.BET:
            continue
        actor, opp = st.to_act, 1 - st.to_act
        if st.seen[actor] and not st.seen[opp]:
            assert A.SHOW not in legal_actions(st)
            checked += 1
    assert checked > 0, "no seen-vs-blind bet nodes found; test is vacuous"


def test_e5_raise_masked_at_the_raise_cap():
    """RULES.md E5: raise is illegal once r == 2, and s never exceeds 4."""
    st = initial_state()
    st = apply_action(st, A.STAY_BLIND)
    st = apply_action(st, A.RAISE)     # r = 1, s = 2
    st = apply_action(st, A.STAY_BLIND)
    st = apply_action(st, A.RAISE)     # r = 2, s = 4
    assert st.r == 2 and st.stake == 4
    st = apply_action(st, A.STAY_BLIND)
    assert A.RAISE not in legal_actions(st)
    with pytest.raises(IllegalAction):
        apply_action(st, A.RAISE)


def test_e6_seen_player_never_faces_the_look_decision():
    """RULES.md E6: conversion is one-way; `see` is unreachable once seen."""
    st = initial_state()
    st = apply_action(st, A.SEE)
    assert st.seen[0] is True
    assert st.phase is Phase.BET
    assert A.SEE not in legal_actions(st)
    with pytest.raises(IllegalAction):
        apply_action(st, A.SEE)

    # And after the turn passes back, P1 goes straight to a bet node.
    st = apply_action(st, A.CHAAL)
    st = apply_action(st, A.STAY_BLIND)
    st = apply_action(st, A.CHAAL)
    assert st.to_act == 0 and st.seen[0]
    assert st.phase is Phase.BET, "a seen player must not get a look node"


def test_e7_seeing_this_turn_prices_at_seen_rates():
    """RULES.md E7: pricing reads status AFTER the look decision resolves."""
    st = initial_state()
    st_seen = apply_action(st, A.SEE)
    assert G.amount_for(st_seen, A.CHAAL) == 2   # 2s
    assert G.amount_for(st_seen, A.RAISE) == 4   # 4s

    st_blind = apply_action(st, A.STAY_BLIND)
    assert G.amount_for(st_blind, A.CHAAL) == 1  # s
    assert G.amount_for(st_blind, A.RAISE) == 2  # 2s


def test_e8_requested_tie_and_forced_tie_resolve_differently():
    """RULES.md E8: requester loses a requested tie; a forced tie splits."""
    tie = Deal(parse_hand("As 9s 5s"), parse_hand("Ah 9h 5h"))

    # Requested show by P1 on turn 1 (both blind), then a tie.
    st = initial_state()
    st = apply_action(st, A.STAY_BLIND)
    st = apply_action(st, A.SHOW)
    assert st.terminal.kind is TerminalKind.SHOW_REQUESTED
    assert st.terminal.actor == 0
    u = utilities(st, tie)
    # P1 paid s=1 so c=(2,1), pot 3; P1 requested and so loses the tie.
    assert st.contrib == (2, 1)
    assert u == (Fraction(-2), Fraction(2)), "the requester must lose the tie"

    # Forced show with a tie splits instead.
    st2 = initial_state()
    for _ in range(8):
        st2 = apply_action(st2, A.STAY_BLIND)
        st2 = apply_action(st2, A.CHAAL)
    assert st2.terminal.kind is TerminalKind.SHOW_FORCED
    assert utilities(st2, tie) == (Fraction(0), Fraction(0))


def test_e10_pack_and_show_are_legal_on_turn_one():
    """RULES.md E10: both are available immediately and are not special-cased."""
    st = apply_action(initial_state(), A.STAY_BLIND)
    legal = legal_actions(st)
    assert A.PACK in legal
    assert A.SHOW in legal
    assert G.amount_for(st, A.SHOW) == 1  # s = 1

    packed = apply_action(st, A.PACK)
    assert packed.is_terminal and packed.terminal.kind is TerminalKind.PACK
    # P1 packs having paid only the boot; P2 wins the 2-chip pot.
    assert utilities(packed) == (Fraction(-1), Fraction(1))


def test_e14_stake_transition_ignores_status():
    """RULES.md E14: chaal leaves s unchanged, raise doubles it, always."""
    for look, expect_chaal_pay, expect_raise_pay in (
        (A.STAY_BLIND, 1, 2),
        (A.SEE, 2, 4),
    ):
        st = apply_action(initial_state(), look)
        after_chaal = apply_action(st, A.CHAAL)
        assert after_chaal.stake == 1, "chaal must not change the stake"
        assert after_chaal.contrib[0] == 1 + expect_chaal_pay

        after_raise = apply_action(st, A.RAISE)
        assert after_raise.stake == 2, "raise must double the stake"
        assert after_raise.contrib[0] == 1 + expect_raise_pay


def test_stake_is_always_in_1_2_4():
    """RULES.md 7.2: with r <= 2 the stake can only be 1, 2 or 4."""
    from tree import walk

    assert {st.stake for st in walk()} == {1, 2, 4}


# --------------------------------------------------------------------------
# Blind information: the structural guarantee (RULES.md 12.4)
# --------------------------------------------------------------------------

def test_blind_infoset_has_no_field_for_cards():
    """The type itself must make card access impossible, not just discouraged."""
    fields = set(BlindInfoSet.__dataclass_fields__)
    assert fields == {"public", "player", "phase"}
    forbidden = {"hand", "hand_index", "cards", "kappa", "bucket", "hole"}
    assert not (fields & forbidden)

    iset = BlindInfoSet((), 0, Phase.LOOK)
    for name in forbidden:
        assert not hasattr(iset, name)
    # __slots__ means one cannot even be attached at runtime.
    with pytest.raises(AttributeError):
        iset.hand = parse_hand("As Ah Ad")


def test_seen_infoset_does_carry_the_hand():
    """The contrast case: a seen player's infoset is indexed by their hand."""
    assert "hand_index" in SeenInfoSet.__dataclass_fields__


def test_blind_observation_is_identical_across_different_deals():
    """RULES.md 12.4: exactly one blind infoset per public history."""
    deals = [
        Deal(parse_hand("Ac Ad Ah"), parse_hand("7d 4c 2s")),
        Deal(parse_hand("2c 3d 5h"), parse_hand("Ks Qs Js")),
        Deal(parse_hand("Ts 9s 8s"), parse_hand("Kc 7d 4h")),
    ]
    st = apply_action(initial_state(), A.STAY_BLIND)
    assert not st.seen[st.to_act]
    observations = [observation(st, d) for d in deals]
    assert all(isinstance(o, BlindInfoSet) for o in observations)
    assert len(set(observations)) == 1, (
        "a blind player's information set varied with their own cards"
    )


def test_deal_refuses_to_hand_over_unseen_cards():
    """The runtime tripwire behind the structural guarantee."""
    deal = Deal(parse_hand("Ac Ad Ah"), parse_hand("7d 4c 2s"))
    st = initial_state()
    with pytest.raises(BlindAccessViolation):
        deal.hand_for(0, st)
    with pytest.raises(BlindAccessViolation):
        deal.hand_for(1, st)

    seen = apply_action(st, A.SEE)
    assert deal.hand_for(0, seen) == parse_hand("Ac Ad Ah")
    with pytest.raises(BlindAccessViolation):
        deal.hand_for(1, seen)  # the opponent is still blind


def test_seen_observation_is_deal_dependent():
    """Sanity check the other direction: a seen infoset must vary with the hand."""
    st = apply_action(initial_state(), A.SEE)
    a = observation(st, Deal(parse_hand("Ac Ad Ah"), parse_hand("7d 4c 2s")))
    b = observation(st, Deal(parse_hand("2c 3d 5h"), parse_hand("Ks Qs Js")))
    assert isinstance(a, SeenInfoSet) and isinstance(b, SeenInfoSet)
    assert a != b


# --------------------------------------------------------------------------
# Randomised property test
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_playthroughs_respect_every_invariant(seed):
    """Play random legal hands; assert nothing illegal is ever offered."""
    rng = random.Random(seed)
    deck = list(range(52))
    n_hands = 400

    for _ in range(n_hands):
        rng.shuffle(deck)
        deal = Deal(tuple(deck[:3]), tuple(deck[3:6]))
        st = initial_state()
        steps = 0

        while not st.is_terminal:
            steps += 1
            assert steps < 40, "hand failed to terminate"
            legal = legal_actions(st)
            assert legal, "a non-terminal state offered no actions"
            actor = st.to_act

            # Phase/status consistency (RULES.md 13.4).
            if st.phase is Phase.LOOK:
                assert not st.seen[actor]
                assert set(legal) == {A.SEE, A.STAY_BLIND}
            else:
                assert A.PACK in legal and A.CHAAL in legal
                assert A.SEE not in legal and A.STAY_BLIND not in legal

            # Caps (RULES.md 9.2) and the mask rules (E4, E5).
            assert 0 <= st.k < G.ACTION_CAP
            assert 0 <= st.r <= G.RAISE_CAP
            if st.r >= G.RAISE_CAP:
                assert A.RAISE not in legal
            if st.phase is Phase.BET and st.seen[actor] and not st.seen[1 - actor]:
                assert A.SHOW not in legal

            # Stacks (RULES.md E1) and pot identity (2.3).
            assert st.stack(0) >= 0 and st.stack(1) >= 0
            assert st.pot == st.contrib[0] + st.contrib[1]
            assert st.stake in (1, 2, 4)

            st = apply_action(st, rng.choice(legal))

        assert st.k <= G.ACTION_CAP
        assert st.r <= G.RAISE_CAP
        assert st.stack(0) >= 0 and st.stack(1) >= 0
        assert max(st.contrib) <= 33  # RULES.md 9.4

        u = utilities(st, deal)
        assert u[0] + u[1] == 0
        # Conservation: chips never appear or vanish.
        assert (st.stack(0) + u[0] + st.contrib[0]) + (
            st.stack(1) + u[1] + st.contrib[1]
        ) == 2 * STARTING_STACK


def test_illegal_action_raises_rather_than_defaulting():
    """RULES.md working style: no silent defaults."""
    st = initial_state()
    assert st.phase is Phase.LOOK
    for action in (A.CHAAL, A.RAISE, A.PACK, A.SHOW):
        with pytest.raises(IllegalAction):
            apply_action(st, action)

    packed = apply_action(apply_action(st, A.STAY_BLIND), A.PACK)
    with pytest.raises(IllegalAction):
        legal_actions(packed)
    with pytest.raises(IllegalAction):
        observation(packed, Deal(parse_hand("Ac Ad Ah"), parse_hand("7d 4c 2s")))


def test_utilities_require_a_deal_at_a_showdown():
    st = apply_action(apply_action(initial_state(), A.STAY_BLIND), A.SHOW)
    with pytest.raises(ValueError):
        utilities(st, None)


def test_utilities_rejected_for_non_terminal_states():
    with pytest.raises(IllegalAction):
        utilities(initial_state())


def test_state_is_immutable():
    st = initial_state()
    with pytest.raises(Exception):
        st.stake = 99


def test_disjoint_hands_enforced():
    with pytest.raises(ValueError):
        Deal(parse_hand("Ac Ad Ah"), parse_hand("Ac 4c 2s"))

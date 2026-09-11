"""Tests for tree.py -- exact enumeration and the Phase 2 gates."""

from fractions import Fraction

import pytest

import tree as T
from game import (
    Action as A,
    BlindInfoSet,
    Deal,
    Phase,
    TerminalKind,
    apply_action,
    initial_state,
    legal_actions,
    observation,
    utilities,
)
from hands import parse_hand

DEALS = [
    Deal(parse_hand("Ah As 9d"), parse_hand("Kc Kd 3s")),   # P1 wins
    Deal(parse_hand("7d 4c 2s"), parse_hand("Ac Ad Ah")),   # P2 wins
    Deal(parse_hand("As 9s 5s"), parse_hand("Ah 9h 5h")),   # exact tie
]


@pytest.fixture(scope="module")
def stats():
    return T.enumerate_tree()


# --------------------------------------------------------------------------
# Exact counts (the RULES.md 9.3 gate)
# --------------------------------------------------------------------------

def test_tree_is_finite_and_terminates(stats):
    assert stats.n_nodes == stats.n_decision_nodes + stats.n_terminals
    assert stats.n_terminals > 0
    assert stats.max_k == 8


def test_exact_decision_node_count_is_below_the_rules_estimate(stats):
    """RULES.md 9.3 predicts the true figure is 'somewhat lower' than ~2.2e3."""
    assert stats.n_decision_nodes == 2014
    assert stats.n_bet_nodes == 1650
    assert stats.n_look_nodes == 364
    assert stats.n_decision_nodes < T.RULES_ESTIMATE, (
        "RULES.md 9.3 predicts an overestimate; if this fails the spec's "
        "reasoning was wrong, not just imprecise"
    )


def test_look_and_blind_bet_nodes_agree(stats):
    """Every look node is followed by exactly one blind and one seen branch.

    A blind actor choosing `stay-blind` reaches a blind bet node; choosing
    `see` reaches a seen bet node. So look nodes and blind bet nodes must be
    equinumerous.
    """
    assert stats.n_look_nodes == stats.n_bet_nodes_blind_actor == 364
    assert stats.n_bet_nodes == (
        stats.n_bet_nodes_blind_actor + stats.n_bet_nodes_seen_actor
    )


def test_every_look_node_has_a_blind_actor():
    for st in T.walk():
        if not st.is_terminal and st.phase is Phase.LOOK:
            assert not st.seen[st.to_act]


def test_terminal_kinds_are_all_reachable(stats):
    for kind in TerminalKind:
        assert stats.terminals_by_kind[kind] > 0, f"{kind.name} unreachable"
    assert sum(stats.terminals_by_kind.values()) == stats.n_terminals


def test_pack_terminal_count_equals_bet_node_count(stats):
    """`pack` is legal at every bet node and nowhere else."""
    assert stats.terminals_by_kind[TerminalKind.PACK] == stats.n_bet_nodes


# --------------------------------------------------------------------------
# Information sets (RULES.md 12.4)
# --------------------------------------------------------------------------

def test_blind_infoset_count(stats):
    assert stats.n_blind_infosets == 728
    assert (
        stats.n_blind_infosets
        == stats.n_look_nodes + stats.n_bet_nodes_blind_actor
    )


def test_seen_public_node_count(stats):
    assert stats.n_seen_public_nodes == 1286


def test_infoset_totals_scale_with_the_abstraction(stats):
    """Only the seen side is multiplied by the hand abstraction."""
    assert stats.seen_infosets(1) == stats.n_seen_public_nodes
    assert stats.total_infosets(22100) == 728 + 1286 * 22100
    assert stats.total_infosets(1755) == 728 + 1286 * 1755
    assert stats.total_infosets(25) == 728 + 1286 * 25


def test_positions_are_asymmetric(stats):
    """RULES.md 3.4: action order is the only asymmetry, and it is real."""
    assert stats.decision_nodes_by_player[0] != stats.decision_nodes_by_player[1]
    assert sum(stats.decision_nodes_by_player.values()) == stats.n_decision_nodes


# --------------------------------------------------------------------------
# Gates required before Phase 2
# --------------------------------------------------------------------------

def test_zero_sum_at_every_terminal_in_a_full_walk():
    """RULES.md 11, over the entire tree and three contrasting deals."""
    checked = T.verify_zero_sum(DEALS)
    assert checked == 3915 * len(DEALS)


def test_blind_infosets_are_card_independent_across_the_whole_tree():
    """RULES.md 12.4, checked structurally at every blind decision node."""
    checked = T.verify_blind_infosets_are_card_independent(DEALS)
    assert checked == 728


def test_blind_infosets_are_card_independent_needs_multiple_deals():
    with pytest.raises(ValueError):
        T.verify_blind_infosets_are_card_independent(DEALS[:1])


def test_max_exposure_is_exhaustively_33():
    """Stronger than RULES.md 9.4, which exhibits a single witness."""
    max_contrib, max_pot = T.verify_max_exposure()
    assert max_contrib == 33
    assert max_pot == 62


def test_no_state_ever_violates_a_cap():
    for st in T.walk():
        assert st.k <= 8
        assert st.r <= 2
        assert st.stake in (1, 2, 4)
        assert st.stack(0) >= 0 and st.stack(1) >= 0


def test_pot_identity_holds_everywhere():
    for st in T.walk():
        assert st.pot == st.contrib[0] + st.contrib[1]


def test_forced_show_happens_exactly_at_the_cap():
    for st in T.walk():
        if st.is_terminal and st.terminal.kind is TerminalKind.SHOW_FORCED:
            assert st.k == 8
            assert st.terminal.actor is None
        if not st.is_terminal:
            assert st.k < 8


def test_history_uniquely_identifies_every_node():
    """Public histories must be distinct, or perfect recall is broken."""
    histories = [st.history for st in T.walk()]
    assert len(histories) == len(set(histories))


def test_seen_status_is_monotone_along_every_path():
    """RULES.md E6: conversion never reverses."""
    def recurse(st):
        if st.is_terminal:
            return
        for action in legal_actions(st):
            nxt = apply_action(st, action)
            for p in (0, 1):
                assert nxt.seen[p] >= st.seen[p], "seen -> blind is impossible"
            recurse(nxt)

    recurse(initial_state())


# --------------------------------------------------------------------------
# Odd pots at a forced show: a case RULES.md 14 does not exhibit
# --------------------------------------------------------------------------

def test_forced_show_pots_can_be_odd():
    """RULES.md 10.6 gives u_i = p/2 - c_i, which need not be integral."""
    odd = T.odd_pot_forced_shows()
    assert odd, "expected reachable odd-pot forced shows"
    assert len(odd) == 396

    smallest = min(odd, key=lambda s: s.pot)
    assert smallest.pot == 11
    u = utilities(smallest, DEALS[2])  # tie -> split
    assert u == (Fraction(1, 2), Fraction(-1, 2))
    assert u[0] + u[1] == 0
    assert u[0].denominator == 2, "a half-integer payoff must be exact"


def test_all_odd_pot_splits_remain_exactly_zero_sum():
    for st in T.odd_pot_forced_shows():
        u = utilities(st, DEALS[2])
        assert u[0] + u[1] == Fraction(0)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_report_mentions_the_headline_numbers(stats):
    text = T.report(stats)
    for token in ("2,014", "1,650", "728", "1,286", "33"):
        assert token in text


def test_n_deals_matches_rules_md():
    """RULES.md 12.2: C(52,3) * C(49,3) = 407,170,400."""
    assert T.N_OPPONENT_HANDS == 18424
    assert T.N_DEALS == 407_170_400

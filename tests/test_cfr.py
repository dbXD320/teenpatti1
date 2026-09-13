"""Tests for cfr.py, exploitability.py and experiments.py.

The load-bearing checks are:

  * `test_cap1_matches_direct_maximisation` -- a variant small enough that the
    exact answer is computable WITHOUT game theory (seat 2 never acts), used as
    an independent cross-check on the whole pipeline;
  * `test_forced_pure_strategy_value_matches_closed_form` -- validates the
    value computation, including exact card removal, against a hand-derived
    expression;
  * the blind-leak tests, which fail if the solver can condition a blind
    decision on unseen cards.
"""

from __future__ import annotations

import numpy as np
import pytest

import cfr
import experiments as XP
import exploitability as EX
from game import (
    Action,
    DEFAULT_CONFIG,
    GameConfig,
    Phase,
    StackWouldBind,
    TerminalKind,
)


@pytest.fixture(scope="module")
def tables():
    return cfr.build_private_tables()


@pytest.fixture(scope="module")
def tiny(tables):
    """action_cap=1: seat 2 never acts, so the optimum needs no game theory."""
    solver = cfr.CFRPlusSolver(GameConfig(action_cap=1), tables=tables)
    solver.iterate(3000)
    return solver


@pytest.fixture(scope="module")
def small(tables):
    solver = cfr.CFRPlusSolver(GameConfig(action_cap=2), tables=tables)
    solver.iterate(400)
    return solver


# ==========================================================================
# Private tables: the disjointness matrix
# ==========================================================================

def test_private_state_is_suit_isomorphic_not_kappa(tables):
    """Gate 4: the solver must not index by strength class."""
    from hands import N_CLASSES

    assert tables.n_classes == 1755
    assert tables.n_classes != N_CLASSES
    assert len(set(tables.kappa_of_class.tolist())) == N_CLASSES


def test_disjointness_matrix_row_sums(tables):
    """Every class must see exactly C(49,3) disjoint opponent hands."""
    assert np.allclose(tables.cond.sum(axis=1), 1.0, atol=1e-12)


def test_disjointness_matrix_is_pair_symmetric(tables):
    M = np.rint(tables.cond * cfr.N_OPPONENT_HANDS).astype(np.int64)
    ordered = tables.class_size[:, None] * M
    assert np.array_equal(ordered, ordered.T)
    assert int(ordered.sum()) == 22100 * 18424


def test_chance_weights_sum_to_one(tables):
    assert abs(tables.chance_w.sum() - 1.0) < 1e-12
    assert int(tables.class_size.sum()) == 22100


def test_win_tie_lose_decomposition_is_complete(tables):
    """win + tie + lose mass must be exactly 1 for every class."""
    lose = 1.0 - tables.unit_win - tables.unit_tie
    assert np.all(lose > -1e-12)
    assert np.allclose(tables.unit_win + tables.unit_tie + lose, 1.0, atol=1e-12)


def test_unit_masses_match_phase1_equity(tables):
    """unit_win + unit_tie/2 must reproduce the Phase 1 equity exactly."""
    import abstraction as AB
    from hands import index_of

    ab = AB.build_abstraction(25, verify=False)
    equity_from_masses = tables.unit_win + tables.unit_tie / 2.0
    for c in (0, 17, 500, 1200, 1754):
        i = index_of(tables.rep_hand[c])
        assert abs(equity_from_masses[c] - ab.equity[i]) < 1e-12


def test_verify_disjointness_matrix_rejects_corruption(tables):
    M = np.rint(tables.cond * cfr.N_OPPONENT_HANDS).astype(np.int64).copy()
    M[3, 4] += 1
    with pytest.raises(AssertionError):
        cfr.verify_disjointness_matrix(M, tables.class_size)


# ==========================================================================
# Tree and terminal coefficients
# ==========================================================================

def test_tree_matches_tree_py_enumeration(tables):
    import tree as T

    stats = T.enumerate_tree()
    built = cfr.build_public_tree(DEFAULT_CONFIG)
    assert built.n_nodes == stats.n_nodes == 5929
    assert int((~built.is_terminal).sum()) == stats.n_decision_nodes == 2014
    assert int(built.is_terminal.sum()) == stats.n_terminals == 3915


def test_terminal_coefficients_are_seat_symmetric(tables):
    """u_0 == -u_1 at every terminal, checked on the coefficients themselves.

    The rules are NOT seat-symmetric (seat 1 acts first, RULES.md 3.4), so
    there is no exact seat-swap invariance of the equilibrium to test. What
    must hold is that the payoff layer treats the seats as exact mirrors.
    """
    tree = cfr.build_public_tree(DEFAULT_CONFIG)
    checked = 0
    for raw in tree.terminal_ids:
        n = int(raw)
        st = tree.states[n]
        c0 = cfr.terminal_coeffs(st, 0)
        c1 = cfr.terminal_coeffs(st, 1)
        assert c0.is_showdown == c1.is_showdown
        if not c0.is_showdown:
            assert c0.const == -c1.const
        else:
            # Player 0 winning is player 1 losing, so the roles swap.
            assert c0.win == -c1.lose
            assert c0.lose == -c1.win
            assert c0.tie == -c1.tie
        checked += 1
    assert checked == 3915


def test_forced_show_tie_can_be_a_half_integer(tables):
    """PHASE1.md 6.1: an odd pot makes the split non-integral."""
    tree = cfr.build_public_tree(DEFAULT_CONFIG)
    halves = 0
    for raw in tree.terminal_ids:
        st = tree.states[int(raw)]
        if st.terminal.kind is TerminalKind.SHOW_FORCED and st.pot % 2 == 1:
            coeff = cfr.terminal_coeffs(st, 0)
            assert abs(coeff.tie * 2 - round(coeff.tie * 2)) < 1e-12
            assert abs(coeff.tie - round(coeff.tie)) > 1e-12
            halves += 1
    assert halves == 396


def test_terminal_coeffs_rejects_decision_nodes(tables):
    tree = cfr.build_public_tree(GameConfig(action_cap=2))
    with pytest.raises(ValueError):
        cfr.terminal_coeffs(tree.states[0], 0)


# ==========================================================================
# The blind information partition -- structural
# ==========================================================================

def test_blind_regret_tables_have_no_class_axis(tables):
    """RULES.md 12.4: one information set per public history, so shape (nA,).

    If a blind node's regret had a class axis, CFR could learn a strategy
    conditioned on cards the player has not seen.
    """
    solver = cfr.CFRPlusSolver(GameConfig(action_cap=4), tables=tables)
    n_blind = n_seen = 0
    for raw in solver.tree.decision_ids:
        n = int(raw)
        shape = solver.regret[n].shape
        if solver.tree.actor_blind[n]:
            assert len(shape) == 1, f"node {n}: blind regret has a class axis"
            assert shape[0] == len(solver.tree.actions[n])
            n_blind += 1
        else:
            assert len(shape) == 2 and shape[0] == tables.n_classes
            n_seen += 1
    assert n_blind == 68 and n_seen == 63  # matches tree.py for action_cap=4


def test_blind_strategy_is_one_distribution_per_public_history(small, tables):
    for raw in small.tree.decision_ids:
        n = int(raw)
        sigma = small.average_strategy(n)
        if small.tree.actor_blind[n]:
            assert sigma.ndim == 1
            assert abs(sigma.sum() - 1.0) < 1e-9
        else:
            assert sigma.ndim == 2
            assert np.allclose(sigma.sum(axis=1), 1.0, atol=1e-9)


def test_blind_reach_stays_scalar_through_a_traversal(small):
    strategies = small.strategy_profile(average=True)
    reach = small._down_pass(strategies)
    for raw in small.tree.decision_ids:
        n = int(raw)
        actor = int(small.tree.actor[n])
        if small.tree.actor_blind[n]:
            assert not isinstance(reach[n][actor], np.ndarray), (
                f"node {n}: a blind player's reach became hand-dependent"
            )


def test_blind_reach_assertion_fires_on_a_leak(tables):
    """Deliberately hand the guard a vector and confirm it refuses."""
    with pytest.raises(AssertionError, match="blind"):
        cfr._assert_blind_reach_is_scalar(np.ones(5), blind=True, node=0)
    # A seen player legitimately still has a scalar reach on the turn they look.
    cfr._assert_blind_reach_is_scalar(1.0, blind=False, node=0)


def test_best_response_at_blind_nodes_is_a_single_action(small):
    for player in (0, 1):
        _, choices = small.best_response_value(average=True, player=player)
        n_blind = EX.verify_blind_best_response_is_single_action(
            small, choices, player
        )
        assert n_blind > 0, "no blind nodes exercised; test is vacuous"


def test_blind_best_response_verifier_catches_a_per_class_choice(small):
    """The verifier must reject a per-class choice at a blind node."""
    _, choices = small.best_response_value(average=True, player=0)
    blind_node = next(
        n for n in choices if small.tree.actor_blind[n]
    )
    corrupted = dict(choices)
    corrupted[blind_node] = np.zeros(small.tables.n_classes, dtype=np.int64)
    with pytest.raises(AssertionError, match="BLIND"):
        EX.verify_blind_best_response_is_single_action(small, corrupted, 0)


# ==========================================================================
# Value computation
# ==========================================================================

def test_game_value_is_zero_sum(small):
    v0 = small.game_value(player=0)
    v1 = small.game_value(player=1)
    assert abs(v0 + v1) < 1e-12


def test_forced_pure_strategy_value_matches_closed_form(tables):
    """Validate the value computation, card removal included, in closed form.

    With action_cap=1, forcing seat 1 to stay blind and chaal makes the whole
    game deterministic given the deal: seat 1 pays the blind stake of 1, the
    cap binds and a forced show follows with contributions (2, 1). So

        u_0 = +1 * P(win) - 2 * P(lose) + ((1 - 2) / 2) * P(tie)

    averaged over the prior. Seat 2 never acts, so nothing else can vary.
    """
    solver = cfr.CFRPlusSolver(GameConfig(action_cap=1), tables=tables)
    tree = solver.tree

    # Force: root -> stay-blind, and the blind bet node -> chaal.
    for raw in tree.decision_ids:
        n = int(raw)
        acts = tree.actions[n]
        want = Action.STAY_BLIND if tree.states[n].phase is Phase.LOOK else Action.CHAAL
        if want not in acts:
            continue
        onehot = np.zeros_like(solver.strat_sum[n])
        idx = acts.index(want)
        if onehot.ndim == 1:
            onehot[idx] = 1.0
        else:
            onehot[:, idx] = 1.0
        solver.strat_sum[n] = onehot

    win = tables.unit_win
    tie = tables.unit_tie
    lose = 1.0 - win - tie
    expected = float(tables.chance_w @ (1.0 * win - 2.0 * lose - 0.5 * tie))

    assert abs(solver.game_value(average=True, player=0) - expected) < 1e-12
    # And the mirror image for seat 2.
    assert abs(solver.game_value(average=True, player=1) + expected) < 1e-12


def test_cap1_matches_direct_maximisation(tiny, tables):
    """The primary independent cross-check.

    With action_cap=1 seat 2 never acts, so seat 1 faces a decision problem
    rather than a game and the optimum is a direct maximisation. CFR+ must find
    the same value.
    """
    w, win, tie = tables.chance_w, tables.unit_win, tables.unit_tie
    lose = 1.0 - win - tie

    def showdown(own, opp, tie_coeff):
        return opp * win - own * lose + tie_coeff * tie

    # Staying blind: one action for every class, so compare prior averages.
    blind_options = {
        "pack": np.full(tables.n_classes, -1.0),
        "chaal": showdown(2, 1, (1 - 2) / 2),      # pays s=1, forced show
        "raise": showdown(3, 1, (1 - 3) / 2),      # pays 2s=2, forced show
        "show": 1 * win - 2 * lose - 2 * tie,      # requester loses ties
    }
    best_blind = max(float(w @ v) for v in blind_options.values())

    # Looking: maximise per class. `show` is ILLEGAL because seat 2 is still
    # blind (RULES.md 10.3), so only pack/chaal/raise are available.
    seen = np.stack(
        [
            np.full(tables.n_classes, -1.0),
            showdown(3, 1, (1 - 3) / 2),
            showdown(5, 1, (1 - 5) / 2),
        ],
        axis=1,
    )
    best_seen = float(w @ seen.max(axis=1))

    exact = max(best_blind, best_seen)
    assert abs(tiny.game_value(average=True, player=0) - exact) < 1e-5
    # Looking strictly beats every blind action even in this degenerate variant.
    assert best_seen > best_blind


def test_cap1_show_is_illegal_for_a_seen_player(tables):
    """Guards the assumption the cross-check above relies on."""
    tree = cfr.build_public_tree(GameConfig(action_cap=1))
    seen_bet = [
        int(n) for n in tree.decision_ids
        if not tree.actor_blind[int(n)]
    ]
    assert seen_bet, "expected a seen bet node"
    for n in seen_bet:
        assert Action.SHOW not in tree.actions[n]


# ==========================================================================
# Convergence
# ==========================================================================

def test_exploitability_is_nonnegative_and_decreases(tables):
    solver = cfr.CFRPlusSolver(GameConfig(action_cap=2), tables=tables)
    run = EX.solve_with_logging(solver, 300, log_every=50, verbose=False)
    vals = [r.exploitability for r in run.records]
    assert all(v >= -1e-12 for v in vals), "exploitability cannot be negative"
    assert run.is_monotone_decreasing()
    assert vals[-1] < vals[0] / 5, "exploitability barely moved"


def test_exploitability_equals_br_sum_over_two(small):
    rep = EX.compute_exploitability(small, average=True)
    assert abs(rep.exploitability - (rep.br_value_p0 + rep.br_value_p1) / 2) < 1e-15


def test_best_response_is_at_least_as_good_as_the_strategy(small):
    """A best response can never do worse than playing the strategy itself."""
    for player in (0, 1):
        br, _ = small.best_response_value(average=True, player=player)
        val = small.game_value(average=True, player=player)
        assert br >= val - 1e-12


def test_average_strategy_differs_from_final_iterate(small):
    """The average is accumulated separately, as required."""
    differs = 0
    for raw in small.tree.decision_ids:
        n = int(raw)
        if not np.allclose(small.average_strategy(n), small.current_strategy(n),
                           atol=1e-6):
            differs += 1
    assert differs > 0, "average and current strategy are identical; the "\
                        "average is probably not being accumulated"


def test_exploitability_units(small):
    rep = EX.compute_exploitability(small, average=True)
    assert abs(rep.milliboots_per_hand - rep.exploitability * 1000) < 1e-12
    assert abs(rep.pct_of_initial_pot
               - 100 * rep.exploitability / (2 * small.config.boot)) < 1e-12


# ==========================================================================
# Checkpointing
# ==========================================================================

def test_checkpoint_roundtrip(tables, tmp_path):
    a = cfr.CFRPlusSolver(GameConfig(action_cap=2), tables=tables)
    a.iterate(30)
    path = tmp_path / "ck.pkl"
    a.save(path)

    b = cfr.CFRPlusSolver(GameConfig(action_cap=2), tables=tables)
    b.load(path)
    assert b.iterations == a.iterations
    assert abs(b.game_value() - a.game_value()) < 1e-15
    for raw in a.tree.decision_ids:
        n = int(raw)
        assert np.array_equal(a.strat_sum[n], b.strat_sum[n])


def test_checkpoint_rejects_a_different_config(tables, tmp_path):
    a = cfr.CFRPlusSolver(GameConfig(action_cap=2), tables=tables)
    a.iterate(2)
    path = tmp_path / "ck.pkl"
    a.save(path)
    b = cfr.CFRPlusSolver(GameConfig(action_cap=4), tables=tables)
    with pytest.raises(ValueError, match="config"):
        b.load(path)


def test_solver_rejects_a_mismatched_tree(tables):
    tree = cfr.build_public_tree(GameConfig(action_cap=2))
    with pytest.raises(ValueError, match="different config"):
        cfr.CFRPlusSolver(GameConfig(action_cap=4), tables=tables, tree=tree)


# ==========================================================================
# Experiments layer
# ==========================================================================

def test_turn_index_matches_seat_alternation():
    from game import apply_action, initial_state

    st = initial_state()
    assert XP.turn_index(st) == 1 and st.to_act == 0
    st = apply_action(apply_action(st, Action.STAY_BLIND), Action.CHAAL)
    assert st.to_act == 1 and XP.turn_index(st) == 1
    st = apply_action(apply_action(st, Action.STAY_BLIND), Action.CHAAL)
    assert st.to_act == 0 and XP.turn_index(st) == 2


def test_node_reach_of_root_is_one(small):
    strategies = small.strategy_profile(average=True)
    reach = small._down_pass(strategies)
    assert abs(XP.node_reach(small, reach, 0) - 1.0) < 1e-12


def test_reach_probabilities_partition_at_the_root(small):
    """The root's children must carry exactly the root's probability mass."""
    strategies = small.strategy_profile(average=True)
    reach = small._down_pass(strategies)
    total = sum(
        XP.node_reach(small, reach, c) for c in small.tree.children[0]
    )
    assert abs(total - 1.0) < 1e-12


def test_blind_statistics_are_probabilities(small):
    stats = XP.blind_statistics(small)
    assert 0.0 <= stats.p1_stay_blind_first <= 1.0
    for p in (0, 1):
        marg = sum(stats.convert_marginal[p].values())
        assert -1e-12 <= marg <= 1.0 + 1e-12
        assert abs(stats.never_convert[p] - (1.0 - marg)) < 1e-12
        for j, v in stats.convert_conditional[p].items():
            assert np.isnan(v) or -1e-12 <= v <= 1.0 + 1e-12


def test_dominant_action_stats_are_fractions(small):
    dom = XP.dominant_action_stats(small)
    assert 0.0 <= dom.overall_unweighted <= 1.0
    assert 0.0 <= dom.overall_weighted <= 1.0
    for d in dom.groups.values():
        assert np.isnan(d["unweighted_dominant"]) or 0.0 <= d["unweighted_dominant"] <= 1.0


def test_stack_above_34_never_binds_and_gives_one_game(tables):
    """RULES.md 3.2's non-binding stack makes stack depth a non-parameter."""
    rows = XP.stack_sweep(tables, iterations=40, stacks=(34, 50, 100))
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["tree_identical"] and r["values_identical"] for r in rows)
    assert len({r["n_nodes"] for r in rows}) == 1


def test_raising_the_boot_alone_leaves_the_specification(tables):
    """RULES.md E1 specifies no all-in rule, so the engine must refuse."""
    with pytest.raises(StackWouldBind):
        cfr.build_public_tree(GameConfig(boot=3, starting_stack=50))


def test_scaling_boot_and_stack_together_is_a_change_of_units(tables):
    a = cfr.CFRPlusSolver(GameConfig(boot=1, starting_stack=50,
                                     action_cap=2), tables=tables)
    b = cfr.CFRPlusSolver(GameConfig(boot=3, starting_stack=150,
                                     action_cap=2), tables=tables)
    a.iterate(60)
    b.iterate(60)
    assert abs(a.game_value() * 3 - b.game_value()) < 1e-9
    for raw in a.tree.decision_ids:
        n = int(raw)
        assert np.allclose(a.average_strategy(n), b.average_strategy(n), atol=1e-9)

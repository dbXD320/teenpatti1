"""Phase 2 entry gate: verify all six preconditions before any solver code.

Run: python gate_phase2.py
Exits non-zero if any gate fails.
"""

from __future__ import annotations

import sys
import traceback
from fractions import Fraction

RESULTS: list[tuple[str, bool, str]] = []


def gate(name):
    def deco(fn):
        try:
            detail = fn()
            RESULTS.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001 - a gate failure must be reported, not raised
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return deco


# --------------------------------------------------------------------------
# Gate 1: all three RULES.md worked examples replay exactly
# --------------------------------------------------------------------------

@gate("1. RULES.md 14 worked examples replay exactly (stake/pot/both stacks/payoff)")
def _gate1():
    from game import apply_action, initial_state, legal_actions, utilities, Deal
    from hands import parse_hand
    from test_game import ALL_EXAMPLES

    n_assertions = 0
    for ex in ALL_EXAMPLES:
        deal = Deal(parse_hand(ex["hands"][0]), parse_hand(ex["hands"][1]))
        st = initial_state()
        assert (st.stake, st.pot, st.contrib, st.stacks) == (1, 2, (1, 1), (49, 49))
        n_assertions += 4
        for i, (action, stake, pot, contrib, stacks) in enumerate(ex["rows"]):
            assert action in legal_actions(st), f"{ex['name']} step {i}: illegal"
            st = apply_action(st, action)
            assert st.stake == stake, f"{ex['name']} step {i}: stake"
            assert st.pot == pot, f"{ex['name']} step {i}: pot"
            assert st.contrib == contrib, f"{ex['name']} step {i}: contrib"
            assert st.stacks == stacks, f"{ex['name']} step {i}: stacks"
            n_assertions += 5
        assert st.is_terminal
        assert st.k == ex["k"] and st.r == ex["r"]
        u = utilities(st, deal)
        assert u == tuple(Fraction(x) for x in ex["utilities"]), f"{ex['name']}: payoff"
        assert u[0] + u[1] == 0
        n_assertions += 4
    return f"3 examples, {n_assertions} per-action assertions, all exact"


# --------------------------------------------------------------------------
# Gate 2: zero-sum at EVERY terminal in a full tree walk
# --------------------------------------------------------------------------

@gate("2. u_1 + u_2 == 0 at every terminal in a full tree walk")
def _gate2():
    import tree as T
    from game import Deal
    from hands import parse_hand

    deals = [
        Deal(parse_hand("Ah As 9d"), parse_hand("Kc Kd 3s")),  # P1 wins
        Deal(parse_hand("7d 4c 2s"), parse_hand("Ac Ad Ah")),  # P2 wins
        Deal(parse_hand("As 9s 5s"), parse_hand("Ah 9h 5h")),  # exact tie
    ]
    stats = T.enumerate_tree()
    checked = T.verify_zero_sum(deals)
    expected = stats.n_terminals * len(deals)
    assert checked == expected, f"checked {checked}, expected {expected}"
    assert stats.n_terminals == 3915
    return f"{stats.n_terminals:,} terminals x {len(deals)} deals = {checked:,} checks, all exact"


# --------------------------------------------------------------------------
# Gate 3: blind decision function STRUCTURALLY unable to see its own cards
# --------------------------------------------------------------------------

@gate("3. Blind infoset structurally cannot carry own cards")
def _gate3():
    import tree as T
    from game import (
        Action,
        BlindAccessViolation,
        BlindInfoSet,
        Deal,
        Phase,
        apply_action,
        initial_state,
    )
    from hands import parse_hand

    # (a) the type has no card field, and __slots__ prevents attaching one
    fields = set(BlindInfoSet.__dataclass_fields__)
    assert fields == {"public", "player", "phase"}, fields
    forbidden = {"hand", "hand_index", "cards", "kappa", "bucket", "hole"}
    assert not (fields & forbidden)
    iset = BlindInfoSet((), 0, Phase.LOOK)
    for attr in forbidden:
        assert not hasattr(iset, attr)
    try:
        iset.hand = parse_hand("As Ah Ad")
        raise AssertionError("attaching a hand to a BlindInfoSet must fail")
    except AttributeError:
        pass

    # (b) Deal refuses to release unseen cards
    deal = Deal(parse_hand("Ac Ad Ah"), parse_hand("7d 4c 2s"))
    st = initial_state()
    for p in (0, 1):
        try:
            deal.hand_for(p, st)
            raise AssertionError("Deal released a hand its owner had not seen")
        except BlindAccessViolation:
            pass
    seen = apply_action(st, Action.SEE)
    assert deal.hand_for(0, seen) == parse_hand("Ac Ad Ah")

    # (c) tree-wide: every blind decision node observes identically across deals
    deals = [
        Deal(parse_hand("Ac Ad Ah"), parse_hand("7d 4c 2s")),
        Deal(parse_hand("2c 3d 5h"), parse_hand("Ks Qs Js")),
        Deal(parse_hand("Ts 9s 8s"), parse_hand("Kc 7d 4h")),
    ]
    n = T.verify_blind_infosets_are_card_independent(deals)
    assert n == 728, n
    return f"type has no card field; Deal gated; all {n} blind nodes deal-invariant"


# --------------------------------------------------------------------------
# Gate 4: suit-isomorphic class count is NOT 741
# --------------------------------------------------------------------------

@gate("4. Suit-isomorphic class count != 741 (not grouping by kappa)")
def _gate4():
    import abstraction as AB
    from hands import N_CLASSES

    ab = AB.build_abstraction(25)
    n_iso = ab.n_iso_classes
    assert n_iso != N_CLASSES, "abstraction is grouping by kappa, not suit isomorphism"
    assert n_iso == 1755, n_iso
    assert AB.burnside_class_count() == n_iso, "Burnside disagrees with canonicalisation"
    # And it must be a strict refinement of kappa, not merely a different count.
    assert n_iso > N_CLASSES
    lossy = AB.kappa_classes_with_varying_equity(ab.wins, ab.ties, ab.losses)
    assert lossy, "kappa appeared lossless; card removal is being discarded"
    return (
        f"1,755 classes (kappa={N_CLASSES}), Burnside-confirmed, "
        f"{len(lossy)} kappa classes proven lossy"
    )


# --------------------------------------------------------------------------
# Gate 5: exact public decision history count <= 2.2e3 estimate
# --------------------------------------------------------------------------

@gate("5. Exact public decision history count <= RULES.md 9.3 estimate")
def _gate5():
    import tree as T

    stats = T.enumerate_tree()
    exact_decisions = stats.n_decision_nodes
    exact_bets = stats.n_bet_nodes
    assert exact_decisions <= T.RULES_ESTIMATE, (
        f"{exact_decisions} > {T.RULES_ESTIMATE}: engine and spec disagree, "
        "most likely on whether stay-blind consumes an action-cap slot"
    )
    assert exact_bets <= T.RULES_ESTIMATE
    assert exact_decisions == 2014 and exact_bets == 1650
    # Direct confirmation of the stay-blind rule (RULES.md 8.1/9.2/E13).
    from game import Action, apply_action, initial_state

    st = initial_state()
    st = apply_action(st, Action.STAY_BLIND)
    assert st.k == 0, "stay-blind must not increment k"
    return (
        f"2,014 decision nodes (1,650 bet) <= 2,200 estimate; "
        f"stay-blind confirmed not to consume a cap slot"
    )


# --------------------------------------------------------------------------
# Gate 6: no silent defaults
# --------------------------------------------------------------------------

@gate("6. No silent defaults: unspecified states raise")
def _gate6():
    from game import (
        Action,
        Deal,
        IllegalAction,
        apply_action,
        initial_state,
        legal_actions,
        observation,
        utilities,
    )
    from hands import classify, index_of, parse_hand

    checks = 0

    def must_raise(exc, fn, label):
        nonlocal checks
        try:
            fn()
        except exc:
            checks += 1
            return
        raise AssertionError(f"{label} did not raise {exc.__name__}")

    st = initial_state()
    # Betting actions at a look node
    for a in (Action.CHAAL, Action.RAISE, Action.PACK, Action.SHOW):
        must_raise(IllegalAction, lambda a=a: apply_action(st, a), f"{a.name} at look node")
    # Look actions at a bet node
    bet = apply_action(st, Action.STAY_BLIND)
    for a in (Action.SEE, Action.STAY_BLIND):
        must_raise(IllegalAction, lambda a=a: apply_action(bet, a), f"{a.name} at bet node")
    # Raise beyond the cap
    capped = st
    for a in (Action.STAY_BLIND, Action.RAISE, Action.STAY_BLIND, Action.RAISE):
        capped = apply_action(capped, a)
    capped = apply_action(capped, Action.STAY_BLIND)
    must_raise(IllegalAction, lambda: apply_action(capped, Action.RAISE), "raise at r=2")
    # Seen player demanding a show against a blind opponent
    sv = apply_action(apply_action(st, Action.SEE), Action.RAISE)
    sv = apply_action(sv, Action.STAY_BLIND)
    sv = apply_action(sv, Action.CHAAL)
    assert sv.seen == (True, False) and sv.to_act == 0
    must_raise(IllegalAction, lambda: apply_action(sv, Action.SHOW), "seen-vs-blind show")
    # Terminal states
    packed = apply_action(bet, Action.PACK)
    must_raise(IllegalAction, lambda: legal_actions(packed), "legal_actions at terminal")
    must_raise(
        IllegalAction,
        lambda: observation(packed, Deal(parse_hand("Ac Ad Ah"), parse_hand("7d 4c 2s"))),
        "observation at terminal",
    )
    must_raise(IllegalAction, lambda: utilities(st), "utilities at non-terminal")
    # Showdown without a deal
    shown = apply_action(bet, Action.SHOW)
    must_raise(ValueError, lambda: utilities(shown, None), "showdown without a deal")
    # Malformed cards / hands / deals
    must_raise(ValueError, lambda: parse_hand("As Ah"), "2-card hand")
    must_raise(ValueError, lambda: parse_hand("As As Kd"), "duplicate card")
    must_raise(ValueError, lambda: classify((0, 1)), "2-card classify")
    must_raise(ValueError, lambda: index_of((0, 1, 1)), "duplicate index_of")
    must_raise(
        ValueError,
        lambda: Deal(parse_hand("Ac Ad Ah"), parse_hand("Ac 4c 2s")),
        "overlapping deal",
    )
    return f"{checks} unspecified/illegal operations all raise"


# --------------------------------------------------------------------------

def main() -> int:
    print("Phase 2 entry gate\n" + "=" * 72)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 72)
    if failed:
        print(f"{len(failed)} GATE(S) FAILED -- do not start CFR:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print(f"All {len(RESULTS)} gates PASSED. Cleared to implement CFR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

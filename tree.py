"""Exact game-tree and information-set enumeration.

RULES.md 9.3 reports ~2.2e3 public decision histories as an analytic estimate
and explicitly flags it as sizing-only, because the conversion-pattern factor is
not fully independent of k. This module produces the exact count by
enumeration, which RULES.md requires as a gate before Phase 2 begins.

It also enumerates information sets under the partition of RULES.md 12.4:

  * a BLIND actor contributes exactly ONE information set per public history,
    with no refinement by their own cards;
  * a SEEN actor contributes one per (public history, own hand).

The blind case is the one that silently invalidates a solve if implemented
wrongly, so it is enforced structurally in `game.py` (see `BlindInfoSet`) and
checked here: `verify_blind_infosets_are_card_independent()` walks the tree
against several different deals and asserts the blind information sets it
observes are literally identical objects by value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

from game import (
    ACTION_CAP,
    DEFAULT_CONFIG,
    BlindInfoSet,
    Deal,
    Phase,
    STARTING_STACK,
    SeenInfoSet,
    State,
    TerminalKind,
    apply_action,
    initial_state,
    legal_actions,
    observation,
    utilities,
)
from hands import N_HANDS

#: Opponent hands consistent with a seen player's own hand: C(49,3).
N_OPPONENT_HANDS = 18424
#: Distinct deals at the chance node: C(52,3) * C(49,3) (RULES.md 12.2).
N_DEALS = N_HANDS * N_OPPONENT_HANDS


# --------------------------------------------------------------------------
# Tree walk
# --------------------------------------------------------------------------

def walk(state: State | None = None, config=DEFAULT_CONFIG):
    """Depth-first generator over every reachable public state.

    Yields both decision and terminal states. Chance is not expanded: these
    are public histories, so each is a single node regardless of the deal.
    """
    stack = [initial_state(config) if state is None else state]
    while stack:
        st = stack.pop()
        yield st
        if st.is_terminal:
            continue
        for action in legal_actions(st):
            stack.append(apply_action(st, action))


@dataclass
class TreeStats:
    """Exact structural counts for the capped heads-up game."""

    n_nodes: int = 0
    n_decision_nodes: int = 0
    n_look_nodes: int = 0
    n_bet_nodes: int = 0
    n_bet_nodes_blind_actor: int = 0
    n_bet_nodes_seen_actor: int = 0
    n_terminals: int = 0
    terminals_by_kind: Counter = field(default_factory=Counter)
    n_blind_infosets: int = 0
    n_seen_public_nodes: int = 0
    decision_nodes_by_player: Counter = field(default_factory=Counter)
    blind_infosets_by_player: Counter = field(default_factory=Counter)
    seen_public_nodes_by_player: Counter = field(default_factory=Counter)
    max_k: int = 0
    max_contribution: int = 0
    max_pot: int = 0
    depth_histogram: Counter = field(default_factory=Counter)

    # -- information-set totals under the three candidate abstractions -------

    def seen_infosets(self, hands_per_node: int) -> int:
        return self.n_seen_public_nodes * hands_per_node

    def total_infosets(self, hands_per_node: int) -> int:
        return self.n_blind_infosets + self.seen_infosets(hands_per_node)


def enumerate_tree(config=DEFAULT_CONFIG) -> TreeStats:
    """Walk the whole public tree and collect exact counts."""
    stats = TreeStats()

    for st in walk(config=config):
        stats.n_nodes += 1
        stats.max_k = max(stats.max_k, st.k)
        stats.max_contribution = max(stats.max_contribution, *st.contrib)
        stats.max_pot = max(stats.max_pot, st.pot)

        if st.is_terminal:
            stats.n_terminals += 1
            stats.terminals_by_kind[st.terminal.kind] += 1
            continue

        actor = st.to_act
        stats.n_decision_nodes += 1
        stats.decision_nodes_by_player[actor] += 1
        stats.depth_histogram[len(st.history)] += 1

        if st.phase is Phase.LOOK:
            # Only a blind player faces the look decision (RULES.md 13.4).
            stats.n_look_nodes += 1
            stats.n_blind_infosets += 1
            stats.blind_infosets_by_player[actor] += 1
            continue

        stats.n_bet_nodes += 1
        if st.seen[actor]:
            stats.n_bet_nodes_seen_actor += 1
            stats.n_seen_public_nodes += 1
            stats.seen_public_nodes_by_player[actor] += 1
        else:
            stats.n_bet_nodes_blind_actor += 1
            stats.n_blind_infosets += 1
            stats.blind_infosets_by_player[actor] += 1

    return stats


# --------------------------------------------------------------------------
# Verification gates
# --------------------------------------------------------------------------

def verify_zero_sum(deals) -> int:
    """Assert u_1 + u_2 == 0 at every terminal, for each deal in `deals`.

    Returns the number of (terminal, deal) pairs checked. RULES.md 11 requires
    this assertion; a split pot of an odd total makes it a Fraction identity,
    not an integer one.
    """
    checked = 0
    terminals = [st for st in walk() if st.is_terminal]
    for deal in deals:
        for st in terminals:
            u0, u1 = utilities(st, deal)
            if u0 + u1 != Fraction(0):
                raise AssertionError(
                    f"non-zero-sum terminal {st.terminal}: {u0} + {u1}"
                )
            if st.stack(0) < 0 or st.stack(1) < 0:
                raise AssertionError(f"negative stack at terminal: {st.stacks}")
            checked += 1
    return checked


def verify_blind_infosets_are_card_independent(deals) -> int:
    """Assert a blind actor's information set never depends on the deal.

    For every public history at which the actor is blind, the observation must
    be byte-identical across completely different deals, and must be a
    `BlindInfoSet` (which structurally has no card field). This is the check
    that fails loudly if RULES.md 12.4 is ever violated.
    """
    if len(deals) < 2:
        raise ValueError("need at least two distinct deals to compare")

    decisions = [st for st in walk() if not st.is_terminal]
    checked = 0
    for st in decisions:
        if st.seen[st.to_act]:
            continue
        observations = [observation(st, d) for d in deals]
        first = observations[0]
        if not isinstance(first, BlindInfoSet):
            raise AssertionError(
                f"blind actor got {type(first).__name__}, expected BlindInfoSet"
            )
        for obs in observations[1:]:
            if obs != first:
                raise AssertionError(
                    "a blind information set varied with the deal, violating "
                    f"RULES.md 12.4: {first} != {obs}"
                )
        checked += 1
    return checked


def verify_max_exposure() -> tuple[int, int]:
    """Exhaustively confirm the RULES.md 9.4 bound of 33.

    RULES.md 9.4 proves max c_i = 33 by exhibiting one witness. Enumerating
    the whole tree gives the bound directly, which is a strictly stronger
    check. Returns (max_contribution, max_pot).
    """
    max_contrib, max_pot = 0, 0
    for st in walk():
        max_contrib = max(max_contrib, *st.contrib)
        max_pot = max(max_pot, st.pot)
    if max_contrib > STARTING_STACK:
        raise AssertionError(
            f"max contribution {max_contrib} exceeds stack {STARTING_STACK}"
        )
    return max_contrib, max_pot


def odd_pot_forced_shows() -> list[State]:
    """Forced-show terminals whose pot is odd.

    RULES.md 10.6 gives the split as u_i = p/2 - c_i. That is well defined for
    an odd pot but not integral, which none of the RULES.md worked examples
    exhibits. Collected here so PHASE1.md can report it.
    """
    return [
        st
        for st in walk()
        if st.is_terminal
        and st.terminal.kind is TerminalKind.SHOW_FORCED
        and st.pot % 2 == 1
    ]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

RULES_ESTIMATE = 2200  # RULES.md 9.3, flagged as sizing-only


def report(stats: TreeStats | None = None, n_buckets: int = 25) -> str:
    stats = stats or enumerate_tree()
    lines = []
    add = lines.append

    add("=== Exact public tree (RULES.md 9.3 gate) ===")
    add(f"  public nodes (all)                  : {stats.n_nodes:>8,}")
    add(f"  public decision nodes               : {stats.n_decision_nodes:>8,}")
    add(f"    look nodes  (blind actor)         : {stats.n_look_nodes:>8,}")
    add(f"    bet nodes                         : {stats.n_bet_nodes:>8,}")
    add(f"      with blind actor                : {stats.n_bet_nodes_blind_actor:>8,}")
    add(f"      with seen actor                 : {stats.n_bet_nodes_seen_actor:>8,}")
    add(f"  terminal nodes                      : {stats.n_terminals:>8,}")
    for kind in TerminalKind:
        add(f"    {kind.name.lower():<34}: {stats.terminals_by_kind[kind]:>8,}")
    add("")
    add(f"  RULES.md 9.3 estimate               : {RULES_ESTIMATE:>8,}  (sizing-only)")
    add(f"  exact bet nodes  (== turns)         : {stats.n_bet_nodes:>8,}"
        f"   ratio {stats.n_bet_nodes / RULES_ESTIMATE:.2f}x")
    add(f"  exact decision nodes (look + bet)   : {stats.n_decision_nodes:>8,}"
        f"   ratio {stats.n_decision_nodes / RULES_ESTIMATE:.2f}x")
    add("")
    add(f"  max k reached                       : {stats.max_k:>8,} "
        f"(cap {ACTION_CAP})")
    add(f"  max contribution by one player      : {stats.max_contribution:>8,} "
        f"(RULES.md 9.4 says 33)")
    add(f"  max pot                             : {stats.max_pot:>8,}")

    add("")
    add("=== Information sets (RULES.md 12.4) ===")
    add(f"  blind infosets (1 per public history): {stats.n_blind_infosets:>10,}")
    add(f"    player 1                          : "
        f"{stats.blind_infosets_by_player[0]:>10,}")
    add(f"    player 2                          : "
        f"{stats.blind_infosets_by_player[1]:>10,}")
    add(f"  seen public nodes                   : {stats.n_seen_public_nodes:>10,}")
    add(f"    player 1                          : "
        f"{stats.seen_public_nodes_by_player[0]:>10,}")
    add(f"    player 2                          : "
        f"{stats.seen_public_nodes_by_player[1]:>10,}")
    add("")
    add("  seen infosets depend on the hand abstraction:")
    for label, per_node in (
        ("exact hands (22,100)", N_HANDS),
        ("suit-isomorphic (1,755)", 1755),
        (f"buckets ({n_buckets})", n_buckets),
    ):
        add(f"    {label:<34}: {stats.seen_infosets(per_node):>10,} seen"
            f"  -> {stats.total_infosets(per_node):>10,} total")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - reporting path
    from hands import parse_hand

    stats = enumerate_tree()
    print(report(stats))

    # Deals chosen to cover a P1 win, a P2 win and an exact tie.
    deals = [
        Deal(parse_hand("Ah As 9d"), parse_hand("Kc Kd 3s")),   # P1 wins
        Deal(parse_hand("7d 4c 2s"), parse_hand("Ac Ad Ah")),   # P2 wins
        Deal(parse_hand("As 9s 5s"), parse_hand("Ah 9h 5h")),   # exact tie
    ]
    print()
    print("=== Verification gates ===")
    n = verify_zero_sum(deals)
    print(f"  zero-sum at every terminal          : OK ({n:,} checks)")
    n = verify_blind_infosets_are_card_independent(deals)
    print(f"  blind infosets card-independent     : OK ({n:,} blind nodes)")
    contrib, pot = verify_max_exposure()
    print(f"  max exposure exhaustive             : {contrib} (RULES.md 9.4: 33)")
    odd = odd_pot_forced_shows()
    print(f"  forced shows with an odd pot        : {len(odd):,}"
          f"  -> split is a half-integer")
    if odd:
        st = min(odd, key=lambda s: s.pot)
        print(f"    smallest example: pot={st.pot} c={st.contrib} "
              f"-> u = {utilities(st, deals[2])}")


if __name__ == "__main__":  # pragma: no cover
    main()

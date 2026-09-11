"""Heads-up Teen Patti game engine.

A faithful, deliberately slow, auditable implementation of the extensive form in
RULES.md sections 3, 6-11 and 12-13. Designed for Phase 2 (tabular CFR+ with
exact best-response exploitability), so every rule is spelled out and every
edge case in RULES.md section 15 is handled explicitly rather than by fallthrough.

Structural guarantee (RULES.md 12.4)
------------------------------------
A blind player has exactly ONE information set per public history. This is
enforced by the type system, not by convention:

  * `BlindInfoSet` has no field for cards and uses __slots__, so a policy
    handed one cannot read the player's hand even by accident.
  * `Deal.hand_for()` raises `BlindAccessViolation` if asked for a hand its
    owner has not seen.
  * `observation()` does not touch the `Deal` at all on the blind branch.

Exact arithmetic
----------------
Utilities are `Fraction`. A cap-triggered forced show splits the pot (RULES.md
10.6) and the pot can be odd, so payoffs are not always integral. Floats would
break the exact zero-sum assertion RULES.md section 11 requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction

from hands import HAND_KAPPA, index_of

# --------------------------------------------------------------------------
# Table constants (RULES.md 3)
# --------------------------------------------------------------------------

BOOT = 1               # RULES.md 3.1: symmetric ante, both players
STARTING_STACK = 50    # RULES.md 3.2
MIN_LEGAL_STACK = 34   # RULES.md 3.2 / E1: max exposure is 33, so reject < 34
INITIAL_STAKE = 1      # RULES.md 7.1
ACTION_CAP = 8         # RULES.md 9.2 C2
RAISE_CAP = 2          # RULES.md 9.2 C1
N_PLAYERS = 2

if STARTING_STACK < MIN_LEGAL_STACK:  # RULES.md E1, checked at import
    raise ValueError(
        f"starting stack {STARTING_STACK} < {MIN_LEGAL_STACK}; RULES.md 3.2 "
        "requires a non-binding stack and this specification has no all-in rule"
    )


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Table parameters. The default is exactly the RULES.md specification.

    Parameterised only so Phase 2 can run the required sweep. Any value other
    than `DEFAULT_CONFIG` is a variant of the specification, not the
    specification, and must be reported as such.
    """

    boot: int = BOOT
    starting_stack: int = STARTING_STACK
    action_cap: int = ACTION_CAP        # RULES.md 9.2 C2
    raise_cap: int = RAISE_CAP          # RULES.md 9.2 C1

    def __post_init__(self):
        if self.boot < 1:
            raise ValueError(f"boot must be >= 1, got {self.boot}")
        if self.starting_stack < self.boot:
            raise ValueError(
                f"stack {self.starting_stack} cannot cover the boot {self.boot}"
            )
        if self.action_cap < 1:
            raise ValueError(f"action_cap must be >= 1, got {self.action_cap}")
        if self.raise_cap < 0:
            raise ValueError(f"raise_cap must be >= 0, got {self.raise_cap}")

    @property
    def initial_stake(self) -> int:
        """RULES.md 7.1: the opening stake equals the boot."""
        return self.boot

    def label(self) -> str:
        return (
            f"boot={self.boot} stack={self.starting_stack} "
            f"k<={self.action_cap} r<={self.raise_cap}"
        )


DEFAULT_CONFIG = GameConfig()


class Phase(IntEnum):
    LOOK = 0   # RULES.md 6.2 step 1, reached only when the actor is blind
    BET = 1    # RULES.md 6.2 step 2


class Action(IntEnum):
    SEE = 0
    STAY_BLIND = 1
    CHAAL = 2
    RAISE = 3
    PACK = 4
    SHOW = 5


ACTION_NAMES = {
    Action.SEE: "see",
    Action.STAY_BLIND: "stay-blind",
    Action.CHAAL: "chaal",
    Action.RAISE: "raise",
    Action.PACK: "pack",
    Action.SHOW: "show",
}

LOOK_ACTIONS = (Action.SEE, Action.STAY_BLIND)
BETTING_ACTIONS = (Action.CHAAL, Action.RAISE, Action.PACK, Action.SHOW)
#: chaal and raise are the only actions that advance the action cap (RULES.md 9.2, E13)
CAP_CONSUMING_ACTIONS = (Action.CHAAL, Action.RAISE)


class IllegalAction(ValueError):
    """Raised when an action not in `legal_actions()` is applied."""


class BlindAccessViolation(RuntimeError):
    """Raised on any attempt to read a hand its owner has not seen."""


class StackWouldBind(RuntimeError):
    """Raised when a configuration would force an all-in.

    RULES.md E1 states that a player being unable to afford a legal action
    "cannot occur" and specifies no side-pot or all-in mechanism. Any
    configuration that reaches this point is outside the specification, so the
    engine refuses rather than inventing a rule.
    """


# --------------------------------------------------------------------------
# Terminal descriptors (RULES.md 11)
# --------------------------------------------------------------------------

class TerminalKind(IntEnum):
    PACK = 0            # a player folded
    SHOW_REQUESTED = 1  # a player paid for a show; ties go against them (10.5)
    SHOW_FORCED = 2     # action cap bound; no fee, ties split (9.5, 10.6)


@dataclass(frozen=True, slots=True)
class Terminal:
    kind: TerminalKind
    #: the packer, or the show requester; None for a forced show (no requester)
    actor: int | None


# --------------------------------------------------------------------------
# Public state (RULES.md 12.3)
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class State:
    """Public state. Every field here is public information (RULES.md 6.3).

    Private information is held separately in `Deal` and is never reachable
    from a `State`.
    """

    stake: int
    contrib: tuple[int, int]
    seen: tuple[bool, bool]
    k: int
    r: int
    to_act: int
    phase: Phase
    history: tuple[Action, ...]
    terminal: Terminal | None
    config: GameConfig = DEFAULT_CONFIG

    # -- derived quantities (RULES.md 2.3: the pot is derived, never tracked) --

    @property
    def pot(self) -> int:
        return self.contrib[0] + self.contrib[1]

    def stack(self, player: int) -> int:
        return self.config.starting_stack - self.contrib[player]

    @property
    def stacks(self) -> tuple[int, int]:
        return (self.stack(0), self.stack(1))

    @property
    def is_terminal(self) -> bool:
        return self.terminal is not None

    def opponent(self, player: int) -> int:
        return 1 - player

    def public_key(self) -> tuple[Action, ...]:
        """Information-set key for the public history (RULES.md 12.4).

        RULES.md 12.4 defines H_pub as the full sequence of actions taken, and
        notes that the state vector is a function of it. Keying on the action
        sequence itself preserves perfect recall (RULES.md 12.6); keying on the
        state vector could merge distinct histories and break it.
        """
        return self.history


def initial_state(config: GameConfig = DEFAULT_CONFIG) -> State:
    """Post the symmetric boot and open the betting (RULES.md 3.1, 3.4, 7.1)."""
    return State(
        stake=config.initial_stake,
        contrib=(config.boot, config.boot),
        seen=(False, False),  # RULES.md 6.1: both players begin blind
        k=0,
        r=0,
        to_act=0,  # RULES.md 3.4: seat 1 acts first
        phase=Phase.LOOK,  # both blind, so the actor faces the look decision
        history=(),
        terminal=None,
        config=config,
    )


# --------------------------------------------------------------------------
# Pricing (RULES.md 6.4, 7.2, 10.2)
# --------------------------------------------------------------------------

def amount_for(state: State, action: Action) -> int:
    """Chips the actor pays for `action`.

    Priced off the actor's status AFTER any `see` this turn (RULES.md E7).
    Because `see` mutates `seen` before the bet node is reached, reading
    `state.seen` here is automatically the post-look status.
    """
    if action in LOOK_ACTIONS:
        return 0
    if action is Action.PACK:
        return 0  # RULES.md 14.1: packing pays nothing

    s = state.stake
    is_seen = state.seen[state.to_act]
    if action is Action.CHAAL:
        return 2 * s if is_seen else s
    if action is Action.RAISE:
        return 4 * s if is_seen else 2 * s
    if action is Action.SHOW:
        # The show fee is exactly the requester's chaal price (RULES.md 10.2),
        # and for a blind requester it is s regardless of the opponent's
        # status (RULES.md E3).
        return 2 * s if is_seen else s
    raise IllegalAction(f"unknown action {action!r}")


# --------------------------------------------------------------------------
# Legality (RULES.md 8.3)
# --------------------------------------------------------------------------

def legal_actions(state: State) -> tuple[Action, ...]:
    """Exact action mask. Order is deterministic for reproducibility."""
    if state.is_terminal:
        raise IllegalAction("terminal state has no legal actions")

    actor = state.to_act
    if state.phase is Phase.LOOK:
        # Reached only when the actor is blind (RULES.md 8.3, E6). A seen
        # player never faces the look decision, so `see` is unreachable for
        # them by construction rather than by a check.
        if state.seen[actor]:
            raise AssertionError(
                "LOOK phase reached with a seen actor; conversion is one-way "
                "(RULES.md 6.2, E6)"
            )
        return LOOK_ACTIONS

    actions = [Action.PACK, Action.CHAAL]  # unconditionally legal (RULES.md 8.3)
    if state.r < state.config.raise_cap:
        actions.append(Action.RAISE)  # RULES.md E5
    if _show_is_legal(state):
        actions.append(Action.SHOW)
    return tuple(actions)


def _show_is_legal(state: State) -> bool:
    """RULES.md 8.3 / 10.3: legal iff the actor is blind, or both are seen.

    A seen player may never demand a show against a blind opponent
    (RULES.md E4) -- "you cannot see a blind man". Heads-up this makes the
    show a unilateral right of the blind player.
    """
    actor = state.to_act
    if not state.seen[actor]:
        return True
    return state.seen[state.opponent(actor)]


# --------------------------------------------------------------------------
# Transition
# --------------------------------------------------------------------------

def apply_action(state: State, action: Action) -> State:
    """Apply `action`, returning the successor state. Never mutates `state`."""
    legal = legal_actions(state)
    if action not in legal:
        raise IllegalAction(
            f"{ACTION_NAMES.get(action, action)!r} is not legal here; "
            f"legal={[ACTION_NAMES[a] for a in legal]} "
            f"(phase={state.phase.name}, seen={state.seen}, r={state.r})"
        )

    actor = state.to_act
    history = state.history + (action,)

    # ---- look decisions: same actor, no chips, no cap consumption ----------
    if action is Action.SEE:
        # RULES.md 6.2: irreversible conversion, then the actor must bet at
        # seen prices in this same turn.
        seen = list(state.seen)
        seen[actor] = True
        return _replace(state, seen=tuple(seen), phase=Phase.BET, history=history)

    if action is Action.STAY_BLIND:
        # RULES.md 8.1, 9.2, E13: does not increment k.
        return _replace(state, phase=Phase.BET, history=history)

    # ---- betting decisions -------------------------------------------------
    paid = amount_for(state, action)
    contrib = list(state.contrib)
    contrib[actor] += paid
    contrib = (contrib[0], contrib[1])

    if state.config.starting_stack - contrib[actor] < 0:  # RULES.md E1
        raise StackWouldBind(
            f"player {actor} overdrew: contribution {contrib[actor]} exceeds "
            f"stack {state.config.starting_stack}. RULES.md 9.4 proves this is "
            f"unreachable at the default config; this configuration "
            f"({state.config.label()}) needs an all-in rule, which RULES.md "
            f"deliberately does not specify (E1)."
        )

    if action is Action.PACK:
        return _replace(
            state,
            contrib=contrib,
            history=history,
            terminal=Terminal(TerminalKind.PACK, actor),
        )

    if action is Action.SHOW:
        return _replace(
            state,
            contrib=contrib,
            history=history,
            terminal=Terminal(TerminalKind.SHOW_REQUESTED, actor),
        )

    # chaal / raise: the stake transition ignores status entirely -- chaal
    # leaves it unchanged, raise doubles it (RULES.md 7.2, E14).
    if action is Action.CHAAL:
        stake, raises = state.stake, state.r
    elif action is Action.RAISE:
        stake, raises = state.stake * 2, state.r + 1
    else:
        raise IllegalAction(f"unhandled betting action {action!r}")

    k = state.k + 1
    cap = state.config.action_cap
    if k > cap:
        raise AssertionError(f"action cap {cap} exceeded (k={k})")

    if k == cap:
        # RULES.md 9.5: forced show, immediately, at no cost, no requester.
        return _replace(
            state,
            stake=stake,
            contrib=contrib,
            k=k,
            r=raises,
            history=history,
            terminal=Terminal(TerminalKind.SHOW_FORCED, None),
        )

    nxt = state.opponent(actor)
    return _replace(
        state,
        stake=stake,
        contrib=contrib,
        k=k,
        r=raises,
        to_act=nxt,
        # A blind next actor faces the look decision; a seen one goes straight
        # to the bet node (RULES.md 13.4).
        phase=Phase.BET if state.seen[nxt] else Phase.LOOK,
        history=history,
    )


def _replace(state: State, **changes) -> State:
    fields = {
        "stake": state.stake,
        "contrib": state.contrib,
        "seen": state.seen,
        "k": state.k,
        "r": state.r,
        "to_act": state.to_act,
        "phase": state.phase,
        "history": state.history,
        "terminal": state.terminal,
        "config": state.config,
    }
    fields.update(changes)
    return State(**fields)


# --------------------------------------------------------------------------
# Private information (RULES.md 12.2, 13.2)
# --------------------------------------------------------------------------

class Deal:
    """The chance outcome: two disjoint 3-card hands.

    RULES.md 12.2: dealing and observing are SEPARATE events. Cards exist from
    the chance node, but a player observes them only via `see`. Access is
    gated accordingly; the engine's showdown path is the one exception
    (RULES.md E2: a forced show must evaluate a hand its owner never saw).
    """

    __slots__ = ("_hands", "_kappa")

    def __init__(self, hand0, hand1):
        h0 = tuple(sorted(hand0))
        h1 = tuple(sorted(hand1))
        if len(set(h0) | set(h1)) != 6:
            raise ValueError(f"hands must be disjoint 3-card sets: {h0}, {h1}")
        self._hands = (h0, h1)
        self._kappa = (HAND_KAPPA[index_of(h0)], HAND_KAPPA[index_of(h1)])

    def hand_for(self, player: int, state: State) -> tuple[int, int, int]:
        """The hand of `player`, only if they have seen it.

        This is the sole player-facing accessor. It raises rather than
        returning anything a blind player is not entitled to.
        """
        if not state.seen[player]:
            raise BlindAccessViolation(
                f"player {player} is blind and has not observed their cards "
                "(RULES.md 12.4); a blind player's decision cannot depend on them"
            )
        return self._hands[player]

    def showdown_kappa(self) -> tuple[int, int]:
        """Both strength classes, for engine-side showdown resolution only.

        Not gated on seen-status: RULES.md E2 requires a forced show to
        evaluate hands neither player looked at.
        """
        return self._kappa

    def hand_unchecked(self, player: int) -> tuple[int, int, int]:
        """Engine/test-only accessor that bypasses the seen-status gate."""
        return self._hands[player]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Deal(<hidden>, <hidden>)"


# --------------------------------------------------------------------------
# Information sets (RULES.md 12.4, 13.4)
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BlindInfoSet:
    """What a blind player knows: the public history, and nothing else.

    There is deliberately NO field for the player's own cards. With __slots__,
    one cannot even be attached at runtime. This is the structural enforcement
    of RULES.md 12.4: exactly one information set per public history,
    independent of the deal.
    """

    public: tuple[Action, ...]
    player: int
    phase: Phase


@dataclass(frozen=True, slots=True)
class SeenInfoSet:
    """What a seen player knows: the public history plus their own hand."""

    public: tuple[Action, ...]
    player: int
    phase: Phase
    hand_index: int


def observation(state: State, deal: Deal) -> BlindInfoSet | SeenInfoSet:
    """The acting player's information set (RULES.md 13.4).

    On the blind branch `deal` is never consulted, so no card information can
    leak into the returned object.
    """
    if state.is_terminal:
        raise IllegalAction("terminal state has no acting player")
    actor = state.to_act
    if not state.seen[actor]:
        return BlindInfoSet(state.public_key(), actor, state.phase)
    return SeenInfoSet(
        state.public_key(),
        actor,
        state.phase,
        index_of(deal.hand_for(actor, state)),
    )


# --------------------------------------------------------------------------
# Terminal utilities (RULES.md 11)
# --------------------------------------------------------------------------

def utilities(state: State, deal: Deal | None = None) -> tuple[Fraction, Fraction]:
    """Zero-sum payoffs in boot units.

    `deal` is required at any showdown and unused at a pack. Returns exact
    `Fraction`s: a forced-show split of an odd pot is a half-integer.
    """
    if not state.is_terminal:
        raise IllegalAction("utilities requested for a non-terminal state")

    term = state.terminal
    pot = Fraction(state.pot)
    c = (Fraction(state.contrib[0]), Fraction(state.contrib[1]))

    if term.kind is TerminalKind.PACK:
        winner = state.opponent(term.actor)
        return _decisive(pot, c, winner)

    if deal is None:
        raise ValueError(f"a {term.kind.name} terminal needs the deal to resolve")
    k0, k1 = deal.showdown_kappa()

    if k0 == k1:
        if term.kind is TerminalKind.SHOW_REQUESTED:
            # RULES.md 10.5: the requester loses the tie outright.
            return _decisive(pot, c, state.opponent(term.actor))
        # RULES.md 10.6: forced show has no requester, so the pot splits.
        # Nonzero payoffs are normal here because contributions differ (7.3).
        half = pot / 2
        return (half - c[0], half - c[1])

    winner = 0 if k0 > k1 else 1
    return _decisive(pot, c, winner)


def _decisive(pot: Fraction, c: tuple[Fraction, Fraction], winner: int):
    loser = 1 - winner
    u = [None, None]
    u[winner] = pot - c[winner]
    u[loser] = -c[loser]
    total = u[0] + u[1]
    if total != 0:  # RULES.md 11: the assertion that catches accounting bugs
        raise AssertionError(f"payoffs not zero-sum: {u[0]} + {u[1]} = {total}")
    return (u[0], u[1])


# --------------------------------------------------------------------------
# Convenience: replay a scripted line (used by the RULES.md example tests)
# --------------------------------------------------------------------------

def replay(actions, state: State | None = None,
           config: GameConfig = DEFAULT_CONFIG) -> list[State]:
    """Apply `actions` in order, returning the state after each one."""
    st = initial_state(config) if state is None else state
    out = []
    for a in actions:
        st = apply_action(st, a)
        out.append(st)
    return out

"""Vanilla full-tree tabular CFR+ for heads-up Teen Patti.

Design
------
Full traversal every iteration -- no sampling, no MCCFR, no function
approximation. The public tree has 5,929 nodes (RULES.md 9.3 / PHASE1.md), so a
complete sweep is cheap and the regrets are exact, which makes the solver
debuggable.

Private state is the SUIT-ISOMORPHIC class (1,755 of them), not the strength
class kappa (741). PHASE1.md section 3 establishes that 442 of the 741 kappa
classes contain hands with different exact equities, so kappa discards
card-removal information and is invalid as a solver abstraction. The 1,755-class
quotient is lossless: hands related by a permutation of suits generate
isomorphic subgames, so an equilibrium symmetric under S_4 exists and
restricting to it is without loss of generality.

Card removal between the two players is handled exactly by `M[c, c']`, the number
of hands in class c' disjoint from a hand in class c. Every row of M sums to
18,424 = C(49,3), and `size[c] * M[c,c'] == size[c'] * M[c',c]`, both asserted at
build time.

Vector form and the blind information structure
-----------------------------------------------
The tree is traversed once per iteration carrying, at each public node, a reach
vector over private classes for each player. This is what makes the blind
mechanic representable:

  * a SEEN player's strategy at a public node is a [n_classes x n_actions]
    matrix, and each class is its own information set;
  * a BLIND player's strategy is a single [n_actions] vector shared by every
    class, because RULES.md 12.4 gives them exactly ONE information set per
    public history.

Consequently a blind player's reach is a SCALAR, not a vector -- every action
they have taken was hand-independent. This is asserted on every traversal
(`_assert_blind_reach_is_scalar`); if it ever becomes a vector, the blind
information set has leaked and the solve is void.

Regrets for a blind node are accumulated from the chance-weighted AGGREGATE over
classes, sum_c w[c] * (v^a[c] - v[c]), never per class. There is no code path
that indexes a blind node's regret by class: those arrays are allocated with
shape (n_actions,), so a per-class blind strategy is unrepresentable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game import (
    Action,
    DEFAULT_CONFIG,
    GameConfig,
    State,
    TerminalKind,
    apply_action,
    initial_state,
    legal_actions,
)
from hands import HAND_KAPPA, HANDS, N_HANDS, canonical_hand, hand_str, index_of

N_OPPONENT_HANDS = 18424  # C(49,3)


def _is_vector(x) -> bool:
    return isinstance(x, np.ndarray) and x.ndim == 1


# ==========================================================================
# Private-state tables (config-independent, built once and shared)
# ==========================================================================

@dataclass
class PrivateTables:
    """Suit-isomorphic private state and exact card-removal structure."""

    n_classes: int
    class_of_hand: np.ndarray      # int32[22100]
    class_size: np.ndarray         # int64[nC], sums to 22100
    kappa_of_class: np.ndarray     # int32[nC]
    rep_hand: list                 # representative hand per class
    chance_w: np.ndarray           # float64[nC] = size/22100, sums to 1
    cond: np.ndarray               # float64[nC,nC] = M/18424, rows sum to 1
    cond_win: np.ndarray           # cond masked to kappa(c') < kappa(c)
    cond_tie: np.ndarray           # cond masked to kappa(c') == kappa(c)
    # Precomputed rows for a BLIND opponent, whose reach is uniform across
    # classes: then the win/tie mass is a fixed vector times that scalar.
    unit_win: np.ndarray           # float64[nC] = cond_win @ 1
    unit_tie: np.ndarray           # float64[nC] = cond_tie @ 1
    #: [3nC, nC] = vstack(cond_win, cond_tie, cond), so a showdown batch needs
    #: ONE contiguous BLAS call instead of three separate bandwidth-bound ones.
    cond_stack: np.ndarray = None

    def describe_class(self, c: int) -> str:
        return hand_str(self.rep_hand[c])


def build_private_tables(verify: bool = True) -> PrivateTables:
    """Build the suit-isomorphic quotient and the disjointness matrix."""
    canon_to_id: dict[tuple, int] = {}
    class_of_hand = np.empty(N_HANDS, dtype=np.int32)
    rep_hand: list = []
    for i, hand in enumerate(HANDS):
        canon = canonical_hand(hand)
        cid = canon_to_id.get(canon)
        if cid is None:
            cid = len(rep_hand)
            canon_to_id[canon] = cid
            rep_hand.append(hand)  # first hand seen in this orbit
        class_of_hand[i] = cid

    n_classes = len(rep_hand)
    class_size = np.bincount(class_of_hand, minlength=n_classes).astype(np.int64)
    kappa_of_class = np.array(
        [HAND_KAPPA[index_of(rep_hand[c])] for c in range(n_classes)], dtype=np.int32
    )

    # Hands containing each card, for the intersection count below.
    card_hands: list[list[int]] = [[] for _ in range(52)]
    for i, hand in enumerate(HANDS):
        for card in hand:
            card_hands[card].append(i)
    card_hands_np = [np.array(v, dtype=np.int64) for v in card_hands]

    # M[c0, c1] = #{h1 in c1 : h1 disjoint from h0}, for any h0 in c0.
    # Well-defined because both classes are S_4-invariant: if h0' = sigma(h0)
    # then the disjoint members of c1 are exactly sigma of those for h0.
    M = np.zeros((n_classes, n_classes), dtype=np.int64)
    for c0 in range(n_classes):
        h0 = rep_hand[c0]
        touching = np.unique(np.concatenate([card_hands_np[card] for card in h0]))
        M[c0] = class_size - np.bincount(class_of_hand[touching], minlength=n_classes)

    if verify:
        verify_disjointness_matrix(M, class_size)

    cond = M.astype(np.float64) / N_OPPONENT_HANDS
    weaker = kappa_of_class[None, :] < kappa_of_class[:, None]
    equal = kappa_of_class[None, :] == kappa_of_class[:, None]
    cond_win = np.ascontiguousarray(np.where(weaker, cond, 0.0))
    cond_tie = np.ascontiguousarray(np.where(equal, cond, 0.0))

    return PrivateTables(
        n_classes=n_classes,
        class_of_hand=class_of_hand,
        class_size=class_size,
        kappa_of_class=kappa_of_class,
        rep_hand=rep_hand,
        chance_w=class_size.astype(np.float64) / N_HANDS,
        cond=np.ascontiguousarray(cond),
        cond_win=cond_win,
        cond_tie=cond_tie,
        unit_win=cond_win.sum(axis=1),
        unit_tie=cond_tie.sum(axis=1),
        cond_stack=np.ascontiguousarray(np.vstack([cond_win, cond_tie, cond])),
    )


def verify_disjointness_matrix(M: np.ndarray, class_size: np.ndarray) -> None:
    """Three independent checks that M is exactly right."""
    row_sums = M.sum(axis=1)
    if not np.all(row_sums == N_OPPONENT_HANDS):
        raise AssertionError(
            f"M rows must each sum to {N_OPPONENT_HANDS}; got "
            f"min={row_sums.min()} max={row_sums.max()}"
        )
    if M.min() < 0:
        raise AssertionError("M has a negative count")
    ordered = class_size[:, None] * M
    if not np.all(ordered == ordered.T):
        raise AssertionError(
            "size[c0]*M[c0,c1] must equal size[c1]*M[c1,c0]; both count the "
            "same set of ordered disjoint hand pairs"
        )
    total = int(ordered.sum())
    expected = N_HANDS * N_OPPONENT_HANDS
    if total != expected:
        raise AssertionError(
            f"total ordered disjoint deals {total} != {expected} (RULES.md 12.2)"
        )


# ==========================================================================
# Flattened public tree
# ==========================================================================

@dataclass
class PublicTree:
    """The public tree of one config, flattened into arrays.

    Node 0 is the root. `children[n][i]` is the node reached by `actions[n][i]`.
    Nodes are numbered in DFS pre-order, so every parent has a lower index than
    all of its descendants. The traversals rely on that.
    """

    config: GameConfig
    states: list[State]
    actions: list[tuple[Action, ...]]
    children: list[tuple[int, ...]]
    actor: np.ndarray              # int8[N], -1 at terminals
    actor_blind: np.ndarray        # bool[N]
    is_terminal: np.ndarray        # bool[N]
    terminal_ids: np.ndarray
    decision_ids: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.states)


def build_public_tree(config: GameConfig = DEFAULT_CONFIG) -> PublicTree:
    states: list[State] = []
    actions: list[tuple[Action, ...]] = []
    children: list[list[int]] = []

    def add(state: State) -> int:
        idx = len(states)
        states.append(state)
        actions.append(())
        children.append([])
        if state.is_terminal:
            return idx
        legal = legal_actions(state)
        actions[idx] = legal
        for action in legal:
            children[idx].append(add(apply_action(state, action)))
        return idx

    add(initial_state(config))

    n = len(states)
    actor = np.full(n, -1, dtype=np.int8)
    actor_blind = np.zeros(n, dtype=bool)
    is_terminal = np.zeros(n, dtype=bool)
    for i, st in enumerate(states):
        if st.is_terminal:
            is_terminal[i] = True
            continue
        actor[i] = st.to_act
        actor_blind[i] = not st.seen[st.to_act]

    # Guard the ordering assumption the traversals depend on.
    for i, kids in enumerate(children):
        for c in kids:
            if c <= i:
                raise AssertionError("tree is not in DFS pre-order")

    return PublicTree(
        config=config,
        states=states,
        actions=actions,
        children=[tuple(c) for c in children],
        actor=actor,
        actor_blind=actor_blind,
        is_terminal=is_terminal,
        terminal_ids=np.flatnonzero(is_terminal),
        decision_ids=np.flatnonzero(~is_terminal),
    )


# ==========================================================================
# Terminal payoff coefficients (RULES.md 11)
# ==========================================================================

@dataclass(frozen=True, slots=True)
class TerminalCoeffs:
    """Payoff to one player at one terminal, as coefficients on opponent mass.

    For a pack the payoff is hand-independent, so only `const` applies. For a
    showdown the payoff is `win` when the player's kappa is higher, `lose` when
    lower, and `tie` when equal.
    """

    is_showdown: bool
    const: float = 0.0
    win: float = 0.0
    lose: float = 0.0
    tie: float = 0.0


def terminal_coeffs(state: State, player: int) -> TerminalCoeffs:
    """Transcribe RULES.md 11's payoff table for one player at one terminal."""
    if not state.is_terminal:
        raise ValueError("terminal_coeffs called on a decision node")
    term = state.terminal
    own = float(state.contrib[player])
    opp = float(state.contrib[1 - player])

    if term.kind is TerminalKind.PACK:
        # The winner takes pot - c_winner == c_loser; the loser pays -c_loser.
        if term.actor == player:
            return TerminalCoeffs(False, const=-own)
        return TerminalCoeffs(False, const=+opp)

    win, lose = +opp, -own
    if term.kind is TerminalKind.SHOW_REQUESTED:
        # RULES.md 10.5: the requester loses the tie outright.
        tie = lose if term.actor == player else win
    elif term.kind is TerminalKind.SHOW_FORCED:
        # RULES.md 10.6: no requester, so the pot splits. Nonzero because
        # contributions need not be equal (RULES.md 7.3), and a half-integer
        # when the pot is odd (PHASE1.md 6.1).
        tie = (opp - own) / 2.0
    else:
        raise ValueError(f"unhandled terminal kind {term.kind!r}")
    return TerminalCoeffs(True, win=win, lose=lose, tie=tie)


# ==========================================================================
# The solver
# ==========================================================================

class CFRPlusSolver:
    """Vanilla full-tree CFR+ with regret matching+ and linear averaging.

    Updates are ALTERNATING (the canonical CFR+ formulation): each iteration
    traverses once per player, updating only that player's regrets.
    """

    def __init__(
        self,
        config: GameConfig = DEFAULT_CONFIG,
        tables: PrivateTables | None = None,
        tree: PublicTree | None = None,
    ):
        self.config = config
        self.tables = tables if tables is not None else build_private_tables()
        self.tree = tree if tree is not None else build_public_tree(config)
        if self.tree.config != config:
            raise ValueError("supplied tree was built for a different config")
        self.iterations = 0
        self._allocate()
        self._plan_terminal_batches()
        self._strat_cache = [None] * self.tree.n_nodes
        self._dirty = {int(n) for n in self.tree.decision_ids}

    # -- allocation --------------------------------------------------------

    def _allocate(self) -> None:
        nC = self.tables.n_classes
        tree = self.tree
        self.regret: list[np.ndarray | None] = [None] * tree.n_nodes
        self.strat_sum: list[np.ndarray | None] = [None] * tree.n_nodes

        for raw in tree.decision_ids:
            n = int(raw)
            n_actions = len(tree.actions[n])
            if tree.actor_blind[n]:
                # RULES.md 12.4: ONE information set, so shape (n_actions,).
                # There is deliberately no class axis available to index.
                shape = (n_actions,)
            else:
                shape = (nC, n_actions)
            self.regret[n] = np.zeros(shape, dtype=np.float64)
            self.strat_sum[n] = np.zeros(shape, dtype=np.float64)

    def _plan_terminal_batches(self) -> None:
        """Group (terminal, player) pairs by whether the opponent is seen.

        A blind opponent's reach is a scalar, so the opponent mass is a
        precomputed vector times that scalar and needs no matrix product. Only a
        seen opponent needs the exact card-removal product, and those are
        batched into one matmul per traversal: a per-terminal matrix-vector
        product would be memory-bandwidth bound and roughly an order of
        magnitude slower.
        """
        tree = self.tree
        self.batch_show: dict[int, np.ndarray] = {}
        self.batch_pack: dict[int, np.ndarray] = {}
        self.cheap_nodes: dict[int, np.ndarray] = {}
        self.coeffs: dict[int, list[TerminalCoeffs]] = {}
        self._term_pos = {int(n): i for i, n in enumerate(tree.terminal_ids)}
        for player in (0, 1):
            show, pack, cheap, coeff_list = [], [], [], []
            for raw in tree.terminal_ids:
                n = int(raw)
                st = tree.states[n]
                coeff = terminal_coeffs(st, player)
                coeff_list.append(coeff)
                if not st.seen[1 - player]:
                    cheap.append(n)          # blind opponent: no matmul at all
                elif coeff.is_showdown:
                    show.append(n)           # needs win, tie and total
                else:
                    pack.append(n)           # hand-independent: total only
            self.batch_show[player] = np.array(show, dtype=np.int64)
            self.batch_pack[player] = np.array(pack, dtype=np.int64)
            self.cheap_nodes[player] = np.array(cheap, dtype=np.int64)
            self.coeffs[player] = coeff_list

    # -- strategies --------------------------------------------------------

    def current_strategy(self, node: int) -> np.ndarray:
        """Regret matching+ on the clipped cumulative regrets."""
        regret = self.regret[node]
        pos = np.maximum(regret, 0.0)
        total = pos.sum(axis=-1, keepdims=True)
        safe = np.where(total > 0.0, total, 1.0)
        return np.where(total > 0.0, pos / safe, 1.0 / regret.shape[-1])

    def average_strategy(self, node: int) -> np.ndarray:
        """The equilibrium approximation: the normalised cumulative strategy.

        This, not the final iterate, is what converges.
        """
        acc = self.strat_sum[node]
        total = acc.sum(axis=-1, keepdims=True)
        safe = np.where(total > 0.0, total, 1.0)
        return np.where(total > 0.0, acc / safe, 1.0 / acc.shape[-1])

    def strategy_profile(self, average: bool = True) -> list:
        get = self.average_strategy if average else self.current_strategy
        return [
            None if self.tree.is_terminal[n] else get(n)
            for n in range(self.tree.n_nodes)
        ]

    def _current_profile_cached(self) -> list:
        """Current strategies, recomputing only nodes whose regrets changed.

        With alternating updates only one player's regrets move per traversal,
        so recomputing every node twice per iteration is wasted work. The cache
        is invalidated node-by-node in `_update_node`.
        """
        for n in self._dirty:
            self._strat_cache[n] = self.current_strategy(n)
        self._dirty.clear()
        return self._strat_cache

    # -- traversal ---------------------------------------------------------

    def _down_pass(self, strategies):
        """Propagate reach down the tree.

        Reaches are scalars for a blind player (all their actions are
        hand-independent) and vectors once a player has acted while seen.
        """
        tree = self.tree
        reach: list[list] = [[None, None] for _ in range(tree.n_nodes)]
        reach[0] = [1.0, 1.0]

        for n in range(tree.n_nodes):  # DFS pre-order: parents before children
            if tree.is_terminal[n]:
                continue
            r = reach[n]
            actor = int(tree.actor[n])
            blind = bool(tree.actor_blind[n])
            _assert_blind_reach_is_scalar(r[actor], blind, n)
            sigma = strategies[n]

            for i, child in enumerate(tree.children[n]):
                child_reach = list(r)
                if blind:
                    # Scalar times scalar: stays hand-independent.
                    child_reach[actor] = r[actor] * float(sigma[i])
                else:
                    # Scalar or vector times a per-class column: becomes a
                    # vector. A player is legitimately still scalar at the bet
                    # node of the turn on which they looked.
                    child_reach[actor] = r[actor] * sigma[:, i]
                reach[child] = child_reach
        return reach

    def _terminal_values(self, reach, player: int) -> list:
        """Value vector for `player` at every terminal, indexed by node.

        Only one vector per terminal is retained; the win/tie/lose
        decomposition is combined immediately to keep peak memory down.
        """
        tb = self.tables
        tree = self.tree
        opponent = 1 - player
        out: list = [None] * tree.n_nodes

        nC = tb.n_classes

        def gather(nodes):
            R = np.empty((nodes.size, nC), dtype=np.float64)
            for i, raw in enumerate(nodes):
                vec = reach[int(raw)][opponent]
                if not _is_vector(vec):
                    raise AssertionError(
                        f"node {int(raw)}: opponent is seen at a terminal but "
                        "their reach is not a vector"
                    )
                R[i] = vec
            return R

        # Showdowns against a seen opponent: one contiguous BLAS call yields
        # win, tie and total together.
        show = self.batch_show[player]
        if show.size:
            blocks = gather(show) @ tb.cond_stack.T      # [T, 3nC]
            win_all = blocks[:, :nC]
            tie_all = blocks[:, nC:2 * nC]
            total_all = blocks[:, 2 * nC:]
            for i, raw in enumerate(show):
                n = int(raw)
                coeff = self.coeffs[player][self._term_pos[n]]
                lose = total_all[i] - win_all[i] - tie_all[i]
                out[n] = (
                    coeff.win * win_all[i]
                    + coeff.lose * lose
                    + coeff.tie * tie_all[i]
                )
            del blocks, win_all, tie_all, total_all

        # Packs against a seen opponent: the payoff is hand-independent, so
        # only the total disjoint mass is needed.
        pack = self.batch_pack[player]
        if pack.size:
            total_all = gather(pack) @ tb.cond.T
            for i, raw in enumerate(pack):
                n = int(raw)
                out[n] = self.coeffs[player][self._term_pos[n]].const * total_all[i]
            del total_all

        # Cheap path: a blind opponent's reach is uniform across classes.
        for raw in self.cheap_nodes[player]:
            n = int(raw)
            scalar = reach[n][opponent]
            if _is_vector(scalar):
                raise AssertionError(
                    f"node {n}: opponent is blind but their reach is a vector "
                    "-- the blind information set has leaked (RULES.md 12.4)"
                )
            scalar = float(scalar)
            coeff = self.coeffs[player][self._term_pos[n]]
            if coeff.is_showdown:
                win = tb.unit_win * scalar
                tie = tb.unit_tie * scalar
                lose = scalar - win - tie
                out[n] = coeff.win * win + coeff.lose * lose + coeff.tie * tie
            else:
                out[n] = np.full(tb.n_classes, coeff.const * scalar)
        return out

    def _up_pass(self, strategies, reach, values, player: int, update: bool,
                 iteration: int, best_response: bool = False):
        """Combine values bottom-up; optionally update regrets.

        `values` must already hold the terminal value vectors. Returns the root
        value vector for `player`. Interior entries are freed as consumed, so
        peak memory stays near one level of the tree.
        """
        tb = self.tables
        tree = self.tree
        w = tb.chance_w

        for n in range(tree.n_nodes - 1, -1, -1):  # children before parents
            if tree.is_terminal[n]:
                continue

            kids = tree.children[n]
            child_vals = [values[c] for c in kids]
            actor = int(tree.actor[n])
            blind = bool(tree.actor_blind[n])

            if actor != player:
                # The opponent's strategy is already folded into their reach on
                # the way down, so branches simply add.
                values[n] = np.add.reduce(child_vals)
            else:
                stacked = np.stack(child_vals, axis=-1)  # [nC, n_actions]
                if best_response:
                    if blind:
                        # ONE information set: a single action for every class,
                        # chosen to maximise the chance-weighted aggregate.
                        # Maximising per class here would grant the responder
                        # information they do not have and overstate
                        # exploitability.
                        agg = w @ stacked
                        best = int(np.argmax(agg))
                        values[n] = stacked[:, best].copy()
                    else:
                        # Each class is its own information set, so maximise
                        # independently per class.
                        best = np.argmax(stacked, axis=-1)
                        values[n] = np.take_along_axis(
                            stacked, best[:, None], axis=-1
                        )[:, 0]
                    self._br_choice[n] = best
                else:
                    sigma = strategies[n]
                    if blind:
                        node_value = stacked @ np.asarray(sigma, dtype=np.float64)
                    else:
                        node_value = np.einsum("ca,ca->c", sigma, stacked)
                    values[n] = node_value
                    if update:
                        self._update_node(n, sigma, stacked, node_value, reach,
                                          actor, blind, iteration)
            for c in kids:
                values[c] = None
        return values[0]

    def _update_node(self, n, sigma, stacked, node_value, reach, actor, blind,
                     iteration):
        """CFR+ regret and average-strategy accumulation at one node."""
        self._dirty.add(n)
        w = self.tables.chance_w
        instant = stacked - node_value[:, None]        # [nC, n_actions]
        own_reach = reach[n][actor]

        if blind:
            # Aggregate over classes with the chance weights: the blind player
            # has one information set, so there is no per-class regret to keep.
            self.regret[n] = np.maximum(self.regret[n] + (w @ instant), 0.0)
            # pi_p(I) for a blind player is the scalar reach, since sum_c w = 1.
            self.strat_sum[n] += iteration * float(own_reach) * np.asarray(sigma)
        else:
            self.regret[n] = np.maximum(
                self.regret[n] + w[:, None] * instant, 0.0
            )
            self.strat_sum[n] += iteration * (w * own_reach)[:, None] * sigma

    # -- public API --------------------------------------------------------

    def iterate(self, n_iterations: int = 1) -> None:
        """Run `n_iterations` CFR+ iterations with alternating updates."""
        for _ in range(n_iterations):
            self.iterations += 1
            t = self.iterations
            for player in (0, 1):
                strategies = self._current_profile_cached()
                reach = self._down_pass(strategies)
                values = self._terminal_values(reach, player)
                self._up_pass(strategies, reach, values, player, update=True,
                              iteration=t)

    def game_value(self, average: bool = True, player: int = 0) -> float:
        """Expected payoff to `player` under the (average) strategy profile."""
        strategies = self.strategy_profile(average=average)
        reach = self._down_pass(strategies)
        values = self._terminal_values(reach, player)
        root = self._up_pass(strategies, reach, values, player, update=False,
                             iteration=self.iterations)
        return float(self.tables.chance_w @ root)

    def best_response_value(self, average: bool = True, player: int = 0):
        """Exact best-response value for `player`, by full traversal.

        The opponent plays their (average) strategy. The responder's blind nodes
        are constrained to a single action per public history, as RULES.md 12.4
        requires.
        """
        strategies = self.strategy_profile(average=average)
        reach = self._down_pass(strategies)
        values = self._terminal_values(reach, player)
        self._br_choice: dict[int, object] = {}
        root = self._up_pass(strategies, reach, values, player, update=False,
                             iteration=self.iterations, best_response=True)
        return float(self.tables.chance_w @ root), dict(self._br_choice)

    # -- checkpointing -----------------------------------------------------

    def snapshot(self) -> dict:
        """Serialisable checkpoint of the average strategy and regrets."""
        return {
            "iterations": self.iterations,
            "config": self.config,
            "strat_sum": [None if a is None else a.copy() for a in self.strat_sum],
            "regret": [None if a is None else a.copy() for a in self.regret],
        }

    def restore(self, snap: dict) -> None:
        if snap["config"] != self.config:
            raise ValueError(
                f"checkpoint config {snap['config'].label()} != "
                f"solver config {self.config.label()}"
            )
        self.iterations = snap["iterations"]
        self.strat_sum = [None if a is None else a.copy() for a in snap["strat_sum"]]
        self.regret = [None if a is None else a.copy() for a in snap["regret"]]

    def save(self, path) -> None:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(self.snapshot(), fh)

    def load(self, path) -> None:
        import pickle

        with open(path, "rb") as fh:
            self.restore(pickle.load(fh))


def _assert_blind_reach_is_scalar(reach, blind: bool, node: int) -> None:
    """The invariant that guards RULES.md 12.4 on every traversal.

    A blind player has taken only hand-independent actions, so their reach must
    be identical for every private class. Representing it as a scalar makes a
    per-class blind strategy unrepresentable rather than merely discouraged: if
    a blind node ever produced a per-class strategy, the reach below it would
    become a vector and this would fire.

    Only the blind direction is checked. A SEEN player's reach is legitimately
    still a scalar at the bet node of the turn on which they looked, because
    they have not yet taken any hand-dependent action.
    """
    if blind and _is_vector(reach):
        raise AssertionError(
            f"node {node}: the acting player is blind but their reach is a "
            "vector, so their strategy has become hand-dependent. This "
            "violates RULES.md 12.4 and voids the solve."
        )

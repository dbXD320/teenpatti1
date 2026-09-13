"""Natural-language templating for TeenPattiBench (Phase 3).

The template follows PokerBench (Zhuang et al., AAAI 2025, arXiv:2501.08328)
closely for methodological continuity: cards are spelled out in words, position
and betting history are given in prose, and the prompt closes with the exact
instruction "Do not explain your answer. Your optimal action is:".

Three things are adapted for Teen Patti:

1. Two players and a symmetric boot, rather than six players and blinds.
2. The blind/seen state, which has no poker analogue, plus an explicit statement
   of what the actor pays now versus what they would pay after looking.
3. A glossary in the system prompt, because models have seen very little Teen
   Patti vocabulary in pretraining and would otherwise be guessing at `chaal`.

THE CRITICAL CONSTRAINT (RULES.md 12.4). A blind player has not looked at their
cards. Their information set is the public history alone. A prompt for a blind
decision must therefore contain no card identity and no hint of hand strength.
`assert_no_private_leak` enforces this and is called on every blind prompt that
this module emits; there is no code path that skips it.

Nothing here silently repairs a spot it cannot render: every failure raises.
"""

from __future__ import annotations

import re

from game import (
    ACTION_NAMES,
    Action,
    GameConfig,
    Phase,
    State,
    amount_for,
    apply_action,
    initial_state,
    legal_actions,
)
from hands import RANK_CHARS, SUIT_CHARS, card_rank, card_suit

__all__ = [
    "SYSTEM_PROMPT",
    "PromptError",
    "PrivateLeakError",
    "spell_card",
    "spell_hand",
    "render_prompt",
    "assert_no_private_leak",
    "prices_for",
    "turn_index",
    "history_prose",
]


class PromptError(ValueError):
    """A spot that cannot be templated cleanly. Never suppressed."""


class PrivateLeakError(PromptError):
    """A blind-state prompt was found to contain private information."""


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

RANK_WORDS = {
    2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine", 10: "Ten", 11: "Jack", 12: "Queen", 13: "King",
    14: "Ace",
}
SUIT_WORDS = {0: "Clubs", 1: "Diamonds", 2: "Hearts", 3: "Spades"}

#: Single words that would betray a card or a hand category in a blind prompt.
#: Matched on word boundaries, so `Ten` does not fire on `often` and `set` does
#: not fire on `subset`. Deliberately over-broad within that: a false positive
#: costs one raised error, a false negative silently invalidates the research
#: object. The rank word `three` is why a blind prompt must not say "your three
#: cards" -- it reads as a rank and the guard cannot tell the difference.
_FORBIDDEN_WORDS = (
    tuple(w.lower() for w in RANK_WORDS.values())
    + tuple(w.lower() for w in SUIT_WORDS.values())
    + (
        "club", "diamond", "heart", "spade",
        "trail", "trio", "set", "sequence", "run", "straight",
        "colour", "color", "flush", "pair", "kicker",
        "strength", "strong", "weak", "equity", "odds", "outs",
        "holding", "bucket",
    )
)

#: Multi-word phrasings that only make sense if the hand is known. Matched as
#: plain substrings.
_FORBIDDEN_PHRASES = (
    "your cards are", "your hand is", "you hold", "you are holding",
    "you were dealt", "high card",
)

SYSTEM_PROMPT = """You are a specialist in playing heads-up (two-player) Teen Patti, the Indian three-card game. You make optimal decisions from the information available to you.

Glossary of Teen Patti terms:
- boot: the mandatory ante both players post before the deal. Here it is 1 unit each, so every hand starts with a pot of 2.
- stake: the current unit price of a betting action, quoted at the blind rate.
- blind: a player who has not looked at their own three cards. A blind player pays HALF what a seen player pays for the same action.
- seen: a player who has looked at their own cards. Looking is voluntary, may be done only at the start of your own turn, and is permanent. Once seen you pay the full rate.
- chaal: to call, matching the current stake and continuing the hand.
- raise: to increase the stake. The stake doubles.
- pack: to fold, forfeiting the pot. Costs nothing further.
- show: to pay the show fee and force an immediate showdown, revealing both hands. A blind player may demand a show against any opponent. A seen player may NOT demand a show against a blind opponent.

Hand ranking, strongest first: trail (three of a kind), pure sequence (three consecutive cards of one suit), sequence (three consecutive cards), colour (three cards of one suit), pair, high card.

Answer with exactly one action word from the list of legal actions given to you."""


# --------------------------------------------------------------------------
# Card rendering
# --------------------------------------------------------------------------

def spell_card(card: int) -> str:
    """`the Ace of Spades`. PokerBench spells cards out rather than using codes."""
    return f"the {RANK_WORDS[card_rank(card)]} of {SUIT_WORDS[card_suit(card)]}"


def spell_hand(hand) -> str:
    """Three cards in prose, strongest-first ordering left to the caller."""
    if len(hand) != 3:
        raise PromptError(f"a Teen Patti hand has three cards, got {len(hand)}")
    parts = [spell_card(c) for c in hand]
    return f"{parts[0]}, {parts[1]}, and {parts[2]}"


# --------------------------------------------------------------------------
# State description
# --------------------------------------------------------------------------

def turn_index(state: State) -> int:
    """Which of the actor's own turns this is, 1-based.

    Seat 1 acts at k = 0, 2, 4, 6 and seat 2 at k = 1, 3, 5, 7, so k // 2 + 1
    gives the actor's own turn number for either seat. Matches
    `experiments.turn_index`; duplicated rather than imported so that the
    templating layer does not depend on the experiment driver.
    """
    return state.k // 2 + 1


def prices_for(stake: int, is_seen: bool) -> dict:
    """What each betting action costs at `stake` for a player of that status.

    Mirrors `game.amount_for` exactly (RULES.md 6.4, 10.2). Kept as a pure
    function of (stake, status) so a look decision can quote both branches
    without constructing the hypothetical child state.
    """
    return {
        "chaal": 2 * stake if is_seen else stake,
        "raise": 4 * stake if is_seen else 2 * stake,
        "show": 2 * stake if is_seen else stake,
    }


def _seat(player: int) -> str:
    return f"Player {player + 1}"


def history_prose(state: State) -> list:
    """Replay the public history into one English sentence per action.

    Replays from the initial state rather than reading `state` alone, because
    the amount paid for each past action depends on the stake and status at the
    time it was taken, neither of which the final state records.
    """
    lines = []
    cur = initial_state(state.config)
    for action in state.history:
        actor = cur.to_act
        who = _seat(actor)
        if action is Action.SEE:
            lines.append(f"{who} looked at their cards and is now seen.")
        elif action is Action.STAY_BLIND:
            lines.append(f"{who} declined to look and remains blind.")
        else:
            amount = amount_for(cur, action)
            status = "seen" if cur.seen[actor] else "blind"
            if action is Action.CHAAL:
                lines.append(f"{who} ({status}) played chaal for {amount}.")
            elif action is Action.RAISE:
                lines.append(f"{who} ({status}) raised for {amount}.")
            elif action is Action.PACK:
                lines.append(f"{who} ({status}) packed.")
            elif action is Action.SHOW:
                lines.append(f"{who} ({status}) demanded a show for {amount}.")
            else:
                raise PromptError(f"cannot describe action {action!r}")
        cur = apply_action(cur, action)
    return lines


# --------------------------------------------------------------------------
# The leakage guard
# --------------------------------------------------------------------------

def assert_no_private_leak(prompt: str, *, where: str = "") -> None:
    """Fail if a blind-state prompt carries card identity or strength language.

    Three layers, because the failure this guards against is invisible
    downstream: a blind prompt that leaks produces a benchmark item whose
    correct answer was computed for an information set the model was not in.

    1. Spelled card names and bare rank/suit words.
    2. Two-character card codes such as `As` or `Td` as standalone tokens.
    3. Hand-category and strength vocabulary, plus second-person phrasings that
       only make sense if the hand is known.
    """
    low = prompt.lower()
    for phrase in _FORBIDDEN_PHRASES:
        if phrase in low:
            raise PrivateLeakError(
                f"blind prompt{' for ' + where if where else ''} contains the "
                f"forbidden phrase {phrase!r}. A blind player's information set "
                "is the public history alone (RULES.md 12.4)."
            )
    for word in _FORBIDDEN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", low):
            raise PrivateLeakError(
                f"blind prompt{' for ' + where if where else ''} contains the "
                f"forbidden word {word!r}. A blind player's information set is "
                "the public history alone (RULES.md 12.4)."
            )
    codes = {
        f"{r}{s}" for r in RANK_CHARS.values() for s in SUIT_CHARS
    }
    for raw in prompt.replace(",", " ").replace(".", " ").split():
        if raw in codes:
            raise PrivateLeakError(
                f"blind prompt{' for ' + where if where else ''} contains the "
                f"card code {raw!r}."
            )


# --------------------------------------------------------------------------
# The template
# --------------------------------------------------------------------------

def render_prompt(state: State, hand=None) -> str:
    """Render one decision into an English prompt.

    `hand` must be supplied for a seen actor and must be None for a blind one.
    Supplying a hand for a blind actor raises rather than being ignored: a
    caller that passes one has misunderstood the information partition, and
    silently dropping it would hide that.
    """
    if state.is_terminal:
        raise PromptError("terminal states are not decisions and cannot be templated")

    actor = state.to_act
    blind = not state.seen[actor]
    opponent = state.opponent(actor)
    opp_status = "seen" if state.seen[opponent] else "blind"
    legal = legal_actions(state)
    if not legal:
        raise PromptError(f"no legal actions at k={state.k} r={state.r}")

    if blind and hand is not None:
        raise PromptError(
            "a hand was supplied for a BLIND actor. A blind player has not "
            "looked; their information set contains no card information "
            "(RULES.md 12.4)."
        )
    if not blind and hand is None:
        raise PromptError("a seen actor needs their hand to be templated")

    cap_left = state.config.action_cap - state.k
    raises_left = state.config.raise_cap - state.r
    j = turn_index(state)
    blind_px = prices_for(state.stake, False)
    seen_px = prices_for(state.stake, True)

    out = []
    out.append(
        f"You are {_seat(actor)} in a heads-up game of Teen Patti. "
        f"{'You act first each round.' if actor == 0 else 'You act second each round.'}"
    )
    out.append(
        f"Both players posted a boot of {state.config.boot}. "
        f"The pot is {state.pot}. The current stake is {state.stake}."
    )

    # -- status ------------------------------------------------------------
    if blind:
        out.append(
            f"You are BLIND: you have not looked at your cards, and you do not "
            f"know what they are. Your opponent is {opp_status}."
        )
    else:
        out.append(
            f"You are SEEN: you have looked at your cards. "
            f"Your opponent is {opp_status}."
        )
        out.append(f"Your cards are {spell_hand(hand)}.")

    # -- history -----------------------------------------------------------
    lines = history_prose(state)
    if lines:
        out.append("The betting so far: " + " ".join(lines))
    else:
        out.append("No action has been taken yet. You are first to act.")

    # -- the clock ---------------------------------------------------------
    out.append(
        f"This is your turn number {j} of this hand. "
        f"{cap_left} betting action{'s' if cap_left != 1 else ''} remain before "
        f"the action cap forces a showdown, and {raises_left} "
        f"raise{'s' if raises_left != 1 else ''} remain."
    )

    # -- pricing -----------------------------------------------------------
    if state.phase is Phase.LOOK:
        # Phrasing note: this must not begin a sentence with "As", which is a
        # valid card code (Ace of Spades) and trips the leakage guard.
        show_note_blind = (
            " While blind you may demand a show."
            if _show_would_be_legal(state, viewer_seen=False)
            else ""
        )
        show_note_seen = (
            " If you look, you may still demand a show."
            if _show_would_be_legal(state, viewer_seen=True)
            else " If you look, you may no longer demand a show against a blind opponent."
        )
        out.append(
            f"You must first decide whether to look at your cards. "
            f"If you stay blind, chaal costs {blind_px['chaal']} and a raise costs "
            f"{blind_px['raise']}.{show_note_blind} "
            f"If you look, you become seen for the rest of the hand and the same "
            f"actions cost {seen_px['chaal']} and {seen_px['raise']}."
            f"{show_note_seen}"
        )
    else:
        px = blind_px if blind else seen_px
        bits = [f"chaal costs {px['chaal']}"]
        if Action.RAISE in legal:
            bits.append(f"a raise costs {px['raise']}")
        if Action.SHOW in legal:
            bits.append(f"a show costs {px['show']}")
        bits.append("packing costs nothing")
        out.append("At this stake, " + ", ".join(bits) + ".")
        if blind:
            out.append(
                f"If you had looked you would be paying {seen_px['chaal']} for "
                f"chaal instead of {px['chaal']}."
            )

    # -- the ask -----------------------------------------------------------
    names = [ACTION_NAMES[a] for a in legal]
    out.append("Your legal actions are: " + ", ".join(names) + ".")
    out.append("Do not explain your answer. Your optimal action is:")

    prompt = "\n".join(out)

    if blind:
        assert_no_private_leak(prompt, where=f"k={state.k} history={state.history}")
    return prompt


def _show_would_be_legal(state: State, viewer_seen: bool) -> bool:
    """Whether `show` would be legal for the actor at the given status.

    RULES.md 10.3: legal iff the actor is blind, or both players are seen.
    Used only to phrase the look decision, which must quote both branches.
    """
    if not viewer_seen:
        return True
    return state.seen[state.opponent(state.to_act)]


def correct_amount(state: State, action: Action) -> int:
    """The chips the labelled action costs, for the Exact Match metric."""
    return amount_for(state, action)

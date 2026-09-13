"""Tests for TeenPattiBench generation (Phase 3).

The test that matters most is `test_no_blind_prompt_contains_card_identities`.
A blind player's information set is the public history alone (RULES.md 12.4); a
blind prompt carrying card identity would be an item whose answer key was
computed for an information set the model was never in, and nothing downstream
would notice. It is checked here against the shipped files rather than against
freshly rendered prompts, so it tests what was actually written to disk.
"""

from __future__ import annotations

import json
import os

import pytest

import cfr
import generate as G
import metrics as M
import prompts as P
from game import ACTION_NAMES, amount_for
from hands import RANK_CHARS, SUIT_CHARS

EVAL = "data/teenpattibench_eval.jsonl"
TRAIN = "data/teenpattibench_train.jsonl"

pytestmark = pytest.mark.skipif(
    not os.path.exists(EVAL),
    reason="datasets not generated; run `python generate.py` first",
)


@pytest.fixture(scope="module")
def eval_items():
    return M.load_jsonl(EVAL)


@pytest.fixture(scope="module")
def train_items():
    return M.load_jsonl(TRAIN)


@pytest.fixture(scope="module")
def solver():
    s, _ = G.load_solver()
    return s


# --------------------------------------------------------------------------
# The blind information partition
# --------------------------------------------------------------------------

def _card_tokens():
    words = [w.lower() for w in P.RANK_WORDS.values()]
    suits = [w.lower() for w in P.SUIT_WORDS.values()]
    codes = {f"{r}{s}" for r in RANK_CHARS.values() for s in SUIT_CHARS}
    return words, suits, codes


def test_no_blind_prompt_contains_card_identities(eval_items, train_items):
    """No blind-state prompt may name a card, in words or in code form."""
    words, suits, codes = _card_tokens()
    checked = 0
    for item in eval_items + train_items:
        if not item["meta"]["blind"]:
            continue
        checked += 1
        prompt = item["prompt"]
        low = prompt.lower()
        for w in words:
            assert f" {w} " not in f" {low} ", (
                f"{item['id']}: blind prompt contains rank word {w!r}"
            )
        for s in suits:
            assert s not in low, (
                f"{item['id']}: blind prompt contains suit word {s!r}"
            )
        for tok in prompt.replace(",", " ").replace(".", " ").split():
            assert tok not in codes, (
                f"{item['id']}: blind prompt contains card code {tok!r}"
            )
    assert checked > 0, "no blind items were checked"


def test_blind_items_carry_no_hand_metadata(eval_items, train_items):
    for item in eval_items + train_items:
        if item["meta"]["blind"]:
            assert item["meta"]["hand"] is None, item["id"]
            assert item["meta"]["iso_class"] is None, item["id"]
            assert item["meta"]["hand_bucket"] is None, item["id"]


def test_leak_guard_catches_injected_card_identity():
    """The guard must fire on the exact failure it exists to prevent."""
    for bad in (
        "You are BLIND. You hold the Ace of Spades.",
        "Your cards are Ah 7d 2c.",
        "You are BLIND but your hand is a strong pair.",
        "You are BLIND. Kh is in your hand.",
    ):
        with pytest.raises(P.PrivateLeakError):
            P.assert_no_private_leak(bad)


def test_render_refuses_a_hand_for_a_blind_actor(solver):
    tree = solver.tree
    node = next(int(n) for n in tree.decision_ids if tree.actor_blind[n])
    with pytest.raises(P.PromptError):
        P.render_prompt(tree.states[node], hand=(0, 1, 2))


def test_every_blind_decision_node_renders_without_leaking(solver):
    """Sweep the whole tree, not just the sampled items."""
    tree = solver.tree
    n = 0
    for raw in tree.decision_ids:
        idx = int(raw)
        if not tree.actor_blind[idx]:
            continue
        P.render_prompt(tree.states[idx])  # raises internally on any leak
        n += 1
    assert n == 728, f"expected 728 blind decision nodes, rendered {n}"


# --------------------------------------------------------------------------
# Dataset integrity
# --------------------------------------------------------------------------

def test_eval_and_train_are_disjoint(eval_items, train_items):
    def keys(items):
        return {(i["meta"]["node"], i["meta"]["iso_class"]) for i in items}
    assert not (keys(eval_items) & keys(train_items))


def test_eval_ids_are_unique(eval_items, train_items):
    assert len({i["id"] for i in eval_items}) == len(eval_items)
    assert len({i["id"] for i in train_items}) == len(train_items)


def test_labels_match_the_solver_distribution(eval_items):
    """The label must be the argmax of the stored distribution, and dominant."""
    for item in eval_items:
        dist = item["solver_distribution"]
        best = max(dist, key=dist.get)
        if not item["is_mixed"]:
            assert item["correct_action"] == best, item["id"]
            assert dist[best] > G.DOMINANCE_THRESHOLD, item["id"]
        assert abs(sum(dist.values()) - 1.0) < 1e-9, item["id"]


def test_labels_match_the_live_solver(eval_items, solver):
    """Re-read the answer key for a sample of items and compare.

    Guards against the stored distribution drifting from the checkpoint, which
    a stale regeneration could otherwise hide.
    """
    tree = solver.tree
    sample = eval_items[::200]
    assert sample
    for item in sample:
        node = item["meta"]["node"]
        sigma = solver.average_strategy(node)
        row = sigma if item["meta"]["blind"] else sigma[item["meta"]["iso_class"]]
        names = [ACTION_NAMES[a] for a in tree.actions[node]]
        live = dict(zip(names, (float(x) for x in row)))
        for k, v in item["solver_distribution"].items():
            assert abs(live[k] - v) < 1e-12, f"{item['id']}: {k}"


def test_correct_action_is_legal(eval_items, solver):
    tree = solver.tree
    for item in eval_items:
        node = item["meta"]["node"]
        names = [ACTION_NAMES[a] for a in tree.actions[node]]
        assert item["legal_actions"] == names, item["id"]
        assert item["correct_action"] in names, item["id"]


def test_correct_amount_matches_the_rules(eval_items, solver):
    tree = solver.tree
    for item in eval_items[::50]:
        node = item["meta"]["node"]
        st = tree.states[node]
        action = tree.actions[node][item["legal_actions"].index(item["correct_action"])]
        assert item["correct_amount"] == amount_for(st, action), item["id"]


def test_prompt_closes_with_the_pokerbench_instruction(eval_items):
    for item in eval_items:
        assert item["prompt"].rstrip().endswith(
            "Do not explain your answer. Your optimal action is:"
        ), item["id"]


def test_conversion_items_expose_turn_and_cap(eval_items):
    """Phase 2 shows the answer depends on both, so both must be legible."""
    look = [i for i in eval_items if i["meta"]["family"] == "look"]
    assert look
    for item in look:
        assert f"turn number {item['meta']['turn_index']}" in item["prompt"], item["id"]
        assert "before the action cap" in item["prompt"], item["id"]


def test_all_four_turn_strata_are_populated(eval_items):
    turns = {i["meta"]["turn_index"] for i in eval_items
             if i["meta"]["family"] == "look"}
    assert turns == {1, 2, 3, 4}, turns


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_solver_policy_scores_100_percent(eval_items):
    preds = {
        i["id"]: {"action": i["correct_action"], "amount": i["correct_amount"],
                  "distribution": i["solver_distribution"]}
        for i in eval_items
    }
    rep = M.score(eval_items, preds)
    assert rep["overall"]["aa"] == rep["overall"]["n"]
    assert rep["overall"]["em"] == rep["overall"]["n"]
    assert rep["mixed"]["mean_tvd"] < 1e-12


def test_no_trivial_baseline_scores_well(eval_items):
    base = M.trivial_baselines(eval_items)
    for key, value in base.items():
        if key.startswith("_"):
            continue
        assert value < 0.35, f"{key} scores {value:.3f}; rebalancing failed"


def test_missing_predictions_raise(eval_items):
    preds = {i["id"]: {"action": i["correct_action"]} for i in eval_items[:-1]}
    with pytest.raises(KeyError):
        M.score(eval_items, preds)


def test_parse_action_prefers_legal_words():
    assert M.parse_action("chaal", ["pack", "chaal"]) == "chaal"
    assert M.parse_action("I would pack here.", ["pack", "chaal"]) == "pack"
    assert M.parse_action("stay-blind", ["see", "stay-blind"]) == "stay-blind"
    assert M.parse_action("nonsense", ["pack", "chaal"]) is None
    # `see` must not be matched inside `stay-blind`
    assert M.parse_action("stay-blind", ["see"]) is None

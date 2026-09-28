"""Build an order-randomised twin of the eval set, to break the menu-position confound.

THE PROBLEM
-----------
Every prompt in `teenpattibench_eval.jsonl` lists the legal actions in the same
order, because that is simply the order `game.legal_actions()` emits them:

    pack, chaal, raise, show          (all 1,838 betting questions)
    see, stay-blind                   (all 162 look questions)

So `pack` is at index 0 in *every* betting prompt and `see` is at index 0 in
*every* look prompt. Menu position and action identity are perfectly confounded,
and a model that merely favours an early slot cannot be told apart from one that
has a view about folding or about looking at its cards.

This is not hypothetical. In the v1 runs Llama-3.2-3B selected index 0 or 1 and
never index 2 or 3 across all 1,838 betting questions -- which by itself accounts
for its zero raises and zero shows, since those two actions never appear earlier
than index 2. It also answered "see" on 99.4% of look questions, where "see" is
index 0, so its contribution to the headline conversion finding is unsafe as it
stands.

THE CONTROL
-----------
This script emits a TWIN of the eval set: identical ids, identical correct
answers, identical solver distributions, identical metadata -- with only the
*presentation order* of the actions permuted, consistently through both the
pricing sentence and the closing "Your legal actions are:" line.

Run the same models on the twin and compare item by item:

  * scores unchanged  -> presentation order is not driving the result, and the
                         v1 numbers can be reported as they stand;
  * scores move       -> the difference is exactly the size of the position
                         artefact, and v1 must be reported with that correction.

Because ids match, the comparison is paired: every question has a before and an
after, so the effect can be measured per item, not just in aggregate.

WHAT IS DELIBERATELY NOT CHANGED
--------------------------------
The answer key. `correct_action`, `correct_amount`, `solver_distribution`,
`legal_actions` (as a set), node, class and every other metadata field are copied
across unchanged, and asserted equal at the end. Presentation order is the single
manipulated variable. Nothing here touches the solver, the abstraction, or
RULES.md.

BALANCE
-------
Permutations are assigned so that each action lands in each position as evenly as
the item count allows, per menu shape, rather than by independent random draws --
which would leave a residual imbalance. The achieved balance is printed and
asserted.

Usage:
    python src/make_eval_shuffled.py                 # writes the twin
    python src/make_eval_shuffled.py --dry-run       # report balance, write nothing
    python src/make_eval_shuffled.py --seed 7        # a different permutation draw
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cfr
import prompts as P
from game import ACTION_NAMES, DEFAULT_CONFIG, Action
from hands import make_card

IN_PATH = "data/teenpattibench_eval.jsonl"
OUT_PATH = "data/teenpattibench_eval_shuffled.jsonl"

#: Distinct from generate.py's SEED: this is a presentation draw, not a sample.
SEED = 20260914

_NAME_TO_ACTION = {name: action for action, name in ACTION_NAMES.items()}
_RANK = {word: rank for rank, word in P.RANK_WORDS.items()}
_SUIT = {word: suit for suit, word in P.SUIT_WORDS.items()}


def _bag(text: str):
    """Multiset of words, punctuation stripped, for the reordering check."""
    return collections.Counter(re.findall(r"[A-Za-z0-9-]+", text))


def unspell_hand(text: str):
    """Recover the card ints from `the Ace of Spades the Two of Clubs ...`.

    The eval item stores the hand only as English. Parsing it back is what lets
    the twin be rendered from the engine rather than by editing prompt strings.
    """
    tokens = text.split()
    if len(tokens) % 4:
        raise ValueError(f"cannot parse hand {text!r}")
    cards = []
    for i in range(0, len(tokens), 4):
        if tokens[i] != "the" or tokens[i + 2] != "of":
            raise ValueError(f"cannot parse hand {text!r}")
        cards.append(make_card(_RANK[tokens[i + 1]], _SUIT[tokens[i + 3]]))
    return tuple(sorted(cards))


def assign_permutations(items, rng):
    """Map item id -> action permutation, balanced within each menu shape.

    For each distinct legal-action set, every permutation of it is cycled through
    in turn over a shuffled ordering of the items holding that set. With 947
    four-action questions and 24 permutations that is ~39 uses of each, so no
    action is systematically favoured by position.
    """
    by_shape = collections.defaultdict(list)
    for item in items:
        by_shape[tuple(item["legal_actions"])].append(item["id"])

    out = {}
    for shape in sorted(by_shape):
        actions = [_NAME_TO_ACTION[n] for n in shape]
        perms = sorted(itertools.permutations(actions))
        rng.shuffle(perms)
        ids = sorted(by_shape[shape])
        rng.shuffle(ids)
        for k, item_id in enumerate(ids):
            out[item_id] = perms[k % len(perms)]
    return out


def position_table(items, order_of=None, shape=None):
    """action -> {position: count}, optionally restricted to one menu shape.

    Balance is only meaningful WITHIN a menu shape. A two-action menu has no
    position 2 to fill, so pooling shapes together makes an even assignment look
    uneven: position 3 is reachable only from the 947 four-action questions.
    """
    table = collections.defaultdict(collections.Counter)
    for item in items:
        if shape is not None and tuple(item["legal_actions"]) != shape:
            continue
        names = (
            list(item["legal_actions"]) if order_of is None
            else [ACTION_NAMES[a] for a in order_of[item["id"]]]
        )
        for pos, name in enumerate(names):
            table[name][pos] += 1
    return table


def format_positions(table, width: int = 4) -> str:
    header = "    action     " + "".join(f"{'pos' + str(p):>8}" for p in range(width))
    lines = [header + f"{'total':>10}"]
    for name in sorted(table):
        row = table[name]
        cells = "".join(f"{row.get(p, 0):>8,}" for p in range(width))
        lines.append(f"    {name:<11}{cells}{sum(row.values()):>10,}")
    return "\n".join(lines)


def check_balance(items, order_of) -> list:
    """Assert evenness within each menu shape. Returns per-shape report rows."""
    shapes = sorted({tuple(i["legal_actions"]) for i in items})
    rows = []
    for shape in shapes:
        n = sum(1 for i in items if tuple(i["legal_actions"]) == shape)
        table = position_table(items, order_of, shape=shape)
        # With n items over len(shape)! permutations, each action should land in
        # each of the len(shape) positions about n/len(shape) times. Allow one
        # incomplete cycle of slack.
        expected = n / len(shape)
        slack = max(1.0, len(shape))
        worst = 0.0
        for name in shape:
            for pos in range(len(shape)):
                worst = max(worst, abs(table[name][pos] - expected))
        rows.append((shape, n, expected, worst, worst <= slack, table))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Emit an order-randomised twin of the eval set (answers unchanged)."
    )
    ap.add_argument("--in", dest="src", default=IN_PATH)
    ap.add_argument("--out", dest="dst", default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="report balance and verify, but write nothing")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.src, encoding="utf-8") if l.strip()]
    print("=" * 78)
    print("Order-randomised twin of the eval set")
    print(f"  source : {args.src}  ({len(items):,} items)")
    print(f"  output : {args.dst if not args.dry_run else '(dry run)'}")
    print(f"  seed   : {args.seed}")
    print("=" * 78)

    print("\nBEFORE -- position of each action in the v1 prompts")
    print(format_positions(position_table(items)))
    print("\n  Every action sits in exactly one position, always. Position and identity")
    print("  are perfectly confounded; this is the defect being corrected.")

    rng = random.Random(args.seed)
    order_of = assign_permutations(items, rng)

    print("\nAFTER -- position of each action in the twin, per menu shape")
    failed = []
    for shape, n, expected, worst, ok, table in check_balance(items, order_of):
        print(f"\n  menu {list(shape)}   n={n:,}   expected {expected:.1f} per cell")
        print(format_positions(table, width=len(shape)))
        print(f"    worst deviation from even: {worst:.1f}  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(shape)
    if failed:
        raise SystemExit(
            f"position balance failed for menu shape(s) {failed}. "
            "Permutation assignment is not even; do not ship this twin."
        )
    print("\n  balance check: PASS in every menu shape "
          "(no action favours any position)")

    # -- re-render --------------------------------------------------------
    tree = cfr.build_public_tree(DEFAULT_CONFIG)
    out_items, n_blind = [], 0
    for item in items:
        meta = item["meta"]
        state = tree.states[meta["node"]]
        hand = None if meta["blind"] else unspell_hand(meta["hand"])
        order = order_of[item["id"]]

        twin = json.loads(json.dumps(item))          # deep copy, answers included
        twin["prompt"] = P.render_prompt(state, hand=hand, action_order=order)
        twin["legal_actions"] = [ACTION_NAMES[a] for a in order]
        twin["meta"]["action_order"] = [ACTION_NAMES[a] for a in order]
        twin["meta"]["v1_action_order"] = list(item["legal_actions"])
        out_items.append(twin)
        if meta["blind"]:
            P.assert_no_private_leak(twin["prompt"], where=twin["id"])
            n_blind += 1

    # -- verify the answer key is untouched --------------------------------
    by_id = {i["id"]: i for i in items}
    changed_prompts = 0
    for twin in out_items:
        src = by_id[twin["id"]]
        for field in ("correct_action", "correct_amount", "solver_distribution",
                      "is_mixed"):
            if twin[field] != src[field]:
                raise SystemExit(f"{twin['id']}: {field} changed -- answer key corrupted")
        if sorted(twin["legal_actions"]) != sorted(src["legal_actions"]):
            raise SystemExit(f"{twin['id']}: legal action SET changed")
        if twin["correct_action"] not in twin["legal_actions"]:
            raise SystemExit(f"{twin['id']}: correct action missing from the menu")
        for key in src["meta"]:
            if twin["meta"][key] != src["meta"][key]:
                raise SystemExit(f"{twin['id']}: meta.{key} changed")
        if twin["prompt"] != src["prompt"]:
            changed_prompts += 1
        # The twin must be a pure reordering: same words, same numbers, only
        # their sequence changed. Compared with punctuation stripped, because
        # commas and the full stop attach to whichever clause ends up last.
        if _bag(twin["prompt"]) != _bag(src["prompt"]):
            raise SystemExit(
                f"{twin['id']}: the twin is not a reordering of the same words -- "
                "something other than presentation order changed"
            )

    print(f"\n  answer key identical on all {len(out_items):,} items: PASS")
    print(f"  prompts that actually differ from v1: {changed_prompts:,} "
          f"({100 * changed_prompts / len(out_items):.1f}%)")
    print(f"  blind prompts re-checked for private leakage: {n_blind:,}: PASS")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    with open(args.dst, "w", encoding="utf-8") as fh:
        for twin in out_items:
            fh.write(json.dumps(twin, ensure_ascii=False) + "\n")

    size = os.path.getsize(args.dst) / 1e6
    print(f"\nwrote {args.dst}  ({len(out_items):,} items, {size:.1f} MB)")
    print("\nNext: run the same models on this file and score both with")
    print("  python src/metrics.py --eval <file> --predictions <preds> --json <report>")
    print("Ids match v1, so the comparison is paired item by item.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

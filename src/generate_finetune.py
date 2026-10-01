"""Generate the TeenPattiBench FINE-TUNING dataset (100k items).

This is the training set designed from the Phase 4 error analyses, not the
benchmark's generic training split. Rationale for every choice is in
docs/FINETUNE_STRATEGY.md; the short version is here.

WHAT IS DIFFERENT FROM teenpattibench_train_100k.jsonl
------------------------------------------------------
1. Action order is randomised, balanced per menu shape. In the v1 data `pack`
   is index 0 in every betting prompt and `raise`/`show` are never below index 2,
   so a model can learn the slot instead of the game. Qwen3-8B's `show` recall
   swings 49.6% -> 1.0% on that alone.

2. The seen portion is built from four components instead of one stratified draw:

     B  contrast   same-node groups: the same betting line with 2 confident hands
                   per correct action at that node, one presentation order per
                   group, written contiguously. The hand is the only thing that
                   varies inside a group -- exactly the input every evaluated
                   model ignores (Mistral: one identical answer in 74/74 nodes).
     C  boundary   targeted draws for the specific failures:
                     F2  raise legal but not correct, equity 0.50-0.80  (Mistral)
                     F1  pack correct on a [pack, chaal] menu     (all four, 39/39)
                     F3  show correct                            (all four)
                     F4  strong hand (>=0.80), raise/show correct   (Llama)
     D  balancer   a stratified draw per label that fills each betting label to
                   an equal share, so no single-word policy scores well.

   C changes the composition *within* each label toward its decision boundary;
   D keeps the label totals balanced. The failures are boundary failures, not
   label-frequency failures, so this is the split that targets them.

3. Low-confidence labels are down-weighted, not dropped. 30.6% of graded eval
   items have label_prob <= 0.60, so dropping them would remove training signal
   exactly where a third of the benchmark is scored. Instead every item carries
   `weight` = label_prob normalised to mean 1.0 *within its label*, so confident
   items count more without shifting the label balance. (Plain label_prob would
   silently down-weight `show`, whose labels are the least confident, and `show`
   is the label models already fail most.)

4. Mixed (non-dominant) items are excluded. Their label is the argmax of a
   genuinely mixed strategy; `solver_distribution` stays on every item for a
   later soft-label stage.

5. The scarce blind pool (128 distinct infosets, 0 node overlap with eval by
   construction) is repeated 30x with a different action order each time, which
   makes the repeats less identical than v1's byte-identical 20x.

WHAT IS UNCHANGED
-----------------
The answer key (1,000-iteration checkpoint, not re-solved), the eval set (read,
never written, and gated on exact RNG reproduction), the templating, the leakage
guard, and disjointness from eval.

Usage:
    python src/generate_finetune.py              # writes the dataset
    python src/generate_finetune.py --dry-run    # plan, gates and composition only
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate as G
from game import ACTION_NAMES
from generate_train_100k import (
    Progress,
    gate_rng_replay,
    load_eval_keys,
    stamp,
    template_with_progress,
    write_jsonl_with_progress,
)

OUT_PATH = "data/teenpattibench_finetune.jsonl"
STATS_PATH = "data/teenpattibench_finetune_stats.json"

#: Independent of generate.SEED: this stream selects and orders training items
#: only. The eval-reproduction gate still replays generate.SEED separately.
SEED = 20260928

TOTAL = 100_000
BLIND_REPEAT = 30

#: Component sizes. Chosen against measured pool sizes (docs/FINETUNE_STRATEGY.md
#: section 2); every one is well inside what its pool can supply.
CONTRAST_TARGET = 24_000
CONTRAST_PER_ACTION = 2          # hands per correct action per group
CONTRAST_CONFIDENT = 0.80        # preferred label_prob for contrast hands
CONTRAST_FLOOR = 0.60            # never build a contrast from coin-flip labels
F2_N = 12_000                    # raise legal, not correct, equity 0.50-0.80
F1_N = 6_000                     # pack correct on a [pack, chaal] menu
F3_N = 8_000                     # show correct
F4_N = 6_000                     # strong hand, raise or show correct

#: raise's share of the whole dataset. 0.21 puts P(raise correct | raise legal)
#: at 32.9% -- against 32.1% on eval's seen items and 34.0% in the v1 training
#: set -- while the worst single-word baseline stays at 25.64%, below eval's own
#: 25.94%. An equal four-way split gives 37.5%, the wrong direction for a model
#: (Mistral) whose failure is raising whenever raise is legal. Swept in
#: docs/FINETUNE_STRATEGY.md section 4.
RAISE_SHARE = 0.21

BET = G.BET_LABELS
NAME_TO_ACTION = {name: action for action, name in ACTION_NAMES.items()}


# --------------------------------------------------------------------------
# Selection helpers
# --------------------------------------------------------------------------

def key(r):
    return (r["node"], r["iso_class"])


def draw(pool, n, taken, rng, label=""):
    """Take `n` records from `pool` not already taken, uniformly at random."""
    cands = [r for r in pool if key(r) not in taken]
    if len(cands) < n:
        raise SystemExit(
            f"component {label!r} needs {n:,} but its pool has only "
            f"{len(cands):,} untaken records. Lower its quota."
        )
    rng.shuffle(cands)
    got = cands[:n]
    taken.update(key(r) for r in got)
    return got


def build_contrast_groups(seen_pool, taken, rng, target):
    """Same-node groups spanning every correct action present at the node.

    A group holds CONTRAST_PER_ACTION hands per action, all from one public
    node, so the betting history is fixed and only the hand varies. Hands are
    confident (>= CONTRAST_CONFIDENT) where the node has enough, never below
    CONTRAST_FLOOR -- a contrast between two coin-flip labels teaches false
    precision (32% of adjacent label flips in the v1 data are exactly that).

    Nodes are visited round-robin so a few large nodes cannot dominate.
    """
    by_node = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in seen_pool:
        if r["label_prob"] > CONTRAST_FLOOR and key(r) not in taken:
            by_node[r["node"]][r["label"]].append(r)

    queues = {}
    for node, by_label in by_node.items():
        if len(by_label) < 2:
            continue
        q = {}
        for label, rows in by_label.items():
            conf = [r for r in rows if r["label_prob"] >= CONTRAST_CONFIDENT]
            rest = [r for r in rows if r["label_prob"] < CONTRAST_CONFIDENT]
            rng.shuffle(conf)
            rng.shuffle(rest)
            q[label] = conf + rest          # confident first
        queues[node] = q

    groups, total = [], 0
    nodes = sorted(queues)
    rng.shuffle(nodes)
    while total < target and nodes:
        still = []
        for node in nodes:
            q = queues[node]
            live = [lab for lab, rows in q.items() if len(rows) >= CONTRAST_PER_ACTION]
            if len(live) < 2:
                continue                    # no contrast left at this node
            group = []
            for lab in sorted(live):
                for _ in range(CONTRAST_PER_ACTION):
                    group.append(q[lab].pop(0))
            groups.append((node, group))
            total += len(group)
            taken.update(key(r) for r in group)
            still.append(node)
            if total >= target:
                break
        nodes = still
    return groups, total


def balanced_orders(units, rng):
    """unit id -> action permutation, cycled evenly within each menu shape."""
    by_shape = collections.defaultdict(list)
    for uid, shape in units:
        by_shape[shape].append(uid)
    out = {}
    for shape in sorted(by_shape):
        perms = sorted(itertools.permutations(NAME_TO_ACTION[n] for n in shape))
        rng.shuffle(perms)
        ids = list(by_shape[shape])
        rng.shuffle(ids)
        for k, uid in enumerate(ids):
            out[uid] = perms[k % len(perms)]
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--stats", default=STATS_PATH)
    ap.add_argument("--eval", default=G.EVAL_PATH)
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--blind-repeat", type=int, default=BLIND_REPEAT)
    ap.add_argument("--contrast", type=int, default=CONTRAST_TARGET)
    ap.add_argument("--f1", type=int, default=F1_N, help="pack on [pack, chaal]")
    ap.add_argument("--f2", type=int, default=F2_N, help="raise legal, not correct")
    ap.add_argument("--f3", type=int, default=F3_N, help="show correct")
    ap.add_argument("--f4", type=int, default=F4_N, help="strong hand, raise/show")
    ap.add_argument("--raise-share", type=float, default=RAISE_SHARE,
                    help="raise's share of the WHOLE dataset; the other three betting "
                         "labels split the rest equally. Default: equal four-way split.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("TeenPattiBench -- fine-tuning dataset")
    print(f"  target {args.total:,} items | seed {SEED} | answer key {G.CHECKPOINT} (not re-solved)")
    print("=" * 78)

    # -- answer key and the full information-set pool ----------------------
    solver, tables = G.load_solver()
    abst, bucket_of_class, hands_by_class = G.class_tables(tables)
    equity = abst.iso_equity
    look, blind_bet, seen_bet = G.enumerate_infosets(solver, bucket_of_class)
    stamp(t0, f"pool: {len(look)} look, {len(blind_bet)} blind-bet, {len(seen_bet):,} seen")

    # -- eval is the authority; reproduce its split exactly or stop --------
    eval_items, eval_keys, eval_blind_nodes, eval_seen_keys, _ = load_eval_keys(args.eval)
    rep = random.Random(G.SEED)
    look_ev, look_tr = G.split_blind_pool(look, rep)
    bb_ev, bb_tr = G.split_blind_pool(blind_bet, rep)
    n_seen_eval = len(eval_items) - len(look_ev) - len(bb_ev) - G.EVAL_MIXED_N
    seen_ev = G.stratified_seen_sample(seen_bet, n_seen_eval // len(BET), rep)
    gate_rng_replay(look_ev + bb_ev, seen_ev, eval_blind_nodes, eval_seen_keys)
    stamp(t0, "GATE PASS: eval split reproduced exactly; eval is read, never written")

    rng = random.Random(SEED)
    taken = set(eval_keys)                      # nothing from eval, ever

    # Dominant seen records only; mixed are excluded from SFT entirely.
    seen_pool = [r for r in seen_bet if r["dominant"] and key(r) not in eval_keys]
    for r in seen_pool:
        r["equity"] = float(equity[r["iso_class"]])

    # -- A: blind ------------------------------------------------------------
    blind_units = []
    for r in look_tr + bb_tr:
        perms = sorted(itertools.permutations(NAME_TO_ACTION[n] for n in r["actions"]))
        rng.shuffle(perms)
        for k in range(args.blind_repeat):
            c = dict(r)
            c["repeat_index"] = k
            c["action_order"] = perms[k % len(perms)]
            c["extra_meta"] = {"component": "A_blind", "group_id": None}
            blind_units.append([c])
    n_blind = sum(len(u) for u in blind_units)
    n_look = len(look_tr) * args.blind_repeat
    stamp(t0, f"A blind      {len(look_tr) + len(bb_tr)} distinct x {args.blind_repeat} "
              f"= {n_blind:,}  (look {n_look:,})")

    # -- B: same-node contrast groups ----------------------------------------
    groups, n_contrast = build_contrast_groups(seen_pool, taken, rng, args.contrast)
    stamp(t0, f"B contrast   {len(groups):,} groups over "
              f"{len({n for n, _ in groups}):,} nodes = {n_contrast:,}")

    # -- C: boundary components ----------------------------------------------
    comps = {
        "C_F3_show": draw([r for r in seen_pool if r["label"] == "show"],
                          args.f3, taken, rng, "F3 show"),
        "C_F1_pack_2menu": draw([r for r in seen_pool if r["label"] == "pack"
                                 and len(r["actions"]) == 2],
                                args.f1, taken, rng, "F1 pack on [pack,chaal]"),
        "C_F4_strong": draw([r for r in seen_pool if r["equity"] >= 0.80
                             and r["label"] in ("raise", "show")],
                            args.f4, taken, rng, "F4 strong raise/show"),
        "C_F2_raise_not_right": draw([r for r in seen_pool if "raise" in r["actions"]
                                      and r["label"] != "raise"
                                      and 0.50 <= r["equity"] < 0.80],
                                     args.f2, taken, rng, "F2 raise legal, not right"),
    }
    for name, rows in comps.items():
        stamp(t0, f"{name:<22} {len(rows):>6,}  {dict(collections.Counter(r['label'] for r in rows))}")

    # -- D: balancer ---------------------------------------------------------
    # Betting labels are balanced over every betting item, blind_bet included;
    # see / stay-blind are the look family and are not part of the balance.
    have = collections.Counter()
    for u in blind_units:
        if u[0]["family"] == "blind_bet":
            have[u[0]["label"]] += 1
    for _, g in groups:
        have.update(r["label"] for r in g)
    for rows in comps.values():
        have.update(r["label"] for r in rows)

    betting_total = args.total - n_look
    if args.raise_share is None:
        base, extra = divmod(betting_total, len(BET))
        target = {lab: base + (1 if i < extra else 0) for i, lab in enumerate(BET)}
    else:
        # Equal four-way balance forces raise to ~25%, which pushes
        # P(raise correct | raise legal) above eval's 32.1% once F1's raise-illegal
        # [pack, chaal] items are in -- the wrong direction for a model whose
        # failure is over-raising. Lowering raise's share corrects the conditional
        # while keeping every single-word baseline low.
        n_raise = round(args.raise_share * args.total)
        rest = [lab for lab in BET if lab != "raise"]
        base, extra = divmod(betting_total - n_raise, len(rest))
        target = {lab: base + (1 if i < extra else 0) for i, lab in enumerate(rest)}
        target["raise"] = n_raise
    over = {lab: have[lab] - target[lab] for lab in BET if have[lab] > target[lab]}
    if over:
        raise SystemExit(f"components already exceed the balanced target for {over}; "
                         "lower the C/B quotas")
    balancer = []
    for lab in BET:
        need = target[lab] - have[lab]
        got = G.stratified_seen_sample([r for r in seen_pool if r["label"] == lab],
                                       need, rng, exclude=frozenset(taken))
        if len(got) < need:
            raise SystemExit(f"balancer: {lab} pool exhausted ({len(got):,} < {need:,})")
        for r in got:
            taken.add(key(r))
        balancer.extend(got)
    stamp(t0, f"D balancer   {len(balancer):,}  "
              f"{dict(collections.Counter(r['label'] for r in balancer))}")

    # -- assemble units, assign presentation orders --------------------------
    # Rows inside a contrast group were built label by label in sorted order
    # (chaal, chaal, pack, pack, ...). Shuffle them, so row order carries no label
    # information. A separate RNG, so selection, presentation orders and hands --
    # everything else in the dataset -- are unchanged.
    group_rng = random.Random(SEED + 2)
    for _, g in groups:
        group_rng.shuffle(g)

    units, shapes = [], []
    for node, g in groups:
        uid = f"g{len(units)}"
        for r in g:
            r["extra_meta"] = {"component": "B_contrast", "group_id": uid}
        units.append(g)
        shapes.append((uid, tuple(g[0]["actions"])))
    for name, rows in list(comps.items()) + [("D_balance", balancer)]:
        for r in rows:
            uid = f"s{len(units)}"
            r["extra_meta"] = {"component": name, "group_id": None}
            units.append([r])
            shapes.append((uid, tuple(r["actions"])))
    orders = balanced_orders(shapes, rng)
    for (uid, _), u in zip(shapes, units):
        for r in u:
            r["action_order"] = orders[uid]     # one order per contrast group
    units.extend(blind_units)

    # -- weights: label_prob normalised to mean 1.0 within each label --------
    flat = [r for u in units for r in u]
    mean_p = {lab: statistics.mean(r["label_prob"] for r in flat if r["label"] == lab)
              for lab in {r["label"] for r in flat}}
    for r in flat:
        r["weight"] = round(r["label_prob"] / mean_p[r["label"]], 4)

    if len(flat) != args.total:
        raise SystemExit(f"assembled {len(flat):,} items, expected {args.total:,}")
    stamp(t0, f"assembled {len(flat):,} items in {len(units):,} units")

    if args.dry_run:
        _report(flat, None, eval_items, args, mean_p, t0, dry=True)
        return 0

    # -- template in unit order so contrast groups stay contiguous -----------
    rng.shuffle(units)
    records = [r for u in units for r in u]
    print()
    items = template_with_progress(records, solver, hands_by_class,
                                   random.Random(SEED + 1), "ft", t0)
    for k, it in enumerate(items):
        it["id"] = f"ft-{k:06d}"

    # -- verify --------------------------------------------------------------
    G.assert_disjoint(eval_items, items)
    n_blind_chk = G.assert_no_blind_leaks(items)
    if len({i["id"] for i in items}) != len(items):
        raise RuntimeError("duplicate ids")
    for i in items:
        if sorted(i["legal_actions"]) != sorted(i["solver_distribution"]):
            raise RuntimeError(f"{i['id']}: displayed menu is not the legal set")
        if i["correct_action"] not in i["legal_actions"]:
            raise RuntimeError(f"{i['id']}: correct action not on the menu")
    stamp(t0, f"GATE PASS: disjoint from eval, {n_blind_chk:,} blind prompts leak-checked, "
              "every menu a permutation of its legal set")

    print()
    write_jsonl_with_progress(args.out, items, t0)
    _report(flat, items, eval_items, args, mean_p, t0, dry=False)
    return 0


def _report(flat, items, eval_items, args, mean_p, t0, dry):
    comp = collections.Counter(r["extra_meta"]["component"] for r in flat)
    labels = collections.Counter(r["label"] for r in flat)
    n = len(flat)

    ra = [r for r in flat if "raise" in r["actions"]]
    raise_right = sum(1 for r in ra if r["label"] == "raise") / len(ra)
    groups = collections.defaultdict(list)
    for r in flat:
        if r["extra_meta"]["group_id"]:
            groups[r["extra_meta"]["group_id"]].append(r)
    gsize = collections.Counter(len(g) for g in groups.values())
    bad_groups = sum(1 for g in groups.values()
                     if len({r["node"] for r in g}) != 1
                     or len({r["label"] for r in g}) < 2
                     or len({tuple(r["action_order"]) for r in g}) != 1)

    base = {f"always-{a}": sum(1 for r in flat if r["label"] == a) / n
            for a in ("pack", "chaal", "raise", "show", "see", "stay-blind")}

    print("\n=== composition ===")
    for c in sorted(comp):
        print(f"  {c:<24} {comp[c]:>7,}  {100 * comp[c] / n:5.2f}%")
    print("\n=== label mix ===")
    for a in ("pack", "chaal", "raise", "show", "see", "stay-blind"):
        print(f"  {a:<11} {labels[a]:>7,}  {100 * labels[a] / n:5.2f}%   "
              f"mean label_prob {mean_p.get(a, 0):.3f}")
    print(f"\n  raise-legal items where raise is correct: {100 * raise_right:.1f}%  "
          f"(eval: 31.5%, v1 train: 34.0%)")
    print(f"  contrast groups: {len(groups):,}  sizes {dict(sorted(gsize.items()))}  "
          f"malformed {bad_groups}")
    print("\n=== trivial baselines (must stay low) ===")
    for k, v in sorted(base.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {100 * v:5.2f}%")

    stats = {
        "generator": os.path.basename(__file__), "seed": SEED, "total": n,
        "blind_repeat": args.blind_repeat,
        "components": dict(comp), "labels": dict(labels),
        "raise_correct_when_legal": raise_right,
        "contrast_groups": len(groups), "contrast_group_sizes": dict(gsize),
        "mean_label_prob_by_label": mean_p,
        "trivial_baselines": base,
        "quotas": {"contrast": args.contrast, "F1": args.f1, "F2": args.f2,
                   "F3": args.f3, "F4": args.f4},
        "mixed_items_included": 0,
    }
    if not dry:
        with open(args.stats, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1)
        size = os.path.getsize(args.out) / 1e6
        print(f"\nDONE in {Progress._hms(time.time() - t0)}: {args.out} "
              f"({len(items):,} items, {size:,.0f} MB) + {args.stats}")
    else:
        print("\n--dry-run: nothing written")


if __name__ == "__main__":
    sys.exit(main())

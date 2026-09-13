"""TeenPattiBench dataset generation (Phase 3).

Pipeline follows PokerBench (Zhuang et al., AAAI 2025, arXiv:2501.08328):
solver as answer key -> prune the tree -> stratify the state space and sample
evenly across strata -> keep only unambiguous (dominant-action) spots ->
rebalance labels so no trivial policy scores well -> template into English ->
split into a small eval set and a large training set.

Four things differ from PokerBench, all forced by the game rather than chosen:

1. The betting tree is already finite and small (5,929 public nodes), because
   RULES.md 9.2 caps it. There is nothing to prune; we enumerate.

2. A *seen* item is an (public node, suit-isomorphic class) pair -- 913,825 of
   them. A *blind* item is a public node alone, because a blind player's
   information set carries no card information at all (RULES.md 12.4). There are
   only 486 reachable blind information sets in the entire game.

3. Because of (2) the blind subset is not sampled, it is ENUMERATED. Every
   reachable, dominant blind decision in Teen Patti is either an eval item or a
   training item. That also means the 30-40% eval share the Phase 3 brief asked
   for is not reachable -- see `report_blind_ceiling` and PHASE3.md.

4. Label rebalancing has to run per family, because the look decision
   (see / stay-blind) and the betting decision (pack / chaal / raise / show)
   have disjoint label spaces.

Usage:
    python generate.py               # writes eval + train JSONL and stats
    python generate.py --dry-run     # inventory and composition only
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time

import numpy as np

import cfr
import prompts as P
from abstraction import build_abstraction
from game import ACTION_NAMES, DEFAULT_CONFIG, Phase
from hands import HANDS

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CHECKPOINT = "data/solution_default.pkl"
EVAL_PATH = "data/teenpattibench_eval.jsonl"
TRAIN_PATH = "data/teenpattibench_train.jsonl"
STATS_PATH = "data/teenpattibench_stats.json"
SYSTEM_PATH = "data/teenpattibench_system_prompt.txt"

SEED = 20260912

#: PokerBench keeps "action lines that choose one dominant action with greater
#: than 50% probability". Same threshold here.
DOMINANCE_THRESHOLD = 0.5

#: Phase 2 used this floor for its reach-weighted dominance analysis. Below it
#: the average strategy is not meaningfully trained -- CFR+ has barely visited
#: the node -- so the label would be noise rather than an answer.
REACH_FLOOR = 1e-12

EVAL_N = 2000
TRAIN_N = 50_000

#: Share of the (tiny, finite) blind information-set pool given to eval rather
#: than training. Eval gets the majority because the per-turn conversion
#: breakdown is the paper's primary diagnostic and eval items must be distinct
#: to carry any information; training items may legitimately repeat.
BLIND_EVAL_SHARE = 0.70

#: Retained mixed (non-dominant) seen decisions, scored by distribution distance
#: rather than accuracy. See the module docstring of metrics.py.
EVAL_MIXED_N = 200
TRAIN_MIXED_N = 2000

#: How many times each training blind information set is emitted. The pool left
#: after eval takes its share is ~130 distinct sets; without repetition the
#: research object would be a rounding error in a 50,000-item training set.
#: Repeats are recorded per item (`meta.repeat_index`) so the distinct count
#: stays auditable.
TRAIN_BLIND_REPEAT = 20

LOOK_LABELS = ("see", "stay-blind")
BET_LABELS = ("pack", "chaal", "raise", "show")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_solver():
    """Load the Phase 2 answer key and re-verify the two gate invariants."""
    tables = cfr.build_private_tables()
    solver = cfr.CFRPlusSolver(DEFAULT_CONFIG, tables=tables)
    solver.load(CHECKPOINT)
    if solver.iterations != 1000:
        raise RuntimeError(
            f"checkpoint has {solver.iterations} iterations, expected the "
            "1,000-iteration Phase 2 solve"
        )
    return solver, tables


def class_tables(tables):
    """Per-class bucket index and the hands belonging to each class."""
    abst = build_abstraction()
    if not np.array_equal(tables.class_of_hand, abst.iso_class_of_hand):
        raise RuntimeError(
            "the solver's private-class indexing does not match the "
            "abstraction's. Prompts would show cards from the wrong class."
        )
    n_classes = tables.n_classes
    bucket_of_class = np.full(n_classes, -1, dtype=np.int32)
    hands_by_class = [[] for _ in range(n_classes)]
    for h in range(len(tables.class_of_hand)):
        c = int(tables.class_of_hand[h])
        hands_by_class[c].append(h)
        b = int(abst.bucket_of_hand[h])
        if bucket_of_class[c] == -1:
            bucket_of_class[c] = b
        elif bucket_of_class[c] != b:
            raise RuntimeError(
                f"class {c} straddles buckets {bucket_of_class[c]} and {b}; "
                "Phase 1 asserts this cannot happen"
            )
    return abst, bucket_of_class, hands_by_class


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------

def enumerate_infosets(solver, bucket_of_class):
    """Every reachable information set, with its solver distribution.

    Returns three lists of records: look, blind betting, seen betting. A record
    is a plain dict so it can go straight into JSON once templated.
    """
    import experiments as XP

    tree = solver.tree
    strategies = solver.strategy_profile(average=True)
    reach = solver._down_pass(strategies)

    look, blind_bet, seen_bet = [], [], []

    for raw in tree.decision_ids:
        n = int(raw)
        st = tree.states[n]
        sigma = strategies[n]
        acts = tree.actions[n]
        names = [ACTION_NAMES[a] for a in acts]
        j = XP.turn_index(st)

        if sigma.ndim == 1:
            # Blind actor: ONE information set for this public history, with no
            # card axis. This is RULES.md 12.4 made physical.
            pi = XP.node_reach(solver, reach, n)
            if pi <= REACH_FLOOR:
                continue
            top = int(sigma.argmax())
            rec = {
                "node": n,
                "iso_class": None,
                "family": "look" if st.phase is Phase.LOOK else "blind_bet",
                "seat": int(st.to_act) + 1,
                "blind": True,
                "turn_index": j,
                "actions": names,
                "distribution": [float(x) for x in sigma],
                "label": names[top],
                "label_prob": float(sigma[top]),
                "dominant": float(sigma.max()) > DOMINANCE_THRESHOLD,
                "reach": float(pi),
                "hand_bucket": None,
                "k": int(st.k),
                "r": int(st.r),
                "stake": int(st.stake),
                "pot": int(st.pot),
                "actions_to_cap": int(st.config.action_cap - st.k),
                "opponent_seen": bool(st.seen[st.opponent(st.to_act)]),
            }
            (look if st.phase is Phase.LOOK else blind_bet).append(rec)
        else:
            w = XP.seen_infoset_reach(solver, reach, n, st.to_act)
            live = np.flatnonzero(w > REACH_FLOOR)
            if live.size == 0:
                continue
            rows = sigma[live]
            tops = rows.argmax(axis=1)
            maxp = rows.max(axis=1)
            for c, t, mp, row, wc in zip(live, tops, maxp, rows, w[live]):
                seen_bet.append({
                    "node": n,
                    "iso_class": int(c),
                    "family": "seen_bet",
                    "seat": int(st.to_act) + 1,
                    "blind": False,
                    "turn_index": j,
                    "actions": names,
                    "distribution": [float(x) for x in row],
                    "label": names[int(t)],
                    "label_prob": float(mp),
                    "dominant": bool(mp > DOMINANCE_THRESHOLD),
                    "reach": float(wc),
                    "hand_bucket": int(bucket_of_class[c]),
                    "k": int(st.k),
                    "r": int(st.r),
                    "stake": int(st.stake),
                    "pot": int(st.pot),
                    "actions_to_cap": int(st.config.action_cap - st.k),
                    "opponent_seen": bool(st.seen[st.opponent(st.to_act)]),
                })
    return look, blind_bet, seen_bet


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def split_blind_pool(records, rng):
    """Split the finite blind pool between eval and train, stratified by turn.

    There is no sampling here in the statistical sense. The pool is small enough
    to enumerate, so this is an allocation: every reachable dominant blind
    information set in the game ends up in exactly one of the two sets.
    """
    by_turn = collections.defaultdict(list)
    for r in records:
        if r["dominant"]:
            by_turn[r["turn_index"]].append(r)
    ev, tr = [], []
    for j in sorted(by_turn):
        pool = sorted(by_turn[j], key=lambda r: r["node"])
        rng.shuffle(pool)
        cut = int(round(BLIND_EVAL_SHARE * len(pool)))
        cut = max(1, min(cut, len(pool) - 1)) if len(pool) > 1 else len(pool)
        ev.extend(pool[:cut])
        tr.extend(pool[cut:])
    return ev, tr


def stratified_seen_sample(records, n_per_label, rng, exclude=frozenset()):
    """Sample seen betting items, balanced by label and stratified within it.

    PokerBench samples an equal number of flops from each of 11 board textures,
    then rebalances labels. Teen Patti has no board, so the analogous texture is
    the public betting line the actor faces: the stratum key pairs the hand
    bucket with the actor's turn number, the stake, the raises used, and whether
    the opponent has looked.

    Stratification and label balance are done in one pass rather than two.
    Sampling across strata first and rebalancing afterwards discards down to the
    rarest label and undershoots the target badly -- `raise` and `show` are rare
    in the natural distribution (60,653 and 51,050 against 454,237 chaal), so a
    texture-first sample starves them. Drawing `n_per_label` from each label's
    own strata gives both properties by construction.
    """
    by_label = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        if not r["dominant"]:
            continue
        if (r["node"], r["iso_class"]) in exclude:
            continue
        key = (r["hand_bucket"], r["turn_index"], r["stake"], r["r"],
               r["opponent_seen"])
        by_label[r["label"]][key].append(r)

    out = []
    for label in sorted(by_label):
        strata = by_label[label]
        keys = sorted(strata)
        rng.shuffle(keys)
        for k in keys:
            rng.shuffle(strata[k])
        taken, i = [], 0
        # Round-robin across strata: equal representation until one runs dry.
        while len(taken) < n_per_label:
            progressed = False
            for k in keys:
                if i < len(strata[k]):
                    taken.append(strata[k][i])
                    progressed = True
                    if len(taken) >= n_per_label:
                        break
            if not progressed:
                break
            i += 1
        out.extend(taken)
    rng.shuffle(out)
    return out


def rebalance(records, labels, rng, cap=None):
    """Downsample to an equal count per label.

    PokerBench found that preserving the natural label distribution let a
    fold-everything policy score close to 90%. Phase 2 reports pack terminals at
    0.767 of all hands, so the equivalent degenerate policy here is "always
    pack". Equalising the label counts caps any single-action policy at 1/L.
    """
    by_label = collections.defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    present = [l for l in labels if by_label[l]]
    if not present:
        return []
    smallest = min(len(by_label[l]) for l in present)
    per = smallest if cap is None else min(smallest, cap)
    out = []
    for l in present:
        pool = by_label[l][:]
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------
# Templating
# --------------------------------------------------------------------------

def template(records, solver, hands_by_class, rng, split):
    """Render each record into a benchmark item. Raises on anything unrenderable."""
    tree = solver.tree
    out = []
    for r in records:
        st = tree.states[r["node"]]
        if r["blind"]:
            hand = None
            hand_str = None
        else:
            members = hands_by_class[r["iso_class"]]
            # Any member of the orbit is strategically identical (Phase 1 proves
            # win/tie/loss is constant within a class), so sampling a member
            # gives surface variety at zero cost to correctness.
            hand_idx = members[rng.randrange(len(members))]
            hand = HANDS[hand_idx]
            hand_str = " ".join(P.spell_card(c) for c in hand)
        prompt = P.render_prompt(st, hand=hand)

        action = tree.actions[r["node"]][r["actions"].index(r["label"])]
        item = {
            "id": f"{split}-{len(out):06d}",
            "prompt": prompt,
            # The system prompt is identical for every item and is written once
            # to SYSTEM_PATH rather than repeated 50,000 times -- inlining it
            # added 60 MB to the training file for no information.
            "correct_action": r["label"],
            "correct_amount": P.correct_amount(st, action),
            "solver_distribution": dict(zip(r["actions"], r["distribution"])),
            "legal_actions": r["actions"],
            "is_mixed": not r["dominant"],
            "meta": {
                "node": r["node"],
                "iso_class": r["iso_class"],
                "family": r["family"],
                "seat": r["seat"],
                "blind": r["blind"],
                "turn_index": r["turn_index"],
                "hand_bucket": r["hand_bucket"],
                "hand": hand_str,
                "betting_history": [ACTION_NAMES[a] for a in st.history],
                "actions_to_cap": r["actions_to_cap"],
                "stake": r["stake"],
                "pot": r["pot"],
                "raises_used": r["r"],
                "opponent_seen": r["opponent_seen"],
                "label_prob": r["label_prob"],
                "reach": r["reach"],
                "repeat_index": r.get("repeat_index", 0),
            },
        }
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def baseline_scores(items):
    """What degenerate single-action policies score on a finished set.

    A policy that always answers X is correct exactly on items labelled X. It is
    scored over the whole set, so answering "pack" on a look decision -- where
    pack is not even legal -- counts as wrong, which is the honest treatment.
    """
    graded = [i for i in items if not i["is_mixed"]]
    if not graded:
        return {}
    out = {}
    for action in ("pack", "chaal", "raise", "show", "stay-blind", "see"):
        hits = sum(1 for i in graded if i["correct_action"] == action)
        out[f"always-{action}"] = hits / len(graded)
    out["_n_graded"] = len(graded)
    return out


def report_blind_ceiling(look, blind_bet, n_seen):
    """The structural fact that caps the blind share of the eval set."""
    n_look_dom = sum(1 for r in look if r["dominant"])
    n_bb_dom = sum(1 for r in blind_bet if r["dominant"])
    total = len(look) + len(blind_bet) + n_seen
    return {
        "look_reachable": len(look),
        "look_dominant": n_look_dom,
        "blind_bet_reachable": len(blind_bet),
        "blind_bet_dominant": n_bb_dom,
        "blind_pool_dominant_total": n_look_dom + n_bb_dom,
        "seen_reachable": n_seen,
        "all_infosets": total,
        "natural_blind_share": (len(look) + len(blind_bet)) / total,
        "max_blind_eval_items": int(
            round(BLIND_EVAL_SHARE * (n_look_dom + n_bb_dom))
        ),
    }


def composition(items):
    """Counts by family, turn index, label and bucket, for PHASE3.md."""
    fam = collections.Counter(i["meta"]["family"] for i in items)
    turn = collections.Counter(
        (i["meta"]["family"], i["meta"]["turn_index"]) for i in items
        if i["meta"]["blind"]
    )
    label = collections.Counter(i["correct_action"] for i in items)
    bucket = collections.Counter(
        i["meta"]["hand_bucket"] for i in items if i["meta"]["hand_bucket"] is not None
    )
    seat = collections.Counter(i["meta"]["seat"] for i in items)
    blind = sum(1 for i in items if i["meta"]["blind"])
    return {
        "n": len(items),
        "by_family": dict(fam),
        "blind_items": blind,
        "blind_share": blind / len(items) if items else 0.0,
        "mixed_items": sum(1 for i in items if i["is_mixed"]),
        "by_label": dict(label),
        "blind_by_turn": {f"{f}:turn{j}": c for (f, j), c in sorted(turn.items())},
        "by_seat": dict(seat),
        "bucket_min": min(bucket.values()) if bucket else 0,
        "bucket_max": max(bucket.values()) if bucket else 0,
        "buckets_covered": len(bucket),
    }


def assert_disjoint(eval_items, train_items):
    """Eval and train must not share an information set. Asserted, not assumed."""
    def keys(items):
        return {(i["meta"]["node"], i["meta"]["iso_class"]) for i in items}
    overlap = keys(eval_items) & keys(train_items)
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} information sets appear in both splits, e.g. "
            f"{sorted(overlap)[:5]}"
        )
    return len(overlap)


def assert_no_blind_leaks(items):
    """Re-run the leakage guard over every finished blind item.

    `render_prompt` already checks on the way out. This checks again on the way
    in to the file, so a future refactor that loosens the templating path still
    cannot ship a leaking dataset.
    """
    n = 0
    for i in items:
        if i["meta"]["blind"]:
            P.assert_no_private_leak(i["prompt"], where=i["id"])
            if i["meta"]["hand"] is not None:
                raise P.PrivateLeakError(f"{i['id']}: blind item carries a hand")
            n += 1
    return n


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report inventory and composition without writing files")
    ap.add_argument("--eval-n", type=int, default=EVAL_N)
    ap.add_argument("--train-n", type=int, default=TRAIN_N)
    args = ap.parse_args()

    rng = random.Random(SEED)
    t0 = time.time()

    solver, tables = load_solver()
    abst, bucket_of_class, hands_by_class = class_tables(tables)
    print(f"[{time.time()-t0:5.1f}s] solver + abstraction ready "
          f"({solver.tree.n_nodes:,} nodes, {tables.n_classes:,} classes)")

    look, blind_bet, seen_bet = enumerate_infosets(solver, bucket_of_class)
    print(f"[{time.time()-t0:5.1f}s] enumerated: {len(look)} look, "
          f"{len(blind_bet)} blind-bet, {len(seen_bet):,} seen-bet infosets")

    ceiling = report_blind_ceiling(look, blind_bet, len(seen_bet))
    print(f"\n  natural blind share of the state space : "
          f"{100*ceiling['natural_blind_share']:.4f}%")
    print(f"  dominant blind information sets in game: "
          f"{ceiling['blind_pool_dominant_total']}")
    print(f"  hard ceiling on blind EVAL items       : "
          f"{ceiling['max_blind_eval_items']} "
          f"({100*ceiling['max_blind_eval_items']/args.eval_n:.1f}% of a "
          f"{args.eval_n}-item eval set)\n")

    # -- allocate the finite blind pool ------------------------------------
    # NOT rebalanced by label. The blind pool is the scarce resource -- 428
    # dominant information sets in the whole game -- and equalising labels
    # inside it discards roughly 40% of the research object to fix a baseline
    # that the seen family already controls. The blind label distribution is
    # reported in the stats instead, and the trivial-baseline check below is run
    # over the finished set, so nothing is taken on trust.
    look_ev, look_tr = split_blind_pool(look, rng)
    bb_ev, bb_tr = split_blind_pool(blind_bet, rng)
    blind_ev = look_ev + bb_ev
    blind_tr = look_tr + bb_tr

    # -- seen items --------------------------------------------------------
    n_seen_eval = args.eval_n - len(blind_ev) - EVAL_MIXED_N
    if n_seen_eval <= 0:
        raise RuntimeError("eval budget exhausted before seen items")
    seen_ev = stratified_seen_sample(seen_bet, n_seen_eval // len(BET_LABELS), rng)

    used = {(r["node"], r["iso_class"]) for r in seen_ev}
    # Training repeats the blind pool: there are only ~130 distinct blind
    # information sets left after eval takes its share, so without oversampling
    # the research object would be 0.3% of the training signal. Repetition is
    # legitimate here in a way it would not be in eval -- a repeated eval item
    # asks the same question twice and yields nothing -- but it is recorded
    # per item so the effective sample size stays visible.
    blind_tr_emitted = []
    for rep in range(TRAIN_BLIND_REPEAT):
        for r in blind_tr:
            copy = dict(r)
            copy["repeat_index"] = rep
            blind_tr_emitted.append(copy)

    n_seen_train = args.train_n - len(blind_tr_emitted) - TRAIN_MIXED_N
    seen_tr = stratified_seen_sample(
        seen_bet, n_seen_train // len(BET_LABELS), rng, exclude=used
    )

    # -- retained mixed items (distribution-distance subset) ---------------
    mixed_pool = [r for r in seen_bet if not r["dominant"]]
    rng.shuffle(mixed_pool)
    mixed_ev = mixed_pool[:EVAL_MIXED_N]
    mixed_tr = mixed_pool[EVAL_MIXED_N:EVAL_MIXED_N + TRAIN_MIXED_N]

    print(f"[{time.time()-t0:5.1f}s] selected: eval blind={len(blind_ev)} "
          f"seen={len(seen_ev)} mixed={len(mixed_ev)} | train blind="
          f"{len(blind_tr)} distinct x{TRAIN_BLIND_REPEAT} = "
          f"{len(blind_tr_emitted)} seen={len(seen_tr)} mixed={len(mixed_tr)}")

    # -- template ----------------------------------------------------------
    eval_items = template(blind_ev + seen_ev + mixed_ev, solver, hands_by_class,
                          rng, "eval")
    train_items = template(blind_tr_emitted + seen_tr + mixed_tr, solver,
                           hands_by_class, rng, "train")
    rng.shuffle(eval_items)
    rng.shuffle(train_items)
    for k, i in enumerate(eval_items):
        i["id"] = f"eval-{k:06d}"
    for k, i in enumerate(train_items):
        i["id"] = f"train-{k:06d}"
    print(f"[{time.time()-t0:5.1f}s] templated {len(eval_items):,} eval and "
          f"{len(train_items):,} train items")

    # -- verify ------------------------------------------------------------
    assert_disjoint(eval_items, train_items)
    n_blind_checked = assert_no_blind_leaks(eval_items) + assert_no_blind_leaks(train_items)
    print(f"[{time.time()-t0:5.1f}s] disjointness OK; "
          f"{n_blind_checked} blind prompts re-checked for leakage")

    stats = {
        "seed": SEED,
        "config": solver.config.label(),
        "solver_iterations": solver.iterations,
        "dominance_threshold": DOMINANCE_THRESHOLD,
        "reach_floor": REACH_FLOOR,
        "blind_eval_share_of_pool": BLIND_EVAL_SHARE,
        "state_space": ceiling,
        "eval": composition(eval_items),
        "train": composition(train_items),
        "baselines_eval": baseline_scores(eval_items),
        "baselines_train": baseline_scores(train_items),
        "train_blind_distinct": len(blind_tr),
        "train_blind_repeat": TRAIN_BLIND_REPEAT,
        "eval_blind_distinct": len(blind_ev),
    }

    print("\n=== EVAL composition ===")
    print(json.dumps(stats["eval"], indent=1))
    print("\n=== trivial baselines on EVAL ===")
    for k, v in sorted(stats["baselines_eval"].items()):
        if not k.startswith("_"):
            print(f"  {k:<18} {100*v:6.2f}%")

    if args.dry_run:
        print("\n--dry-run: no files written")
        return

    with open(SYSTEM_PATH, "w", encoding="utf-8") as fh:
        fh.write(P.SYSTEM_PROMPT)
    write_jsonl(EVAL_PATH, eval_items)
    write_jsonl(TRAIN_PATH, train_items)
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    print(f"\nwrote {EVAL_PATH} ({len(eval_items):,}), "
          f"{TRAIN_PATH} ({len(train_items):,}), {STATS_PATH}, {SYSTEM_PATH}")


if __name__ == "__main__":
    main()

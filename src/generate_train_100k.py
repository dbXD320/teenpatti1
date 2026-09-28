"""Generate a 100,000-item TeenPattiBench TRAINING set, leaving eval untouched.

This is a driver, not a second implementation. Every selection, templating and
verification step is imported from `generate.py` and called unmodified; this
file only changes the training budget, writes to a new path, and reports
progress. If `generate.py` changes, this follows it.

WHY THIS IS NOT `generate.py --train-n 100000`
----------------------------------------------
It would silently rewrite the eval set. In `generate.py.main()` the training
seen-sample is drawn BEFORE `rng.shuffle(mixed_pool)`, and `mixed_ev` is the
first 200 of that shuffled pool. `stratified_seen_sample` ends with
`rng.shuffle(out)`, which consumes a number of random draws proportional to the
sample size -- so a different `--train-n` leaves the RNG in a different state
and the 200 mixed EVAL items come out different. The 1,800 graded eval items are
unaffected (they are drawn earlier in the stream), but the eval FILE would change
and the three scored Phase 4 prediction files would no longer refer to it.

So this script never writes eval. It reads the shipped
`data/teenpattibench_eval.jsonl` as the authority, reproduces the RNG stream up
to the point where the two splits diverge, and GATES on the reproduction:
the blind and seen eval items it recomputes must match the shipped file exactly,
or it refuses to write anything.

NO SOLVE IS NEEDED. The answer key is `data/solution_default.pkl`, the
1,000-iteration Phase 2 checkpoint. `generate.py.load_solver()` loads it and
hard-requires `iterations == 1000`. CFR is not re-run; it is only re-read.

BUDGET (strictly the current implementation's arithmetic)
---------------------------------------------------------
    blind   = 128 distinct x TRAIN_BLIND_REPEAT (20)      =   2,560
    mixed   = TRAIN_MIXED_N                               =   2,000
    seen    = (100,000 - 2,560 - 2,000) // 4 per label    =  95,440
                                                            -------
                                                            100,000

The seen pool supports this: of 727,643 dominant seen information sets the
scarcest label, `show`, has 37,001 against a per-label draw of 23,860.

ONE CONSEQUENCE WORTH READING. `TRAIN_BLIND_REPEAT` is 20 and the blind pool is
exhausted at 128 distinct information sets, so the blind contribution is fixed at
2,560 items no matter how large the training set is. Doubling the set therefore
HALVES the blind share, 5.12% -> 2.56%. That is the current implementation's
behaviour and is what runs by default. `--blind-repeat 40` restores the 5.12%
share by emitting each blind set 40 times instead of 20; it is off by default
because it is a change to the recipe, not to the size.

Usage:
    python src/generate_train_100k.py                 # writes the 100k train file
    python src/generate_train_100k.py --dry-run       # plan + gates, writes nothing
    python src/generate_train_100k.py --train-n 250000
    python src/generate_train_100k.py --blind-repeat 40
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate as G
import prompts as P

TRAIN_PATH = "data/teenpattibench_train_100k.jsonl"
STATS_PATH = "data/teenpattibench_train_100k_stats.json"
EVAL_PATH = G.EVAL_PATH

#: Items rendered between progress refreshes. Templating is the only loop long
#: enough to need one.
PROGRESS_EVERY = 2000


# --------------------------------------------------------------------------
# Terminal progress reporting
# --------------------------------------------------------------------------

class Progress:
    """Elapsed / rate / ETA on one rewritten terminal line.

    Falls back to plain periodic lines when stdout is not a TTY, so piping to a
    log file does not produce a megabyte of carriage returns.
    """

    def __init__(self, total: int, label: str, t_start: float):
        self.total = total
        self.label = label
        self.t_start = t_start
        self.t0 = time.time()
        self.tty = sys.stdout.isatty()

    @staticmethod
    def _hms(seconds: float) -> str:
        seconds = max(0.0, seconds)
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def update(self, done: int, final: bool = False) -> None:
        now = time.time()
        phase = now - self.t0
        rate = done / phase if phase > 0 else 0.0
        remaining = self.total - done
        eta = remaining / rate if rate > 0 else float("inf")
        pct = 100.0 * done / self.total if self.total else 100.0
        line = (
            f"  {self.label}: {done:>7,}/{self.total:,} ({pct:5.1f}%)  "
            f"created={done:>7,}  remaining={remaining:>7,}  "
            f"{rate:8,.0f} items/s  phase={self._hms(phase)}  "
            f"eta={'--:--' if eta == float('inf') else self._hms(eta)}  "
            f"total={self._hms(now - self.t_start)}"
        )
        if self.tty:
            sys.stdout.write("\r" + line + " " * 4)
            if final:
                sys.stdout.write("\n")
            sys.stdout.flush()
        elif final or done % (PROGRESS_EVERY * 5) == 0:
            print(line, flush=True)


def stamp(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Eval set: loaded, never regenerated
# --------------------------------------------------------------------------

def load_eval_keys(path: str):
    """Read the shipped eval file and return the keys train must avoid.

    Returns (all_keys, blind_nodes, seen_keys, mixed_keys). `all_keys` is what
    `generate.assert_disjoint` compares against; the rest are used to gate the
    RNG replay below.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. This script never regenerates eval -- it treats "
            "the shipped file as the authority. Restore it from git before "
            "generating a training set against it."
        )
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    all_keys, blind_nodes, seen_keys, mixed_keys = set(), set(), set(), set()
    for i in items:
        meta = i["meta"]
        key = (meta["node"], meta["iso_class"])
        all_keys.add(key)
        if i["is_mixed"]:
            mixed_keys.add(key)
        elif meta["blind"]:
            blind_nodes.add(meta["node"])
        else:
            seen_keys.add(key)
    return items, all_keys, blind_nodes, seen_keys, mixed_keys


def gate_rng_replay(blind_ev, seen_ev, blind_nodes, seen_keys) -> None:
    """Refuse to continue unless the replayed RNG reproduces the shipped eval.

    This is the check that makes "strictly the current implementation" a
    verified claim rather than an assertion. The blind eval pool and the seen
    eval sample are both drawn before the training budget enters the RNG stream,
    so with SEED unchanged they must come out identical to what shipped.
    """
    got_blind = {r["node"] for r in blind_ev}
    if got_blind != blind_nodes:
        raise RuntimeError(
            "RNG replay did not reproduce the shipped eval's BLIND split: "
            f"{len(got_blind ^ blind_nodes)} nodes differ "
            f"(replayed {len(got_blind)}, shipped {len(blind_nodes)}). "
            "generate.py's selection path or SEED has changed since the eval "
            "file was written; regenerate both splits together instead."
        )
    got_seen = {(r["node"], r["iso_class"]) for r in seen_ev}
    if got_seen != seen_keys:
        raise RuntimeError(
            "RNG replay did not reproduce the shipped eval's SEEN sample: "
            f"{len(got_seen ^ seen_keys)} information sets differ "
            f"(replayed {len(got_seen)}, shipped {len(seen_keys)}). "
            "generate.py's selection path or SEED has changed since the eval "
            "file was written; regenerate both splits together instead."
        )


# --------------------------------------------------------------------------
# Templating with progress
# --------------------------------------------------------------------------

def template_with_progress(records, solver, hands_by_class, rng, split, t0):
    """`generate.template`, called in order on chunks so progress can be shown.

    Chunking is behaviour-preserving: `template` consumes one `rng.randrange`
    per seen record in list order, so slicing the list changes neither the RNG
    stream nor the output. Item ids assigned inside `template` are overwritten
    after the final shuffle, exactly as `generate.main` does.
    """
    out = []
    bar = Progress(len(records), f"templating {split}", t0)
    for start in range(0, len(records), PROGRESS_EVERY):
        chunk = records[start:start + PROGRESS_EVERY]
        out.extend(G.template(chunk, solver, hands_by_class, rng, split))
        bar.update(len(out))
    bar.update(len(out), final=True)
    return out


def write_jsonl_with_progress(path, items, t0):
    bar = Progress(len(items), f"writing {os.path.basename(path)}", t0)
    with open(path, "w", encoding="utf-8") as fh:
        for n, item in enumerate(items, 1):
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            if n % PROGRESS_EVERY == 0:
                bar.update(n)
    bar.update(len(items), final=True)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a large TeenPattiBench training set. Eval is read, never written."
    )
    ap.add_argument("--train-n", type=int, default=100_000,
                    help="target training items (default 100000)")
    ap.add_argument("--blind-repeat", type=int, default=G.TRAIN_BLIND_REPEAT,
                    help=f"times each distinct blind infoset is emitted "
                         f"(default {G.TRAIN_BLIND_REPEAT}, the current implementation)")
    ap.add_argument("--out", default=TRAIN_PATH, help=f"output JSONL (default {TRAIN_PATH})")
    ap.add_argument("--stats", default=STATS_PATH, help=f"output stats JSON (default {STATS_PATH})")
    ap.add_argument("--eval", default=EVAL_PATH, help=f"eval file to stay disjoint from (read-only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every gate and print the plan, but write nothing")
    args = ap.parse_args()

    if args.train_n < 1:
        raise SystemExit("--train-n must be positive")
    if args.blind_repeat < 1:
        raise SystemExit("--blind-repeat must be positive")

    t0 = time.time()
    print("=" * 78)
    print("TeenPattiBench -- large training set")
    print(f"  target items    : {args.train_n:,}")
    print(f"  answer key      : {G.CHECKPOINT} (Phase 2, 1,000 CFR+ iterations; NOT re-solved)")
    print(f"  eval (read-only): {args.eval}")
    print(f"  output          : {args.out if not args.dry_run else '(dry run, nothing written)'}")
    print(f"  seed            : {G.SEED}")
    print("=" * 78)

    # -- stage 1: answer key ------------------------------------------------
    solver, tables = G.load_solver()
    abst, bucket_of_class, hands_by_class = G.class_tables(tables)
    stamp(t0, f"answer key loaded: {solver.iterations} iterations, "
              f"{solver.tree.n_nodes:,} public nodes, {tables.n_classes:,} private classes")

    # -- stage 2: enumerate every reachable information set -----------------
    look, blind_bet, seen_bet = G.enumerate_infosets(solver, bucket_of_class)
    n_dom_seen = sum(1 for r in seen_bet if r["dominant"])
    stamp(t0, f"enumerated {len(look)} look, {len(blind_bet)} blind-bet, "
              f"{len(seen_bet):,} seen-bet information sets "
              f"({n_dom_seen:,} seen dominant)")

    # -- stage 3: the shipped eval set, as the authority --------------------
    eval_items, eval_keys, eval_blind_nodes, eval_seen_keys, eval_mixed_keys = \
        load_eval_keys(args.eval)
    stamp(t0, f"eval loaded: {len(eval_items):,} items "
              f"({len(eval_blind_nodes)} blind, {len(eval_seen_keys):,} seen, "
              f"{len(eval_mixed_keys)} mixed) -- this file is never written")

    # -- stage 4: replay the RNG stream and gate on it ----------------------
    rng = random.Random(G.SEED)
    look_ev, look_tr = G.split_blind_pool(look, rng)
    bb_ev, bb_tr = G.split_blind_pool(blind_bet, rng)
    blind_ev, blind_tr = look_ev + bb_ev, look_tr + bb_tr

    n_seen_eval = len(eval_items) - len(blind_ev) - G.EVAL_MIXED_N
    seen_ev = G.stratified_seen_sample(seen_bet, n_seen_eval // len(G.BET_LABELS), rng)

    gate_rng_replay(blind_ev, seen_ev, eval_blind_nodes, eval_seen_keys)
    stamp(t0, f"GATE PASS: RNG replay reproduced the shipped eval exactly "
              f"({len(blind_ev)} blind nodes, {len(seen_ev):,} seen infosets)")

    # -- stage 5: the training budget ---------------------------------------
    used = {(r["node"], r["iso_class"]) for r in seen_ev}

    blind_tr_emitted = []
    for rep in range(args.blind_repeat):
        for r in blind_tr:
            copy = dict(r)
            copy["repeat_index"] = rep
            blind_tr_emitted.append(copy)

    n_seen_train = args.train_n - len(blind_tr_emitted) - G.TRAIN_MIXED_N
    if n_seen_train <= 0:
        raise SystemExit(
            f"training budget exhausted before seen items: {args.train_n:,} target "
            f"minus {len(blind_tr_emitted):,} blind minus {G.TRAIN_MIXED_N:,} mixed"
        )
    per_label = n_seen_train // len(G.BET_LABELS)

    # Refuse to start a long run that provably cannot be filled.
    by_label = collections.Counter(
        r["label"] for r in seen_bet
        if r["dominant"] and (r["node"], r["iso_class"]) not in used
    )
    print()
    print("  training budget")
    print(f"    blind : {len(blind_tr):>3} distinct x {args.blind_repeat} "
          f"= {len(blind_tr_emitted):>7,}")
    print(f"    mixed : {G.TRAIN_MIXED_N:>7,}")
    print(f"    seen  : {per_label:,} per label x {len(G.BET_LABELS)} = {per_label * 4:>7,}")
    print(f"    total : {len(blind_tr_emitted) + G.TRAIN_MIXED_N + per_label * 4:>7,}")
    print()
    print("  seen pool availability (dominant, excluding eval)")
    short = []
    for label in G.BET_LABELS:
        have = by_label[label]
        ok = have >= per_label
        print(f"    {label:<6} available {have:>7,}  need {per_label:>7,}   "
              f"{'OK' if ok else 'SHORT'}")
        if not ok:
            short.append((label, have))
    n_mixed_avail = sum(
        1 for r in seen_bet
        if not r["dominant"] and (r["node"], r["iso_class"]) not in eval_mixed_keys
    )
    print(f"    {'mixed':<6} available {n_mixed_avail:>7,}  need {G.TRAIN_MIXED_N:>7,}   "
          f"{'OK' if n_mixed_avail >= G.TRAIN_MIXED_N else 'SHORT'}")
    if short:
        raise SystemExit(
            "\nthe seen pool cannot fill this budget: "
            + ", ".join(f"{l} has {h:,}" for l, h in short)
            + f" against {per_label:,} needed per label. Lower --train-n."
        )

    blind_share = len(blind_tr_emitted) / args.train_n
    print()
    print(f"  blind share of the training set: {100 * blind_share:.2f}% "
          f"({len(blind_tr_emitted):,} of {args.train_n:,})")
    if args.blind_repeat == G.TRAIN_BLIND_REPEAT and args.train_n > 50_000:
        print("    NOTE: the blind pool is exhausted at 128 distinct information sets,")
        print("    so this count is fixed by --blind-repeat and does NOT grow with the")
        print("    training set. At 50,000 items the share was 5.12%. Pass")
        print(f"    --blind-repeat {round(20 * args.train_n / 50_000)} to hold that share.")
    print()

    if args.dry_run:
        stamp(t0, "--dry-run: all gates passed, nothing written")
        return 0

    # -- stage 6: draw the seen and mixed training items --------------------
    seen_tr = G.stratified_seen_sample(seen_bet, per_label, rng, exclude=used)
    stamp(t0, f"sampled {len(seen_tr):,} seen training items "
              f"({len(G.BET_LABELS)} labels x {per_label:,})")

    # `generate.main` slices a shuffled pool at [200:2200]; the 200 it skips are
    # the eval mixed items under the ORIGINAL RNG stream. That stream no longer
    # holds here, so the eval mixed keys are excluded by identity instead. Same
    # intent, and it cannot overlap by accident.
    mixed_pool = [r for r in seen_bet if not r["dominant"]]
    rng.shuffle(mixed_pool)
    mixed_tr = [r for r in mixed_pool
                if (r["node"], r["iso_class"]) not in eval_mixed_keys][:G.TRAIN_MIXED_N]
    if len(mixed_tr) < G.TRAIN_MIXED_N:
        raise SystemExit(
            f"only {len(mixed_tr)} mixed items available, need {G.TRAIN_MIXED_N}"
        )
    stamp(t0, f"selected {len(mixed_tr):,} mixed items (eval's {len(eval_mixed_keys)} excluded)")

    # -- stage 7: template ---------------------------------------------------
    records = blind_tr_emitted + seen_tr + mixed_tr
    print()
    train_items = template_with_progress(
        records, solver, hands_by_class, rng, "train", t0
    )
    rng.shuffle(train_items)
    width = max(6, len(str(len(train_items) - 1)))
    for k, item in enumerate(train_items):
        item["id"] = f"train-{k:0{width}d}"
    stamp(t0, f"templated {len(train_items):,} items")

    # -- stage 8: verify ------------------------------------------------------
    G.assert_disjoint(eval_items, train_items)
    n_blind_checked = G.assert_no_blind_leaks(train_items)
    ids = {i["id"] for i in train_items}
    if len(ids) != len(train_items):
        raise RuntimeError("duplicate item ids after renumbering")
    stamp(t0, f"GATE PASS: disjoint from eval; {n_blind_checked:,} blind prompts "
              f"re-checked for private leakage; {len(ids):,} unique ids")

    # -- stage 9: write -------------------------------------------------------
    print()
    write_jsonl_with_progress(args.out, train_items, t0)

    stats = {
        "generator": os.path.basename(__file__),
        "seed": G.SEED,
        "config": solver.config.label(),
        "solver_iterations": solver.iterations,
        "checkpoint": G.CHECKPOINT,
        "dominance_threshold": G.DOMINANCE_THRESHOLD,
        "reach_floor": G.REACH_FLOOR,
        "target_n": args.train_n,
        "train_blind_repeat": args.blind_repeat,
        "train_blind_distinct": len(blind_tr),
        "blind_share": blind_share,
        "eval_file": args.eval,
        "eval_items": len(eval_items),
        "seen_per_label": per_label,
        "train": G.composition(train_items),
        "baselines_train": G.baseline_scores(train_items),
    }
    with open(args.stats, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)

    # -- stage 10: report -----------------------------------------------------
    size_mb = os.path.getsize(args.out) / 1e6
    print()
    print("=== composition ===")
    print(json.dumps(stats["train"], indent=1))
    print()
    print("=== trivial baselines on the training set (must all be low) ===")
    for k in sorted(stats["baselines_train"]):
        if not k.startswith("_"):
            print(f"  {k:<18} {100 * stats['baselines_train'][k]:6.2f}%")
    print()
    print("=" * 78)
    print(f"DONE in {Progress._hms(time.time() - t0)}")
    print(f"  {args.out}  ({len(train_items):,} items, {size_mb:,.1f} MB)")
    print(f"  {args.stats}")
    print(f"  eval untouched: {args.eval}")
    if size_mb > 100:
        print(f"  WARNING: {size_mb:,.0f} MB exceeds GitHub's 100 MB hard limit. "
              "Keep this file out of git.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

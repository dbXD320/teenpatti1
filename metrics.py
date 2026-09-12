"""Scoring for TeenPattiBench (Phase 3).

Metrics follow PokerBench (Zhuang et al., AAAI 2025, arXiv:2501.08328):

- **Action Accuracy (AA)** -- the model chose the solver's action.
- **Exact Match (EM)** -- the model chose the solver's action *and* the wager
  amount is the one the rules price for that action. In Teen Patti the amount is
  a deterministic function of (stake, status, action), so EM differs from AA only
  when a model names a legal action but misprices it.

One metric is added, for the mixed subset that a single-label score cannot
represent:

- **Total variation distance (TVD)** between the model's sampled action
  distribution and the solver's, on retained non-dominant decisions.
  TVD = 0.5 * sum_a |p(a) - q(a)|, which is 0 for a perfect match and 1 for
  disjoint support.

THE BREAKDOWNS ARE THE POINT. The paper's hypothesis is that models transfer
poker knowledge to ordinary betting decisions and fail on conversion timing,
where no corresponding text exists in pretraining and the correct pattern is
non-monotone. That is testable only if conversion accuracy is reported per
own-turn index, never aggregated: Phase 2 found P(look) running 0.000035, 0.786,
0.058, 0.931 across turns 1-4, so an aggregate score averages a dip and a spike
into a number that means nothing.

Usage:
    python metrics.py --eval teenpattibench_eval.jsonl --predictions preds.jsonl
    python metrics.py --eval teenpattibench_eval.jsonl --self-test
"""

from __future__ import annotations

import argparse
import collections
import json
import re

__all__ = [
    "load_jsonl",
    "parse_action",
    "score",
    "format_report",
    "total_variation_distance",
]

ACTION_WORDS = ("stay-blind", "see", "chaal", "raise", "pack", "show")


# --------------------------------------------------------------------------
# IO and parsing
# --------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_action(text, legal):
    """Extract one action word from a model's raw completion.

    Returns None when nothing legal is found, which is scored as wrong rather
    than skipped -- a model that will not answer in the required format has
    failed the item, and silently dropping those would flatter it.

    `stay-blind` is checked before `see` because "stay-blind" does not contain
    "see" but a completion like "see: no, stay-blind" contains both, and the
    later, more specific token is the intended answer.
    """
    if text is None:
        return None
    low = str(text).strip().lower()
    hits = []
    for word in ACTION_WORDS:
        pattern = rf"(?<![a-z-]){re.escape(word)}(?![a-z-])"
        for m in re.finditer(pattern, low):
            hits.append((m.start(), word))
    hits = [(pos, w) for pos, w in hits if w in legal]
    if not hits:
        return None
    # First legal mention wins: models that comply answer with one word, and
    # models that ramble usually lead with their answer.
    hits.sort()
    return hits[0][1]


def total_variation_distance(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _blank():
    return {"n": 0, "aa": 0, "em": 0, "unparsed": 0}


def _rate(d, field):
    return d[field] / d["n"] if d["n"] else float("nan")


def score(items, predictions):
    """Score predictions against the eval set.

    `predictions` maps item id -> dict with at least one of:
      - "action": the chosen action, or raw text to be parsed
      - "amount": the wager amount, for Exact Match
      - "distribution": {action: probability}, for the mixed subset

    Every graded item must have a prediction. A missing one raises rather than
    being treated as wrong: a silently short prediction file would otherwise
    report a plausible score for an incomplete run.
    """
    by_id = {i["id"]: i for i in items}
    missing = [k for k in by_id if k not in predictions]
    if missing:
        raise KeyError(
            f"{len(missing)} eval items have no prediction, e.g. {missing[:5]}"
        )

    overall = _blank()
    by_status = collections.defaultdict(_blank)
    by_family = collections.defaultdict(_blank)
    conversion_by_turn = collections.defaultdict(_blank)
    blind_bet_by_turn = collections.defaultdict(_blank)
    by_bucket = collections.defaultdict(_blank)
    by_seat = collections.defaultdict(_blank)
    tvds, mixed_n = [], 0

    for item_id, item in by_id.items():
        pred = predictions[item_id]
        meta = item["meta"]

        if item["is_mixed"]:
            mixed_n += 1
            dist = pred.get("distribution")
            if dist:
                tvds.append(
                    total_variation_distance(dist, item["solver_distribution"])
                )
            continue

        raw = pred.get("action", pred.get("text"))
        chosen = raw if raw in item["legal_actions"] else parse_action(
            raw, item["legal_actions"]
        )
        correct = chosen == item["correct_action"]
        amount_ok = correct and (
            pred.get("amount") is None
            or int(pred["amount"]) == int(item["correct_amount"])
        )

        buckets = [
            overall,
            by_status["blind" if meta["blind"] else "seen"],
            by_family[meta["family"]],
            by_seat[f"seat {meta['seat']}"],
        ]
        if meta["family"] == "look":
            buckets.append(conversion_by_turn[meta["turn_index"]])
        if meta["family"] == "blind_bet":
            buckets.append(blind_bet_by_turn[meta["turn_index"]])
        if meta["hand_bucket"] is not None:
            buckets.append(by_bucket[meta["hand_bucket"]])

        for b in buckets:
            b["n"] += 1
            b["aa"] += int(correct)
            b["em"] += int(amount_ok)
            b["unparsed"] += int(chosen is None)

    return {
        "overall": overall,
        "by_status": dict(by_status),
        "by_family": dict(by_family),
        "conversion_by_turn": dict(conversion_by_turn),
        "blind_bet_by_turn": dict(blind_bet_by_turn),
        "by_seat": dict(by_seat),
        "by_hand_bucket": dict(by_bucket),
        "mixed": {
            "n": mixed_n,
            "scored": len(tvds),
            "mean_tvd": sum(tvds) / len(tvds) if tvds else float("nan"),
        },
    }


def trivial_baselines(items):
    """Score every degenerate single-action policy on the graded items."""
    graded = [i for i in items if not i["is_mixed"]]
    out = {}
    for action in ACTION_WORDS:
        preds = {i["id"]: {"action": action} for i in items}
        hits = sum(1 for i in graded if i["correct_action"] == action)
        out[f"always-{action}"] = hits / len(graded) if graded else float("nan")
    # A policy that always names the most common label overall.
    common = collections.Counter(i["correct_action"] for i in graded).most_common(1)
    if common:
        out["_majority_label"] = common[0][0]
        out["_majority_rate"] = common[0][1] / len(graded)
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_report(report, baselines=None) -> str:
    lines = []
    o = report["overall"]
    lines.append("=== TeenPattiBench ===")
    lines.append(f"graded items      : {o['n']}")
    lines.append(f"Action Accuracy   : {100*_rate(o,'aa'):6.2f}%")
    lines.append(f"Exact Match       : {100*_rate(o,'em'):6.2f}%")
    if o["unparsed"]:
        lines.append(f"unparseable       : {o['unparsed']} "
                     f"({100*o['unparsed']/o['n']:.2f}%)")

    lines.append("")
    lines.append("--- blind vs seen ---")
    for key in ("blind", "seen"):
        d = report["by_status"].get(key)
        if d:
            lines.append(f"  {key:<6} n={d['n']:<6} AA={100*_rate(d,'aa'):6.2f}%  "
                         f"EM={100*_rate(d,'em'):6.2f}%")

    lines.append("")
    lines.append("--- conversion timing, by the actor's own turn ---")
    lines.append("    (Phase 2 solver: P(look) = 0.000035, 0.786, 0.058, 0.931)")
    for j in sorted(report["conversion_by_turn"]):
        d = report["conversion_by_turn"][j]
        lines.append(f"  turn {j}  n={d['n']:<5} AA={100*_rate(d,'aa'):6.2f}%")
    if not report["conversion_by_turn"]:
        lines.append("  (no look items scored)")

    lines.append("")
    lines.append("--- blind betting, by the actor's own turn ---")
    for j in sorted(report["blind_bet_by_turn"]):
        d = report["blind_bet_by_turn"][j]
        lines.append(f"  turn {j}  n={d['n']:<5} AA={100*_rate(d,'aa'):6.2f}%")

    lines.append("")
    lines.append("--- by family ---")
    for key in sorted(report["by_family"]):
        d = report["by_family"][key]
        lines.append(f"  {key:<10} n={d['n']:<6} AA={100*_rate(d,'aa'):6.2f}%")

    lines.append("")
    lines.append("--- by seat ---")
    for key in sorted(report["by_seat"]):
        d = report["by_seat"][key]
        lines.append(f"  {key:<8} n={d['n']:<6} AA={100*_rate(d,'aa'):6.2f}%")

    m = report["mixed"]
    lines.append("")
    lines.append("--- mixed (non-dominant) subset, distribution distance ---")
    lines.append(f"  items={m['n']}  scored={m['scored']}  mean TVD={m['mean_tvd']:.4f}"
                 if m["scored"] else
                 f"  items={m['n']}  scored=0 (no distributions supplied)")

    if baselines:
        lines.append("")
        lines.append("--- trivial baselines (must all be low) ---")
        for k in sorted(baselines):
            if not k.startswith("_"):
                lines.append(f"  {k:<18} {100*baselines[k]:6.2f}%")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def self_test(items):
    """Score two reference policies without any model, to prove the scorer works.

    The solver policy must reach 100% Action Accuracy by construction -- it is
    the answer key. Anything less means the scorer, the parser or the dataset
    disagree with each other.
    """
    perfect = {
        i["id"]: {"action": i["correct_action"], "amount": i["correct_amount"],
                  "distribution": i["solver_distribution"]}
        for i in items
    }
    rep = score(items, perfect)
    aa = _rate(rep["overall"], "aa")
    em = _rate(rep["overall"], "em")
    tvd = rep["mixed"]["mean_tvd"]
    print("self-test: solver policy against its own answer key")
    print(f"  Action Accuracy {100*aa:.4f}%   Exact Match {100*em:.4f}%   "
          f"mean TVD {tvd:.6f}")
    ok = abs(aa - 1.0) < 1e-12 and abs(em - 1.0) < 1e-12 and tvd < 1e-12
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    # A model that answers in prose rather than a bare word.
    wordy = {
        i["id"]: {"action": f"I would {i['correct_action']} here.",
                  "amount": i["correct_amount"],
                  "distribution": i["solver_distribution"]}
        for i in items
    }
    rep2 = score(items, wordy)
    aa2 = _rate(rep2["overall"], "aa")
    print(f"  prose-wrapped answers parse to {100*aa2:.4f}% "
          f"-> {'PASS' if abs(aa2 - 1.0) < 1e-12 else 'FAIL'}")

    print()
    print(format_report(rep, trivial_baselines(items)))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="teenpattibench_eval.jsonl")
    ap.add_argument("--predictions")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", help="write the report as JSON to this path")
    args = ap.parse_args()

    items = load_jsonl(args.eval)
    if args.self_test:
        self_test(items)
        return
    if not args.predictions:
        ap.error("give --predictions or --self-test")

    raw = load_jsonl(args.predictions)
    preds = {r["id"]: r for r in raw}
    rep = score(items, preds)
    print(format_report(rep, trivial_baselines(items)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()

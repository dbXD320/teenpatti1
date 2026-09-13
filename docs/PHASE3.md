# Phase 3 Report — TeenPattiBench

**Status: eval and training sets generated. 18 Phase 3 tests pass (168 with Phases 1–2).**

Modules: [generate.py](../src/generate.py), [prompts.py](../src/prompts.py), [metrics.py](../src/metrics.py).
Tests: [test_phase3.py](../tests/test_phase3.py).
Outputs: `teenpattibench_eval.jsonl` (2,000), `teenpattibench_train.jsonl` (50,000),
`teenpattibench_stats.json`, `teenpattibench_system_prompt.txt`.

Reproduce with `python generate.py`, then `python metrics.py --self-test`,
then `python -m pytest -q`.

---

## 1. Entry gate

All three checks pass. Reported before anything was generated.

| # | Gate | Result |
|---|---|---|
| 1 | Reading the **average** strategy, not the final iterate | **PASS** — `average_strategy()` used throughout; max \|average − current\| over 400 nodes is 0.999, so the two are genuinely different objects and `average=True` is a real choice rather than an alias |
| 2 | Exploitability at or below the Phase 2 figure | **PASS** — recomputed from the checkpoint: **6.127378e-04 boots/hand**, 0.0306% of pot, identical to Phase 2 to all printed digits |
| 3 | Five sampled information sets match `cfr.py` directly | **PASS** — 2 blind, 3 seen; every distribution bit-identical to `average_strategy()` and summing to 1 |

The checkpoint carries 1,000 iterations and `generate.py` refuses to run against any
other count.

---

## 2. The finding that reshaped this phase

**The blind information-set pool is not merely rare. It is finite, small, and
fully enumerable — and it is smaller than the eval allocation the brief asked
for.**

| | count |
|---|---:|
| Look information sets, reachable (reach > 1e-12) | 251 |
| — of those, dominant (p > 0.5) | **231** |
| Blind betting information sets, reachable | 235 |
| — of those, dominant | **197** |
| **Total dominant blind pool, entire game** | **428** |
| Seen betting information sets, reachable | 913,825 |
| **All information sets** | **914,311** |
| Natural blind share | **0.0532%** |

The brief targets 30–40% of a 2,000–3,000-item eval set on blind and
conversion-timing decisions — that is 600 to 1,200 items. **Only 428 exist in the
whole game**, and they must also supply the training set, which may not overlap.

This is not a sampling shortfall that more compute would fix. A blind player's
information set is the public history alone (RULES.md 12.4), so there is no card
axis to multiply against. A seen item is a (public node, hand class) pair —
1,286 × 1,755 of them. A blind item is a public node. That asymmetry is the
research object, and it is also what caps the dataset.

**Consequence: the blind subset is not sampled, it is enumerated.** Every
reachable dominant blind decision in Teen Patti is either an eval item or a
training item. There is no sampling error on the research object at all, which is
a property PokerBench could not have — but it also means the achieved eval share
is **15.0%**, not 30–40%.

### Why 15% and not more

With a 2,000-item eval set, 70% of the 428-set pool gives 300 items — the largest
share that still leaves a usable training pool. Pushing further trades away the
only thing that makes the training set able to teach the pattern at all.

| option | eval blind share | cost |
|---|---:|---|
| Give eval the whole pool | 21.4% | training gets **zero** distinct blind items |
| **Give eval 70% (chosen)** | **15.0%** | training gets 128 distinct, repeated ×20 |
| Shrink eval to ~1,000 items | 30% | below the brief's 2,000 floor; halves the seen-side sample |
| Duplicate blind items in eval | 30%+ | meaningless — a repeated eval item asks the same question twice |

The last row is the important one. Because a blind prompt is a deterministic
function of the public history, two items drawn from the same blind information
set are **byte-identical prompts**. Inflating the blind share by duplication would
report a larger number for no additional information.

**What matters for the paper is not the share but the per-stratum count**, since
the diagnostic is conversion accuracy broken out by own-turn index. Those counts
are adequate at turns 2, 3 and 4 and thin at turn 1 — see §6.

**This is the one place the delivered dataset departs from the brief, and it is a
property of the game rather than a choice.** Changing `BLIND_EVAL_SHARE` and
`EVAL_N` in `generate.py` moves the trade-off if you would rather have the share
than the seen-side sample.

---

## 3. Composition

### Eval set — 2,000 items

| family | items | share |
|---|---:|---:|
| Seen betting (graded) | 1,500 | 75.0% |
| Look / conversion | 162 | 8.1% |
| Blind betting | 138 | 6.9% |
| Mixed seen (TVD-scored, not graded) | 200 | 10.0% |
| **Blind total** | **300** | **15.0%** |

Natural blind share of the state space: **0.0532%**. Sampled: **15.0%**. That is a
deliberate **282× oversampling** of the research object. A reviewer should read it
as a design choice, not a sampling error: the blind subset is the paper's subject
and at natural frequency a 2,000-item eval set would contain **one** blind item.

### Training set — 50,000 items

| family | items | share |
|---|---:|---:|
| Seen betting | 47,440 | 94.9% |
| Look / conversion | 1,380 | 2.8% |
| Blind betting | 1,180 | 2.4% |
| Mixed seen | 2,000 | 4.0% |
| **Blind total** | **2,560** | **5.1%** |

The 2,560 blind training items are **128 distinct information sets emitted 20
times each**. Repetition is recorded per item in `meta.repeat_index`, so the
effective sample size stays visible rather than being implied by the row count.
Without it the research object would be 0.26% of the training signal. This is
legitimate oversampling for a training set and would not be legitimate for eval.

---

## 4. Stratification achieved

**Conversion items are stratified across own-turn index**, which is the
requirement that makes the non-monotone pattern testable turn by turn rather than
averaged into meaninglessness.

| own turn | look items (eval) | blind bet items (eval) | solver P(look) |
|---:|---:|---:|---:|
| 1 | 4 | 4 | 0.000035 |
| 2 | 19 | 17 | 0.786 |
| 3 | 53 | 41 | 0.058 |
| 4 | 86 | 76 | 0.931 |

All four strata are populated and asserted to be
(`test_all_four_turn_strata_are_populated`).

**Seen items are stratified across hand strength and betting line.** PokerBench
samples equally from 11 board textures; Teen Patti has no board, so the analogous
texture is the public line the actor faces. The stratum key is

> (hand bucket, own-turn index, stake, raises used, opponent seen/blind)

Stratification and label balance are done in **one pass**, not two. Sampling
across textures first and rebalancing afterwards starves the rare labels —
`raise` and `show` are 60,653 and 51,050 of 913,825 against 454,237 `chaal` — and
undershoots the target size badly. Drawing an equal quota from each label's own
strata gives both properties by construction. All 25 hand buckets are covered, at
41–98 items each in eval and 946–3,361 in training.

Seat coverage is 727 / 1,273 (seat 1 / seat 2) in eval. The imbalance is natural,
not introduced: seat 2 acts second so its decision nodes sit deeper and are more
numerous (PHASE1.md §4).

---

## 5. Label balance and trivial baselines

PokerBench found that preserving the natural label distribution let a
fold-everything policy score close to 90%. Phase 2 reports pack terminals at 0.767
of all hands, so the degenerate policy here is "always pack".

Betting items are drawn to an equal quota per label. The blind families are
**deliberately not rebalanced**: the pool is 428 sets in total and equalising
labels inside it discards roughly 40% of the research object to fix a baseline
that the seen family — 85% of the graded set — already controls. The blind label
distribution is reported instead, and the baselines below are measured on the
finished set rather than assumed.

**Trivial baselines on the 1,800 graded eval items:**

| policy | score |
|---|---:|
| always-chaal | **25.94%** |
| always-show | 22.67% |
| always-raise | 21.56% |
| always-pack | **20.83%** |
| always-stay-blind | 5.67% |
| always-see | 3.33% |

No degenerate policy exceeds 26%, against the ~90% PokerBench observed before
rebalancing. `test_no_trivial_baseline_scores_well` asserts every baseline stays
below 35%.

Training baselines are tighter still (23.7% always-pack, 25.3% always-chaal).

Eval label counts: chaal 580, pack 441, show 421, raise 396, stay-blind 102,
see 60.

---

## 6. Mixed (non-dominant) seen decisions — the choice made

**186,182 of 913,825 reachable seen betting information sets (20.4% by count) have
no action above 0.5.** By reach weight they are only ~2.0%, matching Phase 2's
finding that 98.0% of seen betting decisions are reach-weighted dominant. The gap
says the mixing lives in rarely-reached information sets.

**Choice: exclude them from the single-label graded set, and retain a separate
tagged subset scored by distribution distance.**

- 200 in eval, 2,000 in training, flagged `is_mixed: true`.
- They are excluded from Action Accuracy and Exact Match, so no graded item has an
  ambiguous answer.
- `metrics.py` scores them by **total variation distance** against the solver
  distribution, reported separately.

Why both rather than either: excluding them outright would discard the only part
of the tree where the equilibrium genuinely mixes, and a benchmark that never asks
about mixing cannot detect a model that is wrong about it. Scoring them by
accuracy would be incoherent — there is no single correct action. Keeping them
tagged and separately scored costs nothing and leaves Phase 4 the option.

**The full solver distribution is stored on every record regardless**, dominant or
mixed, as required.

---

## 7. Prompts

The template follows PokerBench: cards spelled out in words, position and betting
history in prose, closing with the exact instruction *"Do not explain your answer.
Your optimal action is:"* — asserted on every item by
`test_prompt_closes_with_the_pokerbench_instruction`.

Adapted for Teen Patti: two players and a symmetric boot rather than six players
and blinds; the blind/seen state; and a glossary in the system prompt covering
boot, stake, blind, seen, chaal, raise, pack and show, because models have seen
very little of this vocabulary. The system prompt is written once to
`teenpattibench_system_prompt.txt` rather than repeated on 50,000 records — inlining
it added 60 MB for no information.

Conversion items make the turn index and the remaining cap explicit
("This is your turn number 3 of this hand. 4 betting actions remain before the
action cap forces a showdown, and 2 raises remain"), because Phase 2 shows the
correct answer depends on both. Asserted by `test_conversion_items_expose_turn_and_cap`.

### The blind prompt guard

A blind prompt must contain no card identity and no strength hint. Three layers,
all in `prompts.assert_no_private_leak`, called on every blind prompt with no code
path that skips it:

1. Spelled card names and bare rank/suit words, matched on word boundaries.
2. Two-character card codes (`As`, `Td`) as standalone tokens.
3. Hand-category and strength vocabulary (trail, sequence, colour, pair, flush,
   kicker, equity, strong/weak) and second-person phrasings that only make sense
   if the hand is known (`you hold`, `your cards are`, `your hand is`).

Passing a hand to a blind actor raises rather than being ignored, because a caller
that does so has misunderstood the information partition.

`test_every_blind_decision_node_renders_without_leaking` renders **all 728 blind
decision nodes in the tree**, not just the sampled ones.
`test_no_blind_prompt_contains_card_identities` re-checks every blind prompt in
both shipped files.

### Two phrasings the guard rejected, and what they show

Both were caught during development and are worth recording, because they are
exactly the "implicit leakage" the brief warned about and neither looks like a
leak.

1. **"you have not looked at your three cards"** — the guard fired on *three*,
   which is a rank word. Reworded to "your cards". Harmless in itself, but the
   guard cannot distinguish a counting word from a rank, and loosening it to
   allow this would have opened a real hole.
2. **"As a blind player you may demand a show"** — `As` is the card code for the
   Ace of Spades. Reworded to "While blind you may demand a show". This one is
   genuinely alarming: a sentence that begins with a capitalised "As" is
   indistinguishable from a card code by any automated check, and had the guard
   been written to skip sentence-initial tokens it would have shipped.

Nothing else required rewording. **No spot failed to template**; every reachable
decision node in the tree renders.

---

## 8. Verification

| Property | Test |
|---|---|
| No blind prompt contains card identities | `test_no_blind_prompt_contains_card_identities` — every blind item in both files |
| All 728 blind nodes render without leaking | `test_every_blind_decision_node_renders_without_leaking` |
| Guard catches injected leaks | `test_leak_guard_catches_injected_card_identity` — 4 deliberate injections |
| Blind items carry no hand metadata | `test_blind_items_carry_no_hand_metadata` |
| Eval and train are disjoint | `test_eval_and_train_are_disjoint` — on (node, class) keys; also asserted inside `generate.py` |
| Labels are the solver's argmax and dominant | `test_labels_match_the_solver_distribution` |
| Stored distributions match the live checkpoint | `test_labels_match_the_live_solver` — re-reads `average_strategy()` |
| Amounts match the rules | `test_correct_amount_matches_the_rules` — against `game.amount_for` |
| Solver policy scores 100% | `test_solver_policy_scores_100_percent` |
| No trivial baseline scores well | `test_no_trivial_baseline_scores_well` |
| Missing predictions raise | `test_missing_predictions_raise` |

`metrics.py --self-test` scores the solver against its own answer key: **100.00%
Action Accuracy, 100.00% Exact Match, mean TVD 0.000000**. It also checks that
prose-wrapped answers ("I would pack here.") parse to 100%, so a model that
refuses to answer in one word is not penalised for formatting.

Generation is deterministic: seed 20260912.

---

## 9. Metrics

`metrics.py` reports:

- **Action Accuracy** — the model chose the solver's action.
- **Exact Match** — correct action *and* correctly priced amount. In Teen Patti
  the amount is determined by (stake, status, action), so EM differs from AA only
  when a model names a legal action but misprices it.
- **Total variation distance** on the retained mixed subset.

Required breakdowns, all implemented:

- blind versus seen
- **conversion accuracy by own-turn index**, printed alongside the solver's
  P(look) = 0.000035, 0.786, 0.058, 0.931 so the non-monotone shape is readable
  directly off the results table
- blind betting by own-turn index
- by family, by seat, by hand bucket
- trivial baselines

A model's failure to answer parseably is scored as wrong, not skipped; a missing
prediction raises rather than being treated as wrong, so a short prediction file
cannot report a plausible score for an incomplete run.

---

## 10. The hypothesis this dataset tests

**Models will do relatively well on ordinary betting decisions and fail on
conversion timing.**

Ordinary betting — pack, chaal, raise, show with a known hand — is close enough to
poker that the large volume of poker strategy text in pretraining should transfer.
Conversion timing has no counterpart: no poker variant lets a player bet without
looking, there is no Teen Patti strategy literature to retrieve, and the correct
pattern is not guessable. Phase 2 found P(look | own turn *j* reached while blind)
running **0.000035, 0.786, 0.058, 0.931** across turns 1–4 for seat 1 — no, then
yes, then no, then yes. Nothing in poker intuition predicts that shape.

If the split holds, this is a cleaner probe of reasoning versus retrieval than
PokerBench could run, precisely because poker strategy is abundant in training
data and Teen Patti strategy is not. The per-turn breakdown in `metrics.py` is
what tests it, which is why conversion accuracy is never aggregated.

**A caution on reading turn 1.** Only 5 look information sets exist at turn 1 in
the entire game, of which 4 are in eval. Phase 2 already established that this
decision is near-deterministic (seat 1 stays blind 0.999965, seat 2 looks
0.994824), so it is a control stratum, not a test: a model is expected to pass it
and passing tells us nothing. Turns 2, 3 and 4 carry the diagnostic, with 19, 53
and 86 items. Turn 3 — the dip, where the solver says *don't* look after having
looked willingly at turn 2 — is the single most informative stratum and the one
most likely to separate reasoning from pattern-matching.

---

## 11. Open items for Phase 4

1. **The eval blind share is 15.0%, not the 30–40% requested.** §2 explains why
   the game cannot supply more. If you prefer the share to the seen-side sample,
   lower `EVAL_N` — 1,000 items would give 30% — and regenerate.
2. **Turn 1 conversion accuracy rests on 4 items.** Report it, do not interpret it.
3. **The mixed subset is scored but not graded.** If Phase 4 wants a single
   headline number, decide whether TVD enters it or stays a separate column.
4. **`teenpattibench_train.jsonl` is 80 MB** and is excluded from git. It
   regenerates deterministically from the checkpoint with `python generate.py`.
   The eval set (3.2 MB) is committed.

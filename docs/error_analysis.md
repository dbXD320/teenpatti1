# TeenPattiBench — Error Analysis Across Four Models

**Scope.** All 1,800 graded eval items × 4 models (7,200 decisions). Sources:
`results/preds_*.jsonl`, `data/teenpattibench_eval.jsonl`, `src/metrics.py`.
All runs fp16, unquantised, greedy, `max_new_tokens=16`, identical system prompt,
Tesla T4. No model was modified.

**Bottom line — three findings, in order of how much they should change what you do
next.**

1. **The largest correctable cause is not the model.** Every prompt lists the
   actions in the same order, and `show` recall swings from 49.6% to 1.0% for
   Qwen3-8B purely by moving it one slot later (§4). Fix the ordering before
   spending another GPU-hour on fine-tuning.
2. **The ceiling is much higher than card-blindness implies.** A policy that never
   looks at its cards can score **79.11%** here, because the betting history alone
   is highly informative (§6). The best model reaches 34.56%. The first-order
   failure is that the *public state* is barely used — not that the cards are
   ignored.
3. **These are not near-misses.** Error rate is flat in the solver's confidence, and
   35–44% of every model's errors are actions the solver plays less than 1% of the
   time (§2). Calibration tuning will not touch this.

---

## 1. Three failure modes, separated

| | Definition | Verdict |
|---|---|---|
| **Decision-boundary** | wrong where the solver is nearly indifferent | **Largely absent** — §2 |
| **Generalisation** | blanket policy, insensitive to state | **Dominant** — §3, §6 |
| **Data coverage** | benchmark/training too thin to learn or measure | **Real and severe for conversion** — §7 |

---

## 2. These are not near-misses

If models were roughly right and erring at the margins, error rate would fall as
the solver's confidence rises. It does not.

**Error rate by solver confidence in the correct label (`label_prob`)**

| label_prob | n | Qwen3-8B | Qwen3-4B | Mistral-7B | Llama-3.2-3B |
|---|---:|---:|---:|---:|---:|
| 0.5–0.6 | 550 | 59.6% | 65.1% | 54.2% | 82.5% |
| 0.6–0.7 | 305 | 72.1% | 70.2% | 69.2% | 81.0% |
| 0.7–0.8 | 195 | 70.3% | 64.6% | 67.7% | 73.8% |
| 0.8–0.9 | 174 | 75.3% | 75.3% | 80.5% | 59.8% |
| **0.9–1.0** | 576 | **67.9%** | **65.1%** | **68.9%** | **60.1%** |

Flat, and for the two Qwen models slightly *worse* on the items the solver is most
certain about. Confirmed by scoring each error by the probability the solver
actually assigns the chosen action:

| | mean solver-prob of chosen action | errors the solver plays ≥30% | errors the solver plays <1% |
|---|---:|---:|---:|
| Qwen3-8B | 0.323 | 18.4% | **35.0%** |
| Qwen3-4B | 0.330 | 19.0% | **37.3%** |
| Mistral-7B | 0.310 | 12.7% | **43.0%** |
| Llama-3.2-3B | 0.315 | 19.5% | **44.2%** |

**Over a third of every model's errors are actions the solver essentially never
plays.** Calibration tuning will not help; the models are not close.

---

## 3. Label recall is the whole story

The clearest single table in this analysis. Recall = of the items where this is the
correct answer, how often the model produced it.

| correct answer | n | Qwen3-8B | Qwen3-4B | Mistral-7B | Llama-3.2-3B |
|---|---:|---:|---:|---:|---:|
| see | 60 | 98.3% | 100% | 100% | 100% |
| chaal | 467 | 53.5% | 55.0% | 32.3% | 44.3% |
| raise | 388 | 57.0% | 71.1% | **100%** | **0.0%** |
| pack | 375 | **0.0%** | **0.0%** | **0.0%** | 63.2% |
| show | 408 | 15.0% | **0.7%** | 5.6% | **0.0%** |
| **stay-blind** | 102 | **2.0%** | **0.0%** | **0.0%** | **1.0%** |

Every model has two or three labels it effectively cannot emit. Mistral's 100% on
`raise` is not skill — it raises whenever raise is legal, so it captures every raise
item and pays for it everywhere else.

**Vocabulary ceilings**, before any question is answered:

| | words ever used | ceiling | actual | % of reachable |
|---|---|---:|---:|---:|
| Qwen3-8B | chaal, raise, show, see, stay-blind | 79.17% | 32.94% | 41.6% |
| Qwen3-4B | chaal, raise, show, see | 73.50% | 33.11% | 45.0% |
| Mistral-7B | chaal, pack, raise, show, see | 94.33% | 34.56% | 36.6% |
| Llama-3.2-3B | pack, chaal, see, stay-blind | **55.78%** | 28.06% | 50.3% |

---

## 4. The layout confound is large, measurable, and fixable

Every prompt lists actions in the same order, so `pack` is index 0 in all 1,838
betting items and `see` is index 0 in all 162 look items. Position and identity are
perfectly confounded.

**The decisive test.** `show` appears at index 2 on three-action menus and index 3
on four-action menus. Same action, same meaning, different slot:

| | n | Qwen3-8B | Qwen3-4B | Mistral-7B | Llama-3.2-3B |
|---|---:|---:|---:|---:|---:|
| `show` at index **2** | 117 | **49.6%** | 2.6% | **19.7%** | 0.0% |
| `show` at index **3** | 291 | **1.0%** | 0.0% | **0.0%** | 0.0% |

**Qwen3-8B's recall on `show` collapses from 49.6% to 1.0% — a 50× drop — purely
from moving it one slot later.** Mistral goes 19.7% → 0.0%. This is not a Llama
quirk; it affects the best models in the set.

Menu positions each model will select at all, over 1,838 betting items:

| | positions used | distribution | look items: picks first |
|---|---|---|---:|
| Qwen3-8B | 1, 2, 3 | p1 52.7%, p2 46.9%, **p3 0.4%** | 98.1% |
| Qwen3-4B | 1, 2 | p1 49.1%, p2 50.9% | 100% |
| Mistral-7B | 0, 1, 2 | **p0 0.3%**, p1 22.8%, p2 77.0% | 100% |
| Llama-3.2-3B | **0, 1 only** | p0 71.1%, p1 28.9% | 99.4% |

`show` is the correct answer on **345 of the 720 items all four models fail** (§8).
A large share of the benchmark's headline difficulty is currently measuring page
layout.

**Caveat in the other direction:** Qwen and Mistral pick index 0 at ~0% of betting
nodes but 98–100% of look nodes — so *their* preference for `see` is a real choice,
not a slot habit. Llama picks index 0 at 71% of betting nodes too, so its `see` is
not separable. Only Llama's conversion result is unsafe on these grounds.

---

## 5. Per-model error signatures

### Qwen3-8B — 1,207 errors (67.06%)

| truth → predicted | n | % of its errors |
|---|---:|---:|
| pack → chaal | 247 | 20.5% |
| chaal → raise | 200 | 16.6% |
| show → chaal | 199 | 16.5% |
| raise → chaal | 166 | 13.8% |
| show → raise | 148 | 12.3% |
| stay-blind → see | 100 | 8.3% |

Never folds. Best `show` recall of the four (15.0%), entirely from three-action
menus. Strong seat asymmetry: 55.1% error as seat 1 vs **74.5%** as seat 2.

> `eval-000007` · seat 2, turn 1, pot 6, opponent seen, bucket 12
> menu `[pack, chaal, raise, show]` · truth **pack** (solver 0.823)
> said **chaal** — an action the solver plays 1.9% of the time.

### Qwen3-4B — 1,204 errors (66.89%)

| truth → predicted | n | % |
|---|---:|---:|
| show → raise | 208 | 17.3% |
| pack → chaal | 200 | 16.6% |
| show → chaal | 197 | 16.4% |
| chaal → raise | 194 | 16.1% |
| pack → raise | 175 | 14.5% |

Same family as the 8B and scores 0.17 points higher — within noise; doubling
parameters changed nothing. Narrowest vocabulary of the Qwen pair: `show` recall
0.7%, `stay-blind` 0.0%, `pack` 0.0%.

### Mistral-7B — 1,178 errors (65.44%) — best overall, most card-blind

| truth → predicted | n | % |
|---|---:|---:|
| chaal → raise | 290 | 24.6% |
| show → raise | 289 | 24.5% |
| pack → raise | 266 | 22.6% |
| stay-blind → see | 102 | 8.7% |

**72% of its errors are "said raise".** On menus containing `raise` it raises 1,327
of 1,330 times. It is also the only model to answer *illegally* — 15 times it said
`raise` at a node where the raise cap made raise illegal (6 on graded items).

It gives one identical answer across every hand in **74 of 74** multi-hand nodes
(solver: 11/74) — i.e. never once observed to change its mind because of its cards —
and it tops the leaderboard. **On this benchmark, ranking by Action Accuracy
currently ranks card-blindness upward.** That needs saying in the paper.

### Llama-3.2-3B — 1,295 errors (71.94%) — inverted, and layout-bound

| truth → predicted | n | % |
|---|---:|---:|
| raise → pack | 353 | 27.3% |
| show → pack | 318 | 24.6% |
| chaal → pack | 260 | 20.1% |
| pack → chaal | 138 | 10.7% |

Folds on 65% of all items; never says `raise` or `show`. Both are at index ≥2, and
Llama **never selects an index above 1** in 1,838 items — so this is a layout
artifact, not an opinion about raising. **48.6% of graded betting items have their
correct answer at index ≥2 and are unreachable for it by construction.**

Its folding does not track its hand: 67.7% fold rate on the weakest hands, 75.2% on
the strongest, while correct-to-fold falls 59.2% → 9.0%.

---

## 6. State-feature patterns

**Cards — no model uses them.** Raw error rate appears to improve with hand strength
for Qwen/Mistral, but that is the fold bias in disguise (folding is correct on 55.6%
of the weakest bucket vs 8.2% of the strongest). Excluding every item where `pack` is
correct removes the effect entirely:

| bucket band | n | Qwen3-8B | Qwen3-4B | Mistral-7B | Llama-3.2-3B |
|---|---:|---:|---:|---:|---:|
| 0–4 weakest | 91 | 56.0% | 33.0% | 40.7% | 83.5% |
| 5–9 | 167 | 67.7% | 59.9% | 56.3% | 83.8% |
| 10–14 | 238 | 59.7% | 60.1% | 59.7% | 88.2% |
| 15–19 | 295 | 60.3% | 64.1% | 66.8% | 87.5% |
| 20–24 strongest | 334 | 55.4% | 57.2% | 47.0% | 89.2% |

Flat for three models; **Llama gets monotonically worse as its hand improves.**

### But card-blindness is not what is capping the scores

This is the most important number in the analysis, and it cuts against the obvious
reading. Compute the best policy that **ignores the cards entirely** — one fixed
action per public betting node, chosen with hindsight:

| policy | score |
|---|---:|
| One fixed word for the whole benchmark | 25.94% |
| **Best fixed action per public node (never reads cards)** | **79.11%** |
| Best model today (Mistral-7B) | 34.56% |
| Perfect play | 100% |

**A model that never once looked at its own hand could score 79.11%** on this eval
set, because the public betting history alone is highly informative. The models
reach 34.56% — about **44% of the card-blind ceiling**.

So the binding constraint is *not* that they ignore their cards. It is that they are
not reading the **betting history** either. Their answers do vary across nodes — each
model's single most common answer covers only 154–283 of the 622 distinct nodes — but
that variation is mostly wrong. Teaching card sensitivity is a second-order fix;
the first-order failure is that the public state is barely being used.

**Menu size** — error rises sharply with more options (Qwen3-8B 53.6% → 54.3% →
81.2% for 2/3/4 actions), consistent with §4 rather than with difficulty.

**Position (seat)** — Qwen3-8B 55.1% vs 74.5%, Qwen3-4B 53.9% vs 75.0%, Mistral
51.0% vs 74.5% (seat 1 vs 2). Llama is flat (73.7 vs 70.8). Seat 2's nodes sit
deeper in the tree, so this is confounded with line length; unresolved.

**Stake / raises used** — error falls at stake 4 (raise cap reached, so `raise` is
masked out and the menu shrinks). Again a menu effect, not a pricing effect.

**Betting line** — worst opening line for everyone is `see chaal see` (n=191, 66–87%
error): both players seen, deepest branching. Best is `see raise stay-blind` (n=92,
51.1% for all three raise-leaning models — identically, because they all answer the
same word there).

**Pot size** — no monotone effect; the 7–14 band is worst for three of four models,
which tracks menu size rather than pot.

---

## 7. Coverage problems

**Conversion is 1.4% of the training signal.** Label counts in
`teenpattibench_train_100k.jsonl`:

| chaal | pack | show | raise | stay-blind | see |
|---:|---:|---:|---:|---:|---:|
| 25,792 | 24,474 | 24,334 | 24,020 | **880** | **500** |

The four betting labels are balanced to ~25% each by design, but the two conversion
labels together are **1,380 of 100,000 items**. `stay-blind` recall is 0–2% across
all four models. That is exactly what a 0.88% label share predicts.

This is structural, not an oversight: only **428 dominant blind information sets
exist in the entire game** (300 in eval, 128 in train, emitted 20× each). The blind
training items are 128 distinct prompts — byte-identical within each group, because
a blind prompt is a deterministic function of the public history.

**Eval strata too thin to interpret:** `look` turn 1 (n=4) and `blind_bet` turn 1
(n=4). Only 5 turn-1 conversion situations exist in the game. Report, do not
interpret.

Hand buckets are adequately covered: 36–80 seen items per bucket, all 25 populated.

---

## 8. Shared vs model-specific

| | count | share of graded |
|---|---:|---:|
| Wrong for **all four** models | **720** | 40.0% |
| Wrong for at least one | 1,610 | 89.4% |
| Right for all four | 190 | 10.6% |

**Composition of the universally-failed 720:** `show` 345, `chaal` 138, `pack` 138,
`stay-blind` 99. By family: seen_bet 573, look 99, blind_bet 48.

Errors unique to one model (all others correct):

| | unique errors | truth mix |
|---|---:|---|
| Qwen3-8B | 2 | chaal 1, see 1 |
| Qwen3-4B | 11 | chaal 11 |
| Mistral-7B | 27 | chaal 27 |
| **Llama-3.2-3B** | **185** | **raise 179**, chaal 6 |

**Shared failure modes** (all four): never emit `stay-blind`; miss `show` almost
entirely; answer `see` on ~100% of conversion items; do not use their cards;
under-select late menu positions.

**Model-specific:** Llama's total exclusion of `raise`/`show` (179 of its 185 unique
errors are raise items) and its fold reflex; Mistral's raise reflex and its 15
illegal answers.

Agreement: Qwen3-4B↔Mistral 73.6%, Qwen3-8B↔Qwen3-4B 65.2%, Qwen3-8B↔Mistral 62.5%,
but Llama vs any other 25–29%. Three models cluster on a raise/chaal prior; Llama
sits alone on a fold prior. None is converging on the game.

---

## 9. Seen vs blind

| stratum | n | own baseline | Qwen3-8B | Qwen3-4B | Mistral-7B | Llama-3.2-3B |
|---|---:|---|---:|---:|---:|---:|
| seen | 1,500 | chaal 25.00% | 30.40% (+5.4) | 31.47% (+6.5) | **33.20% (+8.2)** | 25.33% (+0.3) |
| blind | 300 | stay-blind 34.00% | 45.67% (+11.7) | 41.33% (+7.3) | 41.33% (+7.3) | 41.67% (+7.7) |
| — look | 162 | stay-blind **62.96%** | 37.65% (**−25.3**) | 37.04% (**−25.9**) | 37.04% (**−25.9**) | 37.65% (**−25.3**) |
| — blind_bet | 138 | chaal **66.67%** | 55.07% (−11.6) | 46.38% (−20.3) | 46.38% (−20.3) | 46.38% (−20.3) |

The +11.7 on the blind aggregate is an artefact of merging two strata with different
label mixes. **Both halves are failures.** Never quote the combined blind figure.

Conversion is not weak — it is empty. All four answer `see` on essentially every
item (159–162 of 162), so per-turn accuracy equals the key's see-rate to the
decimal at every turn. The score measures the label distribution, not the model.

---

## 10. Actions for the next fine-tuning run

**1. Randomise action order in training data — do this first.**
Without it, fine-tuning will learn slot position rather than the game, and the
result will be uninterpretable. `prompts.py` now takes an `action_order` argument;
`make_eval_shuffled.py` does this for eval. The training generator still emits v1
order. Evidence: §4, a 50× swing in `show` recall from one slot.
*Also re-run all four baselines on `teenpattibench_eval_shuffled.jsonl` so pre/post
numbers are measured on the same footing.*

**2. Upweight the conversion labels.**
`see`/`stay-blind` are 1.4% of the training set and recall is 0–2%. Only 128
distinct blind prompts exist, so more data is impossible — the levers are loss
weighting or a higher `--blind-repeat`. Note the trade-off: those 128 prompts are
byte-identical across repeats, so raising the repeat trades under-representation for
memorisation. Weighting the loss is the safer lever.

**3. Exclude the 2,000 `is_mixed` items from SFT.**
Their label is the argmax of a genuinely mixed distribution. Training on it teaches
false confidence exactly where the solver is deliberately uncertain.

**4. Expect vocabulary collapse and monitor it directly.**
Per-label recall (§3) is a better early-stopping signal than aggregate loss. A run
where `pack` or `stay-blind` recall stays at 0 has failed regardless of its loss
curve.

**5. Do not tune for calibration.** §2 shows errors are flat in solver confidence
and a third land on actions the solver never plays. The gap is representational,
not a threshold.

**6. Aim at the betting history before the cards.** The card-blind ceiling is
**79.11%** (§6) and the best model sits at 34.56% — so 45 points of headroom are
available *without the model ever reading its hand*. Fine-tuning on the public state
is the cheaper and larger win; card sensitivity is the second stage, worth at most
the remaining 21 points.

**7. Track two ceilings, not one.**

| milestone | score | meaning |
|---|---:|---|
| trivial floor | 25.94% | one word, forever |
| current best | 34.56% | Mistral-7B |
| **card-blind ceiling** | **79.11%** | perfect use of betting history alone |
| perfect play | 100% | requires reading the cards |

A fine-tune that has genuinely learned the public state should land well above 50%.
Below ~40% means the betting history is still not being used, whatever the loss
curve says.

**8. Keep a card-sensitivity check in the harness anyway.** The fold-controlled
bucket table (§6) and the multi-hand node test (§5, Mistral 74/74) are the only
measures that separate reading the cards from reading the history — and because the
card-blind ceiling is 79%, aggregate accuracy will keep rewarding card-blind models
for a long way yet. Report both.

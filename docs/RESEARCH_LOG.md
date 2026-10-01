# TeenPattiBench — Research Log (Phase 4 and fine-tuning)

A complete record of the evaluation, data-preparation, fine-tuning and debugging
work from 14–30 September 2026. It includes the small observations, the failed
attempts and the claims that were later overturned. It is meant as the source of
record for the paper: every number carries its context, and every inference says
what evidence supports it.

**How to read the provenance tags**

| tag | meaning |
|---|---|
| **[V]** | Recomputed from the raw files while writing this log (30 Sep 2026). |
| **[R]** | Read from a saved artefact: `run_config.json`, `val_log.jsonl`, a `metrics.py` report or a run log. |
| **[S]** | Computed or reported earlier in the session. The number was not recomputed for this log, so recompute it before citing. |
| **[U]** | Pasted by the user from a Colab run and not independently saved. |

The fixed background (rules, solver, abstraction, Phases 0–3) is in `CLAUDE.md`,
`docs/PHASE1–3.md` and `docs/RESULTS_PHASE2.md`. It is repeated here only where a
result depends on it.

---

## Contents

1. [Fixed context](#1-fixed-context)
2. [Datasets built](#2-datasets-built)
3. [Zero-shot evaluation: ten models](#3-zero-shot-evaluation-ten-models)
4. [Error-analysis series: findings that still stand](#4-error-analysis-series-findings-that-still-stand)
5. [Is the 2,000-item eval a fair test?](#5-is-the-2000-item-eval-a-fair-test)
6. [**Methodological finding: the per-turn conversion strata do not test the non-monotone pattern**](#6-methodological-finding-the-per-turn-conversion-strata-do-not-test-the-non-monotone-pattern)
7. [Fine-tuning Llama 3.2 3B](#7-fine-tuning-llama-32-3b)
8. [Internal-test deep analysis of the fine-tuned model](#8-internal-test-deep-analysis-of-the-fine-tuned-model)
9. [Engineering and debugging incidents that affect reproducibility](#9-engineering-and-debugging-incidents-that-affect-reproducibility)
10. [Hypotheses: status](#10-hypotheses-status)
11. [Corrections register](#11-corrections-register)
12. [Open items and next experiments](#12-open-items-and-next-experiments)
13. [Artefact index](#13-artefact-index)

---

## 1. Fixed context

| quantity | value |
|---|---|
| Solver | CFR+, **1,000 iterations**, 54.9 min, average strategy |
| Exploitability | 6.127378e-04 boots/hand (0.031% of pot) |
| Game value to seat 1 | −0.039781 boots/hand |
| Config | boot 1, stack 50 (≥34 is the same game), `k ≤ 8` betting actions, `r ≤ 2` raises |
| Public tree | 5,929 nodes, 2,014 decision nodes, 3,915 terminals |
| Hand classes | 1,755 suit-isomorphic classes (lossless) |
| Blind information sets | 728 total → **486 reachable** → **428 dominant** (top action p > 0.5) |
| Seen information sets | 2,256,930 (913,825 reachable) |

**Why only 428 blind items can ever exist.** A blind player's information set is
the public history alone. Each blind decision node is therefore one question,
while a seen node is 1,755 questions, one per hand class. The caps (`k ≤ 8`,
`r ≤ 2`) keep the public tree finite. Of the 486 reachable blind infosets, 428 pass
the PokerBench dominance filter (`experiments.dominant_action_stats`,
threshold 0.5). The eval holds 300 of them and training the remaining 128.

**The solver's look behaviour [R, `data/experiments_results.json`]**

| | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---:|---:|---:|---:|
| Seat 1: P(look \| own turn reached while blind) | 0.000035 | 0.786 | 0.058 | 0.931 |
| Seat 2: same | 0.995 | 0.058 | 0.218 | 0.138 |
| Seat 1: P(still blind at own turn) | 1.000 | 0.589 | 0.125 | 0.118 |
| Seat 2: same | 1.000 | 3.7e-04 | 1.4e-06 | 9.6e-08 |

The famous "no / yes / no / yes" pattern is **seat 1's, and weighted by how often
each situation is reached**. Seat 2 looks immediately with p = 0.9948 and is almost
never blind after turn 1. See §6 for why this matters.

---

## 2. Datasets built

### 2.1 Benchmark eval: `data/teenpattibench_eval.jsonl` (v1, unchanged)

2,000 items: 1,500 seen betting, 162 look (conversion), 138 blind betting, and 200
mixed items (TVD only, not graded). **1,800 graded.**

| stratum | one-word baseline | label mix [V] |
|---|---:|---|
| all graded | 25.94% (always chaal) | chaal 467, show 408, raise 388, pack 375, stay-blind 102, see 60 |
| seen betting | 25.00% | balanced by construction |
| conversion | **62.96%** (always stay-blind) | stay-blind 102, see 60 |
| blind betting | **66.67%** (always chaal) | chaal 92, show 33, raise 13 |

Conversion items by own turn [V]: turn 1 n=4, turn 2 n=19, turn 3 n=53, turn 4 n=86.
All 162 are distinct public nodes. None are mixed.

### 2.2 Shuffled eval twin: `data/teenpattibench_eval_shuffled.jsonl`

- **Why:** in v1 every menu lists `pack → chaal → raise → show`, and every look item
  lists `see` first. Action identity and menu position were perfectly confounded.
- **How:** `src/make_eval_shuffled.py`, seed 20260914. Same 2,000 ids, same answer
  key. The menu order and the order of the price clauses are permuted with
  **balanced permutations per menu shape**.
- **Checks:** the answer key is unchanged; each prompt is a pure reordering (a
  punctuation-insensitive word-bag check); and `src/prompts.py` with no
  `action_order` is byte-identical to v1 on 2,000/2,000 prompts.
- **Bug found while building:** the first balance check pooled menu shapes, which
  made it look unbalanced. Fixed to check per shape.
- **Bug found while building:** the word-bag check failed on punctuation. Fixed
  with a regex.

### 2.3 Training set, 100k: `data/teenpattibench_train_100k.jsonl`

- **Generator:** `src/generate_train_100k.py`, seed 20260912, the same
  implementation as the 50k `generate.py`.
- **Gate:** replaying the RNG must reproduce the eval exactly, and the eval is
  never written.
- **Output:** SHA-256 `a1b3c829…3c18053`, 100,000 items (97,440 seen betting,
  1,180 blind betting, 1,380 look).
- **Blind items:** the 128 distinct blind infosets × 20, so 2,560 blind items
  (2.56%).
- **Mixed items:** 2,000 included.
- **Distinctness [S]:** every seen item is a distinct prompt (95,440/95,440). The
  blind copies are byte-identical in this file, because it uses v1 ordering.
- **Status:** superseded for fine-tuning by §2.4. It is kept for reproducibility.

### 2.4 Fine-tuning set: `data/teenpattibench_finetune.jsonl`

- **Generator:** `src/generate_finetune.py`, seed 20260928.
- **Output:** SHA-256 `aeea7b5a395345b1f5c02697082892a3f273d2b86c0ed441d0a3271aab1e5c11`
  [R], 100,000 rows, 182 MB, gitignored. Full design in `docs/FINETUNE_STRATEGY.md`.

**Composition [R, `teenpattibench_finetune_stats.json`]**

| component | rows | purpose |
|---|---:|---|
| A blind (128 infosets × 30) | 3,840 | conversion and blind betting |
| B same-node contrast groups | 24,002 | 4,931 groups over 356 nodes (sizes 4: 3,159 · 6: 1,405 · 8: 367) |
| C·F1 pack on `[pack, chaal]` | 6,000 | never-folds failure |
| C·F2 raise legal, not correct, equity 0.50–0.80 | 12,000 | raise reflex |
| C·F3 show correct | 8,000 | show collapse |
| C·F4 strong hand, raise or show correct | 6,000 | folding strong hands |
| D balancer | 40,158 | fills labels to target |

- **Labels:** pack 25,644 · chaal 25,643 · show 25,643 · raise **21,000** ·
  stay-blind 1,320 · see 750. There are no mixed items.
- **Best one-word policy:** 25.64%, below the eval floor of 25.94%.

**Design decisions and the evidence behind them**

- **Raise share set to 0.21, not 0.25.**
  - With an equal split, P(raise correct | raise legal) came out at 37.5%, against
    32.1% in eval. The cause was F1's 6,000 `[pack, chaal]` rows, which used up
    pack's quota on menus where raise isn't legal.
  - The share was swept: 0.25 → 37.5%, 0.24 → 36.9%, 0.22 → 34.2%, **0.21 → 32.9%**,
    0.20 → 31.5%. At 0.20 the best single word would exceed the eval floor
    (25.98%), so 0.21 was chosen.
- **Per-example weight = `label_prob` / mean `label_prob` of that label.**
  - This gives every label a mean weight of exactly 1.000.
  - Plain `label_prob` would have given show (mean 0.706) and raise (0.696) about
    17% less total gradient than pack (0.840) and chaal (0.839).
- **Low-confidence items kept, only downweighted.** 30.6% of graded eval items
  have `label_prob` ≤ 0.60, and a hard floor starves `show`: the pool has 22,721
  show items at > 0.6 but only 9,431 at > 0.8.
- **Menu order.** A balanced permutation per menu shape. Each contrast group shares
  one order, so the hand is the only thing that varies within it. Each blind repeat
  takes the next permutation in turn.
- **Distinct blind prompts [R/S].** The 3,840 blind rows cover 924 distinct prompts.
  Look nodes have only 2 orderings, so their 30 repeats split 15/15.

**Anomaly found and fixed: labels sorted alphabetically inside every contrast group.**
- Within all 4,931 groups the rows came out ordered by label (chaal, pack, raise,
  show). With a sequential sampler that is a positional pattern the model could
  learn.
- **Fix:** a within-group shuffle, `random.Random(SEED + 2)`. The composition and
  the decisions were verified to be unchanged after the fix.

**Finding while building the groups: per-node strength order is pack < show < chaal < raise.**
- Where both are correct at the same node, the `show` hand is the *weaker* one at
  **98% of 103 nodes**.
- Pooled medians (show 0.737 vs chaal 0.615) had suggested the opposite. That was
  a pooling artefact: show is legal only at deeper nodes, where hands are stronger
  overall. `training_signal.md` §4 was corrected.
- Other per-node orders [S]: pack < chaal 98% (323 nodes), pack < show 99% (118),
  pack < raise 97% (111), chaal < raise 90% (125), show < raise 93% (54).
- **Implication:** any "show with strong hands" curriculum would teach the opposite
  of the solver.

**Verification of the written file [S]**
- 100,000 unique ids.
- 0 infoset overlap with the eval. Re-verified [V] on 30 Sep: 0 shared prompts,
  0 shared (node, hand class) pairs, 0 shared blind nodes, for both v1 and shuffled.
- 0 mixed items; 0 correct actions missing from their menu; 0 labels that differ
  from the solver's argmax.
- 3,840/3,840 blind prompts pass the leakage guard; 4,931/4,931 groups contiguous.

**Code change supporting it:** `src/generate.py` `template()` accepts optional
`action_order`, `extra_meta` and `weight`. The default output is byte-identical:
regenerating the 100k file reproduced SHA `a1b3c829…3c18053`.

### 2.5 What the training data teaches (from `docs/training_signal.md`) [S]

**Within one betting situation, the hand decides the answer.**
- At 349 of 429 nodes (81%) with ≥ 20 training hands, the betting history alone
  can't determine the answer.
- Nodes by number of distinct correct actions: 80 have 1, 188 have 2, 124 have 3,
  37 have 4.

**The rule is clean.** Contiguous strength bands reproduce the labels at **99.1%**
(median node 100%). With the labels shuffled as a control, the same fit is 80.9%, so
hand strength adds +18.2 points. 94% of nodes fit at ≥ 95%.

**Raise depends on hand strength.** It is correct on 34.0% of the 70,082 items where
it is legal:

| hand equity | raise correct |
|---|---:|
| < 0.50 | 12.1% |
| 0.50–0.80 | 20.8% |
| 0.80–0.95 | 58.9% |
| ≥ 0.95 | 83.4% |

**Contrastive pairs** (adjacent in strength at the same node, different correct
action): 1,877 in total.
- 226 (12.0%) have confident labels on both sides.
- 605 (32.2%) have both sides ≤ 0.60 confidence: the argmax of a near-50/50 mix,
  i.e. noise.
- Example: node 1306 labels equity 0.0338 as pack and 0.0340 as raise, both at
  p = 0.64.

**Card-removal effect.** At node 1357, `5♦5♠K♠` (equity 0.7880) is a confident
fold, p = 0.98, while `5♦5♥8♣` (equity 0.7883) is a confident show, p = 0.99.

---

## 3. Zero-shot evaluation: ten models

### 3.1 Run history (why there are several numbers per model)

| run | precision / hardware | status |
|---|---|---|
| Kaggle (Qwen3-8B, Qwen3-4B, Mistral 7B, Llama 3.2 3B) | fp16, 2× Tesla T4 | superseded; `results/old/` |
| Colab single-model (Gemma 3 4B) | bf16, L4 | superseded; 33.94% |
| Colab single-model (Gemma 3 12B) | bf16, A100 40 GB | identical to the batch run: 36.17% |
| **Batch** (nine models) | **bf16, A100 40 GB** (Qwen3-30B-A3B on A100 80 GB) | **reported** |
| Batch (Gemma 4 E2B) | bf16, **L4** | reported with a hardware caveat |

**The same model gives different scores in different setups [V]:**

| model | Kaggle fp16 T4 | L4 bf16 | A100 bf16 |
|---|---:|---:|---:|
| Qwen3-8B | 32.94 | | 33.89 |
| Qwen3-4B | 33.11 | | 33.06 |
| Mistral 7B | 34.56 | | 35.56 |
| Llama 3.2 3B | 28.06 | | 26.78 |
| Gemma 3 4B | | 33.94 | 34.22 |

- **Inference:** changes in precision and hardware move scores by up to ±1.3 points.
  That is comparable to the order effect (§3.4) and to the gaps between the top six
  models.
- **Stronger evidence from Phase 4:** a 4-bit Q4_K_M local run of Qwen3-8B agreed
  with its fp16 run on only 62.5% of items. That artefact was deleted, so the
  number can't be re-derived.
- **Consequence:** every reported number must state dtype and hardware.

### 3.2 Method (all batch runs)

| setting | value |
|---|---|
| Prompt | Chat template, with the system prompt as a `system` turn. All ten models accepted it; none needed it folded into the user turn. |
| Decoding | Greedy, `max_new_tokens=16`, batch 8, left padding, thinking off |
| Answer | The first action word, matched against all six words (not only the legal set) |
| Scoring | `metrics.parse_action` keeps only legal words, so an illegal or unparsed answer is **wrong**, never skipped |
| Output | A metadata header plus `{"id","answer"}` lines |

### 3.3 Leaderboard [V: all 20 files re-scored on 30 Sep 2026]

| # | model | v1 AA | shuffled AA | seen | conversion | blind betting | blind items folded (of 138) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Gemma 3 12B | **36.17** | 34.22 | 31.73 | 62.96 | 52.90 | 20 |
| 2 | Mistral 7B | 35.56 | 33.39 | 33.87 | 37.04 | 52.17 | 1 |
| 3 | Gemma 3 4B | 34.22 | 32.50 | 30.53 | 37.04 | 71.01 | 0 |
| 4 | Qwen3-8B | 33.89 | 32.61 | 31.73 | 37.65 | 52.90 | 0 |
| 5 | Qwen3-14B | 33.67 | 32.06 | **35.07** | 37.04 | 14.49 | **100** |
| 6 | Qwen3-4B | 33.06 | 31.11 | 31.33 | 37.04 | 47.10 | 0 |
| 7 | Phi-4 14B | 31.78\* | 34.11 | 33.27 | 34.57 | 12.32 | **101** |
| 8 | **Gemma 4 E2B** (L4) | 31.56 | 30.72 | 27.33 | **64.81** | 38.41 | 35 |
| 9 | Qwen3-30B-A3B | 29.06 | 28.78 | 24.93 | 37.04 | 64.49 | 0 |
| 10 | Llama 3.2 3B | 26.78 | 26.39 | 24.33 | 37.04 | 41.30 | 67 |
| — | one-word baseline | 25.94 | 25.94 | 25.00 | 62.96 | 66.67 | — |

- **Range:** the field is +0.8 to +10.2 points above answering one word forever.
- **Exact Match = Action Accuracy** in every run, because no amounts were requested.
- **Statistical ties [S]:** by pairwise McNemar tests, the top six are statistically
  indistinguishable. The binomial 95% confidence interval at n = 1,800 is about
  ±2.2 points.
- \* **Phi-4 misspelling.** Phi-4 answered `chal` 203 times on v1 (19 on shuffled).
  These are correctly scored wrong. Counting `chal` as `chaal` gives v1 34.78% and a
  shuffle effect of −0.56 instead of +2.33.

**Gemma 4 E2B (new in this log)** [V]
- It is the first model to give **different conversion answers**:
  - v1: stay-blind 131, see 31
  - shuffled: stay-blind 107, see 55
- It is the only model **above the 62.96% conversion floor** (64.81 on v1, 67.28 on
  shuffled).
- Its conversion answers change with menu order, so part of that variation is
  presentation.
- **Not yet done:** the per-turn and per-seat breakdown. Interpret its conversion
  score only after that breakdown (and read §6 first).
- Its answer mix is the most balanced of any model: raise 506, pack 493, chaal 451,
  show 387.
- It produced poker vocabulary outside the benchmark's words: `call` once (v1) and
  `fold` twice (shuffled). All three are scored wrong.
- Caveat: it ran on an L4, while the other nine ran on A100s.

### 3.4 Findings (details in `docs/RESULTS_NINE_MODELS.md`)

**1. The conversion failure is universal.**
- Eight of the nine batch models give one answer to all 162 conversion items, and
  their per-turn accuracy equals the answer key's label mix to the decimal.
- Answering `see` everywhere scores 75.0 / 68.4 / 50.9 / 19.8 by turn. Answering
  `stay-blind` everywhere scores 25.0 / 31.6 / 49.1 / 80.2.
- Gemma 3 12B's 62.96% is exactly the always-stay-blind floor, not progress.
- **It survives shuffling.** In the shuffled eval `see` is listed first on only
  half the items, yet the see-answering models still answer `see` ≥ 95.7% of the
  time and pick the first-listed option only about 50% of the time.

**2. Models choose the word, not the menu slot.**
- Llama's raise recall is 0.0% under both orderings, including when `raise` is
  listed first.
- The `show` collapse is driven by whether `raise` is on the menu, not by show
  being listed last. For Qwen3-8B, show recall is 53.0% when raise isn't offered
  and 2.4% when it is (21.4% and 0.0% on shuffled).
- Shuffling changes 7–28% of answers but only 1.3–2.2 points of net accuracy for
  seven of nine models.

**3. Scale doesn't help.**
- Qwen3: 4B 33.06 → 8B 33.89 → 14B 33.67 → 30B-A3B 29.06.
- Qwen3-30B-A3B answers chaal on 82.7% of items, and its seen score (24.93%)
  equals always-chaal.
- Gemma 3 gains about 2 points from 4B to 12B.

**4. The models that read their cards fold while blind.**
- Across 74 nodes with ≥ 6 hands in eval, the number where the model gives the
  same answer whatever its hand:

  | | nodes (of 74) |
  |---|---:|
  | Solver | 11 |
  | Phi-4 (with `chal` merged) | 10 |
  | Qwen3-14B | 18 |
  | Gemma 3 12B | 32 |
  | Mistral | 73 |
  | Llama | 70 |

- Qwen3-14B and Phi-4 fold on 100 and 101 of 138 blind-betting items, where
  folding is never correct.
- **Interpretation:** these models fold in response to *uncertainty*, not to a bad
  hand.

**5. The models don't agree with each other.** Pairwise agreement runs from 11.2%
(Gemma 3 12B vs Mistral) to 76.3% (Gemma 3 4B vs Qwen3-4B). Each model has its own
fixed prior.

**6. Llama folds regardless of its hand.** Its fold rate by strength band is
76 / 78 / 81 / 73 / 81%, where the solver's is 56 / 38 / 26 / 13 / 8%.

---

## 4. Error-analysis series: findings that still stand

These come from the Kaggle-era four-model analysis (`error_analysis.md`,
`error_situations.md`, `training_coverage.md`, `training_signal.md`), with fp16 T4
numbers. Claims later overturned are in §11.

- **A policy that ignores its cards could score 79.11%** on the eval, by answering
  the best fixed action per public node. The best zero-shot model reaches 36.17%.
  So the first failure is not using the public state, before any failure to read
  the cards.
- **Most errors aren't near-misses.** Error rate is flat across solver confidence,
  and 35–44% of each model's errors are actions the solver plays less than 1% of
  the time.
- **Each model can only say a subset of the actions** (its vocabulary ceiling):

  | model | words ever used | best possible score |
  |---|---|---:|
  | Qwen3-8B | chaal, raise, show, see, stay-blind | 79.17% |
  | Qwen3-4B | chaal, raise, show, see | 73.50% |
  | Mistral | chaal, pack, raise, show, see | 94.33% |
  | Llama | pack, chaal, see, stay-blind | 55.78% |

- **One situation fails for every model.** When the menu is only `[pack, chaal]`
  and pack is correct, all four models fail all 39 eval items, including one the
  solver folds at p = 0.996.
- **Mistral raises whenever it can:** 1,227 of the 1,230 items where raise is legal.
- **Phase 4 probe, now deleted.** Adding "Your hand is weak and you are very likely
  beaten" to the prompt flipped Qwen3-8B to pack on 5 of 5 items. The model acts on
  a stated conclusion but doesn't derive it from its cards.
- **Coverage.** 100% of the eval's seen nodes appear in training (322/322), and 0%
  of its blind nodes do (0/300, by construction). In the 100k file:
  - Turn-1 `see` has 0 training examples.
  - Turn-2 `stay-blind` has 2 distinct examples.
- **Seat gap (Qwen3-8B, Phase 4):** 44.94% for seat 1 vs 25.45% for seat 2.
  Unexplained. Seat 2's nodes sit deeper in the tree, so this is confounded with
  depth.

---

## 5. Is the 2,000-item eval a fair test? [S]

| measure | value |
|---|---|
| Reachable public nodes covered | 60.5% |
| Equilibrium reach mass covered | 81.6% |
| Scenario cells covered | 112 of 121 |
| Labels | rebalanced deliberately (PokerBench practice); the best one-word score is 25.94% |
| Precision | ±2.2 points at n = 1,800 (95% binomial) |
| Blind stratum | 300 of the 428 dominant blind infosets that exist: an exhaustive sample |

**Verdict:** broad and fair for seen betting. For conversion it is exhaustive over
what exists, but see §6: its per-turn cells measure something different from the
headline equilibrium pattern.

---

## 6. Methodological finding: the per-turn conversion strata do not test the non-monotone pattern

**What was believed.** `CLAUDE.md` calls turn 3 "the most informative stratum: the
solver says *don't* look, having looked willingly at turn 2".
`FINETUNE_STRATEGY.md` §7 says "a flat always-stay-blind gets turn 3 right and
turn 4 wrong."

**What the eval items actually contain [V]:**

| own turn | n | labels | always-see score | unweighted mean P(see) | reach-weighted P(see) | median item reach |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 4 | see 3, stay 1 | 75.0 | 0.617 | 0.995 | 2.2e-04 |
| 2 | 19 | see 13, stay 6 | 68.4 | 0.572 | 0.786 | 3.5e-07 |
| 3 | 53 | see 27, stay 26 | 50.9 | 0.490 | 0.744 | 1.7e-08 |
| 4 | 86 | see 17, stay 69 | 19.8 | 0.252 | **0.993** | 2.6e-09 |

By seat and turn: labels, and reach-weighted P(see) over those same items:

| | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---|---|---|---|
| seat 1 | — | 3 / 3, 0.786 | 9 / 11, 0.744 | 13 / 47, 0.993 |
| seat 2 | 3 / 1, 0.995 | 10 / 3, 0.052 | 18 / 15, 0.267 | 4 / 22, 0.034 |

Each cell shows the number of `see` labels / number of `stay-blind` labels, then the
reach-weighted P(see).

**What this shows**

1. **Item-level labels fall steadily by turn** (see-rate 75 → 68 → 51 → 20). There
   is no dip at turn 3 and no spike at turn 4. The benchmark's per-turn cells
   therefore reward "look early, stay blind late", a *monotone* rule.
2. **The non-monotone pattern exists only once weighted by reach.** At turn 4, the
   few high-reach infosets say "look" (reach-weighted 0.993). But 69 of the 86
   items, all low-reach, are labelled stay-blind.
3. **Why:** the eval samples dominant blind infosets **uniformly**, one item per
   infoset, and most blind infosets at turns 3–4 are extremely rare in actual play
   (median reach 1.7e-08 and 2.6e-09). The headline 0.058 / 0.931 figures are
   reach-weighted over *all* of seat 1's blind reach, including infosets not in the
   eval.
4. **The internal fine-tuning test shows the same inversion.** Its turn-3 look items
   are both `see` and its turn-4 items are all `stay-blind` (§8.6).

**Consequences for the paper**

- Don't describe the per-turn conversion cells as testing the no / yes / no / yes
  pattern.
- A model can score well per turn with a simple monotone heuristic.
- To test non-monotone timing, add either:
  - a **reach-weighted conversion score**: weight each item by `meta.reach`, which
    is already in every record; or
  - a **high-reach subset**, e.g. seat 1 at the highest-reach node of each turn.
- Both are scoring changes only. Neither changes the eval file or the answer key.
- The prompts carry the seat, so a seat-conditional analysis is also possible.
- This is an **inference from label and reach statistics**. It has not yet been
  checked against a model that actually applies the monotone rule.

---

## 7. Fine-tuning Llama 3.2 3B

### 7.1 Constraints set by the user (verbatim)

- "i dont want to use my eval dataset of 2000 questions in any way in finetuning"
- "eval set should not be there ok?"

The fine-tuning notebook never loads the eval. It stops if any `eval-` id appears
in the fine-tuning file. `run_config.json` records `benchmark_eval_used: false` [R].

### 7.2 Split: 80,000 / 10,000 / 10,000, by unit

- **Units** (`make_split`, seed 20260929): a whole blind node (all 30 rows), a whole
  contrast group, or a single seen row. A unit never crosses splits.
- **Blind:** a fixed fraction of each (family × correct action) stratum, giving
  104 / 12 / 12 blind nodes. [R/V] Validation and test each hold **6 look nodes and
  6 blind-betting nodes**.
- **Contrast groups:** 3,944 / 492 / 495.
- **Singles:** these fill validation and test to exactly 10,000 rows, keeping each
  split's answer mix the same as the singles overall.
- **Leakage asserts (all pass):** no shared blind node, seen infoset, prompt or
  contrast group between any two splits.
- **Distinct questions:** validation 9,742 and test 9,742 [R]. The duplicates
  removed are blind rows with the same menu order.
- **During training:** a validation subset of **1,998** [R], namely all 102 distinct
  blind validation prompts plus 474 per seen label (seed `SEED + 1`).

### 7.3 Configuration actually used [R, `run_config.json`]

| setting | value |
|---|---|
| Base | `meta-llama/Llama-3.2-3B-Instruct`, bf16 |
| Method | LoRA r = 16, α = 32, dropout 0.05, on q, k, v, o, gate, up and down projections; PEFT 0.21.1 |
| Adapter | 97,307,544 bytes (safetensors) |
| Optimiser | AdamW (fused), learning rate 2e-4, cosine schedule, 38 warmup steps (3%), weight decay 0, gradient-norm clip 1.0 |
| Batch | **16 per device × 4 accumulation = 64** (the notebook default was 8 × 8; changed for the A100) |
| Schedule | 1 epoch, **1,250 optimiser steps**, 80,000 training rows |
| Loss | cross-entropy on the answer tokens only (the action word plus `<|eot_id|>`), weighted by the per-example `weight`. The LM head runs only at answer positions. |
| Sampler | sequential, so contrast groups stay together within a batch |
| Prompt | system role, `date_string = "26 Jul 2024"`, thinking off |
| Validation | every 250 steps, scored by the first token of each **legal** action; the best checkpoint is chosen by **balanced accuracy** |
| Checkpoints | every 250 steps, keep the last 3 |
| Hardware | **NVIDIA A100-SXM4-40GB**; torch 2.11.0+cu128, transformers 5.16.1, Python 3.13.15 |
| Wall time | **203.3 min** of training [R, `train_done.json`] |

**Discrepancy in the saved config (unresolved).**
- `run_config.optimisation.gradient_checkpointing` is **false**, but
  `run_config.trainer_args.gradient_checkpointing` is **true**.
- The user had been told to disable gradient checkpointing for the A100, and
  probably changed only one of the two places.
- Checkpointing changes memory use and speed, not the maths. Record it as
  uncertain, and fix the notebook so a single flag drives both.
- **Throughput:** about 94 s per 10 optimiser steps (about 6.6 training examples a
  second). Each validation check added about 82 s.

### 7.4 Failed experiment: the sanity check

**Version 1 (as run): 500 training items × 3 epochs = 24 optimiser steps.**
Validation was on the same 500 items.

| check | acc | bal | chaal | pack | raise | show | see | stay-blind | fold weak / strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| step 10 [U] | 43.0 | 44.1 | 9 | 41 | 15 | 100 | 100 | 0 | 48 / 14 |
| step 20 [U] | 53.6 | 52.7 | 24 | 40 | 64 | 88 | 100 | 0 | 50 / 14 |
| final, best step 24 [U] | 54.2 | 53.2 | 20 | 42 | 67 | 89 | 100 | 0 | 51 / 13 |

- **Verdict printed:** "SANITY FAIL". The threshold was about 100% memorisation.
- **Diagnosis:** a design error in the check, not a pipeline bug.
  - 24 steps at LR 2e-4, with warmup taking the first steps, is too few to memorise
    500 items.
  - The model was clearly learning: +31 points of accuracy over the base model
    (22.8% on a comparable subset), with raise going 0 → 67 and show 0 → 89.
- **Observation:** early training produces label collapse on the scarce actions.
  `see` was answered on every look item (100% see, 0% stay-blind), and `show` was
  over-predicted.
- **Version 2 (built, never run):** 128 items × 40 epochs = 80 steps, validation
  every 20 steps, a PASS (≥ 0.9) / PARTIAL (gain ≥ 0.15) / FAIL verdict, and a
  fresh output folder. The user went straight to the full run, and the **pilot**
  mode (10k rows) was also skipped.

### 7.5 Base model before training (same A100, same prompt)

**Validation subset [U]:** acc 22.8, balanced 32.0.

| action | chaal | pack | raise | show | see | stay-blind |
|---|---:|---:|---:|---:|---:|---:|
| recall | 24 | 68 | 0 | 0 | 100 | 0 |

- It folds 82% of blind-betting items.
- It folds **64% of weak hands and 83% of strong hands**, the inverse of the key
  (73.7% and 3.0%, below). This matches the zero-shot finding that Llama folds
  regardless of hand strength.

**Internal test, 9,742 questions [R]:** AA **24.71%**.

| | score |
|---|---:|
| seen | 24.88% |
| blind (all) | 8.82% |
| blind betting | 5.56% |
| look | 33.33% |
| seat 1 | 20.86% |
| seat 2 | 26.96% |

- **Its answers:** pack 7,701 (79%), chaal 2,024, see 12, show 5, and **raise 0**.
- On look items it always answers `see`: turn 3 is 100% (n = 4), turn 4 is 0%
  (n = 8).
- **Illegal answers: 0.**

### 7.6 Training loss [R, `train_log.jsonl`, 125 points]

Weighted cross-entropy on the answer tokens:

| step | 10 | 50 | 100 | 150 | 250 | 500 | 750 | 1000 | 1250 |
|---|---|---|---|---|---|---|---|---|---|
| loss | 0.8135 | 0.4644 | 0.3727 | 0.2149 | 0.1531 | 0.1194 | 0.0661 | 0.0527 | 0.0272 |

- The minimum logged loss is 0.0208, at step 1150.
- The loss falls steeply over the first ~250 steps, then slowly.
- It never rises for long; batch-to-batch noise is about ±0.02. This fits a
  sequential sampler where consecutive batches come from different components.
- The learning rate peaked at 2.0e-4 at step 40 and decays to about 0 by step 1250.

### 7.7 Validation trajectory [R, `val_log.jsonl`; 1,998 items, legal-action first-token scoring]

| step | acc | bal | chaal | pack | raise | show | see | stay-blind | blind fold | fold weak / strong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 22.8 | 32.0 | 24 | 68 | 0 | 0 | 100 | 0 | 82% | 64 / 83 |
| 250 | 86.1 | 70.2 | 73.4 | 85.4 | 89.9 | 97.4 | 0 | 75.0 | 0% | 77.0 / 2.2 |
| 500 | 90.9 | 75.5 | 80.0 | 93.9 | 92.8 | 99.0 | 25 | 62.5 | 0% | 82.1 / 2.7 |
| 750 | 93.5 | 87.6 | 85.4 | 96.8 | 94.7 | 98.4 | 100 | 50.0 | 0% | 78.3 / 3.1 |
| 1000 | 95.6 | 91.0 | 90.1 | 97.5 | 96.4 | 99.6 | 100 | 62.5 | 0% | 76.4 / 3.3 |
| **1250** | **96.4** | **93.6** | 90.8 | 98.3 | 97.5 | 100 | 100 | 75.0 | 0% | 74.5 / 3.0 |
| full val (9,742), best adapter [U] | 96.9 | 93.8 | 92 | 98 | 98 | 100 | 100 | 75 | 0% | 77 / 4 |
| **answer key** [V] | | | | | | | | | | **73.7 / 3.0** |

**Observations**

1. **Every check was a new best**, so `best_adapter/` is the final step 1250.
   Validation never fell, so there's no sign of overfitting in one epoch, and the
   model was still improving at the end.
2. **Ordinary betting improves steadily.** chaal is the slowest (73 → 91). It is the
   middle band, with a boundary on each side, and it remains the largest source of
   errors on the test (§8).
3. **The fold rates converge on the key's.** The final 74.5% / 3.0% matches the
   answer key's 73.7% / 3.0% almost exactly.
   - This is the clearest evidence that the model uses its cards: the base model
     had them inverted (64 / 83).
   - "Fold weak" is not meant to reach 100%, because 26% of weak hands have a
     non-fold correct answer.
4. **The balanced-accuracy jumps are small-sample noise.** The subset's 12 look
   prompts are **6 nodes × 2 menu orders** (§8.6), and `see` has only 4 prompts
   (2 nodes).
   - "see 25%" at step 500 is one ordering of one node.
   - The jump from 75.5 to 87.6 (steps 500 → 750) is mostly two nodes flipping to
     `see`, worth +12.5 balanced points on their own.
   - Conversion prompts correct: 6 → 6 → 8 → 9 → 10 of 12.
5. **Checkpoint selection by balanced accuracy is fragile.** It gives the 4 `see`
   prompts the same weight as 534 `chaal` prompts. It made no difference here,
   because accuracy and balanced accuracy peaked at the same step. Plain accuracy,
   or seen-only balanced accuracy, is the safer selection metric.
6. **The subset tracked the full set well:** 96.4 on the subset vs 96.9 on the full
   validation set.

### 7.8 Is the 96% real? The leakage audit [V]

| check | result |
|---|---|
| Validation prompts also in training | 0 |
| Seen validation items whose (node, hand class) is in training | 0 |
| Seen validation items whose *betting node* is in training | 1,893 / 1,896 (99.8%), median 267 training hands per node |
| Blind validation nodes in training | 0 (all blind situations held out) |
| **Baseline: answer each node's most common training label, ignoring cards** | **66.9%** |
| **Baseline: copy the answer of the closest-strength training hand at the same node** | **99.0%** |
| Fine-tuned model at step 500 | 90.9% (final: 96.4%) |

**Conclusion:** there is no leakage.
- The task is learnable to this level for seen items, because the betting situations
  repeat with different hands and the rule is a per-node strength cutoff (§2.5).
- The model reaches 96–97%, still below the 99.0% of a simple lookup rule that uses
  the cards.
- The benchmark eval has the same structure (100% of its seen nodes appear in
  training), so seen-betting gains should largely carry over.
- Blind items are the only true test of generalising to new situations.

---

## 8. Internal-test deep analysis of the fine-tuned model

Source: `preds_llama3.2_3b_ft_full_l4_internal_test.jsonl` (free generation, greedy,
16 tokens) and the split manifest. All numbers [V] unless marked.

### 8.1 Headline [R]

| | all | seen | look | blind betting | blind, all |
|---|---:|---:|---:|---:|---:|
| one-word baseline | 26.59 | 26.87 | 66.67 | 66.67 | — |
| base | 24.71 | 24.88 | 33.33 | 5.56 | 8.82 |
| **fine-tuned** | **96.52** | **96.56** | **75.00** | **95.56** | 93.14 |

- **Illegal answers: 0.**
- **By seat:** seat 1 94.63%, seat 2 97.62%. Base: 20.86% and 26.96%.
- **Reporting note:** the notebook's "blind" column is blind betting only (95.56).
  The `metrics.py` report's "blind" is all blind items (93.14). Label them clearly
  when citing.

### 8.2 By data component

| component | n | fine-tuned | base |
|---|---:|---:|---:|
| A blind | 102 | 93.14 | 8.82 |
| B contrast groups | 2,400 | 96.75 | 38.12 |
| C·F1 pack on `[pack, chaal]` | 624 | **100.00** | 40.87 |
| C·F2 raise legal, not correct | 1,187 | 96.29 | 40.10 |
| C·F3 show correct | 795 | 99.25 | **0.25** |
| C·F4 strong, raise or show correct | 628 | 98.25 | **0.00** |
| D balancer | 4,006 | 95.18 | 18.72 |

Every failure the components were built to fix is fixed on this internal test:

| failure | before | after |
|---|---:|---:|
| never folds on `[pack, chaal]` | 40.9% | 100% |
| show collapse | 0.25% | 99.25% |
| folding strong hands | 0.0% | 98.25% |
| raise reflex | 40.1% | 96.3% |

**Caveat:** there was no ablation. This doesn't show that the component design
helped more than an unstructured dataset of the same size would.

### 8.3 Recall and confusion

| correct answer | n | fine-tuned recall | base recall |
|---|---:|---:|---:|
| chaal | 2,513 | **91.96** | 25.31 |
| pack | 2,590 | 97.53 | 68.11 |
| raise | 2,089 | 97.32 | 0.00 |
| show | 2,538 | 99.45 | 0.12 |
| stay-blind | 8 | 87.50 | 0.00 |
| see | 4 | 50.00 | 100.00 |

**Largest confusions** (339 errors in total):

| correct → answered | count |
|---|---:|
| chaal → raise | 100 |
| chaal → pack | 77 |
| pack → show | 31 |
| raise → pack | 26 |
| chaal → show | 25 |
| raise → chaal | 24 |
| pack → chaal | 23 |

- chaal errors spread to both neighbours in strength, pack and raise. chaal accounts
  for 202 of the 339 errors.
- **Answer mix:** pack 2,639, show 2,586, chaal 2,361, raise 2,144. The key has
  pack 2,590, show 2,538, chaal 2,513, raise 2,089, so the model slightly
  under-predicts chaal.

### 8.4 The errors are now near-misses (a qualitative change from zero-shot)

- **305 of the 339 errors (90.0%) are the solver's second choice.**
- The mean solver probability of a wrong answer is **0.271**.
- Mean `label_prob` is 0.779 on correct items and **0.685** on wrong ones.

Accuracy by solver confidence in the correct label:

| label_prob | n | acc |
|---|---:|---:|
| < 0.6 | 2,344 | **93.47** |
| 0.6–0.7 | 1,638 | 95.97 |
| 0.7–0.8 | 1,078 | 98.33 |
| 0.8–0.9 | 1,233 | 97.49 |
| 0.9–0.99 | 1,640 | 97.01 |
| ≥ 0.99 | 1,809 | 98.78 |

**Inference:** zero-shot errors were flat across confidence and often "the solver
never plays this". Fine-tuned errors cluster where the solver is nearly indifferent,
which is the signature of a model that learned the decision boundaries. The residual
error sits largely in labels that are themselves argmaxes of mixed strategies.

### 8.5 Accuracy by game state

| feature | values → accuracy (n) |
|---|---|
| menu size | 2 → **98.97** (1,065) · 3 → 97.31 (3,861) · 4 → **95.35** (4,816) |
| own turn, seen | 1 → 99.38 (323) · 2 → 97.39 (2,070) · 3 → 96.93 (4,263) · 4 → 95.14 (2,984) |
| opponent seen | no → 97.93 (2,752) · yes → 95.97 (6,990) |
| actions to cap | 1 → 98.66 · **2 → 92.78 (1,855)** · 3 → 97.17 · 4 → 96.12 · 5 → 97.44 · 6 → 97.39 · 7 → 99.59 · 8 → 98.78 |
| hand bucket (groups of 5) | 95.88 · 97.03 · 95.96 · 97.18 · 96.41 (flat) |

- Accuracy falls with menu size and with depth, as expected: more options, and
  longer histories.
- **Anomaly:** accuracy dips to 92.78% at 2 actions to cap, 4–6 points below its
  neighbours, on a large n. It hasn't been investigated.
  - Hypothesis: near the cap, the forced-showdown rule changes what chaal and show
    are worth, and the prompt only implies this through "N betting actions remain".
- Accuracy is flat across hand strength, so the model isn't systematically worse on
  weak or strong hands.

### 8.6 Conversion and blind items, node by node, and menu-order sensitivity

**Correction.** Blind copies in the fine-tuning file are **not** identical: every
repeat gets a different menu order (`meta.action_order` is set on all 3,840 blind
rows). Distinct prompts per blind node:

| node type | distinct prompts (the rest are duplicates) |
|---|---|
| look | 2 |
| 3-action blind betting | 6 |
| 4-action blind betting | 24 |

So the internal test's "12 conversion questions" are **6 situations, each asked in 2
menu orders**, and its 90 blind-betting prompts are 6 situations.

**Look nodes (test):**

| node | seat | turn | key (p) | base (both orders) | fine-tuned (two orders) |
|---:|---:|---:|---|---|---|
| 1218 | 2 | 3 | see (0.704) | see, see | **see, see** ✓ |
| 2701 | 1 | 3 | see (0.580) | see, see | stay-blind, stay-blind ✗ |
| 2512 | 2 | 4 | stay-blind (0.749) | see, see | stay-blind, stay-blind ✓ |
| 2686 | 1 | 4 | stay-blind (0.875) | see, see | **stay-blind, see** (order-dependent) |
| 2942 | 1 | 4 | stay-blind (1.000) | see, see | stay-blind, stay-blind ✓ |
| 3152 | 2 | 4 | stay-blind (0.938) | see, see | stay-blind, stay-blind ✓ |

**Situation-level conversion:**

| policy | situations correct (of 6) | score |
|---|---:|---:|
| fine-tuned | 4.5 | 75% |
| always stay-blind | 4 | 66.7% |
| base (always see) | 2 | 33% |

- The fine-tuned model's one clear gain over the constant is node 1218, a `see` at
  turn 3.
- It misses the other turn-3 `see` (node 2701, whose label has the lowest
  confidence, p = 0.58).
- **With n = 6, this does not show that timing was learned.**

**Blind-betting nodes (test):**

| node | key | prompts | fine-tuned answers |
|---:|---|---:|---|
| 2651 | chaal | 24 | chaal 20, **raise 4** |
| 3024 | chaal | 24 | chaal 24 |
| 3530 | chaal | 6 | chaal 6 |
| 3570 | show | 24 | show 24 |
| 3808 | chaal | 6 | chaal 6 |
| 5165 | show | 6 | show 6 |

5 of 6 situations are fully consistent and correct. It never folds a blind hand
(0/90). Compare the zero-shot models, which folded blind hands up to 101 times in 138.

**Menu-order sensitivity remains after training on randomised orders.**
- 2 of the 12 held-out blind situations get different answers under different menu
  orders: node 2686 (1 of 2 orders) and node 2651 (4 of 24).
- This is also visible during training: "see 25%" at step 500 was one ordering of
  one node.
- The shuffled benchmark eval will measure this properly on 300 blind items.

**The per-turn label inversion appears here too.** In this test, turn 3 = `see` and
turn 4 = `stay-blind`, the opposite of the headline reach-weighted pattern. This is
consistent with §6.

---

## 9. Engineering and debugging incidents that affect reproducibility

| # | incident | effect | fix / status |
|---|---|---|---|
| 1 | Kaggle prediction files used `answer`, and had a metadata line; the scorer reads `action` | would have scored 0 | normalised and remapped; later notebooks keep `answer`, remapped at scoring |
| 2 | Prediction files were replaced mid-session (18:14, 14 Sep) | stale scores | all re-scored |
| 3 | The first Mistral file was empty | | re-checked after re-upload |
| 4 | The normalisation script vanished from the scratchpad; three models were silently scored from stale files | wrong numbers | regenerated and verified |
| 5 | The best-constant baseline for conversion first omitted `stay-blind` (reported as 37.04%) | understated the baseline | corrected to **62.96%** |
| 6 | Batch notebook: Qwen3-14B and Phi-4 ran out of memory on a 40 GB A100 [R, run_log] | skipped | the previous model wasn't freed (`free(model)` only deleted a local name). Now `model = tok = generate = build = None` before load and in `finally`. Both then ran. |
| 7 | Batch summary overwrote itself with only the `RUN_ONLY` models | lost rows | the summary loops over all `MODELS` |
| 8 | Qwen3-30B-A3B skipped on a 40 GB A100 ("needs ~64 GB, GPU has 42 GB") [R] | | ran on an 80 GB A100 |
| 9 | Gemma 4 E2B needs transformers ≥ 5.5 and a multimodal loader class | | version gate plus a loader chain; ran on an L4 |
| 10 | Llama's chat template puts **today's date** in the system turn | prompts differ from day to day | fine-tuning fixes `date_string = "26 Jul 2024"`. **The zero-shot batch Llama run used its run-day date**, so the new eval notebook re-runs the base model with the training prompt. |
| 11 | Fine-tuning notebook: default batch 8×8 changed to 16×4 for the A100; the gradient-checkpointing flag is inconsistent in `run_config` | provenance | recorded here (§7.3) |
| 12 | Sanity check too short (24 steps) | false FAIL | redesigned (§7.4) |
| 13 | Dashboard: a fake IPython module and a matplotlib backend dependency broke rendering | | render to PNG, then `display(Image)` / update |
| 14 | `pip install -U torchao` added as the first code cell of all four notebooks, at the user's request (30 Sep) | environment | cause not recorded |
| 15 | `metrics.parse_action` docstring contradicts its code on "see: no, stay-blind" (the code returns the first legal hit, `see`) | documentation | flagged, not fixed |
| 16 | `gate_phase2.py` gate 1 has an import bug (`test_game`) | | flagged, not fixed |
| 17 | An "extended test set" (`make_eval_extended.py` plus two files) was created | | removed at the user's request ("undo") |

---

## 10. Hypotheses: status

| # | hypothesis | status | evidence |
|---|---|---|---|
| H1 | (Phase 3) Models do relatively well on ordinary betting and fail on conversion | **Half confirmed.** Conversion fails for all ten; ordinary betting is only +0 to +10 over baseline, so they're weak everywhere | §3.3, §3.4 |
| H2 | Menu position explains show collapse and Llama's missing raises | **Refuted** by the shuffled twin: models follow the word, and order is worth 1.3–2.2 points | §3.4 |
| H3 | Scale improves play | **Not supported** (flat Qwen3 4B–14B; the 30B MoE collapses) | §3.4 |
| H4 | Models fail because they don't read their cards | **Supported for most;** three models read cards but misuse uncertainty (fold while blind) | §3.4 |
| H5 | (FINETUNE_STRATEGY) Fine-tuning fixes seen betting; conversion moves only as far as its prior | **Consistent so far** (internal test: seen 96.56; conversion 4.5 of 6 situations vs 4 for the constant). The benchmark eval is pending | §8 |
| H6 | Weighted loss, contrast groups and failure components help | **Untested.** No ablation was run | — |
| H7 | Training on randomised orders makes the model order-invariant | **Partly refuted:** 2 of 12 held-out blind situations flip with menu order | §8.6 |
| H8 | The per-turn conversion cells test non-monotone timing | **Refuted** as stated. Item labels are monotone by turn; the non-monotone pattern appears only when weighted by reach | §6 |
| H9 | The fine-tuned model's high validation score means leakage | **Refuted:** 0 overlap; the card-ignoring baseline is 66.9%, a strength-lookup rule gets 99.0% | §7.8 |

---

## 11. Corrections register

Claims made during the project that are wrong. Don't cite them.

| claim | where | correction | evidence |
|---|---|---|---|
| Qwen3-8B's show recall falls "50× from one slot" because of menu position | error_analysis §4 | the driver is whether raise is offered (53.0% vs 2.4%) | shuffled twin, §3.4 |
| Llama's zero raises and shows are a layout artefact ("never selects past index 1") | error_analysis §5, error_situations §3.4, RESULTS_EXPLAINED §10a, training_coverage §6–7, training_signal §6.4 | a genuine behaviour: it never raises even when raise is listed first | §3.4 |
| Randomising order is the single highest-value fix | FINETUNE_STRATEGY, training_signal §6.3, training_coverage §7 | hygiene, worth 1.3–2.2 points | §3.4 |
| Pooled strength order is pack < chaal < show < raise | training_signal §4 (since corrected) | per-node order is pack < **show < chaal** < raise (98% of 103 nodes) | §2.4 |
| "A flat always-stay-blind gets turn 3 right and turn 4 wrong" | FINETUNE_STRATEGY §7 | on the eval, always-stay-blind scores 49.1% at turn 3 and 80.2% at turn 4 | §6 |
| "Turn 3 is the most informative stratum: the solver says don't look" | CLAUDE.md (Phase 4 notes) | true reach-weighted for seat 1 only; the item labels at turn 3 are 27 see / 26 stay-blind | §6 |
| Blind copies in the fine-tuning file are "word-for-word identical", and the 258 removed test rows were identical copies | said during the session | copies differ in menu order; the removed rows are copies with the *same* order; the 12 "distinct" look prompts are 6 situations × 2 orders | §8.6 |
| "Conversion rests on 12 questions" (internal test and validation) | said during the session | 12 prompts, **6 situations** | §8.6 |
| Step-750 checkpoint "along with the others" kept on Drive | said during the session | only the last 3 (750, 1000, 1250) are kept; 250 and 500 were deleted | `save_total_limit=3` |
| Conversion baseline 37.04% | early Phase 4 scoring | 62.96% (always stay-blind) | §9 #5 |

The five older docs still contain the first three claims. `RESULTS_NINE_MODELS.md` §11
lists them, but the documents themselves have not been edited.

---

## 12. Open items and next experiments

1. **Benchmark eval of the fine-tuned adapter.**
   `notebooks/teenpattibench_eval_finetuned_llama3.2_3b.ipynb` is written and tested
   offline except the GPU cells. Not yet run.
   - It scores v1 and shuffled, plus the base model with the same training prompt.
   - The comparison to report is the fine-tuned adapter vs **base with the training
     prompt**, not vs the 26.78% batch run (see §9 #10).
   - Predictions, based on the internal test: seen betting high. Conversion should be
     read against 62.96% and per turn with §6 in mind. Blind betting should be well
     above 66.67% if the "never fold blind" behaviour transfers.
2. **Reach-weighted conversion score, and a high-reach subset** (§6). A scoring
   change only.
3. **Gemma 4 E2B per-turn and per-seat conversion breakdown,** before interpreting
   its above-floor 64.81 / 67.28.
4. **Ablations for H6:** the same run with (a) no weights, (b) no contrast grouping,
   i.e. shuffled sampling, and (c) an unstructured, label-balanced 100k.
5. **Checkpoint selection:** switch to accuracy or seen-only balanced accuracy (§7.7).
6. **The actions-to-cap = 2 dip** (§8.5).
7. **The seat gap:** seat 1 is lower after fine-tuning (94.6 vs 97.6); zero-shot
   Qwen3-8B showed the reverse.
8. **A no-menu ablation** (legal-actions line removed), proposed as an option in
   `prompts.py` with the default kept byte-identical. Not built. The fine-tuned model
   has never seen that format.
9. **Continued training / a second epoch:** validation was still rising at step 1250.
   Fresh seen data is plentiful (about 800k unused seen infosets); **no new blind data
   exists**.
10. **Correct the five older docs** (§11), fix the `parse_action` docstring and the
    `gate_phase2` import, add confidence intervals to `metrics.py`, and make one flag
    drive gradient checkpointing.
11. **Uncommitted work:** `src/generate.py` (modified), `src/generate_finetune.py`,
    `docs/FINETUNE_STRATEGY.md`, `docs/RESULTS_NINE_MODELS.md`,
    `docs/training_signal.md`, this log, three notebooks and `data/*_stats.json`.

---

## 13. Artefact index

| artefact | location | notes |
|---|---|---|
| Solver checkpoint | `data/solution_default.pkl` (107 MB, gitignored) | the answer key |
| Eval v1 / shuffled | `data/teenpattibench_eval.jsonl`, `data/teenpattibench_eval_shuffled.jsonl` | committed |
| 100k training set | `data/teenpattibench_train_100k.jsonl` | SHA `a1b3c829…3c18053`, gitignored |
| Fine-tuning set | `data/teenpattibench_finetune.jsonl` | SHA `aeea7b5a…5c11`, gitignored; stats file committed |
| Zero-shot predictions (ten models × 2) | `results/teenpattibench_results (3)/` | includes Gemma 4 E2B and `run_log.jsonl` |
| Superseded predictions | `results/old/` | Kaggle fp16 T4; Gemma 3 4B on L4 |
| Fine-tuning run | `results/llama3.2_3b_full_finetune_results/` | adapter, `run_config.json`, `split_manifest.json`, `train_log.jsonl`, `val_log.jsonl`, `training_progress.png`, internal-test predictions and reports |
| Notebooks | `notebooks/teenpattibench_batch_eval.ipynb`, `teenpattibench_finetune_llama3.2_3b.ipynb`, `teenpattibench_eval_finetuned_llama3.2_3b.ipynb`, `teenpattibench_gemma3_4b.ipynb` | |
| Generators | `src/generate_train_100k.py`, `src/generate_finetune.py`, `src/make_eval_shuffled.py` | seeds 20260912, 20260928, 20260914 |
| Seeds | split 20260929; validation subset 20260930 (`SEED + 1`); within-group shuffle 20260930 (`SEED + 2` of the generator's seed 20260928) | |

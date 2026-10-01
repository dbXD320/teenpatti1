# What the Fine-Tuned Llama 3.2 3B Gets Wrong on TeenPattiBench

**Model.** Llama 3.2 3B Instruct + LoRA adapter (step 1250, adapter SHA-256
`d07beee5…9096`), bf16, run on an NVIDIA L4. Greedy decoding, 16 new tokens, with
the training prompt (`date_string = "26 Jul 2024"`).

**Data.** `results/teenpattibench_finetuned_eval_results/` (the v1 and shuffled
benchmark evals). Training data comes from the 80k training split of
`teenpattibench_finetune.jsonl` (`split_manifest.json`). Hand strength is the exact
equity of each hand class from `abstraction.build_abstraction()`. Every number here
was computed from the raw files, and the eval was never used in training.

**Score.** v1 **89.22%** (1,606 of 1,800 graded), shuffled 89.83%. Base Llama with
the same prompt: 27.28%.

**Bottom line.** The 194 errors are almost all near-misses. Two failures stand out:
1. **Bluff raises.** The solver raises some weak hands. In this eval that is 121 of
   the 375 raise-correct seen items, almost a third. Where those bluffing hands sit
   in narrow, irregular bands, the model folds them instead.
2. **Conversion where looking is right but rare in training:** turn 1 (0 of 3),
   and seat 1 at turn 4.

There are no illegal answers. Menu order barely matters, and the model reads its
cards about as much as the solver does.

---

## 1. Where the errors are

| family | graded | wrong | accuracy | one-word baseline |
|---|---:|---:|---:|---:|
| seen betting | 1,500 | 152 | 89.87 | 25.00 |
| conversion (look) | 162 | 32 | 80.25 | 62.96 |
| blind betting | 138 | 10 | 92.75 | 66.67 |
| **all** | **1,800** | **194** | **89.22** | 25.94 |

**Confusions** (correct → answered):

| | count |
|---|---:|
| **raise → pack** | **45** |
| raise → chaal | 26 |
| chaal → pack | 26 |
| pack → show | 21 |
| **see → stay-blind** | **21** |
| chaal → raise | 20 |
| stay-blind → see | 11 |
| chaal → show | 8 |
| pack → chaal | 6 |
| raise → show | 5 |
| show → chaal | 3 |
| pack → raise | 2 |

**Seen recall:** show 99.7 · pack 92.3 · chaal 86.9 · **raise 80.5**. Raise
accounts for 73 of the 152 seen errors.

---

## 2. The errors are near-misses

- **186 of 194 wrong answers (95.9%) are the solver's second choice.**
- The mean solver probability of the answer the model gave is 0.301.
- 114 errors are actions the solver plays at least 30% of the time. Only 14 are
  actions it plays under 1%.

Accuracy rises steadily with the solver's confidence in the correct answer:

| solver's confidence in the correct answer | n | accuracy |
|---|---:|---:|
| < 0.6 | 550 | 83.45 |
| 0.6–0.7 | 305 | 87.87 |
| 0.7–0.8 | 195 | 90.77 |
| 0.8–0.9 | 174 | 90.23 |
| 0.9–0.99 | 252 | 93.25 |
| ≥ 0.99 | 324 | 95.68 |

**Inference:** this is the opposite of the zero-shot models. Their errors were flat
across confidence, and 35–44% of them were actions the solver almost never plays.
Most of the fine-tuned model's remaining errors sit where the solver itself mixes
between two actions.

---

## 3. Failure 1: bluff raises

**The evidence**

- **121 of the 375 raise-correct seen items have equity below 0.5.** These are
  bluffs. The model raises 85 of them, **folds 34**, and calls 2.
- **Its 73 raise misses have median equity 0.500.** The raises it gets right have
  median equity 0.788.
- **Misses by its answer and by hand strength:**

  | model answered | equity < 0.5 | 0.5–0.8 | ≥ 0.8 |
  |---|---:|---:|---:|
  | pack | 34 | 11 | 0 |
  | chaal | 2 | 19 | 2 |
  | show | 0 | 5 | 0 |

- **Training rarely labels weak hands raise.** Of the 21,458 training hands with
  equity below 0.5, only 6.7% are raise. The eval is balanced across labels, so it
  contains far more bluff raises than training does.
- **The worst four nodes are bluff nodes,** almost all errors raise → pack:

  | node | wrong / items | errors |
  |---:|---:|---|
  | 1598 | 12 / 15 | raise → pack 7, raise → show 5 |
  | 1395 | 12 / 17 | raise → pack 11 |
  | 1359 | 9 / 13 | raise → pack 9 |
  | 1306 | 8 / 47 | raise → pack 7 |

**What the training data looks like at those nodes.** Here are the training labels
for hands with equity below 0.6, ordered by strength (p = pack, R = raise, s = show):

```
node 1598  ppRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRppRRRRppppppppppppppppppsspppppppppppppps
node 1395  ppppRRRRRRRRpRRRRRRRRRppppppppRRppRpppppppppppppppppppppppppp
node 1359  ppRRRRRRRRRRRRRRRpppppppppppppppppRRppppRppRppRp
```

- **Equity isn't the right ruler here.** The solver folds the very weakest hands,
  raises a band of weak hands as a bluff, then folds the next band up. Measured by
  equity, the bluffing hands sit inside the fold region rather than below it. That
  is a non-monotone rule, unlike the simple cutoffs at most nodes.
- **The bands don't line up with equity.** At node 1359 the eval's raise items have
  equity 0.51–0.67, which in training falls in the "pack" stretch. So the bluff
  hands are picked by card composition (blockers and card removal), not raw
  strength.
- **12 training nodes** split their weak hands between raise and pack, at least 3
  of each, switching back and forth a median of 7.5 times along the strength order.
- **When there's enough data, it's learnable.** At node 1306, with 585 training
  hands and 109 bluff raises, the model gets every bluff from equity 0.02 to 0.37
  right. It only fails at the edge of the band (equity 0.44–0.49), where training
  and eval switch to pack.

**How much a lookup rule fails here too.** The rule "copy the label of the
nearest-strength training hand at the same node" scores 95.26% on eval seen items
(the model gets 89.87%).
- In 52 of the model's 152 seen errors, that rule is also wrong.
- In 49 of them, the model's wrong answer *is* the nearest training hand's label.

So about a third of the seen errors are places where the training data, read by
equity, points to the model's answer.

**Turn-3 dip.** Seen accuracy by turn is 95.0 / 94.0 / **84.7** / 91.2. Turn 3's
89 errors are mostly bluff misses (raise → pack 37, raise → chaal 17). It isn't a
seat effect: seat 1 and seat 2 both score 84.7.

**Fold and raise rates by strength band** (seen items):

| band | model folds | key folds | model raises | key raises |
|---|---:|---:|---:|---:|
| 0–4 | 66.3 | 55.6 | 16.1 | 20.5 |
| 5–9 | 45.0 | 37.9 | 13.4 | 17.8 |
| 10–14 | 30.7 | 26.1 | **10.9** | **20.2** |
| 15–19 | 12.4 | 13.2 | 16.5 | 20.9 |
| 20–24 | 5.2 | 8.2 | 44.5 | 40.9 |

The model folds weak and middling hands too often and bluffs too little. With strong
hands it is about right.

---

## 4. Other seen-betting patterns

| feature | accuracy (n) |
|---|---|
| menu `[pack, chaal]` | 95.96 (99) |
| menu `[pack, chaal, show]` | 93.16 (234) |
| menu `[pack, chaal, raise]` | 90.54 (370) |
| menu, all four actions | **87.83 (797)** |
| actions to cap 3–4 | 84.7 (583) |
| actions to cap 1 | 97.0 (134) |
| actions to cap 7 | 96.6 (117) |
| history length 8–9 actions | 83.8 (457) |
| history 3–5, or 12 | 95–97 |
| opponent seen vs blind | 89.0 vs 91.7 |
| seat 1 vs seat 2 | 88.1 vs 90.8 |
| hand strength (groups of 5 buckets) | 88.3–90.9 (flat) |
| stake 4 (no raises left) | 94.0 vs 87.7–89.8 |

- Mid-depth lines with a full menu are where bluff raising happens, and so are most
  errors.
- When no raises remain (stake 4) there is nothing to bluff with, and accuracy is
  94.0%.

**Card sensitivity.** Across the 74 nodes with at least 6 different hands, the model
gives one identical answer at 12, against the solver's 11. Base Llama does so at 70
and the best zero-shot model (Phi-4) at 10. The fine-tuned model reads its cards.

---

## 5. Why the benchmark (89.9% seen) is harder than the internal test (96.6% seen)

| | eval seen | internal test seen |
|---|---:|---:|
| labels with solver confidence < 0.6 | **31.1%** | 24.0% |
| mean solver confidence | 0.739 | 0.776 |
| raise share | **25.0%** | 21.7% |
| nearest-strength-hand rule | **95.26%** | 99.0% (on validation) |

The eval has the same structure (1,497 of 1,500 seen items sit at a node in the
training split, a median of 352 training hands each). But its label balancing
oversamples low-confidence labels and bluff raises. Even the lookup rule drops about
4 points on it. **This is a harder mix of questions, not a sign of overfitting.**

---

## 6. Failure 2: conversion

32 of 162 wrong: see → stay-blind 21, stay-blind → see 11.

**Accuracy by turn:**

| own turn | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| `see` items the model got right | **0 / 3** | 12 / 13 | 21 / 27 | **6 / 17** |
| `stay-blind` items the model got right | 1 / 1 | 5 / 6 | 20 / 26 | 65 / 69 |
| how often the model says `see` | **0.0%** | 68.4% | 50.9% | 11.6% |
| how often the key says `see` | 75.0% | 68.4% | 50.9% | 19.8% |

- **The model isn't answering one word for everything,** unlike every zero-shot
  model: 68 of the 76 turn 2–3 items are right. Its see-rate at turns 2 and 3 equals
  the key's, while it is correct item by item.
- **Turn 1: it never looks.**
  - All three turn-1 `see` items are missed. One of them is the highest-reach
    decision in the whole stratum: seat 2's first decision (reach 1.0, solver
    p(see) = 0.995).
  - The training split contains **no** turn-1 `see` situation. Its only turn-1 look
    node is labelled stay-blind.
  - **This is a data gap, not a reasoning failure.** Only 5 turn-1 blind situations
    exist in the game, and the eval took 4.
- **Turn 4: it under-calls looking.** 11 of the 17 `see` items are missed (10 of
  them seat 1). Training has 7 distinct turn-4 `see` nodes against 24 turn-4
  `stay-blind` nodes, and the model leans stay-blind.
- **Errors happen where the solver is unsure:** the median solver confidence on
  wrong conversion items is 0.64, against 0.98 on correct ones.
- **The opponent matters:** 96.3% when the opponent is still blind (n = 54), 72.2%
  when the opponent has looked (n = 108).
- **Weighting by how often each question is reached in play:**
  - The fine-tuned model scores **35.3%**; always answering `see` scores 99.9%.
  - The miss on the reach-1.0 turn-1 item dominates this score.
  - It rests on very few high-reach items, so read it as a warning, not a
    measurement (see RESEARCH_LOG §6).

---

## 7. Blind betting (10 wrong)

| correct → answered | n | where |
|---|---:|---|
| chaal → show | 3 | seat 1 turn 3; two of them are confident labels (0.99, 1.0) |
| raise → chaal | 3 | seat 1 turns 3–4; solver confidence 0.51–0.55, near coin-flips |
| show → chaal | 2 | seat 2 **turn 1**, confidence 0.85 and 0.93 |
| chaal → raise | 2 | seat 1 turn 4, confidence 0.53–0.60 |

It **never folds a blind hand** (0 of 138). The zero-shot models folded up to 101 of
138. Its blind answer mix (chaal 92, show 34, raise 12) almost exactly matches the
key's (92 / 33 / 13). The two confident turn-1 misses fit the same turn-1 coverage
gap as conversion.

---

## 8. Menu order

- **Same answer under both orderings:** 1,761 of 1,800 (97.8%). The zero-shot
  models ranged from 72% to 93%.
- **Errors:** 170 are wrong under both orderings, 24 only on v1 and 13 only on
  shuffled. **Almost all errors are about the situation, not the menu.**
- **Answers that change with menu order:**
  - 33 seen items.
  - 6 blind items: 3 look and 3 blind betting. For example, a chaal-correct blind
    item became raise when raise moved earlier in the menu.

---

## 9. What would fix what

| failure | cause | lever |
|---|---|---|
| Bluff raises folded (34 at equity < 0.5; 4 worst nodes) | non-monotone, card-composition bands; only 6.7% of weak training hands are raise | upweight or oversample raise-correct hands with equity < 0.5 (as contrast groups at bluff nodes); a second epoch; auxiliary features (hand category) only if that fails |
| Middling hands folded, under-bluffing at bands 10–14 | same as above | same as above |
| Turn-1 look never chosen | **0 training examples** | can't fix with data from the current split. Only 5 such situations exist and the eval holds 4. Moving one into training would remove it from the eval. Report turn 1 as a control. |
| Turn-4 `see` under-called (seat 1) | 7 vs 24 distinct turn-4 nodes for see vs stay-blind | weight `see` inside the look family; the other option, moving the train/eval blind split, changes the benchmark |
| Conversion against a seen opponent (72%) | not diagnosed | inspect the 108 items |
| Low-confidence seen labels (83% below 0.6) | the solver mixes there | expected; consider soft-label training with `solver_distribution` |

**Not problems:** illegal answers (0), menu order (97.8% the same), reading cards
(12 of 74 nodes with one answer, against the solver's 11), and folding blind (0).

---

## 10. Per-action scorecard

### 10.1 Confusion matrix, all 1,800 graded items (v1)

Rows are the correct answer and columns are what the model answered.

| correct ↓ / answered → | pack | chaal | raise | show | see | stay-blind | total | right | wrong | **recall** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **pack** | **346** | 6 | 2 | 21 | 0 | 0 | 375 | 346 | 29 | **92.3** |
| **chaal** | 26 | **413** | 20 | 8 | 0 | 0 | 467 | 413 | 54 | **88.4** |
| **raise** | 45 | 26 | **312** | 5 | 0 | 0 | 388 | 312 | 76 | **80.4** |
| **show** | 0 | 3 | 0 | **405** | 0 | 0 | 408 | 405 | 3 | **99.3** |
| **see** | 0 | 0 | 0 | 0 | **39** | 21 | 60 | 39 | 21 | **65.0** |
| **stay-blind** | 0 | 0 | 0 | 0 | 11 | **91** | 102 | 91 | 11 | **89.2** |
| times answered | 417 | 448 | 334 | 439 | 50 | 112 | 1,800 | 1,606 | 194 | |
| **precision** | 83.0 | 92.2 | 93.4 | 92.3 | 78.0 | 81.2 | | | | |
| **F1** | 87.4 | 90.3 | 86.4 | 95.6 | 70.9 | 85.0 | | | | |

**Accuracy 89.22 · balanced accuracy 85.77 · macro-F1 85.94.**

- The model never confuses a betting action with a look action, in either direction.
- **Recall** is how many of the questions with that correct answer it got right.
  **Precision** is how many of the times it gave that answer were right.
- Pack has the lowest betting precision (83.0%): 71 of its 417 folds should have
  been something else (45 raise, 26 chaal).
- Raise has the highest precision (93.4%) but the lowest betting recall (80.4%).
  When it raises it is nearly always right, but it doesn't raise often enough.

### 10.2 The same on the shuffled eval

| | pack | chaal | raise | show | see | stay-blind |
|---|---:|---:|---:|---:|---:|---:|
| right / total | 347/375 | 408/467 | **326/388** | 405/408 | 40/60 | 91/102 |
| recall | 92.5 | 87.4 | **84.0** | 99.3 | 66.7 | 89.2 |
| precision | 86.3 | 92.1 | 92.6 | 91.8 | 78.4 | 82.0 |

Accuracy 89.83, balanced 86.51, macro-F1 86.67. The only noticeable change is raise:
14 more right and 15 fewer raise → pack errors.

### 10.3 By question type (v1)

| | n | accuracy | balanced | per-action right / total |
|---|---:|---:|---:|---|
| seen betting | 1,500 | 89.87 | 89.87 | pack 346/375 · chaal 326/375 · **raise 302/375** · show 374/375 |
| blind betting | 138 | 92.75 | 88.48 | chaal 87/92 · **raise 10/13** · show 31/33 |
| conversion | 162 | 80.25 | 77.11 | **see 39/60** · stay-blind 91/102 |

### 10.4 Base Llama vs fine-tuned (v1, same prompt)

| correct answer | base right | fine-tuned right |
|---|---:|---:|
| pack | 265 / 375 (70.7) | 346 / 375 (92.3) |
| chaal | 156 / 467 (33.4) | 413 / 467 (88.4) |
| raise | **0** / 388 (0.0) | 312 / 388 (80.4) |
| show | 10 / 408 (2.5) | 405 / 408 (99.3) |
| see | 60 / 60 (100) | 39 / 60 (65.0) |
| stay-blind | **0** / 102 (0.0) | 91 / 102 (89.2) |
| **all** | **491 / 1,800 (27.28)** | **1,606 / 1,800 (89.22)** |

- **Base Llama's answers:** pack 1,263 times, chaal 362, see on every look item, and
  raise and stay-blind never.
- **Item by item, base → fine-tuned:**

  | | items |
  |---|---:|
  | fixed (wrong → right) | **1,174** |
  | broken (right → wrong) | **59** |
  | right both times | 432 |
  | wrong both times | 135 |

  McNemar z ≈ 31.7, so the difference is not noise.
- **The one cost is `see`.** Base always said `see`, so it got all 60 see items. The
  fine-tuned model says stay-blind on 21 of them, but in exchange gets 91 of 102
  stay-blind items, which base got none of.

### 10.5 Recall by game state (seen betting, v1, right / total)

**By seat**

| | pack | chaal | raise | show |
|---|---|---|---|---|
| seat 1 | 68/80 | 144/171 | 190/213 (**89%**) | 57/57 |
| seat 2 | 278/295 | 182/204 | 112/162 (**69%**) | 317/318 |

Seat 2's raise problem isn't only bluffs: only 40 of its 162 raise items are weak
hands, against 81 of seat 1's 213. Its 50 raise misses become pack 26, chaal 19 and
show 5.

**By own turn**

| | pack | chaal | raise | show |
|---|---|---|---|---|
| turn 1 | 58/59 | 22/25 | 36/39 | 18/18 |
| turn 2 | 100/109 | 91/105 | 63/65 | 154/155 |
| turn 3 | 94/109 | 109/124 | **89/148 (60%)** | 202/202 |
| turn 4 | 94/98 | 104/121 | 114/123 | — |

**By menu size**

| | pack | chaal | raise | show |
|---|---|---|---|---|
| 2 actions | 39/39 | 56/60 | — | — |
| 3 actions | 152/162 | 192/211 | 106/128 | 103/103 |
| 4 actions | 155/174 | **78/104 (75%)** | 196/247 | 271/272 |

**By hand strength band** (0 = weakest, 4 = strongest)

| band | pack | chaal | raise | show |
|---|---|---|---|---|
| 0 | 113/114 | 36/49 | 32/42 | — |
| 1 | 99/102 | 70/79 | 34/48 | 40/40 |
| 2 | 81/84 | 67/70 | **35/65 (54%)** | 103/103 |
| 3 | 36/45 | 84/89 | 54/71 | 135/135 |
| 4 | **17/30 (57%)** | 69/88 | 147/149 | 96/97 |

- With strong hands raise is almost perfect (147/149). With middling hands it is 54%.
- Pack with a strong hand is 17/30. These are spots where the solver folds a good
  hand into heavy betting; the model shows instead 11 times.
- Of the 21 pack → show errors, 19 hold hands in bands 3–4. The median solver
  confidence on them is only 0.58, so they're near coin-flips.

**By whether the opponent has looked**

| | pack | chaal | raise | show |
|---|---|---|---|---|
| opponent blind | 127/128 | 197/213 | 106/128 | — |
| opponent seen | 219/247 | **129/162 (80%)** | 196/247 | 374/375 |

### 10.6 Recall by solver confidence in the correct answer (all graded)

| confidence | pack | chaal | raise | show | see | stay-blind |
|---|---|---|---|---|---|---|
| < 0.6 | 49/66 | 74/106 | 173/201 | 141/142 | 7/15 | 15/20 |
| 0.6–0.8 | 90/98 | 91/101 | 87/113 | 154/154 | 11/19 | 12/15 |
| ≥ 0.8 | 207/211 | 248/260 | **52/74 (70%)** | 110/112 | 21/26 | 64/67 |

**Anomaly: confident raises are the hardest raises.** Of the 74 raise items the
solver plays at ≥ 0.8, the model misses 22 (pack 11, chaal 8, show 3). Their median
equity is 0.507, and 10 are below 0.5. The solver bluff-raises these hands with
confidence, and the model's equity-based instinct folds or calls them (§3).

### 10.7 Blind betting by seat and turn (v1, right / total)

| | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---|---|---|---|
| seat 1 | chaal 1/1 | chaal 5/5 | chaal 18/21, **raise 0/2** | chaal 43/45, raise 10/11 |
| seat 2 | **show 1/3** | show 12/12 | show 18/18 | chaal 20/20 |

The solver's blind plan is visible here: seat 2 demands a show at turns 1–3, and
seat 1 mostly chaals. The model follows it everywhere except seat 2's turn 1 (the
same turn-1 data gap) and seat 1's two turn-3 raises.

### 10.8 The 200 mixed items (not graded)

On items where the solver genuinely mixes, the model picks the solver's most
frequent action 134 times (67%), and an action the solver plays at least 20% of
the time 183 times (91.5%). The mean solver probability of its pick is 0.392. These
can't be graded as right or wrong, because no single answer is correct, but the
model stays within the solver's mix.

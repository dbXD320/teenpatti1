# Does the Training Data Contain the Failure Patterns?

Companion to [error_analysis.md](error_analysis.md) (what fails) and
[error_situations.md](error_situations.md) (when it fails). This one asks whether
`teenpattibench_train_100k.jsonl` actually contains the situations the models get
wrong.

**Data.** 98,000 graded training items vs 1,800 graded eval items. Situation
predicates are the same code applied to both files. Every number computed, none
estimated.

---

## 0. The framing correction this question needs

**None of the four models were fine-tuned. They never saw this training data.**

Checked directly against the run metadata: all four are stock Hugging Face instruct
checkpoints — `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, `mistralai/Mistral-7B-Instruct-v0.3`,
`meta-llama/Llama-3.2-3B-Instruct`. No adapter, no LoRA, no checkpoint path, no
fine-tuning field in any of the four files. The training set has never been consumed
by anything.

So the question "was this failure caused by insufficient training data?" has one
answer for all four models and all ten situations: **no — exposure was zero, so every
observed failure is a zero-shot generalisation failure by definition.**

That does not make the analysis useless, it makes it *forward-looking*. The useful
question is: **when you do fine-tune, is this training set able to fix each failure,
or will it still be blind to it?** That is what the rest of this document answers.

---

## 1. Coverage, situation by situation

| situation | eval n | train n | % of 98k | distinct infosets | distinct nodes |
|---|---:|---:|---:|---:|---:|
| S1 conversion turn 3–4, truth `stay-blind` | 95 | **820** | **0.84%** | **41** | **41** |
| S2 `show` correct, 4-action menu | 291 | 14,362 | 14.66% | 14,191 | 70 |
| S3 `show` correct, 3-action menu | 117 | 9,818 | 10.02% | 9,685 | 79 |
| S4 `pack` correct vs a seen opponent | 247 | 15,766 | 16.09% | 15,766 | 347 |
| S5 `chaal` correct, raise available, stake ≤2 | 288 | 16,161 | 16.49% | 15,952 | 199 |
| S6 `raise` correct, 4-menu, both seen | 247 | 17,953 | 18.32% | 17,953 | 130 |
| S7 turn 1–2, full 4-action menu | 279 | 13,218 | 13.49% | 13,085 | 30 |
| S8 opponent just raised, full menu | 127 | 7,890 | 8.05% | 7,757 | 39 |
| S9 `pack` correct, menu only `[pack, chaal]` | 39 | 2,597 | 2.65% | 2,597 | 35 |
| S10 strong hand 17–24, seen betting | 583 | 46,533 | 47.48% | 46,533 | 560 |

**In every situation the correct label is the one the training data teaches.** For the
single-label situations (S1–S6, S9) the training label distribution is 100% the
correct action by construction. For the mixed ones:

| | eval label mix | training label mix |
|---|---|---|
| S7 | show 144, pack 71, raise 36, chaal 28 | show 4,736, pack 4,037, raise 3,166, chaal 1,279 |
| S8 | show 52, pack 39, chaal 26, raise 10 | show 3,496, pack 2,414, chaal 1,090, raise 890 |
| S10 | raise 203, show 181, chaal 142, pack 57 | raise 18,900, show 15,174, chaal 9,631, pack 2,828 |

The proportions track eval closely. There is no label skew inside these situations
that would teach the wrong answer.

---

## 2. "Similar" vs "nearly identical" — the distinction is structural here

| | items | distinct prompts | repeat factor |
|---|---:|---:|---|
| **Seen** training items | 95,440 | **95,440** | **1× — no duplicates at all** |
| **Blind** training items | 2,560 | **128** | **20× — byte-identical** |

This is not a sampling accident. A blind player's information set is the public
history alone (`RULES.md` §12.4), so two items from the same blind infoset are
*necessarily* byte-identical prompts. Seen items pair a public node with a hand class,
so the same betting line recurs with different cards — genuine variety.

**Consequence:** blind coverage is "nearly identical" by construction and cannot be
made "similar" by any amount of generation. Seen coverage is genuinely similar and
abundant.

---

## 3. Node-level exposure — and the one place it is zero

Do the eval failure situations occur at public nodes the model will have seen during
training?

| situation | eval nodes | also in train | median train items at that node |
|---|---:|---:|---:|
| S1 conversion turn 3–4 | 95 | **0 (0%)** | **0** |
| S2 show, 4-menu | 67 | 48 (72%) | 424 |
| S3 show, 3-menu | 54 | 40 (74%) | 179 |
| S4 pack vs seen | 124 | **124 (100%)** | 177 |
| S5 chaal w/ raise | 123 | 92 (75%) | 266 |
| S6 raise, both seen | 65 | **65 (100%)** | 397 |
| S7 turn 1–2 full menu | 37 | 22 (59%) | 601 |
| S8 opp just raised | 31 | 20 (65%) | 250 |
| S9 pack on 2-menu | 21 | **21 (100%)** | 129 |
| S10 strong hand | 206 | **206 (100%)** | 252 |

Overall:

```
SEEN  nodes: eval 322, train 599, shared 322  (100% of eval nodes appear in training)
BLIND nodes: eval 300, train 128, shared   0  (0%)
```

**The blind zero is structural, not a bug.** A blind information set *is* a public
node, and the splits are disjoint at the infoset level — so for blind items, disjoint
infosets means disjoint nodes. There is no way to have both a clean train/eval split
and node overlap on the conversion stratum. A fine-tuned model must **generalise** to
public histories it has never seen; it cannot memorise its way through conversion.

For seen items the opposite holds: every eval node appears in training with a median
of 129–601 other hands. Memorising the betting line is available there.

---

## 4. The conversion pool, in full

This is the only stratum with a real coverage problem, so it is worth the detail.

| | training | eval |
|---|---|---|
| look items | 1,380 (**1.41%** of 98k) | 162 |
| distinct nodes | **69** | 162 |
| truth = `see` | 500 items / **25 distinct** | 60 / 60 |
| truth = `stay-blind` | 880 items / **44 distinct** | 102 / 102 |

Distinct **training** nodes by own-turn, against what eval tests:

| turn | `see` train / eval | `stay-blind` train / eval |
|---|---|---|
| 1 | **0** / 3 | 1 / 1 |
| 2 | 6 / 13 | **2** / 6 |
| 3 | 11 / 27 | 12 / 26 |
| 4 | 8 / 17 | 29 / 69 |

**Two holes worth naming:**

- **Turn-1 `see` has zero training examples.** The training set contains no instance
  at all of "looking at your cards on your first turn is correct". Eval tests 3.
- **Turn-2 `stay-blind` has 2 distinct examples.** Eval tests 6.

Note what is *not* the problem: within the look family, `stay-blind` is the **majority**
label in training (880 vs 500, 64%). So the models' universal "always say `see`" is not
something the training distribution would teach — the issue is that the whole look
family is 1.41% of the training signal, not that it points the wrong way.

Representative training examples:

```
train-000078  node 4399  seat 2  turn 4  pot 15  stake 2   rep=14
              menu [see, stay-blind]   truth=stay-blind (p=0.52)
train-000080  node 5816  seat 1  turn 3  pot 20  stake 4   rep=3
              menu [see, stay-blind]   truth=stay-blind (p=1.00)
```

`rep=14` and `rep=3` are the 20× repeat index — these are the same 41 prompts seen
over and over.

---

## 5. Verdict per situation

| situation | representation | coverage or model problem? |
|---|---|---|
| S1 conversion turn 3–4 | **underrepresented** — 41 distinct, 0.84%, 0 node overlap | **Coverage**, and irreducible: only 428 blind infosets exist in the whole game |
| turn-1 `see` | **missing** — 0 examples | **Coverage**, absolute |
| turn-2 `stay-blind` | **underrepresented** — 2 distinct | **Coverage** |
| S2 show, 4-menu | sufficient — 14,362 items / 70 nodes | **Model** |
| S3 show, 3-menu | sufficient — 9,818 / 79 | **Model** |
| S4 pack vs seen | sufficient — 15,766 / 347, 100% node overlap | **Model** |
| S5 chaal w/ raise | sufficient — 16,161 / 199 | **Model** |
| S6 raise, both seen | sufficient — 17,953 / 130, 100% overlap | **Model** |
| S7 turn 1–2 full menu | sufficient — 13,218 / 30 | **Model** |
| S8 opp just raised | sufficient — 7,890 / 39 | **Model** |
| S9 pack on 2-menu | sufficient — 2,597 / 35, 100% overlap | **Model** |
| S10 strong hand | sufficient — 46,533 / 560, 100% overlap | **Model** |

**Nine of eleven are amply covered.** The training set is not the bottleneck for them.

---

## 6. Per-model reading

Since exposure was zero, the per-model dimension is: *for the failure each model
actually shows, will this training set be able to correct it?*

| model | its signature failure | training support | prognosis |
|---|---|---|---|
| **Mistral-7B** | raises 99.8% of the time raise is legal; `chaal` recall 32.3% | S5: **16,161** items where chaal beats raise, across 199 nodes | **Fixable.** Abundant counter-evidence; this is a prior to override, not a gap. Contrastive pairs will beat volume. |
| **Qwen3-8B / 4B** | never folds (`pack` recall 0.0%) | S4: **15,766** pack-vs-seen items, 100% node overlap; S9: 2,597 more | **Fixable.** The data says fold in 24,474 places. |
| **Qwen3-4B** | `show` recall 0.7% | S2+S3: **24,180** show items | **Fixable**, but re-measure after randomising action order — half of this is layout (§7). |
| **Llama-3.2-3B** | never selects menu index ≥2 → zero raise, zero show | S6: 17,953 raise items, S2/S3: 24,180 show items | **Fixable only if the ordering is randomised first.** Fine-tuning on the current fixed-order data would teach the position, not the game. |
| **all four** | `stay-blind` recall 0–2% | **41 distinct** examples for turn 3–4, **0** for turn-1 `see` | **Not fixable by data volume.** This is the one real gap. |

---

## 7. What to do about it

**1. Nine of the ten situations need no new data.** They need the ordering confound
removed and then a fine-tune. Adding examples to S2–S10 would be spending effort on
the part that is already covered 8–48×.

**2. Randomise action order before generating the training set.** For Llama this is
decisive — S6 and S2 have ~18k and ~14k examples each, but every one of them lists
`raise` and `show` at index ≥2, so fine-tuning on the current file would reinforce
exactly the behaviour that caps it at 55.78%. `prompts.py` now accepts `action_order`;
`generate_train_100k.py` does not yet pass it. That is the single highest-value change.

**3. Conversion cannot be fixed by generating more data, so use the levers that
exist.** Only 428 dominant blind information sets exist in the entire game; eval holds
300 and training 128. The options, in order of preference:

- **Loss weighting** on the look family — it is 1.41% of tokens-that-matter and
  deserves 10–20× that weight. Safer than repetition because it does not multiply
  byte-identical strings.
- **Rebalance the split.** Eval currently takes 70% of the blind pool. Moving to 50/50
  would give training 214 distinct blind infosets instead of 128 — a 67% increase in
  the only scarce resource — at the cost of eval precision on the per-turn breakdown.
  Worth considering explicitly, since eval's turn-3 and turn-4 cells (26 and 69 items)
  have room to give.
- **Do not simply raise `--blind-repeat`.** Going from 20× to 40× doubles the item
  count and changes nothing about the 128 distinct prompts; it buys memorisation.

**4. Accept that turn-1 `see` cannot be taught from this data.** There are zero
training examples and only 5 such situations in the entire game. Eval's 3 items should
be reported as a control, never interpreted — consistent with what `PHASE3.md` already
says about turn-1 thinness.

**5. Expect the post-fine-tune failure profile to invert.** If the reading above is
right, a fine-tune on the corrected data should largely fix S2–S10 (abundant, covered,
100% node overlap on the seen side) while conversion stays weak. If conversion improves
*and* the seen situations do not, something is wrong with the training run rather than
with this analysis.

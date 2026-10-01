# Fine-Tuning Data Strategy

**Status: dataset built and verified. No model has been fine-tuned.**

Output: `data/teenpattibench_finetune.jsonl` — 100,000 items, 182 MB, gitignored,
regenerates deterministically in ~12 s with `python src/generate_finetune.py`.
Composition record: `data/teenpattibench_finetune_stats.json` (committed).

This turns the four analyses — [error_analysis.md](error_analysis.md),
[error_situations.md](error_situations.md), [training_coverage.md](training_coverage.md),
[training_signal.md](training_signal.md) — into a concrete dataset. Every quota below
was set against a measured pool size, and every claim about the output was checked
against the written file, not the generator's own report.

---

## 1. The decisions, answered directly

| # | question | decision |
|---|---|---|
| 1 | Upweight | Same-node contrast groups (24%), and four boundary components for specific failures (32%) — §2 |
| 2 | Downweight / remove | **Remove** the 2,000 mixed items. **Downweight, don't drop**, low-confidence labels — §5 |
| 3 | Contrastive groups | 4,931 groups over 356 nodes: same betting line, 2 confident hands per correct action, one presentation order per group, contiguous in the file — §3 |
| 4 | Weight per failure mode | Set by component size; table in §4 |
| 5 | Low-confidence / mixed labels | `weight` = `label_prob` normalised to mean 1.0 *within each label*; mixed excluded — §5 |
| 6 | Action ordering | Balanced permutation per menu shape; one per contrast group; cycled across blind repeats — §6 |
| 7 | Scarce `see` / `stay-blind` | 30× repeats with varied order (2.07% of the set, up from 1.38%); no new data exists — §7 |
| 8 | Keep 100k or regenerate | **Regenerate, keep 100k.** Randomising the order means re-rendering every prompt anyway — §8 |

---

## 2. Composition

| component | items | share | what it targets | labels |
|---|---:|---:|---|---|
| **A** blind (look + blind betting) | 3,840 | 3.84% | conversion — the research object | stay-blind 1,320 · chaal 1,170 · see 750 · show 480 · raise 120 |
| **B** same-node contrast | 24,002 | 24.00% | card blindness, all four models | pack 8,936 · chaal 8,872 · raise 3,266 · show 2,928 |
| **C·F1** pack on `[pack, chaal]` | 6,000 | 6.00% | never folds (Qwen ×2, Mistral); 39/39 universal eval failure | pack 6,000 |
| **C·F2** raise legal, not correct, equity 0.50–0.80 | 12,000 | 12.00% | Mistral's raise reflex | chaal 6,644 · pack 3,958 · show 1,398 |
| **C·F3** show correct | 8,000 | 8.00% | show collapse, all four | show 8,000 |
| **C·F4** strong hand (≥0.80), raise/show correct | 6,000 | 6.00% | Llama folds strong hands | raise 4,792 · show 1,208 |
| **D** balancer | 40,158 | 40.16% | keeps label totals balanced | raise 12,822 · show 11,629 · chaal 8,957 · pack 6,750 |
| **total** | **100,000** | | | |

**Resulting label mix**

| pack | chaal | show | raise | stay-blind | see |
|---:|---:|---:|---:|---:|---:|
| 25,644 (25.64%) | 25,643 (25.64%) | 25,643 (25.64%) | **21,000 (21.00%)** | 1,320 (1.32%) | 750 (0.75%) |

Best single-word policy: **25.64%**, below eval's own trivial floor of 25.94%.

### How the components interact

C shifts composition *within* each label toward its decision boundary. D then fills
each label to its target, so the totals stay balanced. That split matches what the
analyses found: these are **boundary** failures (fold below the cutoff, raise only
when strong), not **label-frequency** failures. Adding more of a label a model under-
produces would not fix it; putting its examples where the decision is hard does.

### Pool headroom

Every component draws well inside its pool (dominant seen infosets, eval excluded):

| component | quota | pool | used |
|---|---:|---:|---:|
| F2 | 12,000 | 63,706 | 18.8% |
| F1 | 6,000 | 12,164 | 49.3% |
| F3 | 8,000 | 36,626 | 21.8% |
| F4 | 6,000 | 60,853 | 9.9% |

F1 is the tightest, at half its pool. It can go up to about 12,000 if the first run
shows folding on two-action menus still failing.

---

## 3. Contrastive same-node groups

**Construction.** For each of the 358 multi-action nodes, build groups of 2 hands per
correct action present, drawn from confident labels (`label_prob` ≥ 0.80) and never
below 0.60. Nodes are visited round-robin so no single node dominates.

**Result.** 4,931 groups over 356 nodes, a median of 14 groups per node. Sizes 4 (3,159),
6 (1,405) and 8 (367), which fit ordinary batch sizes. Verified on the written file:
all 4,931 are **contiguous**, all come from **one node**, all use **one presentation
order**, and all carry **at least two correct actions**.

A real group — same betting line, only the hand varies:

```
the Two of Hearts, Five of Hearts, Ten of Hearts          → chaal  p=1.00
the Two of Diamonds, Six of Diamonds, Eight of Diamonds   → chaal  p=1.00
the Four of Hearts, Five of Clubs, King of Spades         → pack   p=0.80
the Four of Hearts, Five of Diamonds, King of Hearts      → pack   p=0.81
the Six of Diamonds, Nine of Clubs, Nine of Diamonds      → show   p=1.00
the Four of Spades, Seven of Diamonds, Seven of Spades    → show   p=0.87
```

Two colours chaal, two high cards fold, two pairs show. **The colours are the stronger
hands and they chaal; the pairs are weaker and they show.**

### The strength order is not what intuition says — and it matters

That group led to a measurement that overturns an earlier conclusion. Node by node,
comparing the median hand strength of two correct actions:

| naive order (left weaker) | nodes | holds |
|---|---:|---:|
| pack < chaal | 323 | 98% |
| pack < show | 118 | 99% |
| pack < raise | 111 | 97% |
| **chaal < show** | **103** | **2%** |
| chaal < raise | 125 | 90% |
| show < raise | 54 | 93% |

**Within a situation the order is pack < show < chaal < raise.** Where both are correct,
the `show` hand is the *weaker* one at 98% of nodes. It's value-betting logic: a
strong hand keeps betting to build the pot, and a medium hand takes a cheap showdown
before facing more bets.

[training_signal.md](training_signal.md) originally read the pooled medians
(show 0.737 above chaal 0.615) as a strength order. That was wrong, and it has been
corrected there. `show` is legal only at a restricted, deeper set of nodes where hands
are strong overall, which inflates its pooled median. **Any curriculum hint of the form
"show with strong hands" would teach the opposite of the solver.** This is also why the
contrast groups matter: they're the only form of data that shows this ordering, because
the reversal only appears within a node.

**Why one order per group.** If the menu order also changed inside a group, the hand
would no longer be the only thing that varies, and the contrast would teach ordering
along with strength.

---

## 4. Weight per failure mode

| failure mode | models | items aimed at it | share |
|---|---|---:|---:|
| Layout / menu position | all, decisively Llama | **all 100,000** (randomised order) | 100% |
| Card blindness | all four | 24,002 (B) | 24.0% |
| Raise reflex | Mistral | 12,000 (F2) + raise-share reduction | 12.0% |
| Never folds | Qwen ×2, Mistral | 6,000 (F1) + 8,936 pack in B + 3,958 pack in F2 | 18.9% |
| Show collapse | all four | 8,000 (F3) + 2,928 show in B + 1,398 in F2 | 12.3% |
| Folds strong hands | Llama | 6,000 (F4) | 6.0% |
| Conversion | all four | 3,840 (A) — look 2,070 | 3.8% |

*(Overlaps are intentional — a pack example inside a contrast group serves both card
blindness and never-folds.)*

### Raise gets 21%, not 25% — the one balance deviation

With an equal four-way split, P(raise correct | raise legal) comes out at **37.5%**.
That's higher than eval's 32.1% on seen items, and it pushes the prior toward raising,
the wrong direction for a model that raises 99.8% of the time raise is legal. The cause
is F1: its 6,000 `[pack, chaal]` items use up pack's quota on menus where raise isn't
legal at all. Cutting F1 would give up the case all four models fail 39/39. Enlarging
F2 barely helps (37.5% → 35.4% at 24,000, and it overflows at 30,000). What does work is
raise's own share:

| raise share | P(raise correct \| legal) | best single-word policy |
|---:|---:|---:|
| 0.25 (equal) | 37.5% | 24.48% |
| 0.24 | 36.9% | 24.64% |
| 0.22 | 34.2% | 25.31% |
| **0.21 (chosen)** | **32.9%** | **25.64%** |
| 0.20 | 31.5% | 25.98% |

**0.21** puts the conditional at 32.9%, against eval's 32.1% and v1's 34.0%, and keeps
every single-word baseline below eval's 25.94%. 0.20 would go slightly past that floor.
21,000 raise examples is still ample for Llama.

---

## 5. Low-confidence and mixed labels

**Mixed items (non-dominant): removed.** Their label is the argmax of a genuinely mixed
strategy. `solver_distribution` stays on every item, so a soft-label stage can use them
later.

**Low-confidence dominant items: kept, with lower weight.** I'm reversing the earlier
advice to drop items below ~0.60, for two measured reasons:

- **30.6% of graded eval items have `label_prob` ≤ 0.60.** Dropping them from training
  would remove signal exactly where a third of the benchmark is scored.
- **A hard floor can't keep `show` balanced.** At `label_prob` > 0.6 the pool holds
  22,721 `show` items; at > 0.8 only 9,431. `show` is the least confident label
  (mean 0.706) and the one models already fail most.

**How the weight is set.** `weight = label_prob / mean(label_prob of that label)`. That
gives every label a mean weight of exactly 1.000 (verified), so confident items count
more *within* a label without changing the balance *between* labels. Plain `label_prob`
would have given `show` and `raise` about 17% less total gradient than `pack` and `chaal`,
because their labels are less confident — a silent imbalance against the two labels
models fail most.

| label | mean label_prob | weight range |
|---|---:|---|
| pack | 0.840 | 0.595 – 1.190 |
| chaal | 0.839 | 0.596 – 1.191 |
| show | 0.706 | 0.708 – 1.417 |
| raise | 0.696 | 0.719 – 1.437 |
| stay-blind | 0.838 | 0.601 – 1.194 |
| see | 0.766 | 0.676 – 1.305 |

The one exception is the contrast groups, which *do* use a floor (≥0.60, preferring
≥0.80). A contrast between two near-50/50 labels teaches false precision: 32% of
adjacent label flips in the v1 data are that kind of case.

---

## 6. Action ordering

Each item's menu order is a permutation assigned by cycling through every permutation
evenly within its menu shape, the same method `make_eval_shuffled.py` uses for eval.
The order is applied consistently to the pricing sentence and the closing menu line.

| | v1 train | fine-tune set |
|---|---|---|
| `pack` at index 0 | 100% | 30.7% |
| `raise` at index ≥ 2 | 100% | 45.7% |
| `see` at index 0 | 100% | 50.0% |

Pooled, index 3 looks under-used (`pack` 12.0%) because only four-action menus have an
index 3. Within each menu shape the cycle is balanced.

**Special cases:**
- **Contrast groups:** one order per group (§3).
- **Blind repeats:** each repeat gets the next permutation in turn. The 3,840 blind items
  now cover **924 distinct prompts**; v1 had 128 for 2,560. Look nodes have only two
  orderings, so their 30 repeats split 15/15.

---

## 7. The scarce `see` / `stay-blind` data

Nothing here can fix the underlying shortage. Only 428 dominant blind information sets
exist in the whole game, eval holds 300, and a blind infoset *is* a public node, so train
and eval share **zero** conversion nodes by construction.

**What was done.** The repeat factor goes from 20× to 30×, with a new order on each
repeat. Look items go from 1.38% to **2.07%** of the set: stay-blind 1,320, see 750.

**Why not more.** A per-example loss weight and repetition give the same expected
gradient under SGD. So the earlier recommendation to prefer loss weighting over
repetition for conversion was only half right: once the order is randomised, repetition
also adds surface variety, which weighting can't. But look nodes have only two
orderings, so beyond two repeats they're byte-identical again. The limit is the **69
distinct look infosets**, not the repeat factor. More repeats buy memorisation of 69
histories the eval set never tests.

**Not recommended: moving the eval/train blind split.** Taking eval from 70% to 50% of
the blind pool would give training 214 distinct blind infosets instead of 128. But it
changes the eval file and invalidates all four scored runs and the numbers in
PHASE4.md. That's a paper-level decision, not a data-pipeline one.

### Read conversion results with this in mind

Within the training look family, `stay-blind` is **64%** of labels, and on eval it's
63%. A fine-tuned model that learns only that prior, answering `stay-blind` every
time, scores **62.96%** on eval conversion. Today's models score 37%, so that would look
like a 26-point gain, while learning **nothing** about timing.

**After fine-tuning, compare conversion against 62.96%, not 37%, and read it per turn.**
The real test is turn 3 vs turn 4, where the solver says don't look and then does look.
A flat "always stay-blind" gets turn 3 right and turn 4 wrong; learning the timing
means getting both right.

---

## 8. Why regenerate rather than reweight the existing 100k

Keeping the existing file and adding weights is not an option. Randomising the order
means re-rendering every prompt, so regeneration is required either way. Once
regenerating, reshaping the composition costs nothing. The size stays at 100k because
the analyses found the seen situations already covered 8–48×. Going bigger would only
dilute the blind share further.

---

## 9. Code changes

### `src/generate.py` — `template()` extended, backward-compatible

Three optional record keys, each added only when present:

| key | effect |
|---|---|
| `action_order` | passed to `render_prompt`; `legal_actions` and `meta.action_order` record the displayed order |
| `extra_meta` | merged into `meta` (component, group id) |
| `weight` | emitted as a top-level field |

**Verified byte-identical:** regenerating `teenpattibench_train_100k.jsonl` after the change
reproduces SHA-256 `a1b3c829…3c18053` exactly, so every existing dataset is still
reproducible. The 168 tests pass.

### `src/generate_finetune.py` — new

Builds components A–D, assigns orders and weights, templates in unit order so groups
stay contiguous, and verifies. It reuses from `generate.py`: `load_solver`,
`class_tables`, `enumerate_infosets`, `split_blind_pool`, `stratified_seen_sample`,
`template`, `assert_disjoint` and `assert_no_blind_leaks`. From
`generate_train_100k.py` it reuses the eval-reproduction gate and the progress
reporting. Every quota is a CLI flag:

```bash
python src/generate_finetune.py                    # the dataset described here
python src/generate_finetune.py --dry-run          # composition and gates, writes nothing
python src/generate_finetune.py --raise-share 0.20 --f2 18000 --blind-repeat 20
```

It refuses to write if the eval split fails to reproduce, if a component overdraws its
pool, if components exceed a label's balanced target, if the total isn't exactly 100,000,
or if any menu isn't a permutation of its legal set.

### `.gitignore`

The 182 MB dataset is ignored, and its stats file is committed.

### Verification of the written file

| check | result |
|---|---|
| items / unique ids | 100,000 / 100,000 |
| overlap with eval (infosets) | **0** |
| mixed items | 0 |
| correct action not on its menu | 0 |
| label ≠ argmax of solver distribution | 0 |
| prompt menu line matches `legal_actions` order | all |
| blind prompts pass the leakage guard | 3,840 / 3,840 |
| contrast groups well-formed and contiguous | 4,931 / 4,931 |
| mean weight per label | 1.000 for every label |

---

## 10. What the trainer must do for this dataset to work as designed

These are requirements on the fine-tuning run, recorded here because the dataset
assumes them. Nothing has been trained.

1. **Mask the loss to the answer only.** The target is one action word. Without masking,
   almost all of the gradient goes into predicting a ~940-character prompt.
2. **Read `weight`.** Standard `SFTTrainer` ignores it. Multiply the per-example loss by
   `weight`, which takes a small `compute_loss` override. If that isn't possible, the
   dataset still works unweighted, since the label balance doesn't depend on it.
3. **Keep contrast groups together.** The file is written with groups contiguous and
   units shuffled. Use a sequential sampler (`shuffle=False`) with a batch size of at
   least 8, or a group-aware sampler over `meta.group_id`. A fully shuffled sampler
   still trains correctly but loses the within-batch contrast.
4. **Use the system prompt from `data/teenpattibench_system_prompt.txt`** exactly as eval
   does.
5. **Evaluate on both eval files.** v1 (fixed order) and `teenpattibench_eval_shuffled.jsonl`.
   Since training used shuffled orders, v1 is just one permutation, but the paired
   comparison shows whether any order sensitivity remains.
6. **Track per-label recall during training, not just loss.** A run where `pack` or
   `stay-blind` recall stays at 0 has failed, whatever the loss curve says.

### Expected outcome, stated so it can be falsified

If the analyses are right, the fine-tune should mostly fix the seen-betting failures,
which are abundant, correctly labelled, and share 100% of their nodes with eval.
Conversion should move only as far as its 62.96% prior unless timing is genuinely
learned. **If conversion improves and the seen situations don't, suspect the training
run, not the data.**

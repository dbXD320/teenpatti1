# What the Training Data Can Teach — Per Failure Mode

Fourth in the series: [error_analysis.md](error_analysis.md) (what fails),
[error_situations.md](error_situations.md) (when), [training_coverage.md](training_coverage.md)
(whether the data contains it). This one reads the training examples themselves and
asks *what the model needs to learn from them*.

**Scope.** The situations with abundant training data — the seen-betting failures.
95,440 seen training items vs 1,500 seen eval items. Hand strength is the exact
equity of each suit-isomorphic class against a uniform opponent, from
`abstraction.build_abstraction()`. Every number computed; a control is run wherever a
result could be an artefact.

**The one-line answer.** The training data teaches a clean, learnable rule — *rank
your hand, then apply a cutoff that depends on the betting situation* — and the
models fail because they never do the first step. Nothing below requires new data.
It requires reweighting, filtering and grouping what already exists.

---

## 1. Is the information the decision needs actually in the prompt?

A full seen prompt, verbatim:

```
You are Player 2 in a heads-up game of Teen Patti. You act second each round.
Both players posted a boot of 1. The pot is 4. The current stake is 1.
You are SEEN: you have looked at your cards. Your opponent is seen.
Your cards are the Five of Diamonds, the Six of Diamonds, and the Ace of Diamonds.
The betting so far: Player 1 looked at their cards and is now seen. Player 1 (seen)
played chaal for 2. Player 2 looked at their cards and is now seen.
This is your turn number 1 of this hand. 7 betting actions remain before the action
cap forces a showdown, and 2 raises remain.
At this stake, chaal costs 2, a raise costs 4, a show costs 2, packing costs nothing.
Your legal actions are: pack, chaal, raise, show.
Do not explain your answer. Your optimal action is:
```

| information | in the prompt? |
|---|---|
| betting history, with amounts and who looked | **yes**, verbatim |
| pot, stake, action costs, turn, cap, raises left | **yes**, explicit |
| legal actions | **yes** |
| own cards | **yes**, as spelled names |
| **hand category / strength** | **no** — must be derived |
| opponent's cards or range | no, correctly |

That hand is a **colour** (three diamonds) in the 23rd of 25 strength buckets — among
the strongest hands a player can hold. The prompt never says so. To use it, a model
must (1) notice all three cards share a suit, (2) look up that a colour ranks fourth
of six in the system prompt's list, and (3) place it relative to the pot odds.

**Every input the decision depends on is present.** The one that matters most is
present only in raw form. That is the step the evaluated models never take — the
multi-hand node test in [error_analysis.md](error_analysis.md) found Mistral gives one
identical answer across every hand in 74 of 74 nodes.

---

## 2. Within a single situation, the hand decides the answer

Fix the betting situation (the public node) and vary only the hand. Among the 429
nodes with at least 20 training hands:

| distinct correct actions at the node | nodes |
|---|---:|
| 1 — the hand does not matter here | 80 |
| 2 | 188 |
| 3 | 124 |
| 4 | 37 |

**At 349 of 429 nodes (81%), the betting history alone cannot determine the answer.**
The training data is saturated with exactly the contrast the models fail to learn:
same situation, different cards, different correct action.

---

## 3. And the rule is clean — the data does not contain conflicting patterns

Sort the hands at each node by strength and ask how well contiguous blocks — *pack
below here, chaal between here and here, raise above* — reproduce the labels.

| | rule accuracy |
|---|---:|
| labels sorted by hand strength | **99.1%** (median 100%) |
| **control:** the same labels, randomly shuffled | 80.9% |
| lift attributable to hand strength | **+18.2 pts** |

The control matters: allowing four blocks inflates any fit, and shuffled labels still
score 80.9%. Sorting by strength is what lifts it to 99.1%. **94% of nodes are fit at
≥95%.**

What that rule looks like at one node (pot 4, stake 2, menu `[pack, chaal, raise]`,
1,437 training hands):

```
equity 0.001 – 0.605   835 hands  →  pack
equity 0.646 – 0.984   563 hands  →  chaal
equity 0.987 – 1.000    39 hands  →  raise
```

Three bands, no overlap. That is the shape of what the model has to learn, repeated
across 599 nodes with different cut points.

### The residual 1%

Two sources of genuine difficulty, both small:

- **5.7% of (node, strength-bucket) cells hold more than one correct action.** A model
  that reasons at 25-bucket resolution cannot separate these — but the prompt gives
  the exact cards, so a model reasoning at full resolution can.
- **A handful of confident boundaries do not follow uniform equity.** At node 1357,
  `5♦5♠K♠` (equity 0.7880) is a confident fold at p=0.98 while `5♦5♥8♣` (equity
  0.7883) is a confident show at p=0.99. Uniform equity is nearly identical; what
  differs is card composition against the range the opponent's betting implies —
  the card-removal effect described in `RULES.md` §12.5. These are real, but rare.

---

## 4. Which hands each action needs — and which side of the line the models land on

**Training, pooled across all nodes:**

| correct action | n | 10th pct | **median** | 90th pct | typical hands |
|---|---:|---:|---:|---:|---|
| pack | 23,860 | 0.053 | **0.320** | 0.717 | high card 92% |
| chaal | 23,860 | 0.212 | **0.615** | 0.919 | high card 67%, pair 21% |
| show | 23,860 | 0.507 | **0.737** | 0.913 | high card 52%, pair 38% |
| raise | 23,860 | 0.465 | **0.897** | 0.981 | pair 31%, colour 30% |

> **Correction (added after [FINETUNE_STRATEGY.md](FINETUNE_STRATEGY.md) §3).** An
> earlier version of this paragraph read these pooled medians as a strength order —
> "fold weak, call middling, show strong, raise very strong". **That is wrong for
> `show` vs `chaal`.** Measured node by node, where both are correct the `show` hands
> are the *weaker* ones at **98% of nodes** (103 nodes). The pooled median puts `show`
> above `chaal` only because `show` is legal at a restricted, deeper set of nodes
> whose hands are strong overall — a pooling artefact. The per-node order is
> **pack < show < chaal < raise**.

Pooled, the bands overlap (pack's 90th percentile 0.717 exceeds show's 10th, 0.507),
because different nodes put the cut points in different places. **Within a node they
separate cleanly**: at the 117 nodes where both pack and show are correct for some
hand, show hands are a median **+0.364** stronger than pack hands, and the order is
inverted at only **1 of 117** nodes.

**Raise is strength-gated, and the data says so unambiguously.** Of 70,082 training
items where raise is legal, raise is correct on only 34.0%:

| hand strength | n | raise correct |
|---|---:|---:|
| 0.00 – 0.50 | 23,543 | 12.1% |
| 0.50 – 0.80 | 21,083 | 20.8% |
| 0.80 – 0.95 | 18,834 | 58.9% |
| 0.95 – 1.00 | 6,622 | **83.4%** |

### The models are on the wrong side of the threshold

For each failure, the hand strength of the items where the model gave its reflex
answer:

| failure | model's reflex | median equity where it did so | what training says at that strength |
|---|---|---:|---|
| `chaal` correct, raise legal | Mistral **raises** (257×) | 0.560 | raise is right **20.8%** of the time here |
| same | Qwen3-8B raises (176×) | 0.525 | same |
| `pack` correct | Qwen3-8B **calls** (247×) | 0.316 | pack's own median is **0.320** |
| same | Mistral calls (97×) | 0.331 | same |
| `raise` correct | Llama **folds** (341×) | 0.695 | raise at 0.50–0.80 is 20.8%, but pack's median is 0.320 |

These are not boundary cases. Mistral raises mid-strength hands where the data says
raise is wrong four times in five. Qwen calls with hands sitting exactly at the median
of the fold distribution. Llama folds hands more than twice as strong as a typical
fold.

---

## 5. Contrastive pairs — plentiful at the node level, scarce at the boundary

A contrastive pair: two hands at the same node, adjacent in strength, with different
correct actions. There are **1,877** in the training set. But they are not all signal:

| both sides of the pair | pairs | share | use |
|---|---:|---:|---|
| confident (label_prob ≥ 0.80) | 226 | 12.0% | **real boundary — emphasise** |
| one confident, one not | 1,046 | 55.7% | usable with care |
| both marginal (≤ 0.60) | 605 | **32.2%** | **argmax of a near-50/50 mix — noise** |

A third of the adjacent flips are the solver being nearly indifferent, with the argmax
landing on different sides by a hair. Training on those teaches false precision:

```
node 1306   equity 0.0338  4♦6♥7♦  → pack   (p=0.64)
            equity 0.0340  4♣6♥7♠  → raise  (p=0.64)
```

Two hands 0.0002 apart, labelled with opposite actions, at 64% confidence. No model
should be asked to learn that as a sharp rule.

A confident pair, by contrast, is a genuine lesson:

```
node 1508   equity 0.0128  3♥5♠6♥  → pack   (p=0.99)
            equity 0.0165  2♦3♦7♠  → chaal  (p=0.97)
```

**Confident pairs by boundary:**

| boundary | all flips | confident | relevant failure |
|---|---:|---:|---|
| chaal vs raise | 476 | **84** | Mistral raise reflex |
| chaal vs pack | 497 | **66** | Qwen / Mistral never fold |
| chaal vs show | 178 | 35 | show collapse |
| pack vs show | 420 | 27 | show collapse |
| pack vs raise | 247 | **8** | Llama folds strong hands |
| raise vs show | 59 | **6** | show collapse |

The *adjacent* confident pairs are thin for two boundaries. But adjacency is not
required for contrast. §2 shows 349 nodes carry multiple correct actions, with a median of **148 hands**
each (interquartile range 66–341) — **every one of those nodes is a ready-made
contrast set**, with the betting history held fixed and only the hand varying.

---

## 6. What the fine-tune should teach, per failure mode

### 6.1 Qwen3-8B / Qwen3-4B / Mistral — never fold (`pack` recall 0.0%)

**The lesson:** *fold when your hand is below this situation's cutoff* — not "fold
more". A blanket fold rate would just swap one constant for another.

**Evidence the data teaches it:** 23,860 `pack` items; 66 confident chaal/pack pairs;
the fold band is the cleanest in the set (92% high card, median equity 0.320).

**Emphasise:** node-grouped examples where weak hands fold and stronger hands at the
same node continue. The two-action `[pack, chaal]` situations deserve specific weight
— 2,597 training items across 35 nodes. In eval, all four models fail all 39 such
items, including a fold the solver plays at p=0.996.

### 6.2 Mistral — raise reflex (raises 99.8% of the time raise is legal)

**The lesson:** *raise being legal is not raise being right.* Raise is gated on hand
strength, and below ~0.80 it is usually wrong.

**Evidence:** 70,082 items where raise is legal; raise correct on only 34.0% of them;
12.1% at equity below 0.50. The data refutes the reflex **46,222** times.

**Emphasise:** the 84 confident chaal/raise pairs, and more broadly the 46,222
raise-legal items where the answer is not raise. The critical examples are
**mid-strength hands (0.50–0.80) with raise available** — exactly where Mistral's
errors sit (median 0.560) and where training says raise is right only 20.8% of the
time.

### 6.3 All four — `show` collapse (recall 0.0–15.0%)

**The lesson (corrected):** *show is for medium hands that want a cheap showdown
now* — stronger than a fold, **weaker than a chaal**. Within a situation, a strong hand
keeps betting (chaal) to grow the pot, and a medium hand ends it. Show is offered only
when the actor is blind or both players are seen. An earlier version of this section
said show sits between chaal and raise; that was a pooling artefact (see the
correction in §4).

**Evidence:** 23,860 `show` items; within a node show hands are +0.364 stronger than
pack hands, inverted at 1 of 117 nodes.

**Emphasise:** the show-vs-chaal boundary above all — it runs *opposite* to naive
strength intuition (the stronger hand chaals), which is precisely what a model reasoning
"show = strong" would get backwards. Only 35 confident adjacent pairs, so lean on the
node-grouped contrast sets instead.

**Caveat:** a large share of this failure is presentation, not knowledge — Qwen3-8B's
show recall is 49.6% at menu index 2 and 1.0% at index 3. Randomise the action order
before judging how much show-specific training is needed.

### 6.4 Llama-3.2-3B — folds strong hands, never raises or shows

**The lesson:** *continue with strong hands.* Its folds have median equity 0.695.

**Evidence:** 23,860 raise items and 23,860 show items, but only 8 confident
pack/raise adjacent pairs.

**Emphasise:** node-grouped sets spanning the full strength range, so the model sees
weak hands fold and strong hands raise at the same node.

**Prerequisite, not optional:** randomise action order first. Llama never selects
beyond menu index 1, and every raise and show example in the current file lists those
actions at index ≥ 2. Fine-tuning on the current file would reinforce the positional
habit rather than correct it.

---

## 7. Concrete changes to the training set

None of these adds data.

**1. Randomise action order.** The only prerequisite for the rest. `prompts.py` accepts
`action_order`; `generate_train_100k.py` does not yet pass it.

**2. Drop or down-weight marginal labels.** 32% of adjacent label flips are the argmax
of a near-50/50 solver mix. Weight each example by `label_prob`, or drop items below
~0.60. The field is already in every item's metadata.

**3. Group examples by public node.** Batch or order training so the model sees the
same betting line with different hands → different actions. That isolates the hand
as the only varying input — the exact variable the models ignore. The data supports
it: 349 nodes carry multiple correct actions, with a median of 148 hands each.

**4. Upweight the specific boundaries each model gets wrong:**

| model | upweight |
|---|---|
| Mistral | raise-legal items where the answer is not raise, especially equity 0.50–0.80 |
| Qwen ×2, Mistral | `pack` items, especially the `[pack, chaal]` two-action nodes |
| all | `show` items, after re-measuring on the shuffled eval |
| Llama | strong-hand items spanning the full strength range at a node |

**5. Keep the 226 confident contrastive pairs together** in the same batch where
possible — they are the sharpest boundary examples the data contains.

### One option worth considering, with a trade-off

The single missing input is the hand's category. An auxiliary training target — have
the model emit the category before the action (`colour → raise`) — would teach the
card → strength derivation directly, without changing the eval prompt. The cost is a
different output format during training, which must be removed before the final
stage so that eval conditions match. Not needed if 1–5 are enough; worth trying if the
card-sensitivity check (74/74 for Mistral) does not move after the first run.

### Where the data genuinely is lacking

Only conversion — and that is covered in [training_coverage.md](training_coverage.md):
41 distinct stay-blind examples, zero node overlap with eval, zero turn-1 `see`
examples, and no way to generate more because only 428 blind information sets exist in
the whole game. Everything in this document concerns situations where the data is
already abundant and already correct.

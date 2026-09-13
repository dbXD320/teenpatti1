# Phase 4 Report — LLM evaluation: Qwen3-8B on TeenPattiBench

**Status: first model evaluated. 2,000/2,000 eval items answered, all legal, none missing.**

Scorer: [metrics.py](../src/metrics.py). Answer key: [teenpattibench_eval.jsonl](../data/teenpattibench_eval.jsonl).
Predictions: [preds_qwen3_8b_kaggle.jsonl](../results/preds_qwen3_8b_kaggle.jsonl).

Reproduce the scoring with:

```bash
python src/metrics.py --eval data/teenpattibench_eval.jsonl \
                      --predictions <predictions.jsonl> --json <report.json>
```

---

## 1. What was run

| | |
|---|---|
| Model | Qwen3-8B (`Qwen3ForCausalLM`) |
| Precision | **torch.float16** (parameter dtype confirmed on `model.embed_tokens.weight`) |
| Hardware | Kaggle GPU notebook |
| Items | all 2,000 eval items; 1,800 graded, 200 mixed |
| Thinking | disabled |
| Coverage | 2,000 unique ids, 0 missing, 0 duplicated, **every answer legal for its own item** |

The system prompt ([teenpattibench_system_prompt.txt](../data/teenpattibench_system_prompt.txt)) supplies the
glossary. Prompts were used verbatim; no prompt text was modified for this run.

---

## 2. Headline

**Action Accuracy 32.94%** on 1,800 graded items. Exact Match is identical, because the
prediction file carries no `amount` field — see §8.

The best trivial policy on the same items scores 25.94%, so the model clears the
degenerate floor by 7 points. That is the only sense in which this is a passing result.

---

## 3. Read every stratum against its own baseline

This is the single most important instruction for interpreting the table below. The
strata have very different label mixes, so the global baseline (25.94%) is the wrong
comparison for all but the first row.

| stratum | n | model | best constant policy | lift |
|---|---:|---:|---:|---:|
| All graded | 1,800 | 32.94% | 25.94% (chaal) | **+7.0** |
| Seen betting | 1,500 | 30.40% | 25.00% (raise) | **+5.4** |
| Blind, total | 300 | 45.67% | 34.00% (stay-blind) | **+11.7** |
| — conversion / look | 162 | 37.65% | 62.96% (stay-blind) | **−25.3** |
| — blind betting | 138 | 55.07% | 66.67% (chaal) | **−11.6** |

Both blind sub-families are **below** the policy of answering one word forever, by 25
and 12 points. The favourable-looking +11.7 on the blind aggregate is an artefact of
combining them; it should never be quoted on its own.

---

## 4. The conversion result

Conversion timing is the benchmark's primary target, and the model fails it completely.

| own turn | n | model answers "see" | key answers "see" | model AA |
|---:|---:|---:|---:|---:|
| 1 | 4 | 50.0% | 75.0% | 75.0% |
| 2 | 19 | **100.0%** | 68.4% | 68.4% |
| 3 | 53 | **100.0%** | 50.9% | 50.9% |
| 4 | 86 | **98.8%** | 19.8% | 20.9% |

**The model's accuracy equals the answer key's see-rate, to the decimal, at turns 2 and
3.** It answered "see" on 159 of 162 conversion items and on every item at turns 2 and 3.
Its score is therefore a readout of how the correct answers happen to be distributed in
each stratum, not a measurement of judgement.

The apparent decline from 68.4% to 20.9% is **not the model degrading**. It is the
solver's answer shifting toward stay-blind while the model never moves. Aggregated into
a single conversion number this would read as 37.65% and look like partial competence.
This is why [metrics.py](../src/metrics.py) never averages conversion accuracy across turns.

**Turn 1 is a control stratum**, 4 items, near-deterministic. Report it, do not interpret it.

---

## 5. The same artefact inflates blind betting at turn 4

| own turn | n | model AA | model answers | key answers |
|---:|---:|---:|---|---|
| 1 | 4 | 25.0% | chaal 4 | chaal 1, show 3 |
| 2 | 17 | 17.6% | raise 10, chaal 6, show 1 | chaal 5, show 12 |
| 3 | 41 | 31.7% | raise 16, chaal 25 | chaal 21, show 18, raise 2 |
| 4 | 76 | **77.6%** | chaal 56, raise 18, show 2 | chaal 65, raise 11 |

The 77.6% at turn 4 is the model's chaal-heavy prior meeting a chaal-heavy answer key
(65 of 76). It is alignment, not skill — the same mechanism as §4, running the other way.
Any single blind-betting figure should be reported with this table beside it.

---

## 6. It never packs

**0 packs in 2,000 items.**

Pack is the correct answer on **375 of the 1,500 seen betting items (25.0%)**, so the
model's ceiling on that family was 75.0% before the run began. It scored 30.40%, i.e.
**40.5% of what was reachable**.

Answer distribution across all 2,000 items:

| chaal | raise | show | see | stay-blind | pack |
|---:|---:|---:|---:|---:|---:|
| 969 | 703 | 166 | 159 | 3 | **0** |

Against a key distribution of chaal 467, show 408, raise 388, pack 375, stay-blind 102, see 60.

A hard zero across 2,000 opportunities is structural, not a preference. §7 gives the cause.

---

## 7. The model does not consult its own cards

Grouping the graded seen-betting items by public betting node isolates the effect: same
situation, same betting history, different hands dealt.

| | nodes with ≥ 6 hands |
|---|---:|
| Model gives **one identical answer** across every hand | **40 / 74** |
| Solver gives one identical answer across every hand | 11 / 74 |

In 54% of these nodes the model answers the same word whether it holds the weakest or the
strongest hand in the deck. The solver does so in 15%. The model is responding to the
betting history and largely ignoring its private cards.

This explains §6 directly. **Folding is the only action that requires knowing your hand is
bad.** Every other action has a defensible rationale without looking. A policy that does
not consult its cards therefore produces zero folds rather than merely few — and forfeits
the 25% of seen items where packing is correct.

Probing carried out during development (on the quantised local model, artefacts since
deleted — see §9) supported this reading: appending a single sentence,
*"Your hand is weak and you are very likely beaten,"* to an unmodified benchmark prompt
flipped the answer to pack on 5 of 5 items where pack was correct. Supplied with the
conclusion, the model acts on it; required to derive it from three named cards, it does not.

---

## 8. Other breakdowns

**By seat.** Seat 1 44.94% (n=692), seat 2 25.45% (n=1,108) — a 19.5-point gap.
Seat 2's decision nodes sit deeper in the tree (PHASE1.md §4), so this is confounded with
depth and betting-line length. Not yet investigated.

**Mixed subset not scored.** All 200 mixed items were answered, but the prediction file
carries no `distribution` field, so mean TVD is unreported. Scoring them requires the
model to emit probabilities, which this run did not request.

**Exact Match is not independent here.** The prompts ask only for an action, and the
prediction file carries no `amount`, so [metrics.py](../src/metrics.py) collapses EM onto AA.
Both read 32.94%. EM should not be presented as a second metric for this run.

---

## 9. Caveats that must travel with these numbers

1. **Precision matters more than expected.** An exploratory local run of the same model
   through Ollama at **Q4_K_M 4-bit** agreed with this fp16 run on only **62.5%** of the
   2,000 items, with very different answer mixes (that run chose raise 1,112 times against
   this run's 703). Same model name, same temperature, same seed. Quantisation is not a
   detail here; every reported result must state the dtype. Those local artefacts were
   deleted at the user's request and the figure is not re-derivable — it is recorded as
   provenance, not as a result.

2. **The decoding method is not recorded.** All 2,000 answers were legal for their own
   item, which implies constrained decoding or post-hoc filtering, but the notebook's
   mechanism was not captured. It must be, before publication: whether the model chose
   freely or was constrained to the legal set changes what the score means.

3. **The prediction file uses the key `answer`; the scorer reads `action`.** It must be
   remapped before scoring. Future runs should emit `action` directly.

4. **One model, one condition.** No claim about scale or about LLMs generally follows from
   a single 8B model at one precision.

---

## 10. What this says about the Phase 3 hypothesis

PHASE3.md §10 predicted: relatively good on ordinary betting, failing on conversion timing.

**The prediction is confirmed on the conversion half and unconfirmed on the other.** The
model is 25 points below baseline on conversion — a clean, large failure exactly where no
poker text exists to transfer from. But it is only +5.4 over baseline on seen betting,
which is not the "relatively well" the hypothesis assumed. It is weak everywhere and
catastrophic on conversion.

§7 supplies a cause the hypothesis did not anticipate: the model largely does not read its
own hand. That is a *separate* deficit from the blind mechanic, and it depresses the seen
side while leaving the blind side untouched — blind decisions have no cards to read. The
seen-versus-conversion contrast is therefore **cleaner than the raw accuracies suggest**,
because the two strata fail for different reasons.

---

## 11. Open items

1. **Scale.** Whether card-insensitivity (§7) is a capability threshold a larger model
   crosses is the obvious next experiment, and it runs off the same eval file and scorer.
2. **Record the decoding method and dtype for every future run** (§9.1, §9.2).
3. **Emit `action`, and optionally `amount` and `distribution`**, to make EM meaningful and
   the mixed subset scorable.
4. **The seat gap** (§8) is unexplained.
5. **No runner ships with the repository.** The benchmark and scorer are here; producing a
   predictions file is left to the caller. The required format is one JSON object per line
   with `id` and `action`, optionally `amount` and `distribution`. A missing prediction
   raises rather than scoring as wrong.

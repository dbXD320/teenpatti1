# TeenPattiBench — Nine-Model Evaluation

**Scored:** 29 September 2026. **Source:** `results/teenpattibench_results (2)/` —
18 prediction files (9 models × 2 eval sets), their `metrics.py` reports and `run_log.jsonl`.
Every number below was recomputed from the raw prediction files; all 18 agree exactly
with the saved `metrics.py` reports. `summary.csv` / `summary.json` were not used.

---

## 1. Setup

| | |
|---|---|
| Eval sets | `teenpattibench_eval.jsonl` (**v1**, fixed action order) and `teenpattibench_eval_shuffled.jsonl` (**shuffled**, same 2,000 items, only the order of listed actions permuted) |
| Graded items | 1,800 per eval (200 mixed items excluded, as always) |
| Precision | `torch.bfloat16` for all 18 runs, unquantised |
| Decoding | greedy, `max_new_tokens=16`, thinking off, batch 8, left padding |
| Prompt | chat template, system prompt as a `system` turn — accepted by all nine models |
| Answer | first action word in the completion, **not** restricted to the legal set |
| Hardware | A100 40 GB for eight models; A100 80 GB for Qwen3-30B-A3B (needs ~61 GB) |
| Notebook | `notebooks/teenpattibench_batch_eval.ipynb` |

All 18 files: 2,000 records, 2,000 unique ids, full coverage of the eval set.

---

## 2. Leaderboard

Answering one fixed word to everything scores **25.94%** (always `chaal`).

| # | model | params | v1 | vs trivial | shuffled | Δ shuffle |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Gemma 3 12B** | 12.2B | **36.17%** | +10.23 | 34.22% | −1.94 |
| 2 | Mistral 7B | 7.2B | 35.56% | +9.62 | 33.39% | −2.17 |
| 3 | Gemma 3 4B | 4.3B | 34.22% | +8.28 | 32.50% | −1.72 |
| 4 | Qwen3-8B | 8.2B | 33.89% | +7.95 | 32.61% | −1.28 |
| 5 | Qwen3-14B | 14.8B | 33.67% | +7.73 | 32.06% | −1.61 |
| 6 | Qwen3-4B | 4.0B | 33.06% | +7.12 | 31.11% | −1.94 |
| 7 | Phi-4 14B | 14.7B | 31.78% \* | +5.84 | 34.11% | +2.33 \* |
| 8 | Qwen3-30B-A3B | 30.5B | 29.06% | +3.12 | 28.78% | −0.28 |
| 9 | Llama 3.2 3B | 3.2B | 26.78% | +0.84 | 26.39% | −0.39 |

**The whole field lies between +0.8 and +10.2 points above a one-word policy.** Exact
Match equals Action Accuracy in every run (no amounts were requested).

\* **Phi-4 spelling artefact.** On v1, Phi-4 answered `chal` instead of `chaal` 203 times
(all on seen-betting items) — its only non-vocabulary answer, and correctly scored wrong.
On the shuffled run it did so only 19 times. Counting `chal` as `chaal`:

| | strict | `chal` = `chaal` |
|---|---:|---:|
| Phi-4 v1 | 31.78% | 34.78% |
| Phi-4 shuffled | 34.11% | 34.22% |
| Δ shuffle | +2.33 | **−0.56** |

The strict score is the reported one. Its apparent gain under shuffling is mostly
spelling, and it otherwise follows the same pattern as the rest of the field.

### Paired effect of shuffling (same 1,800 questions)

| model | answer unchanged | right → wrong | wrong → right |
|---|---:|---:|---:|
| Gemma 3 12B | 83.0% | 110 | 75 |
| Mistral 7B | 93.3% | 54 | 15 |
| Gemma 3 4B | 74.9% | 140 | 109 |
| Qwen3-8B | 74.1% | 142 | 119 |
| Qwen3-14B | 72.3% | 156 | 127 |
| Qwen3-4B | 79.1% | 115 | 80 |
| Phi-4 14B | 78.3% | 59 | 101 |
| Qwen3-30B-A3B | 78.2% | 119 | 114 |
| Llama 3.2 3B | 82.2% | 116 | 109 |

Presentation order changes between 7% and 28% of answers, but the net effect is small:
seven of nine models lose 1.3–2.2 points, and the two near-constant models (Qwen3-30B-A3B,
Llama) barely move.

---

## 3. By stratum, against each stratum's own one-word baseline

Baselines: seen betting **25.00%**, conversion **62.96%** (always `stay-blind`), blind betting **66.67%** (always `chaal`).

| model | seen | conversion | blind betting |
|---|---:|---:|---:|
| Gemma 3 12B | 31.73 (+6.7) | 62.96 (**+0.0**) | 52.90 (−13.8) |
| Mistral 7B | 33.87 (+8.9) | 37.04 (−25.9) | 52.17 (−14.5) |
| Gemma 3 4B | 30.53 (+5.5) | 37.04 (−25.9) | **71.01 (+4.3)** |
| Qwen3-8B | 31.73 (+6.7) | 37.65 (−25.3) | 52.90 (−13.8) |
| Qwen3-14B | **35.07 (+10.1)** | 37.04 (−25.9) | 14.49 (**−52.2**) |
| Qwen3-4B | 31.33 (+6.3) | 37.04 (−25.9) | 47.10 (−19.6) |
| Phi-4 14B | 33.27 (+8.3) | 34.57 (−28.4) | 12.32 (**−54.3**) |
| Qwen3-30B-A3B | 24.93 (**−0.1**) | 37.04 (−25.9) | 64.49 (−2.2) |
| Llama 3.2 3B | 24.33 (**−0.7**) | 37.04 (−25.9) | 41.30 (−25.4) |

(v1 eval. Gemma 3 4B's blind-betting lead comes from answering `chaal` on 130 of 138
items against a 67%-`chaal` key — a prior, not a strategy.)

---

## 4. Finding 1 — the conversion failure is universal and order-independent

On the 162 conversion (look / stay-blind) items, every model gives essentially **one answer
to every question, under both orderings**:

| model | v1 | shuffled | AA by own turn 1 / 2 / 3 / 4 |
|---|---|---|---|
| Gemma 3 12B | stay-blind 162 | stay-blind 162 | 25.0 / 31.6 / 49.1 / 80.2 |
| Mistral 7B | see 162 | see 162 | 75.0 / 68.4 / 50.9 / 19.8 |
| Gemma 3 4B | see 162 | see 162 | 75.0 / 68.4 / 50.9 / 19.8 |
| Qwen3-8B | see 159, stay-blind 3 | same | 75.0 / 68.4 / 50.9 / 20.9 |
| Qwen3-14B | see 162 | see 162 | 75.0 / 68.4 / 50.9 / 19.8 |
| Qwen3-4B | see 162 | see 162 | 75.0 / 68.4 / 50.9 / 19.8 |
| Phi-4 14B | see 146, stay-blind 16 | see 155, stay-blind 7 | 50.0 / 42.1 / 47.2 / 24.4 |
| Qwen3-30B-A3B | see 162 | see 162 | 75.0 / 68.4 / 50.9 / 19.8 |
| Llama 3.2 3B | see 162 | see 162 | 75.0 / 68.4 / 50.9 / 19.8 |

The key's `see` rate by turn is 75.0 / 68.4 / 50.9 / 19.8, and its `stay-blind` rate
25.0 / 31.6 / 49.1 / 80.2. **Eight models' per-turn accuracy equals one of those rows to the
decimal** — the score is a readout of the label mix, not a decision. Phi-4 is the only
model to vary its answer at all, and it scores worse than either constant.

**Gemma 3 12B's 62.96% is not progress.** It is exactly the always-`stay-blind` baseline.

**It is not a position artefact.** On the shuffled eval, `see` is listed first on only half
the conversion items, yet eight models still answer `see` on ≥95.7% of them and pick the
first-listed option only ~50% of the time. This settles the one open doubt — Llama,
whose v1 behaviour could not separate "prefers `see`" from "prefers the first slot": it
answers `see` on 100% of shuffled items while `see` is first on 50%.

The benchmark's central claim — that no model has learned conversion timing — holds for
all nine models.

---

## 5. Finding 2 — scale does not help, and the largest model collapses

| family | v1 Action Accuracy |
|---|---|
| Qwen3 | 4B **33.06** → 8B **33.89** → 14B **33.67** → 30B-A3B **29.06** |
| Gemma 3 | 4B **34.22** → 12B **36.17** |

Within Qwen3, scale buys nothing from 4B to 14B and then hurts. **Qwen3-30B-A3B answers
`chaal` on 82.7% of all items** and raises on only 12.0% of items where raise is legal. Its
seen-betting score, 24.93%, is indistinguishable from always answering `chaal` (25.00%).
The largest model in the set is one of the two most trivial.

Only Gemma improves with size, by about two points.

---

## 6. Finding 3 — mid-size models read their cards, then misapply it

### Card sensitivity

Across the 74 betting nodes with at least six different hands in eval, how many does each
model answer identically regardless of the hand? (Solver: **11/74**.)

| model | identical answer regardless of cards |
|---|---:|
| Phi-4 14B | **10/74** (with `chal` merged into `chaal`) |
| Qwen3-14B | **18/74** |
| Gemma 3 12B | **32/74** |
| Qwen3-8B | 39/74 |
| Qwen3-30B-A3B | 49/74 |
| Qwen3-4B | 60/74 |
| Gemma 3 4B | 66/74 |
| Llama 3.2 3B | 70/74 |
| Mistral 7B | 73/74 |

### Folding tracks hand strength — for these three

Fold rate by hand-strength band, weakest to strongest (seen betting, v1):

| | 0–4 | 5–9 | 10–14 | 15–19 | 20–24 |
|---|---:|---:|---:|---:|---:|
| **Solver (should fold)** | 55.6 | 37.9 | 26.1 | 13.2 | 8.2 |
| Gemma 3 12B | 63.9 | 42.8 | 34.2 | 24.7 | 12.6 |
| Phi-4 14B | 56.1 | 54.3 | 45.0 | 29.4 | 1.9 |
| Qwen3-14B | 30.7 | 15.6 | 15.8 | 5.9 | 3.0 |
| Llama 3.2 3B | 76.1 | 78.4 | 81.1 | 73.2 | 80.5 |

Gemma 3 12B, Phi-4 and Qwen3-14B fold the weak hands and keep the strong ones. Llama folds
~75% of the time whatever it holds. Qwen3-14B also has the **best seen-betting score of all
nine models** (35.07%, +10.1 over baseline).

### …and then fold while blind

On blind-betting items — where the player has *not* looked and folding is never the
correct answer —

| model | folds on blind-betting items | blind-betting AA |
|---|---:|---:|
| Qwen3-14B | **100 of 138** | 14.49% |
| Phi-4 14B | **101 of 138** | 12.32% |

The two most card-sensitive models post the two worst blind-betting scores in the table.
The pattern is consistent with folding as a response to *uncertainty* rather than to a bad
hand: not knowing one's cards reads to them as a reason to quit. That is exactly the
decision the blind/seen mechanic makes costly, and the most direct evidence in this study
that these models do not understand what staying blind is for.

---

## 7. Finding 4 — models choose the word, not the slot

On the shuffled eval, answers spread across every menu position while the action words
chosen stay nearly the same. Menu-position shares on betting items:

| model | v1 | shuffled |
|---|---|---|
| Qwen3-30B-A3B | p1 90.0, p2 9.5 | p0 22.4, p1 26.3, p2 29.7, p3 21.6 |
| Llama 3.2 3B | p0 74.9, p1 24.5 | p0 18.8, p1 31.1, p2 36.8, p3 13.3 |
| Mistral 7B | p1 22.7, p2 77.3 | p0 27.5, p1 30.1, p2 27.7, p3 14.6 |
| Qwen3-8B | p1 50.9, p2 48.2 | p0 22.2, p1 31.3, p2 30.4, p3 16.1 |

A slot-picker would keep the v1 positions; these models follow the action. Two direct
consequences:

- **Llama never raises, in either ordering.** Raise recall is 0.0% on v1 and 0.0% on
  shuffled, including when `raise` is listed first.
- **`show` is suppressed by `raise` being on the menu, not by being listed last.** For
  Qwen3-8B, `show` recall on show-correct items is **53.0%** when `raise` is not offered and
  **2.4%** when it is (21.4% and 0.0% on shuffled). In v1, "show is 4th" and "raise is
  offered" were the same condition, which is what made them indistinguishable.

Menu order is a real but secondary effect — worth 1.3–2.2 points for most models.

---

## 8. Finding 5 — the models do not agree with each other

Pairwise agreement on the same 2,000 items (v1):

- highest: Gemma 3 4B ↔ Qwen3-4B **76.3%**, Gemma 3 4B ↔ Mistral 72.7%
- lowest: Gemma 3 12B ↔ Mistral **11.2%**, Phi-4 ↔ Qwen3-30B-A3B 24.6%
- Llama agrees with every other model on 24.7–32.6%

The field is not converging on a strategy. Each model carries its own fixed prior: Mistral
raises (1,336 of 2,000), Qwen3-30B-A3B calls (1,654), Llama folds (1,376), Gemma 3 12B
calls and folds but almost never raises (14).

---

## 9. Answer vocabulary

Recall on the correct label (v1):

| model | chaal | pack | raise | show | see | stay-blind |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 12B | 72.2 | 55.5 | 1.0 | 0.0 | 0.0 | 100.0 |
| Mistral 7B | 34.9 | 0.0 | 100.0 | 7.1 | 100.0 | 0.0 |
| Gemma 3 4B | 56.5 | 0.0 | 75.3 | 0.0 | 100.0 | 0.0 |
| Qwen3-8B | 53.3 | 0.0 | 59.5 | 16.9 | 98.3 | 2.0 |
| Qwen3-14B | 38.5 | 22.9 | 70.4 | 1.7 | 100.0 | 0.0 |
| Qwen3-4B | 56.5 | 0.0 | 69.3 | 0.5 | 100.0 | 0.0 |
| Phi-4 14B | 5.1 \* | 56.0 | 68.8 | 3.7 | 83.3 | 5.9 |
| Qwen3-30B-A3B | 86.3 | 0.0 | 14.9 | 0.5 | 100.0 | 0.0 |
| Llama 3.2 3B | 34.5 | 67.2 | 0.0 | 2.2 | 100.0 | 0.0 |

\* depressed by the `chal` spelling.

**No model finds `show` more than 17% of the time.** Five models never fold correctly
(recall 0.0%). Apart from Gemma 3 12B, which says it every time, `stay-blind` is almost
absent: Phi-4 says it 16 times, Qwen3-8B 3 times, the other six never.

---

## 10. Caveats

1. **Different GPU for one model.** Qwen3-30B-A3B ran on an A100 80 GB; the rest on an A100
   40 GB. Same precision and code, but hardware can shift a few greedy outputs — Gemma 3 4B
   scored 33.94% on an L4 and 34.22% here at identical settings.
2. **These supersede the earlier Kaggle files** in `results/old/`. Those were float16 on a
   T4; this batch is bfloat16 throughout. Re-runs differ slightly (Mistral 34.56 → 35.56,
   Qwen3-8B 32.94 → 33.89). Report this batch, where all nine share one setup.
3. **Single-shot, no reasoning.** Thinking off, 16 new tokens. Nothing here speaks to what
   these models could do with chain-of-thought.
4. **Small strata.** Conversion turn 1 has 4 items and blind-betting turn 1 has 4. Report,
   don't interpret.
5. **Exact Match and TVD are not measured.** No amounts or distributions were requested.

---

## 11. Corrections to earlier documents

This batch overturns one claim made in several earlier documents. **Those statements are
wrong and should not be cited:**

| earlier claim | what the data shows |
|---|---|
| Qwen3-8B's `show` recall falls "50× from one slot" because of menu position (error_analysis §4) | The driver is whether `raise` is offered (53.0% vs 2.4%). Position and raise-availability were confounded in v1. |
| Llama's zero raises and shows are a layout artefact, since it "never selects past index 1" (error_analysis §5, error_situations §3.4, RESULTS_EXPLAINED §10a) | Llama still never raises when `raise` is listed first. It is a genuine behaviour. |
| Llama's contribution to the conversion finding is "unsafe" | Resolved: its `see` answer follows the word, not the slot. |
| Randomising action order is the single highest-value fix before fine-tuning (FINETUNE_STRATEGY, training_signal §6.3) | Still worthwhile, but worth only 1.3–2.2 points. It is a hygiene fix, not the main lever. |

Affected files: `error_analysis.md`, `error_situations.md`, `RESULTS_EXPLAINED.md`,
`training_signal.md`, `FINETUNE_STRATEGY.md`.

---

## 12. Summary for the paper

- **Headline.** Nine open models, 3B–30B, score 26.8%–36.2% against a 25.9% one-word
  floor. Best: Gemma 3 12B.
- **Robust.** No model has learned conversion timing; all nine give a single fixed answer,
  and this survives shuffling the action order.
- **Scale.** No reliable benefit. Qwen3 is flat from 4B to 14B and the 30B MoE collapses to
  near-constant `chaal`.
- **New.** Gemma 3 12B, Phi-4 and Qwen3-14B genuinely condition on their cards — and the two
  most card-sensitive fold on ~73% of blind-betting items, where folding is never correct.
- **Methodological.** Presentation order matters by 1–2 points; it does not explain any of
  the major failure modes.

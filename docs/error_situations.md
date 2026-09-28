# Situation-Level Error Analysis — What Triggers Each Model's Mistakes

Companion to [error_analysis.md](error_analysis.md), which covers *what* the models
get wrong. This one covers *when*.

**Data.** 1,800 graded eval items × 4 models. All features derived from
`meta.betting_history` by replaying the actor sequence — validated against
`meta.turn_index` with **0 mismatches in 1,800 items**. No speculation: every
number below is computed from `results/*.jsonl` against
`data/teenpattibench_eval.jsonl`.

---

## 0. Two structural facts that must come first

**Number of players is not a variable.** The game is heads-up by construction
(`RULES.md` §1.3). It is 2 in all 1,800 items and cannot explain any variance. Listed
here only because it was asked for.

**Stake, raises-used and "both seen" are not independent causes — they are the menu.**
Verified exactly true across all 1,638 graded betting nodes:

| identity | holds? |
|---|---|
| `stake == 4` ⟺ `raise` illegal | **True**, 1638/1638 |
| `raises_used == 2` ⟺ `raise` illegal | **True**, 1638/1638 |
| `show` on the menu ⟺ (I am blind **or** opponent is seen) | **True**, 0 exceptions |

So any apparent "error rate falls at stake 4" finding is really "error rate falls when
`raise` is removed from the menu". Throughout this document the **menu is the master
variable** and stake/raises/both-seen are reported as aliases, not as separate causes.

*(The equivalence fails only if look nodes are included, because their menu is
`[see, stay-blind]` regardless of stake. That is a definitional artefact, not a
counterexample.)*

---

## 1. The master rule: what happens when `raise` is on the menu

`raise` is legal on 1,230 of the 1,800 graded items. Each model's entire betting
personality is a fixed response to its presence.

| | says `raise` when offered | error **with** raise on menu | error **without** | Δ |
|---|---:|---:|---:|---:|
| Qwen3-8B | 53.2% | 74.7% | 50.5% | **+24.2** |
| Qwen3-4B | 69.3% | 69.9% | 60.4% | +9.6 |
| Mistral-7B | **99.8%** | 68.4% | 59.1% | +9.3 |
| Llama-3.2-3B | **0.0%** | 79.0% | 56.7% | **+22.4** |

Every model is substantially more accurate once `raise` is removed. Mistral takes it
1,227 of 1,230 times; Llama never takes it at all. Neither is a judgement about
raising.

---

## 2. Within a fixed menu — what still separates?

Conditioning on the single largest menu, `[pack, chaal, raise, show]` (n=860), removes
the confound. What survives is genuine state sensitivity — and it splits the models
into two camps.

| feature | Qwen3-8B | Qwen3-4B | Mistral-7B | Llama-3.2-3B |
|---|---|---|---|---|
| opponent's last action | **+16 pts** | **+19 pts** | **+26 pts** | 1 pt |
| own turn index | **+35 pts** | **+35 pts** | **+43 pts** | 9 pts |
| seat | **+24 pts** | +19 pts | +20 pts | 7 pts |
| opponent ever raised | 10 pts | +19 pts | +18 pts | 2 pts |
| **hand strength** | 10 pts | 12 pts | 14 pts | **+27 pts** |

*(spread = worst-cell minus best-cell error rate)*

**Qwen3-8B / Qwen3-4B / Mistral react to the betting history and ignore their cards.
Llama does the reverse — it ignores the history entirely and its only sensitivity is to
its hand, in the wrong direction.**

Direction of the surviving effects, within this fixed menu:

- **Opponent just raised → worse.** Qwen3-8B 95% error vs 79%; Mistral 92% vs 66%.
- **Early turns → worse.** Turn 1: 97% / 92% / 76% error; turn 4: 62% / 61% / 47%.
- **Seat 2 → worse.** Qwen3-8B 91% vs 67% for seat 1.
- **Llama, stronger hand → worse.** 68% error on weak hands, **88%** on strong.

---

## 3. Per-model situation breakdown

### 3.1 Qwen3-8B — 1,207 errors

| situation | correct | it says | n | miss rate |
|---|---|---|---:|---:|
| `show` correct, 4-action menu | show | raise / chaal | 291 | **99.0%** |
| `show` correct, 3-action menu | show | chaal | 117 | 50.4% |
| `pack` correct vs a seen opponent | pack | chaal / raise / show | 247 | **100%** |
| `chaal` correct while raise available, stake ≤ 2 | chaal | raise | 288 | 69.8% |
| `raise` correct, full menu, both seen | raise | chaal | 247 | 57.5% |
| turn 1–2 on the full menu | — | raise / chaal | 279 | **93.9%** |

`show` recall is 49.6% when show sits at index 2 and **1.0%** at index 3 — the same
action, one slot later. Its `raise` misses concentrate where both players are seen
(57.5% vs 17.7% otherwise), i.e. deep in the tree.

> `eval-000005` · seat 1, turn 3, pot 14, both seen, bucket 14
> history `[stay-blind, raise, see, chaal, stay-blind, chaal, chaal, see]`
> menu `[pack, chaal, raise, show]` · truth **pack** (solver 0.762) → said **chaal**

### 3.2 Qwen3-4B — 1,204 errors

Same family, same shape, narrower vocabulary. `show` recall is 0.7% overall and 2.6%
even at index 2 — it has essentially lost the action that the 8B still half-retains.

| situation | correct | it says | n | miss rate |
|---|---|---|---:|---:|
| `show` correct, any menu | show | raise / chaal | 408 | **99.3%** |
| `pack` correct vs a seen opponent | pack | chaal / raise | 247 | **100%** |
| `chaal` correct, raise available, stake ≤ 2 | chaal | raise | 288 | 67.4% |
| turn 1–2 on the full menu | — | raise / chaal | 279 | **95.0%** |
| opponent just raised, full menu | — | raise | 127 | 92.1% |

Its `raise` errors invert the 8B's: worst at turn 4 (38.8%) and stake 2 (42.2%), best
at turn 3 (17.3%).

### 3.3 Mistral-7B — 1,178 errors — a single rule explains most of it

**It raises 1,227 of the 1,230 times raise is legal (99.8%).** Almost every error
follows mechanically.

| situation | correct | it says | n | miss rate |
|---|---|---|---:|---:|
| `chaal` correct while raise available, stake ≤ 2 | chaal | **raise** | 288 | **99.7%** |
| `show` correct, 4-action menu | show | **raise** | 291 | **100%** |
| `pack` correct vs a seen opponent | pack | **raise** | 247 | **100%** |
| `raise` correct, full menu, both seen | raise | — | 247 | **0.0%** |
| turn 1–2 on the full menu | — | raise | 279 | 86.7% |

The 0.0% row is the tell: perfect recall on `raise` is not skill, it is the same rule
paying off. When `raise` is removed, its `chaal` miss rate collapses from ~100% to
22.7% — so the competence is there and is being overridden.

It is also the only model to answer **illegally**: 15 times it said `raise` where the
raise cap made raise illegal (6 on graded items).

### 3.4 Llama-3.2-3B — 1,295 errors — layout-bound, and inverted

It never selects a menu index above 1, across all 1,838 betting items. `raise` and
`show` never appear below index 2.

| situation | correct | it says | n | miss rate |
|---|---|---|---:|---:|
| correct answer at menu index ≥ 2 | raise / show | pack / chaal | 796 | **100%** |
| `raise` correct, any menu | raise | pack | 388 | **100%** |
| `show` correct, any menu | show | pack | 408 | **100%** |
| strong hand (bucket 17–24), seen betting | various | pack | 583 | **82.3%** |
| `chaal` correct, full menu | chaal | pack | 135 | 97.8% |

**48.6% of graded betting items are unreachable for it by construction.**

Its folding is anti-correlated with hand strength — 68% error on the weakest hands,
**89%** on the strongest — and unlike the other three it is insensitive to the
opponent's last action (1-point spread).

One exception worth noting: Llama is the *only* model that folds correctly at all
(63.2% recall on `pack`), and it does so best on the full 4-action menu (9.8% miss)
and worst on the 2-action menu (100% miss). Same inversion as everyone else — see S9.

---

## 4. Shared vs model-specific situations

### Shared by all four

| # | situation | n | Q8B | Q4B | Mis | Lla |
|---|---|---:|---:|---:|---:|---:|
| **S9** | `pack` correct, menu is only `[pack, chaal]` | 39 | **100%** | **100%** | **100%** | **100%** |
| **S1** | conversion at turn 3–4, truth `stay-blind` | 95 | 98.9% | 100% | 100% | 98.9% |
| **S2** | `show` correct on the 4-action menu | 291 | 99.0% | 100% | 100% | 100% |
| **S7** | turn 1–2 with the full menu | 279 | 93.9% | 95.0% | 86.7% | 80.3% |
| **S8** | opponent has just raised, full menu | 127 | 95.3% | 92.1% | 92.1% | 82.7% |

**S9 is the sharpest result in this analysis.** When the menu is only `[pack, chaal]`
and folding is correct, **all four models fail all 39 items, all saying `chaal`** —
including Llama, which folds 65% of the time everywhere else. Stripping the menu to two
options removes every model's ability to fold.

> `eval-000196` · seat 2, turn 3, pot 12, stake 4, opponent blind, bucket 2
> history `[stay-blind, chaal, stay-blind, chaal, stay-blind, raise, stay-blind, chaal, stay-blind, raise, see]`
> menu `[pack, chaal]` · truth **pack**, solver **0.996** → all four said **chaal**

A near-certain fold, missed unanimously.

### Model-specific

| model | situation | n | miss | others |
|---|---|---:|---:|---|
| Mistral-7B | any menu containing `raise` | 1,230 | raises 99.8% | Qwen 53–69%, Llama 0% |
| Llama-3.2-3B | correct answer at index ≥ 2 | 796 | **100%** | Qwen ~65%, Mistral 48% |
| Llama-3.2-3B | strong hand, seen betting | 583 | 82.3% | others 58–63% |
| Qwen3-8B | `show` at index 2 | 117 | 50.4% | Q4B 97.4%, Mistral 80.3%, Llama 100% |
| Qwen3-8B / Q4B | `pack` vs seen opponent | 247 | **100%** | Llama 26.3% |

Qwen3-8B retaining half its `show` recall at index 2 is the only place any model shows
a capability the others lack.

---

## 5. The 10 situations to target with training data

Ranked by (items affected × universality × whether training can actually fix it).

| # | situation | eval n | worst / best miss | why it matters |
|---|---|---:|---|---|
| **1** | `show` correct with 4 actions on the menu | 291 | 100% / 99% | Largest single universal failure. 345 of the 720 universally-failed items are `show`. Partly layout — fix ordering **and** add data. |
| **2** | `chaal` correct while `raise` is legal | 288 | 99.7% / 67.4% | The raise reflex. Needs examples where calling beats raising at stake 1–2. |
| **3** | `pack` correct against a **seen** opponent | 247 | 100% / 26.3% | Three models never fold here. Deep-tree folds vs an informed opponent. |
| **4** | `pack` correct on a **2-action** menu | 39 | 100% / 100% | Small but 100% universal, solver confidence up to 0.996. Pure, unambiguous signal — cheapest fix in the list. |
| **5** | conversion at turn 3–4, `stay-blind` correct | 95 | 100% / 98.9% | The research object. `stay-blind` is 0.88% of the 100k training set; recall is 0–2%. |
| **6** | turn 1–2 with the full 4-action menu | 279 | 95% / 80% | Early-street play is worst for everyone; the tree is widest and least constrained here. |
| **7** | opponent has just **raised**, full menu | 127 | 95% / 83% | The only opponent-action feature that survives menu conditioning. No model responds correctly to aggression. |
| **8** | `raise` correct, full menu, both seen | 247 | 100% (Llama) / 0% (Mistral) | Maximally discriminating: one model gets all, one gets none. Good curriculum item. |
| **9** | strong hand (bucket 17–24), seen betting | 583 | 82% / 58% | Where card-reading should pay. Llama is *anti*-correlated here. |
| **10** | correct answer at menu index ≥ 2 | 796 | 100% (Llama) / 48% | Not a game situation — a layout artefact. Fix by randomising order, **not** by adding data. |

### How to use this list

**Items 1, 3, 4, 5, 7** are genuine knowledge gaps: the situations exist, the solver
has a clear answer, and no model produces it. These deserve oversampling in the
training mix.

**Items 2, 8** are reflex-override problems — the model has the right action in
vocabulary but a competing default wins. Contrastive pairs (near-identical states where
the answer flips between chaal and raise) will do more than volume.

**Item 10 is not a training target at all.** Adding data for it would teach the model
to compensate for a formatting artefact. Randomise the action order first
(`prompts.py` now accepts `action_order`; `make_eval_shuffled.py` does eval), then
re-measure — several of the rates above will move, and items 1 and 8 in particular
should be re-derived afterwards.

**Item 6** is the broadest and the least diagnosed: early-turn play is worst for all
four but no single feature explains it. Worth a dedicated look before spending data
on it.

---

## 6. What this changes about the previous conclusions

[error_analysis.md](error_analysis.md) concluded the models "barely use the public
state". This analysis refines that: **three of the four do use it** — turn index,
opponent's last action and seat all survive menu conditioning with 16–43 point
spreads. What they lack is not sensitivity to the betting history but the *right*
mapping from it, plus a hard vocabulary block on 1–3 actions.

Llama is the exception and should be treated as a separate case: it is insensitive to
the history, sensitive to its hand in the wrong direction, and capped at 55.78% by a
layout artefact before it answers anything.

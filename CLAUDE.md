# CLAUDE.md — heads-up Teen Patti solver and benchmark

Read this first. It is the handoff between sessions.

---

## What this project is

An academic paper on **heads-up (two-player) Teen Patti**, the Indian three-card
game. There is essentially no prior AI or game-theory work on it: no solver, no
benchmark, no OpenSpiel or RLCard implementation, no published equilibrium.

**The research object is the blind/seen mechanic.** A player may bet without
looking at their cards at **half** the stake a seen player pays, and may convert
to seen at the start of any of their own turns, irreversibly. In game-theoretic
terms a player may voluntarily place themselves in a *coarser information
partition* in exchange for a price discount. No studied poker variant has this.

Methodology follows **PokerBench** (Zhuang et al., AAAI 2025, arXiv:2501.08328):
solve the game exactly, then use the solution as an answer key to grade LLMs.

`docs/RULES.md` (Phase 0) is the frozen contract. **Do not change it, the solver, or
the abstraction** without the user explicitly asking.

---

## Status

| Phase | What | State |
|---|---|---|
| 0 | Frozen rules specification | Complete — `docs/RULES.md` |
| 1 | Evaluator, engine, abstraction, tree enumeration | Complete — `docs/PHASE1.md`, 113 tests |
| 2 | CFR+ solve, exploitability, sweeps | Complete — `docs/PHASE2.md`, `docs/RESULTS_PHASE2.md`, 37 tests |
| 3 | TeenPattiBench dataset + metrics | Complete — `docs/PHASE3.md`, 18 tests |
| 4 | LLM evaluation | First model scored — `docs/PHASE4.md` |

**168 tests pass, ~86 s.**

**Phase 4 so far.** Qwen3-8B (fp16, Kaggle GPU) over all 2,000 eval items:
**32.94% Action Accuracy**, against 25.94% for the best trivial policy. Read every
stratum against *its own* baseline — conversion is **−25.3** and blind betting
**−11.6** relative to theirs, i.e. both are worse than answering one word forever.
Two structural findings: the model **never packs** (0 of 2,000, though pack is correct
on 25% of seen items), and it gives **one identical answer across every hand** in 40 of
74 public nodes — it is not reading its own cards. `docs/PHASE4.md` has the detail and
the caveats, which matter: a 4-bit run of the same model agreed with the fp16 run on
only 62.5% of items.

`docs/EXPLAINER.md` is a plain-language walkthrough of Phases 1–2 for non-specialists
(supervisor, viva, reviewer). Useful for recovering the *why* quickly.

---

## The numbers that matter

| Quantity | Value |
|---|---|
| Exploitability of the solution | **6.127378e-04 boots/hand** = 0.031% of pot |
| Game value to seat 1 | **−0.039781** boots/hand |
| Solve | 1,000 CFR+ iterations, 54.9 min |
| Public tree | 5,929 nodes, 2,014 decision nodes, 3,915 terminals |
| Hand abstraction | 22,100 hands → **1,755** suit-isomorphic classes, **lossless** |
| Seen information sets | 1,286 × 1,755 = **2,256,930** (913,825 reachable) |
| Blind information sets | **728** total, **486** reachable, **428** dominant |
| Max exposure in one hand | 33 units, so any stack ≥ 34 is the identical game |
| Caps | `k ≤ 8` betting actions, `r ≤ 2` raises |

**Headline equilibrium findings.** Seat 1 stays blind at its first decision with
p = 0.999965; seat 2 looks with p = 0.994824. Conversion timing for seat 1 is
**non-monotone**: P(look | own turn *j* reached while blind) = **0.000035, 0.786,
0.058, 0.931** for turns 1–4. No, yes, no, yes. Nothing in poker predicts this,
and it is the benchmark's primary target.

---

## Invariants that must never break

### 1. The peeking trap (the single most important thing here)

A blind player has not looked. Their information set is the **public history
alone** (`docs/RULES.md` §12.4). If a blind player's strategy, regrets, or prompt are
ever indexed by their own cards, the program learns a fantasy — *"bet huge while
blind, but only holding three aces"* — and **it does not crash, does not warn,
converges beautifully, and produces a confident worthless answer** that is
*better* than achievable. It corrupts precisely the question the paper asks.

Enforced structurally in several independent places. Do not weaken any of them:

- `BlindInfoSet` has **no card field** and uses `__slots__`.
- `Deal.hand_for()` raises `BlindAccessViolation` for an unseen player.
- Blind regret arrays are allocated with shape `(n_actions,)` — **no class axis**.
- `_assert_blind_reach_is_scalar` fires on every traversal.
- `verify_blind_best_response_is_single_action` — the best response must pick
  **one** action at a blind node; per-class there would *overstate* exploitability.
- `tree.verify_blind_infosets_are_card_independent()` — all 728 blind nodes
  against three disjoint deals.
- Phase 3: `prompts.assert_no_private_leak` on every blind prompt, plus
  `test_every_blind_decision_node_renders_without_leaking`.

**Diagnostic tell:** the blind player has **235** reachable betting infosets
against the seen player's **913,825**. If that number is ever in the hundreds of
thousands, something is peeking.

### 2. Always use the **average** strategy, never the final iterate

`solver.average_strategy(node)` / `strategy_profile(average=True)`. The guarantee
attaches to the average. They genuinely differ (max delta ≈ 0.999).

### 3. Never group by κ (741 strength classes) for the solver

κ is correct for *who wins* and wrong as a solver abstraction — 442 of 741 κ
classes (60%) contain hands with different equities. Use the 1,755 suit-isomorphic
classes. `cfr.PrivateTables.class_of_hand` and
`abstraction.iso_class_of_hand` **must stay identical**; `src/generate.py` asserts it.

### 4. Exact arithmetic at terminals

396 of 925 forced-show terminals (42.8%) have an **odd pot**, so §10.6's `p/2 − c`
split is a half-integer. `Fraction` is used throughout. Integer division silently
breaks zero-sum on 43% of forced showdowns.

---

## Module map

| File | Owns | Notes |
|---|---|---|
| `src/hands.py` | Deck, ranking, κ classes, exact equity | Import-time self-checks |
| `src/game.py` | State machine, legality, pricing, payoffs, info-set types | `Fraction` payoffs |
| `src/abstraction.py` | 1,755 suit-isomorphic orbits, 25 equity buckets | Burnside cross-check |
| `src/tree.py` | Public tree enumeration, blind-infoset verification | |
| `src/cfr.py` | CFR+ solver, private tables, card-removal matrix | `average_strategy()` |
| `src/exploitability.py` | Best response, exploitability, convergence logging | |
| `src/experiments.py` | Sweeps, blind stats, dominance; `turn_index`, `node_reach`, `seen_infoset_reach` | Phase 3 reuses these three |
| `src/gate_phase2.py`, `src/run_main_solve.py` | Phase 2 gate and solve driver | |
| **`src/prompts.py`** | English templating + the leakage guard | Phase 3 |
| **`src/generate.py`** | Dataset generation pipeline | Phase 3 |
| **`src/metrics.py`** | AA / EM / TVD scoring and breakdowns | Phase 3 |

---

## Environment and commands

Python 3.14 venv at `.venv`; deps are just `numpy` and `pytest`.
**Run everything from `~/Projects/teenpatti`** — no `PYTHONPATH` needed.

```bash
source .venv/bin/activate          # or prefix with .venv/bin/
python -m pytest -q                # 168 tests, ~86 s
python src/generate.py --dry-run       # inventory + composition, writes nothing
python src/generate.py                 # regenerate datasets (deterministic, seed 20260912)
python src/metrics.py --self-test      # must print 100.00% AA / 100.00% EM / TVD 0
python src/run_main_solve.py 1000 50   # re-solve from scratch (~55 min) — rarely needed
```

### Files NOT in git (regenerable, gitignored)

- `data/solution_default.pkl` — **107 MB checkpoint, the answer key.** Everything
  depends on it. Regenerate with `src/run_main_solve.py 1000 50` (~55 min). Exceeds
  GitHub's 100 MB hard limit, so it can never be committed without Git LFS.
- `data/teenpattibench_train.jsonl` — 80 MB, regenerates in ~10 s.
- `.venv/`, `__pycache__/`, `.pytest_cache/`

Committed: `data/teenpattibench_eval.jsonl` (3.2 MB), `data/teenpattibench_stats.json`,
`data/teenpattibench_system_prompt.txt`.

### Git identity

`~/Projects/**` is wired to the personal GitHub account **dbXD320** via an
`includeIf` in `~/.gitconfig` plus a URL rewrite to the SSH alias
`github.com-dbxd320` (key `~/.ssh/id_ed25519_dbxd320`). Everything outside
`~/Projects` stays on `devansh.baghla@fitsol.green`. Remote:
https://github.com/dbXD320/teenpatti1 — **public**.

---

## Phase 3 dataset, in brief

`data/teenpattibench_eval.jsonl` — **2,000 items**: 1,500 seen betting, 162 look
(conversion), 138 blind betting, 200 mixed (TVD-scored, not graded).
`data/teenpattibench_train.jsonl` — **50,000 items**, disjoint at the (node, class)
level. Each record carries prompt, correct action, correct amount, the **full
solver distribution**, and metadata (seat, blind/seen, own-turn index, hand
bucket, betting history, actions to cap).

**The blind pool is enumerated, not sampled.** Only 428 dominant blind infosets
exist in the entire game, so eval takes 70% (300 items = **15.0%** of eval,
against 0.0532% natural — a 282× oversampling) and training gets the remaining
128 distinct, emitted 20× each. **The 30–40% blind eval share in the Phase 3
brief is unreachable**; `docs/PHASE3.md` §2 documents the ceiling and the trade-offs.
Duplicating blind items would be meaningless — a blind prompt is a deterministic
function of public history, so two items from one infoset are byte-identical.

Trivial baselines on eval: always-chaal 25.94%, always-pack 20.83%,
always-stay-blind 5.67%. (PokerBench saw ~90% for fold-everything before
rebalancing.)

---

## Traps already hit — don't rediscover these

1. **`docs/RULES.md` §5.3's card-removal intuition is inverted.** It says K♠K♥7♠
   "removes a third spade". It holds **two** spades and performs **worse** than
   K♠K♥7♦ (equity 0.89619518 vs 0.89679223, exactly 11 more losses). Concentrating
   removals in one suit leaves the others *fuller*, and three-card flush counts
   grow faster than linearly. The spec's conclusion is right, its stated reason is
   backwards. docs/PHASE1.md §3 has the arithmetic.

2. **`docs/RULES.md` §10.3's prediction is wrong.** It says a solved strategy not using
   the blind show right is "almost certainly an implementation bug". Measured at
   equilibrium: 1.1e-05 (seat 1) and 5.1e-03 (seat 2), robust across all six cap
   configurations. Seen players demand shows 10–100× more often. Not a bug — the
   hands that most want a showdown are the hands most worth looking at, and
   looking forfeits the right. Recommendation is to soften the prediction, change
   no rule.

3. **Blind prompts: two phrasings that look innocent and are not.**
   *"your **three** cards"* — "three" is a rank word. *"**As** a blind player…"* —
   `As` is the Ace of Spades card code. Both were caught by the leakage guard and
   reworded. Do not loosen the guard to allow sentence-initial tokens.

4. **Stratify-then-rebalance starves rare labels.** `raise` and `show` are 60,653
   and 51,050 of 913,825 against 454,237 `chaal`. Sampling across textures first
   and rebalancing after undershoots badly. `src/generate.py` draws an equal quota
   from *each label's own strata* in one pass.

5. **Unresolved:** `docs/PHASE1.md` §7 gives 2♣2♦3♥ an equity of 0.7403; the code gives
   **0.74080**. The A♠K♥J♦ figure (0.7428) matches. The finding (weakest pair has
   lower equity than best high card) holds either way. Don't know which is the typo.

---

## Phase 4 — what's next

Run the LLM evaluation. The dataset and scorer are ready and were built so that
no model call is needed to validate them.

**The hypothesis to test:** models do relatively well on ordinary betting
decisions — transferable from abundant poker text in pretraining — and **fail on
conversion timing**, where no corresponding text exists and the correct pattern is
non-monotone and unguessable. If it holds, this is a cleaner probe of reasoning
versus retrieval than PokerBench could run.

`src/metrics.py` reports the per-turn conversion breakdown that tests this. **Never
aggregate conversion accuracy across turns** — it averages the turn-3 dip and the
turn-4 spike into a meaningless number.

Two cautions carried forward: **turn-1 conversion rests on 4 eval items** (only 5
exist in the game) and is a control stratum, not a test — report it, don't
interpret it. **Turn 3 is the most informative stratum**: the solver says *don't*
look, having looked willingly at turn 2.

Prediction file format for `src/metrics.py --predictions`: JSONL with `id`, `action`
(raw text is parsed), optional `amount` for Exact Match, optional `distribution`
for the mixed subset. A missing prediction raises rather than scoring as wrong.

---

## Working style the user expects

- Report pass/fail on gates before generating anything; stop if one fails.
- No silent defaults — raise rather than emit something malformed.
- Flag discrepancies in the source material rather than smoothing them over; a
  flagged gap is more useful than a confident guess.
- If a strategy looks implausible or degenerate on inspection, stop and say so
  rather than building on it.
- Verify claims against the code, not against the reports.

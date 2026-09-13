# Phase 2 — Results

Distilled results only. Methodology, solver design and justification are in
[PHASE2.md](PHASE2.md); this file is the numbers.

All figures are from the **average strategy** of the default configuration
`boot=1 stack=50 k<=8 r<=2`, 1,000 CFR+ iterations, 54.9 min wall time.
Source data: [convergence_default.json](../data/convergence_default.json),
[experiments_results.json](../data/experiments_results.json),
[solve_default.log](solve_default.log), [experiments.log](experiments.log).

---

## 1. Headline

| Quantity | Value |
|---|---:|
| Exploitability | **6.127e-04 boots/hand** |
| — in milli-boots | 0.613 mb/hand |
| — as % of starting pot | **0.0306%** |
| Game value to seat 1 | **−0.039781 boots/hand** |
| Iterations / wall time | 1,000 / 54.9 min (3,293.9 s) |
| Per-iteration cost | 1.82 s |
| Public nodes / private classes | 5,929 / 1,755 |
| Tests passing | **150** (37 Phase 2 + 113 Phase 1) |

The solver is converged to an answer key at 0.031% of the pot.

---

## 2. Convergence

| iter | exploitability | mb/hand | % of pot | v₀ | current iterate |
|---:|---:|---:|---:|---:|---:|
| 50 | 9.340e-02 | 93.400 | 4.6700% | −0.012970 | 1.487e-01 |
| 100 | 4.415e-02 | 44.147 | 2.2073% | −0.021181 | 7.054e-02 |
| 200 | 1.130e-02 | 11.303 | 0.5651% | −0.035671 | 4.182e-03 |
| 300 | 5.086e-03 | 5.086 | 0.2543% | −0.038060 | 1.533e-03 |
| 400 | 2.922e-03 | 2.922 | 0.1461% | −0.038876 | 1.156e-03 |
| 500 | 1.927e-03 | 1.927 | 0.0963% | −0.039251 | 7.040e-04 |
| 600 | 1.397e-03 | 1.397 | 0.0698% | −0.039457 | 6.264e-04 |
| 700 | 1.079e-03 | 1.079 | 0.0540% | −0.039586 | 4.988e-04 |
| 800 | 8.694e-04 | 0.869 | 0.0435% | −0.039673 | 4.292e-04 |
| 900 | 7.205e-04 | 0.720 | 0.0360% | −0.039735 | 3.685e-04 |
| 1000 | **6.127e-04** | **0.613** | **0.0306%** | **−0.039781** | 3.241e-04 |

- **Monotone decreasing** at all 20 logged points; no plateau (last/third-last ratio 0.776).
- Fitted decay **~ T^−1.78**.
- Iterations to 10⁻²: **250**. To 10⁻³: **750**. 10⁻⁴ not reached in 1,000.
- Game value stabilises long before exploitability: v₀ moves only −0.039673 → −0.039781
  over the final 200 iterations.
- The **current iterate** is more exploitable early, less exploitable late (1.49e-1 vs
  9.34e-2 at iter 50; 3.24e-4 vs 6.13e-4 at iter 1,000). All reported statistics use the
  average, which is the strategy carrying the regret bound.

---

## 3. Central finding — equilibrium blind play is positional

| | seat 1 (acts first) | seat 2 (acts second) |
|---|---:|---:|
| **P(stays blind at own first decision)** | **0.999965** | **0.005176** |
| P(looks at own first decision) | 0.000035 | 0.994824 |

Seat 1 opens blind essentially always; seat 2 looks essentially always. **Neither decision
is mixed.** Seat 1's blind opening holds at 0.9999–1.0000 across every cap tested, so it is
not an artefact of the k ≤ 8 truncation. Seat 2's looking is *not* universal — see §7(b).

### 3.1 Conversion timing

`P(reach blind)` = probability the hand reaches that player's turn *j* with them still blind.

**Seat 1** — conversion concentrated at **turn 2**, right after seat 2 has revealed
information by acting:

| own turn | P(reach blind) | P(convert there) | P(look \| reached) |
|---:|---:|---:|---:|
| 1 | 1.000000 | 0.000035 | 0.000035 |
| 2 | 0.588575 | 0.462748 | **0.786217** |
| 3 | 0.125448 | 0.007328 | 0.058415 |
| 4 | 0.118116 | 0.109967 | **0.931006** |
| never looks | | **0.419922** | |

**Seat 2** — conversion is immediate:

| own turn | P(reach blind) | P(convert there) | P(look \| reached) |
|---:|---:|---:|---:|
| 1 | 0.999986 | **0.994811** | **0.994824** |
| 2 | 0.000372 | 0.000022 | 0.057836 |
| 3 | 0.000001 | 0.000000 | 0.218210 |
| 4 | 0.000000 | 0.000000 | 0.138042 |
| never looks | | **0.005167** | |

Seat 1's 42% "never looks" is mostly hands that *end* before a second turn — not hands
played blind to the end. Seat 2 packs after one betting action with probability 0.407.

---

## 4. Is the blind discount used?

| | seat 1 | seat 2 |
|---|---:|---:|
| Expected betting actions at the **half price** | **1.252** | 0.006 |
| P(demands a show while blind — the blind-only right) | 0.000011 | 0.005142 |
| P(demands a show while seen) | 0.102764 | 0.076012 |
| P(packs while still blind) | 0.000004 | 0.000012 |

- Seat 1 pays the blind rate on **~1.25 betting actions per hand**. Staying blind acts as a
  deferral of a decision that can be taken later, not as a gamble.
- **The blind player's exclusive show right is essentially never exercised** (1.1e-5 and
  5.1e-3). At equilibrium *seen* players demand shows 10–100× more often. See §9.

---

## 5. Hand outcomes

| terminal type | probability |
|---|---:|
| pack (fold) | 0.766584 |
| requested show | 0.183930 |
| cap-forced show | 0.049486 |
| **total** | **1.000000** |

Most common endings by public history:

| ending | mass |
|---|---:|
| seat 2 packs after 1 betting action | 0.4066 |
| seat 1 packs after 2 | 0.2321 |
| seat 1 demands a show after 2 | 0.0827 |
| seat 2 demands a show after 3 | 0.0596 |
| seat 2 packs after 3 | 0.0471 |

P(packs while seen): 0.295 seat 1, **0.471** seat 2. The modal hand is: seat 1 opens blind
at half price, seat 2 looks, seat 2 folds the bottom ~41% immediately.

**Position.** Acting first is worth **−0.039781 boots/hand** — about −4% of a boot. Seat 1
both stays blind and loses; the blind opening reads as the cheapest response to a structural
positional disadvantage, not a source of edge.

---

## 6. Dominant-action share (Phase 3 readiness)

| group | reachable infosets | unweighted | reach-weighted |
|---|---:|---:|---:|
| look (see / stay-blind) | 251 | 0.9203 | **1.0000** |
| bet, actor blind | 235 | 0.8383 | **1.0000** |
| bet, actor seen | 913,825 | 0.7963 | 0.9800 |
| **overall** | | **0.7963** | **0.9932** |

- Reach-weighted, **99.3%** of decisions have an action above probability 0.5.
- The see/stay-blind decision is **100%** dominant by reach weight, 92% unweighted.
- The unweighted/weighted gap means the mixing lives in rarely-reached infosets.
- 251 of 364 look nodes are reachable; the other 113 carry reach below 1e-12.

**A single-label benchmark is viable.** The residual ~2% of seen betting decisions need
either exclusion or a distribution metric.

---

## 7. Parameter sweeps

### 7.1 Stack depth and boot size are degenerate

Neither is a free parameter in this specification — this is a finding, not an omission.

**Stack sweep — all identical to 16 significant figures**, trees matching field by field
(40 iterations per cell; these test identity across configs, not convergence):

| stack | tree identical | game value to seat 1 | strategy identical |
|---:|:---:|---:|:---:|
| 34 | — (reference) | −0.014049237479203846 | — |
| 40 | yes | −0.014049237479203846 | yes |
| 50 | yes | −0.014049237479203846 | yes |
| 75 | yes | −0.014049237479203846 | yes |
| 100 | yes | −0.014049237479203846 | yes |

`RULES.md` §3.2 fixes the stack at 50 and proves maximum exposure is 33, so every stack ≥ 34
gives the identical game.

**Boot sweep** — scaling boot and stack together is a pure change of units; value per boot
invariant to ~1e-13, strategies identical:

| boot | at stack 50 | stack scaled to 50·boot | value per boot |
|---:|---|---|---:|
| 1 | ok | ok | −0.0140492374792038 |
| 2 | **outside RULES.md** (E1: no all-in rule) | ok | −0.0140492374792038 |
| 3 | **outside RULES.md** | ok | −0.0140492374792298 |
| 5 | **outside RULES.md** | ok | −0.0140492374792281 |

Raising the boot alone makes exposure exceed the stack; the engine raises `StackWouldBind`
rather than inventing an all-in rule §E1 does not specify. A genuine stack-depth sweep
requires adding all-in and side-pot rules to `RULES.md` — a Phase 0 change.

### 7.2 Cap sweep — the parameters that bite

| k | r | nodes | iters | exploitability | v₀ | P1 stays blind | P2 stays blind | seat-1 blind bets | seat-1 blind-show | look dominant (rw) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 48 | 3000 | 6.91e-06 | −0.000011 | **1.0000** | **0.9994** | 1.000 | 2.8e-08 | 1.0000 |
| 4 | 2 | 399 | 1500 | 8.42e-05 | +0.000131 | **1.0000** | **0.9999** | 1.937 | 1.3e-07 | 1.0000 |
| 6 | 2 | 1832 | 600 | 1.40e-03 | −0.011897 | **0.9999** | **0.0464** | 1.314 | 1.8e-05 | 1.0000 |
| 8 | 0 | 311 | 1500 | 9.33e-05 | −0.052877 | **1.0000** | **0.0002** | 1.258 | 4.4e-06 | 1.0000 |
| 8 | 1 | 1925 | 600 | 9.60e-04 | −0.039139 | **0.9999** | **0.0089** | 1.249 | 2.9e-05 | 1.0000 |
| 8 | 2 | 5929 | 1000 | 6.13e-04 | −0.039781 | **0.999965** | **0.0052** | 1.252 | 1.1e-05 | 1.0000 |

Seat-1 conditional conversion `P(look | own turn j reached while blind)`:

| k | r | turn 1 | turn 2 | turn 3 | turn 4 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.0000 | — | — | — |
| 4 | 2 | 0.0000 | 0.0000 | — | — |
| 6 | 2 | 0.0001 | 0.4234 | 0.9681 | — |
| 8 | 0 | 0.0000 | 0.8145 | 0.0003 | 0.7714 |
| 8 | 1 | 0.0001 | 0.7919 | 0.0359 | 0.9103 |
| 8 | 2 | 0.0000 | 0.7862 | 0.0584 | 0.9310 |

**(a) Seat 1 opens blind at every configuration** — 0.9999 to 1.0000 across all six.

**(b) Seat 2 undergoes a sharp transition between k = 4 and k = 6.** At short caps seat 2
*also* stays blind (0.9994, 0.9999); at k ≥ 6 it looks essentially always (0.046, 0.009,
0.005). Below the threshold the equilibrium is **both players blind for the whole hand** —
seat 1's conversion rate is 0.0000 at every turn for k ≤ 4. The "seat 1 blind, seat 2
sighted" split is therefore **emergent from betting room**, not a fixed property of the game.
This is the most consequential result in the sweep.

**(c) The first-mover penalty grows with betting room:** −1e-5 (k≤2), +1e-4 (k≤4), −0.0119
(k≤6), −0.0398 (k≤8). The k≤4 sign flip (+1.3e-4 against exploitability 8.4e-5) is within
solver noise, not a real reversal. The raise cap runs the other way — at k ≤ 8, forbidding
raises (r ≤ 0) makes seat 1 *worse* off (−0.0529) than allowing two (−0.0398).

**(d) The blind show right is never used at any configuration** — 2.8e-8 to 2.9e-5 across
the sweep. The §9 discrepancy is robust, not a k ≤ 8 artefact.

**(e) The look decision is 100% reach-weighted dominant at every configuration**, so the
single-label benchmark design is safe across the whole parameter range.

---

## 8. Validation

**Closed-form cross-check at `action_cap = 1`** (seat 2 never acts, so this is a decision
problem with a directly computable optimum — no game theory involved):

| | value |
|---|---:|
| Best blind action (`chaal`) | −0.50000000 |
| Look, then maximise per class | **−0.49077534** |
| CFR+ after 3,000 iterations | **−0.49077730** |
| Difference | 1.96e-06 |

Exploitability there is 9.8e-07.

**Forced pure strategy vs closed form** — matches to **1e-12**, validating the value
computation including exact card removal.

**Internal consistency at the solution** (at Nash, `BR_0 ≈ v₀` and `BR_1 ≈ −v₀`):

```
br_p0 = -0.039444   br_p1 = +0.040669   v_0 = -0.039781
|br_p0 - v_0| = 3.4e-04     |br_p1 + v_0| = 8.9e-04
```

Both residuals are of the order of the exploitability itself, as required. Terminal
probability mass sums to exactly 1.000000.

**Disjointness matrix** verified three independent ways: every row sums to 18,424 = C(49,3);
`size[c]·M[c,c'] == size[c']·M[c',c]`; grand total exactly 407,170,400 = C(52,3)·C(49,3).

---

## 9. Discrepancy with `RULES.md`

**§10.3's expectation about the blind show right is not borne out.** The spec states the
blind player's exclusive right to force a showdown is "the mechanism by which remaining
uninformed can be *positively* valuable", and that "any solved strategy that does not use it
is almost certainly an implementation bug".

Measured: **1.1e-05** (seat 1) and **5.1e-03** (seat 2), while *seen* players demand shows
10–100× more often. This is not a bug:

- the rule is implemented and tested (`test_e3`, `test_e4`, tree-wide mask check);
- the `action_cap = 1` cross-check confirms `show` is correctly illegal for a seen player
  facing a blind one;
- the game is verifiably near-equilibrium (exploitability 6e-4, `BR ≈ v` both sides);
- there is a coherent mechanism — the hands that most want to force a showdown are the hands
  most worth looking at, and looking forfeits the right.

**§10.3's rule is right; its prediction is wrong.** Suggested resolution: soften the
prediction, change no rule. Everything else in `RULES.md` and the Phase 1 modules held up;
no errors found.

---

## 10. Artifacts

| Item | Where |
|---|---|
| Convergence curve (20 points, avg + current iterate) | `convergence_default.json` |
| All sweep results | `experiments_results.json` |
| Solve log | `solve_default.log` |
| Experiment log | `experiments.log` |
| Solver checkpoint (112 MB, `CFRPlusSolver.load`) | `solution_default.pkl` — **not in git**, regenerate with `run_main_solve.py 1000 50` |

Reproduce: `python gate_phase2.py`, `python run_main_solve.py 1000 50`,
`python experiments.py`, `python -m pytest -q`.

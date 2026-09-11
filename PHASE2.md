# Phase 2 Report — CFR+ Solver, Exact Exploitability, and the Equilibrium

**Status: solver complete and converged. 37 Phase 2 tests pass (150 with Phase 1).**

Modules: [cfr.py](cfr.py), [exploitability.py](exploitability.py),
[experiments.py](experiments.py), [gate_phase2.py](gate_phase2.py),
[run_main_solve.py](run_main_solve.py). Tests: [test_cfr.py](test_cfr.py).

Reproduce with `python gate_phase2.py`, `python run_main_solve.py 1000 50`,
`python experiments.py`, `python -m pytest -q`.

---

## 1. Entry gate

All six preconditions were verified and reported before any solver code was written.

| # | Gate | Result |
|---|---|---|
| 1 | `RULES.md` §14 examples replay exactly | **PASS** — 184 per-action assertions on stake, pot and *both* stacks, plus terminal payoff |
| 2 | `u_1 + u_2 == 0` at every terminal, full walk | **PASS** — 3,915 terminals × 3 deals = 11,745 exact checks |
| 3 | Blind decision structurally cannot see own cards | **PASS** — type has no card field, `__slots__` blocks attaching one, `Deal` gated, all 728 blind nodes deal-invariant |
| 4 | Suit-isomorphic class count ≠ 741 | **PASS** — **1,755**, Burnside-confirmed, 442 κ classes proven lossy |
| 5 | Exact decision count ≤ 2.2 × 10³ | **PASS** — **2,014** (1,650 bet nodes); `stay-blind` confirmed not to consume a cap slot |
| 6 | No silent defaults | **PASS** — 17 illegal/unspecified operations all raise |

---

## 2. Solver design

**Vanilla full-tree tabular CFR+.** Full traversal every iteration; no sampling, no MCCFR,
no function approximation. Regret matching+ (cumulative regrets clipped at zero), linear
averaging, alternating updates. The average strategy is accumulated separately and is the
equilibrium approximation.

**Private state is the 1,755-class suit-isomorphic quotient**, per gate 4 and `PHASE1.md` §3 —
not the 741 κ classes, which discard card-removal information. The quotient is lossless:
hands related by a permutation of suits generate isomorphic subgames, so an `S_4`-symmetric
equilibrium exists and restricting to it is without loss of generality. Exploitability
therefore converging to zero in this representation means converging to zero in the true
22,100-hand game.

**Card removal between the two players is exact**, via `M[c,c']` = the number of hands in
class `c'` disjoint from a hand in class `c`. Well-defined because both classes are
`S_4`-invariant. Verified three independent ways at build time:

- every row sums to exactly 18,424 = C(49,3);
- `size[c] · M[c,c'] == size[c'] · M[c',c]` (both count the same ordered disjoint pairs);
- the grand total is exactly 407,170,400 = C(52,3) · C(49,3).

**Vector form.** The tree is traversed once per iteration carrying a reach vector over
private classes for each player. This is what makes the blind mechanic representable at all:
a seen player's strategy at a node is a `[1755 × n_actions]` matrix, while a blind player's is
a single `[n_actions]` vector shared by every class.

### 2.1 How the blind information set is enforced

`RULES.md` §12.4 is the rule most likely to be violated silently, so it is enforced by
representation rather than convention. Four independent mechanisms:

1. **A blind player's reach is a scalar, not a vector.** Every action they have taken was
   hand-independent, so their reach is identical across all 1,755 classes. Storing it as a
   scalar makes a per-class blind strategy *unrepresentable*.
2. **Blind regret arrays are allocated with shape `(n_actions,)`** — there is no class axis to
   index. Asserted in `test_blind_regret_tables_have_no_class_axis`.
3. **Regrets at a blind node come from the chance-weighted aggregate** `Σ_c w[c]·(v^a[c] − v[c])`,
   never per class.
4. **An assertion fires on every traversal** if a blind actor's reach is ever a vector
   (`_assert_blind_reach_is_scalar`), which would mean the strategy had become hand-dependent.

The best response is constrained the same way: at a blind node it must choose **one** action
maximising the chance-weighted aggregate. Maximising per class there would let the responder
condition on unseen cards and would *overstate* exploitability — silently.
`verify_blind_best_response_is_single_action` checks every blind node in every computed best
response, and a test deliberately corrupts one into a per-class array to confirm the verifier
rejects it.

### 2.2 Performance

1.82 s per iteration for the default config (5,929 public nodes, 1,755 classes, 112 MB of
regret and strategy tables). The cost is concentrated in the 1,568 showdown terminals where
*both* players are seen; those are batched into one contiguous BLAS call per traversal, since
a per-terminal matrix-vector product is memory-bandwidth bound and roughly an order of
magnitude slower. Terminals facing a *blind* opponent need no matrix product at all — that
opponent's reach is a scalar, so the mass vectors are precomputed constants times it.

---

## 3. Convergence

**Final exploitability: 6.13 × 10⁻⁴ boots/hand = 0.613 milli-boots/hand = 0.031% of the
starting pot**, after 1,000 iterations (54.9 min).

| iter | exploitability (boots/hand) | mb/hand | % of pot | v₀ | current iterate |
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

- **Monotone decreasing at every logged point.** No plateau: the ratio of the last
  exploitability to the one three logs earlier is 0.776, still falling.
- Fitted decay **exploitability ~ T^−1.78** over the whole run.
- Iterations to reach 10⁻²: **250**. To reach 10⁻³: **750**.
- The game value stabilises to six figures well before exploitability does: v₀ moves only from
  −0.039673 to −0.039781 over the last 200 iterations.

Neither failure mode the brief warns about occurred. There is no plateau above zero, and
exploitability is not implausibly small — 6e-4 after 1,000 iterations is an ordinary CFR+
value, reached by a smooth power-law decay rather than a collapse.

### 3.1 The average strategy versus the final iterate

Both were logged. The **current iterate is consistently the more exploitable early and the
less exploitable late** (1.49e-1 vs 9.34e-2 at iteration 50; 3.24e-4 vs 6.13e-4 at 1,000).

This is worth stating plainly because it cuts against the brief's expectation that the average
is "what converges". The average is what carries the *guarantee* — its exploitability is
bounded by average regret and is monotone here — whereas the current iterate has no bound and
merely happens to do well, a documented empirical property of CFR+ rather than something to
rely on. All reported statistics use the **average** strategy. The two are genuinely distinct
(`test_average_strategy_differs_from_final_iterate`), so the averaging is real and not an
alias of the final iterate.

### 3.2 Independent validation

Exploitability → 0 computed by an independent best-response traversal is already strong
evidence, but two further checks pin the pipeline down.

**A variant with no game theory in it.** At `action_cap = 1`, seat 2 never acts, so seat 1
faces a decision problem rather than a game and the optimum is a direct maximisation over the
win/tie/lose masses. That value is computable in closed form:

| | value |
|---|---:|
| Best blind action (`chaal`) | −0.50000000 |
| Look, then maximise per class | **−0.49077534** |
| CFR+ after 3,000 iterations | **−0.49077730** |
| Difference | 1.96 × 10⁻⁶ |

Exploitability there is 9.8 × 10⁻⁷. This validates terminal payoffs, blind aggregation,
per-class seen play, the `show`-illegality rule and the best-response path together.

**A forced pure strategy against a closed form.** Forcing seat 1 to stay blind and chaal at
`action_cap = 1` makes the whole game deterministic given the deal, with contributions (2, 1)
after a cap-forced show. The value must be
`Σ_c w[c]·(1·P(win|c) − 2·P(lose|c) − ½·P(tie|c))`, and it matches to 1e-12
(`test_forced_pure_strategy_value_matches_closed_form`). This is the check on the value
computation including exact card removal.

**Internal consistency at the solution.** At a Nash equilibrium `BR_0 ≈ v₀` and `BR_1 ≈ −v₀`:

```
br_p0 = -0.039444   br_p1 = +0.040669   v_0 = -0.039781
|br_p0 - v_0| = 3.4e-04     |br_p1 + v_0| = 8.9e-04
```

Both residuals are of the order of the exploitability itself, as they must be. Terminal
probability mass sums to exactly 1.000000.

---

## 4. The headline result: does equilibrium play stay blind?

**Yes — but only in seat 1, and the split is nearly total.** This is the paper's central
empirical finding, and it is a positional result that no prior work could have reported.

| | seat 1 (acts first) | seat 2 (acts second) |
|---|---:|---:|
| **P(stays blind at own first decision)** | **0.999965** | **0.005176** |
| P(looks at own first decision) | 0.000035 | 0.994824 |

Seat 1 opens blind essentially always. Seat 2, facing an opponent who has already acted, looks
essentially always. Neither decision is mixed.

Two qualifications from the sweep (§6.3), both important:

- **Seat 1's blind opening is universal** — probability 0.9999 to 1.0000 at every cap tested,
  so it is not an artefact of the k ≤ 8 truncation.
- **Seat 2's looking is *not* universal.** It appears only once the action cap reaches 6. At
  k ≤ 2 and k ≤ 4 the equilibrium has *both* players blind for the entire hand. The split
  reported above is an emergent property of having enough betting room, not a fixed feature
  of the game.

### 4.1 Conversion-timing distribution

`P(reach blind)` is the probability the hand reaches that player's turn *j* with them still
blind; `P(convert there)` is the unconditional probability they look at that turn;
`P(look | reached)` is the conditional rate.

**Seat 1:**

| own turn | P(reach blind) | P(convert there) | P(look \| reached) |
|---:|---:|---:|---:|
| 1 | 1.000000 | 0.000035 | 0.000035 |
| 2 | 0.588575 | 0.462748 | **0.786217** |
| 3 | 0.125448 | 0.007328 | 0.058415 |
| 4 | 0.118116 | 0.109967 | **0.931006** |
| never looks | | **0.419922** | |

**Seat 2:**

| own turn | P(reach blind) | P(convert there) | P(look \| reached) |
|---:|---:|---:|---:|
| 1 | 0.999986 | **0.994811** | **0.994824** |
| 2 | 0.000372 | 0.000022 | 0.057836 |
| 3 | 0.000001 | 0.000000 | 0.218210 |
| 4 | 0.000000 | 0.000000 | 0.138042 |
| never looks | | **0.005167** | |

Seat 1's conversion is concentrated at **turn 2** — immediately after seat 2 has revealed
something by acting. The 42% who never look is almost entirely hands that *end* before seat 1
gets a second turn, not hands played blind to the end: seat 2 packs after a single betting
action with probability 0.407 (see §4.3).

### 4.2 Is the blind discount actually exploited?

Yes, by seat 1, and essentially not at all by seat 2.

| | seat 1 | seat 2 |
|---|---:|---:|
| Expected betting actions taken at the **half price** | **1.252** | 0.006 |
| P(demands a show while blind — the blind-only right) | 0.000011 | 0.005142 |
| P(demands a show while seen) | 0.102764 | 0.076012 |
| P(packs while still blind) | 0.000004 | 0.000012 |

Two things stand out.

**The discount is used, and the option value is why.** Seat 1 pays the blind rate on ~1.25
betting actions per hand. Looking would double the price of the very same action, and a player
can always look later — so at turn 1, staying blind is close to free preservation of an
option. The mechanic functions less like a gamble than like a deferral, and equilibrium takes
the deferral every time.

**The blind player's exclusive show right is essentially never exercised** (1e-5 and 5e-3).
This directly contradicts the expectation recorded in `RULES.md` §10.3, which says a solved
strategy that does not use it "is almost certainly an implementation bug". At equilibrium,
shows are demanded by *seen* players (0.103 and 0.076) — an order of magnitude more often
than by blind ones. The right is a real asset in principle, but the hands that would want to
force a showdown are exactly the hands worth looking at first, and once you look you lose the
right. That tension resolves against using it. See §7 for what this implies for the spec.

### 4.3 What the hands actually look like

| terminal type | probability |
|---|---:|
| pack (fold) | 0.766584 |
| requested show | 0.183930 |
| cap-forced show | 0.049486 |
| **total** | **1.000000** |

Most common single ending, by public history:

| ending | mass |
|---|---:|
| seat 2 packs after 1 betting action | 0.4066 |
| seat 1 packs after 2 | 0.2321 |
| seat 1 demands a show after 2 | 0.0827 |
| seat 2 demands a show after 3 | 0.0596 |
| seat 2 packs after 3 | 0.0471 |

P(packs while seen) is 0.295 for seat 1 and **0.471** for seat 2. So the modal hand is: seat 1
opens blind at the half price, seat 2 looks, and seat 2 folds the bottom ~41% immediately.

### 4.4 Position

**The game value is −0.039781 boots per hand to seat 1.** With a symmetric boot and only
action order distinguishing the seats (`RULES.md` §3.4), acting first is worth about −4% of a
boot per hand. Seat 1 both stays blind and loses; the blind opening looks like the cheapest
available response to a structural positional disadvantage rather than a source of edge.

---

## 5. For Phase 3: dominant-action share

PokerBench filters its benchmark to decisions with a dominant action. The brief asked for this
early in case the blind/seen decisions turn out heavily mixed, which would force a
distribution-distance metric instead.

**They are not mixed. A single-label benchmark is viable.**

| group | reachable infosets | unweighted | reach-weighted |
|---|---:|---:|---:|
| look (see / stay-blind) | 251 | 0.9203 | **1.0000** |
| bet, actor blind | 235 | 0.8383 | **1.0000** |
| bet, actor seen | 913,825 | 0.7963 | 0.9800 |
| **overall** | | **0.7963** | **0.9932** |

- Reach-weighted, **99.3%** of decisions have an action above probability 0.5. The number that
  matters most for Phase 3 — the see/stay-blind decision — is **100%** dominant by reach
  weight and 92% unweighted.
- Unweighted, 79.6% of the ~914k reachable seen infosets have a dominant action. The gap
  between unweighted and reach-weighted says the mixing lives in rarely-reached information
  sets, which a benchmark would sample infrequently anyway.
- 251 of the 364 look nodes are reachable at all under the equilibrium; the other 113 carry
  reach below 1e-12 and are excluded.

**Recommendation for Phase 3.** Filter on the reach-weighted dominant action as PokerBench
does. Do *not* filter blind/seen decisions out for being mixed — they are the least mixed part
of the tree. Do expect the residual 2% of seen betting decisions to need either exclusion or a
distribution metric.

---

## 6. The parameter sweep

### 6.1 Stack depth and boot size are degenerate — a finding, not an omission

The brief asked for a sweep over stack depth and boot size. **In this specification neither is
a free parameter**, and reporting a flat sweep would misrepresent why.

**Stack depth.** `RULES.md` §3.2 fixes the stack at 50 and proves maximum exposure is 33, so
the stack provably never binds. Every stack ≥ 34 therefore produces *the identical game*. This
is verified two ways rather than argued from the bound: the public trees are compared
structurally field by field, and short solves are compared numerically.

**Boot size.** The boot is the unit of account (§3.1) and the opening stake equals it (§7.1),
so scaling the boot and the stack together is a pure change of units — identical strategies,
payoffs scaled by the factor. Raising the boot *alone* eventually makes exposure exceed the
stack, which requires an all-in rule that §E1 deliberately does not specify. The engine raises
`StackWouldBind` there rather than inventing one.

*A genuine stack-depth sweep would require adding all-in and side-pot rules to `RULES.md`.*
That is a real extension with real research content — the blind discount interacts with a
binding stack in a way this specification cannot express — but it is a Phase 0 change, not
something Phase 2 should improvise. Flagged for your decision.

### 6.2 The caps are the parameters that bite

`RULES.md` §9.1 states the two caps are modelling devices rather than rules of Teen Patti, so
varying them measures how the blind/seen equilibrium depends on how much betting room the
game is given.

**Stack sweep — all identical, as predicted:**

| stack | tree identical | game value to seat 1 | strategy identical |
|---:|:---:|---:|:---:|
| 34 | — (reference) | −0.014049237479203846 | — |
| 40 | yes | −0.014049237479203846 | yes |
| 50 | yes | −0.014049237479203846 | yes |
| 75 | yes | −0.014049237479203846 | yes |
| 100 | yes | −0.014049237479203846 | yes |

Identical to all 16 significant figures, and the public trees match field by field. (These
cells run only 40 iterations — they test identity across configs, not convergence, so their
absolute numbers are not the converged ones from §3.)

**Boot sweep:**

| boot | at the RULES.md stack of 50 | with the stack scaled to 50·boot | value per boot |
|---:|---|---|---:|
| 1 | ok | ok | −0.0140492374792038 |
| 2 | **outside RULES.md** (E1: no all-in rule) | ok | −0.0140492374792038 |
| 3 | **outside RULES.md** | ok | −0.0140492374792298 |
| 5 | **outside RULES.md** | ok | −0.0140492374792281 |

Scaling boot and stack together leaves the value per boot invariant to ~1e-13 and the strategy
identical, confirming it is a change of units. Raising the boot alone is refused by the engine.

### 6.3 Cap sweep results

| k | r | nodes | iters | exploitability | v₀ | P1 stays blind | P2 stays blind | seat-1 blind bets | seat-1 blind-show | look dominant (rw) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 48 | 3000 | 6.91e-06 | −0.000011 | **1.0000** | **0.9994** | 1.000 | 2.8e-08 | 1.0000 |
| 4 | 2 | 399 | 1500 | 8.42e-05 | +0.000131 | **1.0000** | **0.9999** | 1.937 | 1.3e-07 | 1.0000 |
| 6 | 2 | 1832 | 600 | 1.40e-03 | −0.011897 | **0.9999** | **0.0464** | 1.314 | 1.8e-05 | 1.0000 |
| 8 | 0 | 311 | 1500 | 9.33e-05 | −0.052877 | **1.0000** | **0.0002** | 1.258 | 4.4e-06 | 1.0000 |
| 8 | 1 | 1925 | 600 | 9.60e-04 | −0.039139 | **0.9999** | **0.0089** | 1.249 | 2.9e-05 | 1.0000 |
| 8 | 2 | 5929 | 1000 | 6.13e-04 | −0.039781 | **0.999965** | **0.0052** | 1.252 | 1.1e-05 | 1.0000 |

Seat-1 conditional conversion rate, `P(look | own turn j reached while blind)`:

| k | r | turn 1 | turn 2 | turn 3 | turn 4 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.0000 | — | — | — |
| 4 | 2 | 0.0000 | 0.0000 | — | — |
| 6 | 2 | 0.0001 | 0.4234 | 0.9681 | — |
| 8 | 0 | 0.0000 | 0.8145 | 0.0003 | 0.7714 |
| 8 | 1 | 0.0001 | 0.7919 | 0.0359 | 0.9103 |
| 8 | 2 | 0.0000 | 0.7862 | 0.0584 | 0.9310 |

Four results, in descending order of how much they matter.

**(a) Seat 1 opens blind at every single configuration** — probability 0.9999 to 1.0000 across
all six. The headline finding of §4 is not an artefact of the k ≤ 8 cap.

**(b) Seat 2's behaviour undergoes a sharp transition between k = 4 and k = 6.** At the short
caps seat 2 *also* stays blind (0.9994, 0.9999); at k ≥ 6 seat 2 looks essentially always
(stay-blind 0.046, 0.009, 0.005). So the "seat 1 blind, seat 2 sighted" split reported in §4
is not a general property of the game — it *emerges* once there is enough betting room for
information to be worth paying double for. Below that threshold the equilibrium is
**both players blind for the whole hand**: at k ≤ 2 and k ≤ 4 seat 1's conversion rate is
0.0000 at every turn.

This is the most interesting result in the sweep and it is a genuinely new fact. Teen Patti's
blind mechanic is not simply "worth using" or not; whether *either* player pays for
information depends on how long the betting can run.

**(c) The game is near-fair at short caps and the first-mover penalty grows with room**:
−1e-5 (k≤2), +1e-4 (k≤4), −0.0119 (k≤6), −0.0398 (k≤8). The k≤4 sign flip is +1.3e-4 against
an exploitability of 8.4e-5, so it is within solver noise and not a real reversal. The
positional disadvantage of §4.4 is therefore also an emergent property of betting room, not a
fixed feature.

Note the raise cap runs the other way: at k ≤ 8, forbidding raises entirely (r ≤ 0) makes seat
1 *worse* off (−0.0529) than allowing two (−0.0398). Raising is a tool the first mover needs.

**(d) The blind show right is never used at any configuration** — between 2.8e-8 and 2.9e-5
across the whole sweep. §7's discrepancy with `RULES.md` §10.3 is robust and not a k ≤ 8
artefact.

**(e) For Phase 3: the look decision is 100% reach-weighted dominant at every configuration.**
The single-label benchmark design is safe across the whole parameter range, not just at the
default.

---

## 7. Discrepancies with `RULES.md`

One substantive item, raised rather than worked around.

**§10.3's expectation about the blind show right is not borne out.** The spec says the blind
player's exclusive right to force a showdown is "the mechanism by which remaining uninformed
can be *positively* valuable", and that "any solved strategy that does not use it is almost
certainly an implementation bug".

At equilibrium the right is exercised with probability **1.1 × 10⁻⁵** (seat 1) and
**5.1 × 10⁻³** (seat 2), while *seen* players demand shows 10–100× more often. I have not
worked around this, and I do not believe it is a bug:

- the rule itself is implemented and tested (`test_e3`, `test_e4`, and a tree-wide mask check);
- the `action_cap = 1` cross-check confirms `show` is correctly *illegal* for a seen player
  facing a blind one;
- the game is verifiably near-equilibrium (exploitability 6e-4, `BR ≈ v` both sides);
- there is a coherent mechanism: the hands that most want to force a showdown are the hands
  most worth looking at, and looking forfeits the right.

So §10.3's *rule* is right and its *prediction* is wrong. That is a finding worth a sentence in
the paper — the blind player's show right is a real option that equilibrium play declines —
and I would suggest softening the prediction in §10.3 rather than changing any rule.

Everything else in `RULES.md` and the Phase 1 modules held up. No errors found.

---

## 8. Test coverage

**37 Phase 2 tests, all passing** (150 including Phase 1).

| Requirement | Test |
|---|---|
| Trivially small variant checked independently | `test_cap1_matches_direct_maximisation` — closed-form optimum, no game theory |
| Value computation incl. card removal | `test_forced_pure_strategy_value_matches_closed_form` — matches to 1e-12 |
| Average, not final iterate, is accumulated | `test_average_strategy_differs_from_final_iterate` |
| Seat symmetry where the rules are symmetric | `test_terminal_coefficients_are_seat_symmetric` — `u_0 == −u_1` at all 3,915 terminals |
| Exploitability non-negative and decreasing | `test_exploitability_is_nonnegative_and_decreases` |
| Blind cannot condition on own cards | 5 tests: no class axis, scalar reach, guard fires on a leak, BR single-action, verifier catches a corrupted BR |
| Disjointness matrix exact | 4 tests: row sums, pair symmetry, grand total, corruption rejected |
| Sweep degeneracy | `test_stack_above_34_never_binds_and_gives_one_game`, `test_scaling_boot_and_stack_together_is_a_change_of_units`, `test_raising_the_boot_alone_leaves_the_specification` |
| Checkpointing | roundtrip, and config-mismatch rejection |

The seat-symmetry requirement needs a note: the rules are **not** seat-symmetric — seat 1 acts
first (§3.4) and the game value is −0.0398 to them — so there is no exact seat-swap invariance
of the equilibrium to test. What is testable, and tested, is that the payoff layer treats the
seats as exact mirrors at every terminal.

---

## 9. Status for Phase 3

| Item | Value |
|---|---|
| Exploitability (default config) | **6.13e-4 boots/hand, 0.031% of pot** |
| Game value to seat 1 | **−0.039781 boots/hand** |
| Iterations / wall time | 1,000 / 54.9 min |
| Checkpoint | `solution_default.pkl` (112 MB), reload via `CFRPlusSolver.load` |
| Convergence curve | `convergence_default.json` |
| Sweep results | `experiments_results.json` |
| Dominant-action share (reach-weighted) | **99.3%** overall, **100%** for see/stay-blind |
| Single-label benchmark viable? | **Yes** |

The solver is an answer key at 0.031% of the pot. For a benchmark whose labels are discrete
actions filtered to dominant decisions, that is far inside the margin where the label is
unambiguous.

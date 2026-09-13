# Phase 1 Report — Simulator, Evaluator, Abstraction, Enumeration

**Status: complete. 113 tests pass. No departures from `RULES.md`.**

Modules: [hands.py](../src/hands.py), [game.py](../src/game.py), [abstraction.py](../src/abstraction.py),
[tree.py](../src/tree.py). Tests: [test_hands.py](../tests/test_hands.py), [test_game.py](../tests/test_game.py),
[test_abstraction.py](../tests/test_abstraction.py), [test_tree.py](../tests/test_tree.py).

Reproduce with `python tree.py`, `python abstraction.py`, `python -m pytest -q`.

---

## 1. Exact public tree vs. the `RULES.md` estimate

`RULES.md` §9.3 estimates ~2.2 × 10³ public decision histories and flags the figure as
sizing-only, predicting the true count is "somewhat lower". **Confirmed.**

| Quantity | Exact | vs. 2,200 estimate |
|---|---:|---:|
| Public nodes (all) | 5,929 | — |
| **Public decision nodes (look + bet)** | **2,014** | **0.92×** |
| — look nodes (blind actor only) | 364 | |
| — bet nodes | 1,650 | 0.75× |
| —— with blind actor | 364 | |
| —— with seen actor | 1,286 | |
| Terminal nodes | 3,915 | — |
| — pack | 1,650 | |
| — show, requested | 1,340 | |
| — show, forced by cap | 925 | |

**Which number answers the estimate.** `RULES.md` §9.3 derives its figure by multiplying
betting-action sequences by conversion patterns, which counts *turns* — and each turn is one
bet node (1,650). Read literally, "decision histories" includes look nodes (2,014). The spec
does not distinguish the two, so both are reported. **Under either reading the exact count is
below 2,200, as §9.3 predicted.** The 2,014 figure is within 8.5% of the estimate.

Structural identity worth noting: `look nodes == bet nodes with a blind actor == 364`. Every
look node branches into exactly one blind bet node (`stay-blind`) and one seen bet node
(`see`), so the two must be equinumerous. This is asserted in `test_tree.py`.

### Cross-checks against `RULES.md` §9.4

| Claim | `RULES.md` | Exhaustive enumeration |
|---|---:|---:|
| Max contribution by one player | 33 (one witness) | **33** ✓ |
| Max `k` | 8 | 8 ✓ |
| Stake domain | `{1, 2, 4}` | `{1, 2, 4}` ✓ |
| Max pot | not stated | **62** |

§9.4 proves `max c_i = 33` by exhibiting a single line. Walking the whole tree gives the bound
directly, which is strictly stronger. Max pot is 62 rather than 66 because the two per-player
maxima are not simultaneously achievable: seat 1 opens at `s = 1` and peaks at 27.

---

## 2. Suit-isomorphic class count — the number Phase 2 should use

**1,755 classes.** Not 741.

Computed two independent ways, which agree:

- canonicalisation: minimum image of each hand under all 24 permutations of suits;
- **Burnside's lemma**: mean number of hands fixed by a group element, over `S₄`.

The quotient decomposes exactly, and the decomposition is fully explained by
orbit–stabiliser:

| Orbit size | Stabiliser | Orbits | Hands | Which hands |
|---:|---:|---:|---:|---|
| 4 | 6 | 299 | 1,196 | trails (13) + flushes (286) — one suit distinguished, `S₃` on the rest |
| 12 | 2 | 1,170 | 14,040 | two-suited non-flush hands |
| 24 | 1 | 286 | 6,864 | rainbow hands, three distinct suits, trivial stabiliser |
| | | **1,755** | **22,100** | |

And how the 741 κ classes refine into those 1,755 orbits:

| κ classes | split into | Which κ classes | Contribution |
|---:|---:|---|---:|
| 299 | 1 orbit | 13 trail + 274 colour + 12 pure sequence | 299 |
| 156 | 2 orbits | all 156 pair classes | 312 |
| 286 | 4 orbits | 274 high card + 12 sequence | 1,144 |
| **741** | | | **1,755** |

`299 × 1 + 156 × 2 + 286 × 4 = 1,755`. A pair class splits in two according to whether the
odd card shares a suit with one of the pair; a three-distinct-rank non-flush class splits in
four according to its suit pattern.

---

## 3. `RULES.md` §5.3 verified in both directions

The spec's central warning is not taken on trust. Both halves are established empirically.

**κ is lossy as a state abstraction.** **442 of 741 κ classes (60%) contain hands with
different exact equities.** The spec's own named example behaves as claimed:

| Hand | κ | Orbit | Wins / Ties / Losses | Equity |
|---|---:|---:|---|---:|
| `K♠K♥7♦` | 411 | 1483 | 16521 / 3 / 1900 | 0.89679223 |
| `K♠K♥7♠` | 411 | 1480 | 16510 / 3 / 1911 | 0.89619518 |

Same κ, different orbits, different equity. Using κ as the solver's information-set index
would have merged these.

**The direction is the opposite of the naive reading, and the size of the gap is exactly
accounted for.** `RULES.md` §5.3 says `K♠K♥7♠` "removes a third spade from the deck", which
suggests it should block opponent flushes and therefore lose *less*. It loses *more*, by
exactly 11 hands. The reason is convexity. Opponent flushes total
`Σ_suits C(13 − used, 3)`, and `C(n,3)` is convex, so *concentrating* removals in one suit
leaves the other suits fuller and more total flushes alive:

```
K♠K♥7♦  suits used [1,1,1]  ->  C(12,3)*3 + C(13,3)        = 946
K♠K♥7♠  suits used [2,1]    ->  C(11,3) + C(12,3) + C(13,3)*2 = 957
                                                    difference =  11
```

Both hands hold the same ranks `{K,K,7}`, so every rank-determined count is identical and the
entire 11-hand loss difference *is* those flushes. Asserted in
`test_the_rules_md_worked_example_of_kappa_lossiness`.

The spec's conclusion is right; only the intuition offered for it is inverted. **No change to
`RULES.md` §5.3 is needed beyond, optionally, correcting that parenthetical.**

**The suit-isomorphic quotient is lossless.** Exact win/tie/loss counts are verified constant
within all 1,755 orbits (`_verify_losslessness`, run on every build).

---

## 4. Information sets

Blind and seen sides scale completely differently, which is the whole point of §12.4.

| | Player 1 | Player 2 | Total |
|---|---:|---:|---:|
| Blind information sets (1 per public history, **no card refinement**) | 260 | 468 | **728** |
| Seen public nodes (each × the hand abstraction) | 468 | 818 | **1,286** |
| Decision nodes | 728 | 1,286 | 2,014 |

Seat 2 has more decision nodes than seat 1 throughout, because seat 1 acts first and so seat
2's nodes sit deeper in the tree. This is the positional asymmetry `RULES.md` §3.4 warns must
be evaluated from both seats.

**Totals under each candidate abstraction:**

| Hand abstraction | Seen infosets | Blind | **Total** | CFR+ memory¹ |
|---|---:|---:|---:|---:|
| Exact hands (22,100) | 28,420,600 | 728 | **28,421,328** | ~1.8 GB |
| **Suit-isomorphic (1,755)** — lossless | 2,256,930 | 728 | **2,257,658** | **~145 MB** |
| 25 buckets | 32,150 | 728 | **32,878** | ~2.1 MB |

¹ regret + cumulative-strategy accumulators, ≤ 4 actions per bet node, float64.

The lossless solve is comfortably tractable. **Recommendation for Phase 2: solve at 1,755
(lossless), and use the 25-bucket abstraction only if exact best-response computation over the
full 22,100-hand opponent range proves too slow.** Note the 728 blind information sets are a
rounding error in the total but carry the entire research object.

---

## 5. Bucketing scheme

- **Equity** is exact: for each of the 22,100 hands, win/tie/loss counted against **all 18,424**
  opponent hands drawable from the remaining 49 cards. No Monte Carlo. The 407,170,400
  pairwise comparisons are avoided by inclusion–exclusion over the hand's three cards, which
  makes the whole computation run in ~1.4 s; every hand's `w + t + l == 18424` is asserted, and
  6 randomly chosen hands are re-verified by brute-force `O(18424)` enumeration in the tests.
- **Equity metric**: `(wins + ties/2) / 18424`.
- **Bucketing unit**: suit-isomorphic orbits, never κ classes. No orbit straddles a bucket
  (asserted).
- **Boundaries**: orbits sorted by equity, split to equalise *hand mass* (~884 hands/bucket).
  Because an orbit is never split, sizes are approximate.
- `n_buckets` is a parameter, default 25.

| Bkt | Hands | Orbits | Equity range | Strongest member |
|---:|---:|---:|---|---|
| 0 | 900 | 60 | 0.0007–0.0315 | 7h 6d 3c |
| 1 | 876 | 59 | 0.0338–0.0678 | 8d 7c 5c |
| 2 | 876 | 58 | 0.0680–0.1079 | 9c 7d 6c |
| 3 | 888 | 59 | 0.1079–0.1519 | Th 6d 4c |
| 4 | 900 | 60 | 0.1533–0.1853 | Th 9d 4c |
| 5 | 864 | 58 | 0.1868–0.2351 | Jc 7d 3c |
| 6 | 888 | 59 | 0.2353–0.2678 | Jd 9c 7c |
| 7 | 888 | 59 | 0.2678–0.3228 | Qh 5d 4c |
| 8 | 876 | 59 | 0.3242–0.3556 | Qc 8d 7c |
| 9 | 888 | 59 | 0.3559–0.3889 | Qc Td 9c |
| 10 | 888 | 59 | 0.3890–0.4530 | Kd 6c 2c |
| 11 | 888 | 59 | 0.4533–0.4847 | Kh 8d 7c |
| 12 | 876 | 59 | 0.4879–0.5186 | Kd Tc 9c |
| 13 | 888 | 59 | 0.5190–0.5539 | Kd Qc 7c |
| 14 | 876 | 58 | 0.5541–0.6233 | Ah 7d 3c |
| 15 | 900 | 60 | 0.6248–0.6564 | Ah 9d 7c |
| 16 | 876 | 59 | 0.6575–0.6917 | Ad Jd 7c |
| 17 | 876 | 58 | 0.6920–0.7274 | Ad Kc 3c |
| 18 | 888 | 65 | 0.7275–0.7580 | 5h 3d 3c |
| 19 | 888 | 74 | 0.7580–0.7950 | 6h 6d 3c |
| 20 | 876 | 73 | 0.8014–0.8390 | 9d 9c 6c |
| 21 | 888 | 74 | 0.8392–0.8833 | Qh Qd 5c |
| 22 | 880 | 98 | 0.8834–0.9188 | 9c 5c 3c |
| 23 | 884 | **221** | 0.9189–0.9599 | Ac Qc 4c |
| 24 | 884 | 89 | 0.9600–1.0000 | Ah Ad Ac |

Sizes are tight (864–900 against a target of 884). Orbit counts per bucket are flat at ~59
through bucket 17 — those buckets are all high-card hands — then climb as pairs (18–21) and
flushes and sequences (22–24) compress into narrow equity ranges. Bucket 23 holds 221 orbits
because nearly every colour and sequence lands in a 0.04-wide equity band.

Reverse mapping (`hands_in_bucket`, `iso_classes_in_bucket`, `representative_hands`) is
exposed so Phase 2 and the Phase 3 benchmark can recover concrete hands from a bucket.

---

## 6. Ambiguities and gaps found in `RULES.md`

Four items. **None is a contradiction, and none required a departure from the spec.** Listed
in descending order of how much I think they matter.

### 6.1 A forced show can split an odd pot, giving half-integer payoffs

`RULES.md` §10.6 gives the split as `u_i = p/2 − c_i`. That is unambiguous, but it is not
always integral, and **§14 contains no example of it**.

**396 of the 925 forced-show terminals (42.8%) have an odd pot.** The smallest is `p = 11`,
`c = (5, 6)`, giving `u = (+1/2, −1/2)`. Reachable line: seat 1 blind-chaals at `s = 1` (an odd
payment of 1), seat 2 raises to `s = 2`, and every subsequent payment is even.

This is a live trap: an implementation using integer arithmetic will silently truncate and
break zero-sum on 43% of forced showdowns. **I implemented §10.6 exactly as written, using
`fractions.Fraction`**, so payoffs are exact and `u_1 + u_2 == 0` holds identically. Phase 2
can scale utilities by 2 to work in integers.

**Suggested `RULES.md` amendment:** one line in §10.6 noting that `p` may be odd and that the
split is therefore a half-integer, with a pointer that engines must not use integer division.

### 6.2 §9.3's `|I|` range of 10⁵–10⁶ is now the wrong range

§9.3 concludes `|I|` lands in 10⁵–10⁶. That was correct *given* κ: `1,286 × 741 + 728 =
953,454`. But §5.3 correctly says κ is not a valid abstraction, and the lossless quotient
gives **2,257,658** — just above the stated range. Unabstracted it is 28.4M.

The estimate and the warning were consistent with each other only by accident. Still
comfortably tractable, so this changes nothing operationally.

**Suggested amendment:** update the §9.3 range to ~2.3 × 10⁶ once 1,755 is substituted for
741.

### 6.3 The §9.4 witness table omits seat 1's `see` action

The table labels seat 1 as "P1 (S)" from turn 3 onward but annotates only seat 2's conversion
("P2 → see, raise"). Reproducing the line requires inserting a `see` for seat 1 at the start of
turn 3.

Harmless: seat 2's payments depend only on the stake and seat 2's own status, so `c_2 = 33`
either way. Presentational only — noted because it cost me one failing test before I spotted it.

### 6.4 "Public decision history" is not defined against the look/bet split

§9.3's estimate counts turns; the phrase suggests all decision nodes. The spec's own §13.4
establishes that a blind turn contains *two* decision points, so the two readings differ by the
364 look nodes. Both are reported in §1 above. No consequence beyond needing to say which
number is being compared.

### Explicitly not ambiguous

The task flagged **whether `stay-blind` consumes an action-cap slot** as a possible stop-and-ask.
It is not ambiguous: `RULES.md` states it three times — §8.1 ("does **not** advance the turn or
increment `k`"), §9.2 ("`see` and `stay-blind` do not count toward `k`"), and E13, which names
the opposite behaviour as a bug and nominates Example C as its regression test. Implemented
accordingly; `test_example_c_see_does_not_consume_a_cap_slot` asserts 15 player decisions
against `k == 8`.

---

## 7. New finding for the paper: equity order ≠ hand-strength order

Not a spec issue, but it bears on Phase 3 and I did not see it anticipated.

**Equity is not monotone in κ. 208 adjacent κ classes have overlapping equity ranges** (max
overlap 0.0025). The cleanest instance:

| Hand | κ | Category | Equity |
|---|---:|---|---:|
| `2♣2♦3♥` | 274 | weakest pair | 0.7403 |
| `A♠K♥J♦` | 273 | strongest high card | 0.7428 |

`2♣2♦3♥` beats `A♠K♥J♦` **every single time they meet**, yet has lower equity against a
uniform opponent. The cause is card removal: `A♠K♥J♦` strips an ace, a king and a jack from
the opponent's range, blocking many of their strong hands, while `2♣2♦3♥` leaves every high
rank intact.

Category *means* are strictly increasing (0.372 → 0.828 → 0.937 → 0.979 → 0.996 → 0.999), so the
ordering is right in aggregate and only inverts locally.

**Consequences.** For Phase 2 this is harmless — buckets are a strategic abstraction, not a
ranking. For Phase 3 it is a real hazard: a benchmark question of the form "which hand is
stronger?" must be answered from **κ**, never from equity or bucket index, or the answer key
will be wrong for these 208 boundaries. Worth a sentence in the paper, since it is a second
instance of the general theme in `RULES.md` §5.2 — that Teen Patti's hand ranking and its
underlying probabilities are not aligned.

---

## 8. Departures from the specification

**None.** Every rule is implemented as `RULES.md` states it. Three implementation choices are
additions rather than departures:

1. **`Fraction` arithmetic** throughout, forced by §6.1. §10.6's formula is implemented
   verbatim; only the numeric type is a choice.
2. **Information-set types live in [game.py](../src/game.py), not [tree.py](../src/tree.py)** — `BlindInfoSet`,
   `SeenInfoSet` and `observation()` are inseparable from the state machine that gates card
   access, so splitting them across modules would have weakened the structural guarantee.
   `tree.py` owns enumeration, as specified.
3. **Import-time self-checks** in `hands.py` (§5.1 counts, 12 runs, `2-A-K` rejected). These
   fail loudly at import rather than letting a skewed table reach Phase 2.

### How the blind-information guarantee is enforced structurally

§12.4's contract is that a blind player has exactly one information set per public history.
Four independent mechanisms, so that violating it cannot be done quietly:

1. `BlindInfoSet` **has no field for cards** and uses `__slots__` — one cannot be attached at
   runtime. A policy handed one has nothing to read.
2. `observation()` **never touches the `Deal`** on the blind branch.
3. `Deal.hand_for(player, state)` **raises `BlindAccessViolation`** if that player has not seen.
4. `tree.verify_blind_infosets_are_card_independent()` walks **all 728 blind decision nodes**
   against three disjoint deals and asserts the observations are identical by value.

Tests: `test_blind_infoset_has_no_field_for_cards`,
`test_blind_observation_is_identical_across_different_deals`,
`test_deal_refuses_to_hand_over_unseen_cards`, plus the tree-wide check.

The one deliberate bypass is `Deal.showdown_kappa()`, which is not gated because §E2 requires a
cap-forced show to evaluate hands neither player ever looked at. It is engine-side and is never
reachable from an information set.

---

## 9. Test coverage against the required list

| # | Requirement | Test |
|---|---|---|
| 1 | Combinatorial counts by brute force | `test_hands.py` — category counts, flush and straight cross-checks |
| 2 | 12 valid runs; `2-A-K` rejected | `test_exactly_twelve_valid_runs`, `test_two_ace_king_is_not_a_run` |
| 3 | 741 κ classes | `test_741_kappa_classes`, `test_class_counts_per_category` |
| 4 | **All three §14 examples, action by action** | `test_worked_example_traces_exactly` — asserts stake, pot, both contributions and both stacks after **every** action, plus terminal kind, `k`, `r`, payoff and final stacks |
| 5 | §9.4 witness reaches `c_2 = 33` | `test_max_exposure_witness_reaches_33`; plus `test_max_exposure_is_exhaustively_33` over the whole tree |
| 6 | `u_1 + u_2 == 0` at every terminal | `test_zero_sum_at_every_terminal_in_a_full_walk` — 3,915 terminals × 3 deals = 11,745 checks |
| 7 | Randomised property test | `test_random_playthroughs_respect_every_invariant` — 5 seeds × 400 hands; asserts action legality, phase/status consistency, caps, masks, non-negative stacks, pot identity, chip conservation |
| 8 | **Fails if a blind policy can see its own cards** | four tests, §8 above |

Additionally: one named test per `RULES.md` edge case E2–E14, and `test_e4` checks the
seen-vs-blind show mask at **every** such node in the tree rather than at one.

**113 tests, all passing, ~3 s.**

---

## 10. Gate status for Phase 2

| Gate | Result |
|---|---|
| Exact public decision history count | **2,014** (1,650 bet nodes) vs. 2,200 estimate — below, as predicted |
| Suit-isomorphic class count | **1,755**, Burnside-confirmed |
| Total information sets (lossless) | **2,257,658** — 728 blind, 2,256,930 seen |
| Zero-sum at every terminal | verified, 11,745 checks |
| Blind infosets card-independent | verified, all 728 nodes |
| Max exposure ≤ stack | 33 < 50, exhaustive |
| Tree finite and terminating | verified, max `k` = 8 |

**Phase 2 is unblocked.** Two things to carry forward: solve at 1,755 rather than 741, and use
exact or doubled-integer arithmetic because of the odd-pot split (§6.1).

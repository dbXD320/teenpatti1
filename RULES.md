# Heads-Up Teen Patti: Frozen Rules Specification

**Version 1.0 — Phase 0 deliverable.**
This document is the normative rules specification for a two-player, finite-tree variant of
Teen Patti. It is the implementation contract for the Phase 1 simulator and the Phase 2 CFR+
solver, and is intended for inclusion as a paper appendix.

Every rule below is stated as a commitment. Where real-world play varies, the body of this
document does not hedge; all divergences are collected in §17.

---

## 1. Scope and Design Principles

### 1.1 What this document specifies

A complete, deterministic, unambiguous ruleset for **heads-up (two-player) Teen Patti with
the blind/seen mechanic retained, bet sizes discretised, and betting capped**, together with
its formal extensive-form representation.

### 1.2 Design principles

The specification is governed by four principles, applied in this priority order:

1. **Fidelity to the real game.** Where a real convention can be retained without breaking
   finiteness, it is retained, and cited to a source (§18).
2. **Finiteness.** The game tree must be finite, so that tabular CFR+ converges and
   exploitability is exactly computable.
3. **Zero-sum closure.** Every terminal node must satisfy `u_1(z) + u_2(z) = 0`.
4. **Determinism.** No rule may depend on implementer discretion. Any ambiguity is a defect
   in this document.

### 1.3 Scoping decisions taken as given

- **Two players only.** Heads-up gives a two-player zero-sum game, hence CFR convergence
  guarantees and exactly computable exploitability. Multiplayer is future work.
- **The blind/seen mechanic is retained and is the core research object.** It is a voluntary,
  costly, irreversible choice to remain uninformed, with no analogue in any studied poker
  variant. It is modelled precisely in §6 and §12.
- **Betting is capped and bet sizes discretised** (§9, §8.4) purely to bound the tree.

---

## 2. Primitives and Notation

### 2.1 Deck

A standard 52-card French deck `D`. No jokers.

- **Ranks** `R = {2, 3, ..., 10, J, Q, K, A}`, `|R| = 13`, ordered `2 < 3 < ... < K < A`.
- **Suits** `S = {♣, ♦, ♥, ♠}`, `|S| = 4`. **Suits are unordered.** No rule in this
  specification ever compares suits. Consequently exact ties are possible (§4.4).
- A card is a pair `(r, s) ∈ R × S`.

A **hand** is an unordered 3-subset of `D`. There are `C(52,3) = 22,100` hands.

### 2.2 Ace duality

The ace is high for all rank comparisons (§4), and additionally acts as low **only** inside
the sequence `A-2-3` (§4.2). It has no other low behaviour.

### 2.3 Symbols

| Symbol | Meaning |
|---|---|
| `s` | current stake, in *blind-equivalent* units (§7) |
| `p` | pot |
| `c_i` | total contribution of player `i` to the pot this hand |
| `b_i` | status of player `i`, `b_i ∈ {B, S}` (Blind / Seen) |
| `k` | number of betting actions taken so far this hand (both players) |
| `r` | number of raises taken so far this hand (both players) |
| `h_i` | private hand of player `i` |
| `κ(h)` | strength class of hand `h` (§5.3) |
| `i`, `j` | the player to act, and their opponent |

**Invariant:** `p = c_1 + c_2` at all times. The pot is therefore redundant state and MUST
be derived, not tracked independently.

---

## 3. Table Stakes

### 3.1 Boot (ante)

**Commitment: the boot is 1 unit, and both players post it before the deal.**

All monetary quantities in this document are expressed in boot units. Posting is symmetric —
there is no small/big blind asymmetry, because Teen Patti has no blind structure; the boot is
a true ante paid by every player. After posting, `p = 2`, `c_1 = c_2 = 1`.

*Justification:* the boot is the natural unit of account, it makes the initial stake exactly
1 (§7.1), and symmetry means the only positional asymmetry in the game is action order,
which isolates the effect we want to study.

### 3.2 Starting stacks

**Commitment: each player starts every hand with 50 units.**

*Justification:* under the discretisation (§8.4) and caps (§9), a player's maximum possible
contribution to a single hand is exactly **33 units** (proof in §9.4). A 50-unit stack
therefore **can never bind**. This is deliberate: it removes stack depth as a state variable
and eliminates all-in branches, leaving a pure betting tree. PokerBench used 100 big blinds
because solvers compute at that depth and all-ins matter there; in a capped game the
analogous correct choice is a stack that provably never binds, so that the solved strategy is
a function of the betting history alone.

Implementations MUST reject any configured stack below 34 units (§15, E1).

### 3.3 Stack reset

**Commitment: stacks reset to 50 units at the start of every hand.**

There is no carry-over of winnings between hands and no bankroll dynamics. Each hand is an
independent instance of the same finite game. This is required for solver consistency: CFR+
solves one fixed game, and a drifting stack would make each hand a different game.

### 3.4 Seats and action order

Seat 1 acts first. Seat 2 is the dealer (in the physical game the dealer deals and the player
to their left begins). Turn order strictly **alternates** thereafter.

Because the boot is symmetric, action order is the *only* asymmetry between seats. The game is
therefore not symmetric, and evaluation MUST play both seats.

---

## 4. Hand Ranking

### 4.1 Category order

Fixed, highest to lowest:

| # | Category | Definition |
|---|---|---|
| 1 | **Trail** (trio, prial) | three cards of equal rank |
| 2 | **Pure sequence** (straight run, running flush) | three consecutive ranks, all one suit |
| 3 | **Sequence** (normal run) | three consecutive ranks, not all one suit |
| 4 | **Colour** (flush) | three cards of one suit, not consecutive |
| 5 | **Pair** | exactly two cards of equal rank |
| 6 | **High card** | none of the above |

A hand belongs to exactly one category. Categories are mutually exclusive by construction
(the "not" clauses in rows 3 and 4 are load-bearing).

### 4.2 Sequence validity and order

**Commitment: there are exactly 12 valid runs.** A run is three consecutive ranks. The ace may
close a run at the top (`A-K-Q`) or at the bottom (`A-2-3`), and **`A-2-3` is the highest run**.

**`2-A-K` is NOT a valid run.** Wrap-around is prohibited. An implementation MUST NOT generate
it and MUST classify e.g. `{2♠, A♥, K♦}` as High Card.

Complete ordering, highest to lowest:

```
A-2-3  >  A-K-Q  >  K-Q-J  >  Q-J-10  >  J-10-9  >  10-9-8
       >  9-8-7  >  8-7-6  >  7-6-5   >  6-5-4   >  5-4-3  >  4-3-2
```

This ordering applies identically to Pure Sequence and to Sequence.

### 4.3 Tie-breaking within every category

Tie-breaking is **total** within a category and is specified exhaustively:

| Category | Tie-break rule | Highest | Lowest | Distinct classes |
|---|---|---|---|---|
| Trail | rank of the triplet | `A-A-A` | `2-2-2` | 13 |
| Pure sequence | position in the §4.2 order | `A-2-3` | `4-3-2` | 12 |
| Sequence | position in the §4.2 order | `A-2-3` | `4-3-2` | 12 |
| Colour | highest card; then second; then lowest | `A-K-J` | `5-3-2` | 274 |
| Pair | rank of the pair; then rank of the odd card | `A-A-K` | `2-2-3` | 156 |
| High card | highest card; then second; then lowest | `A-K-J` | `5-3-2` | 274 |

Notes on the extremes, which are useful implementation assertions:

- Colour and High Card share the same rank-set space: three *distinct, non-consecutive* ranks.
  Hence both have 274 classes and identical extremes. `A-K-Q` cannot be a Colour or High Card
  because it is a run, so the maximum is `A-K-J`; `4-3-2` and `5-4-3` are runs, so the minimum
  is `5-3-2`.
- The Pair maximum is `A-A-K`, not `A-A-A`, which is a Trail.

**Aces high in trails.** `A-A-A` is the highest trail and `2-2-2` the lowest. This is the
Teen Patti convention and **diverges from three-card brag**, where the prial of threes beats
the prial of aces. See §17, D1.

### 4.4 Exact ties

Because suits are unordered, two hands may be exactly equal in strength.

| Category | Tie possible? | Reason |
|---|---|---|
| Trail | **No** | two equal trails need 6 cards of one rank; only 4 exist |
| Pure sequence | Yes | e.g. `A♠2♠3♠` vs `A♥2♥3♥` |
| Sequence | Yes | same run, different suit multisets |
| Colour | Yes | e.g. `A♠9♠5♠` vs `A♥9♥5♥` |
| Pair | Yes | e.g. `K♠K♥7♦` vs `K♦K♣7♠` |
| High card | Yes | same ranks, different suits |

Ties are therefore a live case, not a theoretical one, and MUST be handled (§10.5, §10.6).
Comparison MUST be implemented on strength classes (§5.3), never on raw cards.

### 4.5 Excluded ranking variants

No jokers, no wild cards, no Muflis (lowball), no AK47, no 999, no Kiss-Land-Screw, no
"joker hunt", no bonus payouts. The §4.1 order is fixed for the entire study. See §16.

---

## 5. Combinatorial Enumeration

### 5.1 The table

All counts are over `C(52,3) = 22,100` hands. Let `V = 12` be the number of valid runs (§4.2).

| Rank | Category | Derivation | Count | Probability |
|---:|---|---|---:|---:|
| 1 | Trail | `13 × C(4,3)` | 52 | 0.2353% |
| 2 | Pure sequence | `V × 4` | 48 | 0.2172% |
| 3 | Sequence | `V × (4³ − 4)` | 720 | 3.2579% |
| 4 | Colour | `4 × C(13,3) − 48` | 1,096 | 4.9593% |
| 5 | Pair | `13 × C(4,2) × 12 × 4` | 3,744 | 16.9412% |
| 6 | High card | `(C(13,3) − V) × (4³ − 4)` | 16,440 | 74.3891% |
| | **Total** | `C(52,3)` | **22,100** | **100.0000%** |

**Verification.** `52 + 48 + 720 + 1096 + 3744 + 16440 = 22,100.` ✓

Three independent cross-checks:

- *Flushes:* all same-suit hands number `4 × C(13,3) = 4 × 286 = 1,144`, split into
  48 pure sequences + 1,096 colours. ✓
- *Straights:* all consecutive-rank hands number `V × 4³ = 12 × 64 = 768`, split into
  48 pure sequences + 720 sequences. ✓
- *High card, computed forwards:* non-run rank-triples `= C(13,3) − V = 286 − 12 = 274`;
  non-flush suitings per triple `= 4³ − 4 = 60`; `274 × 60 = 16,440`. ✓

### 5.2 Remark: ranking is misaligned with rarity

**The Trail (52 hands) is strictly more common than the Pure Sequence (48 hands), yet ranks
above it.** The hand ranking is therefore *not* a rarity ordering. This is a genuine defect
inherited from three-card brag, and it survives in essentially every commercial Teen Patti
implementation.

Two consequences worth stating in the paper:

1. It is a real, quantifiable inconsistency in a game played for money at scale — the ranking
   inverts frequency at the very top of the scale, exactly where stakes are highest.
2. It is a useful probe for LLM evaluation. A model that has memorised "rarer beats commoner"
   as a general card-game heuristic will rank Pure Sequence above Trail. That is a
   discriminating error: it is wrong under the actual rules but correct under the heuristic,
   so it separates rule recall from heuristic transfer.

Everywhere else the ranking is rarity-consistent (48 < 720 < 1,096 < 3,744 < 16,440), which
makes the single inversion at the top sharper rather than softer.

### 5.3 Strength classes

Define `κ : hands → Z` mapping a hand to its **strength class**: the pair (category,
tie-break key). From the last column of §4.3:

```
|range(κ)| = 13 + 12 + 12 + 274 + 156 + 274 = 741
```

**`κ` is lossless for hand comparison.** Player `i` beats player `j` iff `κ(h_i) > κ(h_j)`,
and ties are exactly `κ(h_i) = κ(h_j)`. The winner-determination function MUST be implemented
as an integer comparison on `κ`.

**`κ` is NOT a lossless information abstraction.** This distinction matters for Phase 1 and is
easy to get wrong. Two hands in the same class may have different *card-removal* effects on
the opponent's hand distribution, because they are not related by a suit permutation. For
example `K♠K♥7♦` and `K♠K♥7♠` are both class `K-K-7`, but the second removes a third spade
from the deck and so induces a different conditional distribution over the opponent's flushes.

Therefore:

- Use `κ` for **terminal utility** — exact, 741 classes.
- Do **not** assume `κ` is a valid state abstraction for the solver without separately
  establishing it. A genuinely lossless information abstraction requires either all 22,100
  hands or a proper suit-isomorphism quotient. Computing that quotient is a **Phase 1**
  deliverable and is out of scope here.

---

## 6. The Blind/Seen Mechanic

This is the core research object; it is specified before the betting rules because the
betting rules depend on it.

### 6.1 Status

Each player carries `b_i ∈ {B, S}`. Both players begin **Blind** (`b_1 = b_2 = B`). A Blind
player has not looked at their cards.

### 6.2 Conversion is voluntary, one-way, and taken at the start of one's own turn

**Commitment: a Blind player may convert to Seen only at the start of their own turn, before
selecting a betting action, and the conversion is irreversible.**

Concretely, a turn for a player with `b_i = B` has two decision points:

1. **Look decision:** choose `see` or `stay-blind`.
2. **Betting decision:** choose a betting action, priced by the status resulting from step 1.

A player who chooses `see` observes `h_i`, sets `b_i ← S` permanently, and **must then act at
Seen prices in that same turn** (§8.4). They cannot see and then bet at Blind prices.

A player with `b_i = S` has only decision point 2.

A player may not convert outside their own turn, and may not convert after `S`.

### 6.3 Status is public

**Commitment: `b_1` and `b_2` are public information at all times.** The `see` action is
publicly observed.

This is faithful — physically looking at one's cards is visible — and it is essential to the
game: bet *amounts* differ by status (§7), so status is inferable from the betting stream even
if it were not directly observed. Status is therefore part of the public history (§13).

### 6.4 What Blind play costs and buys

| | Blind | Seen |
|---|---|---|
| Pays per betting action | `s` (chaal) or `2s` (raise) | `2s` (chaal) or `4s` (raise) |
| Show cost | `s` | `2s` |
| May demand a show against a Blind opponent | **Yes** | **No** |
| Knows own hand | No | Yes |

A Blind player bets at **half** a Seen player's price and holds an **exclusive right to force
a showdown** (§10.3). Against these, they forgo all private information. The strategic tension
is that the discount and the show right are both increasing in how long one stays uninformed.

---

## 7. Stake Mechanics

### 7.1 Definition

`s` is the **current stake, in blind-equivalent units**: the amount a *Blind* player must pay
to chaal. A Seen player always pays `2s` to chaal. Initially `s = 1`, equal to the boot.

### 7.2 Update rule

Following pagat (§18): after a Blind player bets, the new stake is *the amount they put in*;
after a Seen player bets, the new stake is *half the amount they bet*. Under the
discretisation of §8.4 this yields exactly:

| Actor status | Action | Pays | New stake |
|---|---|---:|---:|
| Blind | chaal | `s` | `s` (unchanged) |
| Blind | raise | `2s` | `2s` (doubled) |
| Seen | chaal | `2s` | `s` (unchanged) |
| Seen | raise | `4s` | `2s` (doubled) |

**The two rules collapse to one:** *chaal leaves `s` unchanged; raise doubles `s`.* Status
affects only the amount paid, never the stake transition. Implementations SHOULD encode it
this way — it is the single most error-prone rule in the game, and this form is hard to get
wrong.

Since `r ≤ 2` (§9.2), `s ∈ {1, 2, 4}` always.

### 7.3 Contributions are not equalised

**This is the single largest structural difference from poker and MUST be understood before
implementing.**

Teen Patti has **no betting round and no pot-equalisation condition**. Each betting action is
a fresh contribution of the current amount; it is not a "call" that levels contributions.
There is no point at which the betting "closes" and play advances to a next street — there are
no streets. Betting alternates indefinitely until a player packs, a show occurs, or the cap
binds.

Consequences:

- `c_1 ≠ c_2` is normal at terminal nodes, including at showdowns. Example C (§14.3) ends with
  `c_1 = 7` and `c_2 = 5` at a showdown.
- Do **not** implement an "amount to call" or "bet matched" variable. There is none.
- The absence of a closing condition is precisely why an artificial cap (§9) is required for
  finiteness. Unlike PokerBench's raise cap, which truncates an already-finite tree, our cap
  makes an infinite tree finite.

---

## 8. Action Set

### 8.1 The actions

**Commitment: six actions.**

| Action | Type | Effect |
|---|---|---|
| `see` | look | `b_i ← S`; reveals `h_i` to `i`; does **not** advance the turn or increment `k` |
| `stay-blind` | look | no effect; does not advance the turn or increment `k` |
| `chaal` | betting | pay `s` (Blind) or `2s` (Seen); `s` unchanged; `k += 1` |
| `raise` | betting | pay `2s` (Blind) or `4s` (Seen); `s ← 2s`; `k += 1`; `r += 1` |
| `pack` | betting, terminal | fold; opponent wins the pot |
| `show` | betting, terminal | pay `s` (Blind) or `2s` (Seen); compare hands (§10) |

### 8.2 Correction to the working action set

The working set in the research plan was `{blind-chaal, see, chaal, raise, pack, show}`. Two
adjustments:

- **`blind-chaal` and `chaal` are merged.** They are one action whose *amount* is a function of
  `b_i`. Keeping them separate duplicates every node in the tree and invites the bug where a
  player who has just seen still pays the Blind price. Amount is state-derived, never
  action-derived.
- **`stay-blind` is added.** The look decision needs an explicit complement so that it is a
  well-defined choice at a well-defined node. Without it, "not seeing" is implicit, and the
  information-set structure of §13 cannot be stated cleanly.

### 8.3 Legality

Let `i` be the player to act, `j` the opponent.

**Look decision** (reached iff `b_i = B`): `{see, stay-blind}`, both always legal.

**Betting decision** (always reached):

| Action | Legal iff |
|---|---|
| `pack` | always |
| `chaal` | always |
| `raise` | `r < 2` |
| `show` | `b_i = B`, **or** (`b_i = S` and `b_j = S`) |

The `show` condition is the formal statement of "a seen player may not demand a show against a
blind player" (§10.3). Note it is stated on `b_i` *after* any `see` in this turn: a player who
sees this turn is Seen for legality purposes, and so loses the ability to show against a Blind
opponent.

`pack` and `chaal` are unconditionally legal because the stack provably never binds (§3.2).

### 8.4 Bet-size discretisation

**Commitment: exactly two betting sizes — `chaal` (match) and `raise` (double). Confirmed as
proposed.**

*Justification, which is unusually strong here:* the real game's legal bet interval is
**`[s, 2s]` for a Blind player and `[2s, 4s]` for a Seen player** (pagat, §18). The maximum is
exactly twice the minimum in both cases. Taking `{chaal, raise}` therefore selects **the two
endpoints of the actual legal interval** — it is not an imposed grid but the natural boundary
discretisation of the rule as written. Teen Patti's own betting structure is
minimum-or-maximum-shaped, so a two-action abstraction loses far less than the analogous
abstraction does in no-limit poker.

Intermediate amounts are excluded. This is the only respect in which our betting differs from
the real game's action space, and it is a strictly coarsening restriction.

---

## 9. Betting Cap and Termination

### 9.1 Why a cap is needed

Per §7.3, Teen Patti's betting never closes. Two players may alternate `chaal` forever. The
tree is infinite without an artificial bound. The cap is a **modelling device, not a rule of
Teen Patti**, and the paper must say so.

### 9.2 The caps

**Commitment: two caps.**

- **C1 — Raise cap: `r ≤ 2`.** At most two raises per hand, by either player in any
  combination.
- **C2 — Action cap: `k ≤ 8`.** At most eight betting actions per hand, i.e. four turns per
  player.

`see` and `stay-blind` do not count toward `k`. `pack` and `show` are terminal and so the cap
never blocks them.

*Justification for C1:* the stake doubles on each raise, so raises drive exponential stake
growth and are what actually make stacks bind. Two raises bounds `s ≤ 4`. This mirrors
PokerBench's cap of two pre-flop raises.

*Justification for C2, and why not 4:* the proposed cap of 4 total actions gives each player
only **two** turns. That is too few for the research object: a player who stays Blind on turn 1
and converts on turn 2 has no remaining turns at which to exploit either the Blind discount or
the Blind show right, so the conversion-timing decision collapses to near-triviality. Eight
actions give each player four turns — enough for "stay blind, stay blind, convert, bet"
sequences to be distinguishable — while keeping the tree small (§9.3).

### 9.3 Resulting tree size

Only `chaal` and `raise` continue the hand, so the non-terminal spine branches at most 2 ways
per turn, subject to C1. The number of betting-action sequences of length `k` with at most 2
raises is `1 + k + C(k,2)`:

| `k` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sequences | 1 | 2 | 4 | 7 | 11 | 16 | 22 | 29 | 37 |

Each spine node is further refined by the **conversion pattern**: each player converts at one
of their turns or never, and conversion is one-way and monotone. Over four turns each, that is
at most 5 options per player.

Multiplying and summing over `k` gives an estimate of

```
|public decision histories|  ≈  2.2 × 10³
```

This is an **estimate, not an exact count** — the conversion-pattern factor is not fully
independent of `k` (a player can only have converted at a turn that was actually reached), so
the true figure is somewhat lower. Phase 1 MUST compute it exactly by enumeration; this
figure is for sizing only.

Information sets then follow from §13:

- a Blind actor's node carries **no** private refinement → 1 information set per history;
- a Seen actor's node is refined by their hand → up to 741 (by class) or 22,100 (exact).

So `|I|` lands in the **10⁵–10⁶** range. That is comfortably within tabular CFR+ with exact
best-response exploitability, which is the Phase 2 requirement. The cap could be loosened
later if the tree proves smaller than estimated.

### 9.4 Maximum exposure (bound used in §3.2)

A player pays at most `4s` in one action, and `s ≤ 4` by C1, but `s = 4` requires both raises
already spent, so a raise at `s = 4` is illegal. The maximum single payment is therefore **8**
(a Seen raise at `s = 2`, or a Seen chaal at `s = 4`). With the boot and four turns:

```
max c_i  =  1 + 4 × 8  =  33
```

This bound is **achieved**, so it is exact. Witness (maximising seat 2):

| Turn | Actor | Action | Pays | `s` after | `c_2` |
|---:|---|---|---:|---:|---:|
| — | both | boot | 1 | 1 | 1 |
| 1 | P1 (B) | raise | 2 | 2 | 1 |
| 2 | P2 → see, raise | Seen raise | 8 | 4 | 9 |
| 3 | P1 (S) | chaal | 8 | 4 | 9 |
| 4 | P2 (S) | chaal | 8 | 4 | 17 |
| 5 | P1 (S) | chaal | 8 | 4 | 17 |
| 6 | P2 (S) | chaal | 8 | 4 | 25 |
| 7 | P1 (S) | chaal | 8 | 4 | 25 |
| 8 | P2 (S) | chaal | 8 | 4 | **33** |

Since `33 < 50`, no player can ever be unable to act. ∎

### 9.5 What happens at the cap

**Commitment: when `k` reaches 8 and the hand is still live, a forced show occurs immediately,
at no cost to either player.**

Both hands are revealed and compared under §4. No player requested it, so:

- no show fee is charged;
- the "requester loses ties" rule (§10.5) **cannot apply**; ties are resolved by **splitting
  the pot** (§10.6).

*Justification, and why not a forced pack:* a forced pack would have to designate a loser, and
whichever rule was chosen would create a perverse incentive — a player could win, or avoid
losing, merely by driving the hand to the cap. That would make the artificial cap
strategically load-bearing and would corrupt the solved strategy. A forced show is
outcome-neutral with respect to the cap: it resolves the hand on card strength, which is what
the uncapped game would eventually do anyway.

*On consistency with "a blind player cannot be forced to show":* that rule (§10.3) restricts
**opponent-initiated** shows — it denies the Seen player an action. A cap-triggered show is not
an action of either player, so it does not violate the rule. A player still Blind at the cap has
their cards revealed without ever having looked at them; this is mechanically well-defined and
is required to be handled (§15, E2).

---

## 10. Show Mechanics

### 10.1 Availability

In the real game "a show cannot occur until all but two players have dropped out". Heads-up,
that condition holds from the deal, so **`show` is available from turn 1**, subject only to
§10.3. An immediate turn-1 Blind-vs-Blind show costs 1 unit and is legal; it is retained
rather than special-cased away, because suppressing it would be an unmotivated deviation.

### 10.2 Cost

Paid into the pot by the requester, as part of their turn:

| Requester | Cost |
|---|---:|
| Blind | `s` |
| Seen | `2s` |

The show fee is exactly the requester's `chaal` price. A show is therefore best understood as
"chaal, and then resolve" — which is also why no separate matching requirement is needed.

Per pagat, a Blind requester pays `s` **irrespective of the opponent's status**.

### 10.3 Who may request: the Blind player's exclusive right

**Commitment:**

- Both Blind → **either** player may request, cost `s`.
- Both Seen → **either** player may request, cost `2s`.
- Requester Blind, opponent Seen → **legal**, cost `s`.
- Requester **Seen**, opponent **Blind** → **ILLEGAL**. The Seen player must `chaal`, `raise`,
  or `pack`.

The last row is the formalisation of "a blind player cannot be forced to show" (brag: *"you
cannot see a blind man"*).

**Implication heads-up, which is a central research point.** With only two players, this rule
does not merely protect the Blind player — it hands them a **unilateral and exclusive right to
end the hand at a showdown**, at half price, which the Seen player cannot refuse, cannot
pre-empt, and cannot mirror. Staying Blind therefore purchases:

1. a 50% discount on every betting action, and
2. sole control of when the hand is resolved on cards.

Against a Seen opponent who has, by revealing their status, publicly signalled that they hold
information, the Blind player's show right is the mechanism by which remaining uninformed can
be *positively* valuable rather than merely cheap. Any solved strategy that does not use it is
almost certainly an implementation bug.

### 10.4 Reveal order

**Commitment: reveal is simultaneous.**

The show is terminal — no player acts after it — so reveal order carries no game-theoretic
content and cannot affect any strategy or utility. Simultaneous reveal is therefore chosen as
the representation with no spurious structure. Table conventions that stage the reveal
(§17, D4) are behaviourally equivalent and are not modelled.

### 10.5 Ties in a requested show

**Commitment: the requester loses.** If `κ(h_i) = κ(h_j)` and player `i` paid for the show,
player `j` wins the entire pot.

Source: pagat Teen Patti, *"If the hands are equal, the player who did not pay for the show
wins the pot"*; brag, *"the calling player ... loses"*. Both sources agree.

This is strategically meaningful and must not be simplified to a split: it prices the show
option, making it costly to demand a showdown with a marginal hand. Removing it would
materially change the solution.

### 10.6 Ties in a forced (cap-triggered) show

**Commitment: split the pot.** Each player receives `p/2`, so `u_i = p/2 − c_i`.

There is no requester, so §10.5 has no referent. Zero-sum is preserved:
`(p/2 − c_1) + (p/2 − c_2) = p − p = 0`. ✓

Note that because contributions need not be equal (§7.3), a split pot generally yields
**nonzero** payoffs — the player who contributed less profits. Example C (§14.3) exhibits this.

### 10.7 Trail ties

Impossible (§4.4). Implementations SHOULD assert this rather than handle it.

---

## 11. Terminal Utilities

There are three kinds of terminal node. Let `p = c_1 + c_2` be the final pot.

| Terminal | Winner | `u_i` | `u_j` |
|---|---|---|---|
| `pack` by `i` | `j` | `−c_i` | `p − c_j` |
| show, `κ(h_i) ≠ κ(h_j)` | higher `κ` | `p − c_i` if `i` wins else `−c_i` | complement |
| requested show, tie | non-requester | requester: `−c` | non-requester: `p − c` |
| forced show, tie | none | `p/2 − c_i` | `p/2 − c_j` |

**Zero-sum check.** For a decisive result with winner `i`: `u_i + u_j = (p − c_i) + (−c_j)
= p − c_i − c_j = 0`. ✓ For a split: shown in §10.6. ✓

Payoffs are in boot units. An implementation MUST assert `u_1 + u_2 = 0` at every terminal
node; this single assertion catches almost every stake- and pot-accounting bug.

---

## 12. Extensive-Form Game Definition

### 12.1 Tuple

```
G = ( N, H, Z, P, f_c, {I_i}, {u_i} )
```

- **Players** `N = {1, 2}`, plus the chance player `c`.
- **Histories** `H`: sequences beginning with a chance outcome and continuing with actions
  from §8.1. `Z ⊂ H` are terminal histories (§11).
- **Turn function** `P : H \ Z → N ∪ {c}`. `P(∅) = c`; thereafter turns alternate between
  1 and 2, with both the look and betting decision of a turn assigned to the same player.
- **Chance** `f_c`: a single node at the root.

### 12.2 The chance node

`f_c` deals disjoint 3-subsets `h_1, h_2 ⊂ D` uniformly at random:

```
|outcomes| = C(52,3) × C(49,3) = 22,100 × 18,424 = 407,170,400
```

each with probability `1 / 407,170,400`. There is exactly **one** chance node — Teen Patti has
no community cards, no streets, and no further dealing. All subsequent nodes are decision nodes.

`h_1` and `h_2` are **not disclosed** to anyone at this point, including to their holders. This
is the formal seat of the blind mechanic: in poker the dealing of hole cards is simultaneously
the *observation* of hole cards, so the two are conflated. **Here they are separate events.**
Dealing happens at the chance node; observation happens only if and when that player takes the
`see` action (§6.2). Conflating them is the single most likely modelling error, and it silently
destroys the entire research object.

### 12.3 State

Each decision history `h` determines the state vector

```
σ(h) = ( s, c_1, c_2, b_1, b_2, k, r, P(h), phase )
```

with `phase ∈ {look, bet}` and `p = c_1 + c_2` derived. All components of `σ` are **public**
(§6.3). Private information consists solely of `h_i`, and only for players with `b_i = S`.

### 12.4 Information partition — the critical definition

Write `H_pub(h)` for the **public history**: the full sequence of actions taken (including
`see`, `stay-blind`, and all bet sizes), with the chance outcome removed. Note `σ(h)` is a
function of `H_pub(h)`.

**For a player `i` with `b_i = S`** (at a betting node, having seen either earlier or this turn):

```
I_i(h) = { h' ∈ H : H_pub(h') = H_pub(h)  and  h_i(h') = h_i(h) }
```

The information set fixes their own hand and leaves the opponent's free: it contains 18,424
histories, one per possible opponent hand. Their belief over their **own** hand is a **point
mass** on `h_i`.

**For a player `i` with `b_i = B`** (at a look node, or at a betting node after `stay-blind`):

```
I_i(h) = { h' ∈ H : H_pub(h') = H_pub(h) }
```

with **no conditioning on `h_i` whatsoever.** The information set contains **all 407,170,400
deals**. Their belief over their **own** hand is **not** a point mass:

```
P( h_i = x | I_i )  =  Σ_y  P( h_i = x, h_j = y | I_i )
```

which under the prior is uniform over all 22,100 hands.

**This is the formal content of the blind mechanic and it must be implemented exactly.** A
Blind player's strategy at a given public history is a *single* distribution over actions —
not a per-hand distribution — because they have exactly one information set there. Equivalently:

> **Implementation contract.** The number of information sets at a Blind decision node is
> **one**, independent of the deal. Any implementation in which a Blind player's regret or
> strategy is indexed by their own cards is incorrect, and will produce a strategy that is
> strictly better than any achievable strategy, silently invalidating both the equilibrium and
> the exploitability number.

### 12.5 Remark: a Blind player's beliefs are not static

A subtlety worth a line in the paper. A Blind player's posterior over their **own** hand is not
frozen at the uniform prior. The opponent's actions are informative about `h_j`, and `h_i` and
`h_j` are dependent through the shared deck (card removal). So observing the opponent bet
strongly shifts the Blind player's belief about `h_j`, which in turn shifts their belief about
their own `h_i` — slightly, and in the opposite direction.

The effect is small but it is real, and it is handled by CFR automatically provided §12.4 is
implemented faithfully. It has no analogue in poker, where a player's belief about their own
hole cards is degenerate at all times. It is, as far as we are aware, the only vying game in
common play where a player performs nontrivial inference about their own holding.

### 12.6 Perfect recall

Each player observes all public actions, and observes `h_i` from the moment they see it
onwards. Status is monotone and public. The game therefore has **perfect recall**, which is
required for CFR's regret bound to apply. Implementations MUST NOT prune or merge histories in
a way that breaks it.

---

## 13. Information Set Specification

What each player observes, and does not, at every decision point.

### 13.1 Always observed by both players (public)

- Both boot postings, and `p`, `c_1`, `c_2`.
- The complete action sequence: every `pack`/`chaal`/`raise`/`show`, **with amounts**.
- Every `see` / `stay-blind` decision by either player — hence `b_1`, `b_2` at all times.
- Derived: `s`, `k`, `r`, whose turn it is, and which actions are legal.

### 13.2 Observed by player `i` only

- `h_i`, **if and only if** `b_i = S`, from the turn on which they chose `see` onwards.

### 13.3 Never observed before a terminal show

- `h_j`, the opponent's hand, under all circumstances.
- Any card not in one's own observed hand. There are no community cards and no exposed cards.

### 13.4 Decision points, enumerated

| Point | Reached when | Observes | Information set key |
|---|---|---|---|
| Look | turn of `i`, `b_i = B` | public history only | `H_pub` |
| Bet (blind) | after `stay-blind` | public history only | `(H_pub, stay-blind)` |
| Bet (just seen) | after `see` | public history **+ `h_i`** | `(H_pub, see, h_i)` |
| Bet (already seen) | turn of `i`, `b_i = S` | public history **+ `h_i`** | `(H_pub, h_i)` |

Rows 1 and 2 have information-set keys **independent of the deal**. Rows 3 and 4 are indexed by
`h_i` (or by `κ(h_i)` under abstraction, with the §5.3 caveat).

---

## 14. Worked Examples

Normative implementation test cases. Both players start with stack 50. All amounts in boot
units. `s` and `p` are shown **after** the action on that row.

### 14.1 Example A — Blind/Blind, conversion reveals a weak hand, early pack

Deal: `h_1 = 7♦ 4♣ 2♠` (High card 7-4-2), `h_2` irrelevant (never revealed).

| # | Actor | `b` | Action | Pays | `s` | `p` | `c_1` | `c_2` | Stack 1 | Stack 2 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | both | B,B | boot | 1 each | 1 | 2 | 1 | 1 | 49 | 49 |
| 1 | P1 | B | stay-blind, chaal | 1 | 1 | 3 | 2 | 1 | 48 | 49 |
| 2 | P2 | B | stay-blind, raise | 2 | **2** | 5 | 2 | 3 | 48 | 47 |
| 3 | P1 | B→**S** | **see**, then pack | 0 | 2 | 5 | 2 | 3 | 48 | 47 |

`k = 2`, `r = 1`. **Terminal: P1 packed. P2 wins `p = 5`.**

```
u_1 = −c_1 = −2
u_2 =  p − c_2 = 5 − 3 = +2        u_1 + u_2 = 0 ✓
Final stacks: P1 = 48, P2 = 47 + 5 = 52.   Sum = 100 ✓
```

*Tests:* blind raise doubles `s` (row 2); `see` costs nothing and does not increment `k`;
seeing and then packing in the same turn; `pack` pays nothing.

### 14.2 Example B — Blind→Seen conversion, then the Blind player exercises the show right

Deal: `h_1 = A♥ A♠ 9♦` (Pair, `A-A-9`), `h_2 = K♣ K♦ 3♠` (Pair, `K-K-3`).

| # | Actor | `b` | Action | Pays | `s` | `p` | `c_1` | `c_2` |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | both | B,B | boot | 1 each | 1 | 2 | 1 | 1 |
| 1 | P1 | B | stay-blind, chaal | 1 | 1 | 3 | 2 | 1 |
| 2 | P2 | B | stay-blind, chaal | 1 | 1 | 4 | 2 | 2 |
| 3 | P1 | B→**S** | **see**, then raise (Seen: `4s`) | 4 | **2** | 8 | 6 | 2 |
| 4 | P2 | B | stay-blind, chaal (Blind: `s`) | 2 | 2 | 10 | 6 | 4 |
| 5 | P1 | S | chaal (Seen: `2s`) — **`show` is ILLEGAL here** | 4 | 2 | 14 | 10 | 4 |
| 6 | P2 | B | **show** (Blind: `s`) | 2 | 2 | 16 | 10 | 6 |

`k = 5`, `r = 1`. **Terminal: show.** Reveal `A-A-9` vs `K-K-3` → `κ(h_1) > κ(h_2)`, **P1 wins.**

```
u_1 =  p − c_1 = 16 − 10 = +6
u_2 = −c_2 = −6                    u_1 + u_2 = 0 ✓
Final stacks: P1 = 50 − 10 + 16 = 56,  P2 = 50 − 6 = 44.   Sum = 100 ✓
```

*Tests, and this is the most important case in the document:*

- **Row 3:** a player who sees pays the **Seen** raise `4s = 4`, not the Blind raise `2s`, in
  the very same turn. The stake goes to `2s = 2` — *half* what was paid.
- **Row 4:** P2, still Blind, pays only `s = 2` where a Seen player would pay `2s = 4`.
- **Row 5:** P1 is Seen and P2 is Blind, so `show` **must be masked out** of P1's action set
  (§8.3). P1 cannot force the issue and must chaal, raise, or pack.
- **Row 6:** P2, Blind, exercises the exclusive show right for `s = 2` — half the `2s = 4` that
  P1 would have had to pay, and an action P1 never had available at all.
- P2 loses here, but note `c_2 = 6` against `c_1 = 10`: staying Blind bought the whole hand at
  60% of the price, and bought the right to end it.

### 14.3 Example C — Cap-triggered forced show, exact tie, split pot

Deal: `h_1 = A♠ 9♠ 5♠` (Colour, `A-9-5`), `h_2 = A♥ 9♥ 5♥` (Colour, `A-9-5`). Equal `κ`.

| # | Actor | `b` | Action | Pays | `s` | `p` | `c_1` | `c_2` | `k` |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | both | B,B | boot | 1 each | 1 | 2 | 1 | 1 | 0 |
| 1 | P1 | B | stay-blind, chaal | 1 | 1 | 3 | 2 | 1 | 1 |
| 2 | P2 | B | stay-blind, chaal | 1 | 1 | 4 | 2 | 2 | 2 |
| 3 | P1 | B | stay-blind, chaal | 1 | 1 | 5 | 3 | 2 | 3 |
| 4 | P2 | B | stay-blind, chaal | 1 | 1 | 6 | 3 | 3 | 4 |
| 5 | P1 | B→**S** | **see**, then chaal (Seen: `2s`) | 2 | 1 | 8 | 5 | 3 | 5 |
| 6 | P2 | B | stay-blind, chaal (Blind: `s`) | 1 | 1 | 9 | 5 | 4 | 6 |
| 7 | P1 | S | chaal (Seen: `2s`) — `show` illegal | 2 | 1 | 11 | 7 | 4 | 7 |
| 8 | P2 | B | stay-blind, chaal (Blind: `s`) | 1 | 1 | 12 | 7 | 5 | **8** |

`k = 8` → **cap C2 binds. Forced show, no fee, no requester.** `r = 0` throughout.

Reveal: both `A-9-5` Colour → **exact tie** → **split pot** (§10.6), `p/2 = 6` each.

```
u_1 = p/2 − c_1 = 6 − 7 = −1
u_2 = p/2 − c_2 = 6 − 5 = +1       u_1 + u_2 = 0 ✓
Final stacks: P1 = 50 − 7 + 6 = 49,  P2 = 50 − 5 + 6 = 51.   Sum = 100 ✓
```

*Tests, several of which nothing else in this document covers:*

- The **action cap** triggering a forced show at `k = 8`, with `see` correctly **not** counted
  (there were 9 player decisions but only 8 betting actions).
- **Unequal contributions at a showdown**: `c_1 = 7 ≠ c_2 = 5`. Any implementation carrying a
  poker-style "amount to call" will fail here.
- **An exact tie**, which is only reachable because suits are unordered (§4.4).
- **A split pot yielding nonzero payoffs** — `±1`, arising purely from the contribution
  asymmetry. A naive "split ⇒ both get 0" shortcut fails here.
- **P2's cards are revealed although P2 never looked at them**, and P2 still profits.
- The full run of Blind chaals at `s = 1`: P2 played the entire hand for 5 units.

---

## 15. Edge Cases for the Implementer

| # | Case | Resolution |
|---|---|---|
| **E1** | A player cannot afford a legal action | **Cannot occur.** Max exposure is 33 (§9.4) against a stack of 50. There is no all-in in this specification. Implementations MUST reject configured stacks `< 34` at startup and MUST assert `stack_i ≥ 0` after every payment. Real Teen Patti has no side-pot mechanism, so silently permitting a short stack would leave the rules undefined. |
| **E2** | Both players still Blind when the cap binds | Forced show (§9.5). Both hands are revealed although neither player ever looked. No fee. Tie → split. The engine must be able to evaluate `κ` for a hand its owner never observed — so hand evaluation must not be gated on `b_i = S`. |
| **E3** | Show requested by a Blind player against a Seen player | **Legal.** Cost `s`, not `2s`, per pagat's "irrespective of whether the other player is blind or seen". |
| **E4** | Show requested by a Seen player against a Blind player | **Illegal.** Must be absent from the action mask (§8.3), not merely rejected on attempt. This is the rule most often mis-implemented; make it a mask-level invariant and unit-test it directly. |
| **E5** | `raise` attempted when `r = 2` | Illegal; masked out. `s` can never exceed 4. |
| **E6** | `see` attempted by a player with `b_i = S` | Illegal. The look decision is reached **only** when `b_i = B`. Conversion is one-way; assert `b_i` never transitions `S → B`. |
| **E7** | A player sees, then wants to bet at Blind prices | Illegal. Pricing reads `b_i` **after** the look decision resolves (§6.2). Same for show legality (§8.3). |
| **E8** | Tie, requested show vs. forced show | Different rules by design: requested → **requester loses** (§10.5); forced → **split** (§10.6). Both must be implemented; do not unify them. |
| **E9** | Two equal trails | **Impossible** — needs 6 cards of one rank. Assert rather than handle (§10.7). |
| **E10** | `pack` on turn 1, or `show` on turn 1 | Both **legal**. Turn-1 `show` costs `s = 1` and is available because heads-up satisfies "all but two have dropped out" from the deal (§10.1). Do not special-case them away. |
| **E11** | `2-A-K` classified as a sequence | **Bug.** Wrap-around runs are invalid (§4.2). `{2,A,K}` is High Card. Unit-test this explicitly; it is the most common hand-evaluation error in three-card games. |
| **E12** | Suits used as a tie-break | **Bug.** Suits are unordered (§2.1). If a comparison ever returns a decision that `κ` says is a tie, the evaluator is wrong. |
| **E13** | `see` / `stay-blind` incrementing `k` | **Bug.** Only `chaal` and `raise` increment `k` (§9.2). Example C is the regression test: 9 decisions, `k = 8`. |
| **E14** | Deriving the stake from the amount paid | Do **not**. `chaal` leaves `s` unchanged and `raise` doubles it, regardless of status (§7.2). Deriving `s` from the payment requires a status-dependent halving and is a frequent source of off-by-2× errors. |

---

## 16. Explicitly Excluded Mechanics

Stated so that the paper's scope is defensible.

| Mechanic | Justification for exclusion |
|---|---|
| **Sideshow / compromise** | Heads-up it is **redundant**: a compromise costs `2s`, requires both players Seen, and resolves ties against the requester — identical to a Seen-vs-Seen `show` (§10). It adds only a decline branch. See §17, D2 — this exclusion rests on redundancy, **not** on a player-count requirement. |
| **3+ players** | Destroys the two-player zero-sum setting on which CFR's convergence guarantee and exact exploitability both depend. Future work. |
| **Jokers / wild cards** | Not part of standard Teen Patti; would change §4 and §5 entirely. |
| **Muflis (lowball)** | Inverts the hand ranking. A separate game requiring its own solve. |
| **AK47, 999, K-Little, Kiss-Land-Screw, and other "variations"** | House variants that add wild ranks or alternate rankings; none is standard, and each would need its own §4 and §5. |
| **Pot limit / betting limits / table maximum** | Superseded by the caps of §9.2, which bound the tree more tightly and more usefully. |
| **Progressive boot, bring-in, straddles** | Not standard; would break the symmetric-ante property of §3.1. |
| **Multi-hand bankroll dynamics** | Excluded by the stack reset of §3.3. Each hand is an independent game instance. |
| **Intermediate bet sizes** | Discretised to the endpoints of the real legal interval (§8.4). The only deviation from the real action space. |
| **Side pots / all-in** | Unreachable by construction (§15, E1). Real Teen Patti has no side-pot rule, so this avoids specifying one. |

---

## 17. Variant Divergences

Real Teen Patti varies regionally. The body of this document commits to one choice throughout;
every known alternative is recorded here.

| # | Issue | **Our commitment** | Alternative(s) | Note |
|---|---|---|---|---|
| **D1** | Highest trail | **`A-A-A`** (aces high, `2-2-2` lowest) | **Three-card brag ranks the prial of threes highest**, above `A-A-A` | A real and clean divergence between the two games. We follow the Teen Patti convention. Worth flagging in the paper, since brag-trained priors would get this wrong. |
| **D2** | Sideshow player count | Excluded on **redundancy** grounds | Pagat describes the compromise as available with **two** remaining players, both Seen, requested immediately after betting the minimum | This **contradicts the widespread claim that a sideshow requires 3+ active players**. Our exclusion does not depend on the disputed point: heads-up, a compromise is behaviourally identical to a Seen-vs-Seen show. |
| **D3** | Show tie | **Requester loses** | Some house rules **split the pot** on any tie | Both sources (pagat Teen Patti and brag) agree on requester-loses; we follow them. We use split **only** for cap-forced shows, where there is no requester. |
| **D4** | Reveal order at a show | **Simultaneous** | Various table conventions have the non-requester expose first | No game-theoretic content — the show is terminal (§10.4). Behaviourally equivalent. |
| **D5** | Blind bet range | **`{s, 2s}`** | Real game permits **any** amount in `[s, 2s]`; some houses fix it at exactly `s` | Ours is the endpoint discretisation of the real interval (§8.4). |
| **D6** | Seen bet range | **`{2s, 4s}`** | Real game permits any amount in `[2s, 4s]` | As D5. |
| **D7** | Betting cap | **`k ≤ 8`, `r ≤ 2`** | The real game is **uncapped**; betting may continue indefinitely | Purely an artefact of finiteness (§9.1). This is the largest deviation from real play and must be disclosed as such. |
| **D8** | Cap resolution | **Forced show, no fee, ties split** | No real-world analogue, since the real game is uncapped | Chosen for strategic neutrality (§9.5). |
| **D9** | Boot | **1 unit, both players, symmetric ante** | Some tables use a dealer-posted boot or a rotating single boot | Symmetric posting isolates action order as the sole positional asymmetry (§3.1). |
| **D10** | Stack depth | **50 units, non-binding, reset each hand** | Real play has persistent, unequal, binding stacks | Deliberate: removes stack depth as a state variable (§3.2). |
| **D11** | `A-2-3` ranking | **Highest run** | Universally agreed in both Teen Patti and brag | No real divergence; recorded for completeness. |
| **D12** | Suit ranking | **None** | Some unrelated three-card games rank suits to break ties | Both sources state there is no order of suits; this is what makes exact ties possible (§4.4). |

---

## 18. Sources

Hand rankings, the blind/seen stake conventions, the stake-update rule, and the show rules were
verified against:

1. **Teen Patti**, *pagat.com* — <https://www.pagat.com/vying/teen_patti.html>
   Used for: the boot; the Blind bet interval `[s, 2s]` and Seen interval `[2s, 4s]`; the
   stake-update rule ("the current stake for the next player is then the amount that you put
   in" / "becomes half the amount that you bet"); initial stake of one unit; the show cost of
   `s` for a Blind requester "irrespective of whether the other player is blind or seen" and
   `2s` when both are Seen; the prohibition "if you are a seen player and the other player is
   blind, you are not allowed to demand a show"; the tie rule "the player who did not pay for
   the show wins the pot"; the blind→seen conversion timing ("you may choose to look at your
   cards when your turn comes to bet ... from that turn onwards you must bet at least twice the
   current stake"); `A-A-A` highest and `2-2-2` lowest trail; per-category tie-breaking; the
   compromise/sideshow rule.

2. **Brag**, *pagat.com* — <https://www.pagat.com/vying/brag.html>
   Used for: the run order "`A-2-3` is the highest run or running flush, `A-K-Q` of a suit is
   the second highest, then `K-Q-J`, and so on down to `4-3-2`, which is the lowest. **`2-A-K`
   is not a valid run or running flush**"; "there is no order of suits"; the tie rule "the
   calling player ... loses"; "**you cannot see a blind man**"; and the divergent convention
   that the prial of threes is the highest prial (D1).

Methodological precedent for capping and discretising the betting tree, and for the
solver-as-answer-key pipeline:

3. Zhuang et al., **PokerBench: Training Large Language Models to become Professional Poker
   Players**, AAAI 2025. arXiv:2501.08328.

Related work on Teen Patti specifically:

4. Banerjee, Maitra, De & Mukherjee, arXiv:2410.14363 — a statistical skill-versus-chance
   study, and the only prior academic treatment known to us. It is statistical rather than
   game-theoretic, and provides no solver, benchmark, or equilibrium analysis. *(Citation
   details to be completed from the arXiv record.)*

---

## 19. Change Control

This specification is **frozen** as of Version 1.0. Phases 1 and 2 are implemented against it.

Any change to §§3–11 alters the game and **invalidates all solver output computed under the
previous version**. Such changes require a version increment, a dated changelog entry, and a
full re-solve. §§12–13 (formal representation) may be clarified without a re-solve provided the
induced game is unchanged.

Implementations MUST record the specification version alongside every solved strategy and
exploitability figure.

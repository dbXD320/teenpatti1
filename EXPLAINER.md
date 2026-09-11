# Phases 1 and 2, explained plainly

*For a reader who knows how to play cards and nothing else. No game theory, no machine
learning, no poker-solver background assumed.*

---

## The short version

We took a card game nobody had solved, and solved it. First we rewrote the game in a form a
computer can reason about exhaustively — which meant deciding what a player *knows* at each
moment, and grouping the 22,100 possible hands into manageable categories without destroying the
answer. Then a program played the game against itself a thousand times over, improving each
round, until it reached a way of playing that essentially cannot be beaten. We can prove that: we
measured how much a perfect opponent shown our entire strategy in advance could still win, and
the answer is about three hundredths of one percent of the pot. That strategy is now an answer
key, and Phase 3 will use it to test whether AI language models can play the game well.

---

## What game, and why it's worth a paper

Teen Patti is an Indian three-card game. Two players, three cards each, betting back and forth,
best hand wins.

The interesting part is a mechanic no poker variant has. **You may bet without looking at your
cards.** A player who hasn't looked is *blind*; one who has is *seen*. Being blind buys two
things:

1. **Every bet costs half price.** A blind player pays 1 unit where a seen player pays 2.
2. **You alone can force a showdown.** A blind player can demand that cards be turned over.
   A seen player facing a blind player cannot. As the saying goes, *you cannot see a blind man*.

You may look at the start of any of your own turns. Once you look you are seen forever, and you
immediately start paying double.

So there is a decision here that no poker player faces: *how long do I stay ignorant, and what
is ignorance worth?* That is the research object. Almost nothing exists in the literature on
Teen Patti — no solver, no benchmark, no published analysis.

The plan follows a 2025 paper called PokerBench: solve the game, then use the solution as an
answer key to grade AI language models. Phases 1 and 2 are the "solve it" half. Phase 0 froze the
rules into a written specification first, so nothing could change underneath us later.

---

# Phase 1 — turning a card game into something solvable

## Problem 1: the game never ends

Two players can raise each other forever; nothing in Teen Patti forces betting to close. As a
game tree — every possible sequence of moves — it is infinite, and you cannot solve an infinite
thing by enumeration.

**What we decided:** cap it — at most 8 betting actions per hand (four turns each) and at most 2
raises. Looking at your cards doesn't count toward the limit.

**Why this is uncomfortable:** the cap is not a rule of Teen Patti. We invented it. Anything we
conclude is a conclusion about *capped* Teen Patti, and we have to say so.

**What would have gone wrong otherwise:** with no cap, no computation at all. With a cap set too
low, something worse — a plausible-looking answer to the wrong question. A cap of 4 actions gives
each player two turns. A player who stays blind on turn 1 and looks on turn 2 then has no turns
left to *use* the discount or the show right. The blind/seen decision, the entire thing we are
studying, collapses into triviality.

**So the cap has to be checked against the specific thing being measured.** Phase 2 did this by
re-solving at caps of 2, 4, 6 and 8 and watching whether the answer moved. It did — see the
findings. That check is what makes the cap honest rather than convenient.

## Problem 2: what does a player actually know?

Here is the idea the whole project rests on, built from the ground up.

You hold the seven of clubs and the two of hearts. Your opponent has just bet. You don't know
their cards. You might be in a world where they hold two aces, or one where they hold a five and
a nine. **Those are different situations, but you cannot tell them apart.** Whatever you decide,
you must do the same thing in all of them — you have no way of knowing which one you're in.

That bundle of situations-you-can't-distinguish is an **information set**. It is the right unit
for reasoning about a card game: not "the true state of the world", which no player observes, but
"everything consistent with what I have actually seen".

A rule follows immediately, and it is the rule that matters: **a strategy must give the same
answer everywhere inside one information set.** If your plan says "bet here, fold there" for two
situations you cannot tell apart, that plan is not playable. You'd have to know which one you
were in, and you don't.

## Problem 3: the blind player is strange

Now apply that to a blind player.

A seen player's information set bundles all the hands their *opponent* might hold. They know
their own three cards exactly. Standard.

A blind player hasn't looked, so situations differing *in their own hand* are indistinguishable
to them too. Holding three aces, or holding a seven-two-four, with the same public history, is
the *same situation from where they stand*. They must play them identically, because nothing
they have observed differs.

This is genuinely unusual. In poker you always know your own hole cards. Here a player's
uncertainty about *their own hand* is real and does work. A stranger consequence follows: when
the opponent bets strongly, that says something about the opponent's cards, which — because both
hands come from one deck — shifts what the blind player should believe about *their own* cards,
slightly and in the opposite direction. Teen Patti may be the only common game where a player
does real detective work about what they themselves hold.

Concretely, in our capped game:

| | number of distinct decision situations |
|---|---:|
| Blind player | **728** |
| Seen player | **2,256,930** |

The blind number doesn't depend on the cards at all — 728 is just the number of public betting
positions a blind player can be in. The seen number is the same kind of count multiplied by the
number of hand categories, because a seen player's decision legitimately depends on what they
hold.

## Problem 4: the peeking trap

**This is the most important idea in the project. Everything else is craft; this one is the
difference between a result and a worthless number.**

Suppose you write the program slightly wrong: you store the blind player's strategy per-hand,
the way the seen player's is. Nothing about that looks broken. The data structure is just bigger.

What happens is the program learns a strategy like:

> *"While blind, bet enormously — but only when I'm secretly holding three aces."*

That is not playable by a human. A real blind player has not looked — they cannot condition on
holding three aces, because they don't know. But the *program* knows: the deal is sitting in
memory, and if you indexed the strategy by the hand, the program will happily use it.

Here is what makes it a trap rather than a bug:

- **It does not crash.**
- **It does not warn you.**
- **It converges beautifully** — smoothly, and to a better-looking number than the correct
  version, because a cheating player really can do better.
- **It produces a confident, precise, completely worthless answer.**

Every diagnostic you would normally trust says things are going well. The exploitability curve
slides down. The numbers are stable to six figures. And the strategy is a fantasy: it describes a
player who peeks and pretends they didn't.

Worse, it corrupts the headline result specifically. The paper is about whether staying blind is
worth it. A program that secretly peeks concludes staying blind is *wonderful* — of course it
does, it gets the discount *and* the information. The one question we're asking is the one this
bug answers wrongly and confidently.

**What we decided:** don't rely on being careful. Make the mistake structurally impossible, four
independent ways:

1. The blind player's decision-situation type **has no field for cards**. Nowhere to put them.
2. Python normally lets you bolt a field onto an object at runtime. That's switched off here
   (`__slots__`), so you can't add one later either.
3. The object holding the deal **refuses to hand over cards** to a player who hasn't looked — it
   raises an error rather than answering.
4. A check walks **all 728 blind decision points against three different deals** and asserts the
   blind player sees identical information in all three. Any leak would show up as a difference.

And in Phase 2, two more: the blind player's regret tables are allocated with **no card
dimension at all** — there is no axis to index — and an assertion fires during every single pass
through the game if a blind player's position ever becomes hand-dependent.

The same discipline applies to the opponent used for measuring exploitability: at a blind
decision it must pick *one* action for the whole bundle. Letting it choose per-hand would make
our strategy look *worse* than it is — silently, in the other direction.

Four mechanisms for one rule may look paranoid. It's proportionate: the failure mode is
invisible, and it falsifies exactly the claim the paper exists to make.

## Problem 5: too many hands, and the lie we tell to cope

There are 22,100 possible three-card hands. Multiply that by every betting position and the
bookkeeping explodes.

The standard response is **grouping**: treat similar hands as one.

**Grouping is a lie you tell the solver.** You tell it two hands are the same situation. If they
really are interchangeable, the lie is free — same answer, less work. If they aren't, you've
forced one strategy onto two situations that deserved different ones, and the answer is wrong in
a way the solver cannot detect. It converges confidently on whatever grouping you defined.

So the only question that matters is: **are these hands genuinely interchangeable?**

## Problem 6: two different meanings of "the same hand"

This is where it gets subtle, and it's the nicest idea in Phase 1.

Take two hands:

> **K♠ K♥ 7♦**  and  **K♠ K♥ 7♠**

Both are a pair of kings with a seven. In terms of **who beats whom**, they are identical.
Any hand that beats one beats the other. Any hand that loses to one loses to the other. By
strength, they are the same hand.

But they are **not the same situation**, and the reason is the deck.

Your three cards are gone from the deck — they cannot be in your opponent's hand. So *which*
cards you hold changes what your opponent can possibly have. This is **card removal**.

The second hand holds **two spades**; the first holds one spade, one heart, one diamond. That
changes which flushes your opponent can make — and in the opposite direction to what most people
guess:

| your hand | suits you use | opponent flushes still possible |
|---|---|---:|
| K♠ K♥ 7♦ | one spade, one heart, one diamond | **946** |
| K♠ K♥ 7♠ | two spades, one heart | **957** |

Holding *two* spades leaves **more** opponent flushes alive, not fewer. The intuition that
"using up spades blocks spade flushes" is real, but outweighed: concentrating your removals in
one suit leaves the other three *fuller*, and the count of three-card flushes grows faster than
linearly as a suit fills up. Eleven extra flushes survive — and that shows up in the results:

| hand | wins | ties | losses | equity |
|---|---:|---:|---:|---:|
| K♠ K♥ 7♦ | 16,521 | 3 | 1,900 | 0.89679223 |
| K♠ K♥ 7♠ | 16,510 | 3 | 1,911 | 0.89619518 |

Eleven more losses for the second hand. Exactly the eleven extra flushes. Same ranks, so every
other count is identical — the whole difference *is* card removal.

> **A flagged correction.** The frozen spec (`RULES.md` §5.3) uses this example but says K♠K♥7♠
> "removes a third spade from the deck", implying it should block flushes and do *better*. It
> holds **two** spades, and it does *worse*. **The spec's conclusion is right; its stated reason
> is inverted.** Phase 1 verified this both ways and recommends correcting the parenthetical. I
> re-ran both hands against the code to confirm the numbers above.

**The conclusion for the solver:** there are two notions of "same hand" and they are not
interchangeable. Grouping by **strength** (741 categories) is exactly right for deciding who
takes the pot, and **wrong** as a grouping for the solver, because it merges hands that face
different opponents. We measured this: **442 of the 741 strength classes — 60% — contain hands
with genuinely different chances of winning.**

We grouped by **suit symmetry** instead: two hands are one group if renaming suits turns one into
the other. That is a *safe* lie, because the game doesn't care what suits are called, only which
cards coincide. It collapses 22,100 hands into **1,755 groups**, every hand inside a group having
identical win, tie and loss counts. Counted two independent ways — direct canonicalisation, and a
classical counting argument called Burnside's lemma — both giving 1,755.

So: 22,100 down to 1,755, with **nothing lost**. That is what made this tractable, and why we
solved the real game rather than an approximation of it.

The same work turned up a related trap for Phase 3. **A stronger hand can have worse odds.**
2♣2♦3♥ beats A♠K♥J♦ *every time the two meet*, yet does worse against a random opponent — the
ace-king-jack strips three high cards from the deck, blocking the opponent's good hands, while
the deuces block nothing. So a benchmark question like "which hand is stronger?" must be answered
from the strength ranking, never from the odds, or the answer key is wrong at 208 boundaries.

---

# Phase 2 — actually solving it

## What "solving" even means when luck is involved

It cannot mean "the right move in each position", because the right move depends on cards you
can't see. There is no move that's correct against every possible opponent hand.

What it means instead is **a way of playing** — a complete policy, covering every situation you
could find yourself in, saying what to do. And crucially, the answer is often **deliberately
random**: "in this spot, raise 70% of the time and fold 30%".

Randomness isn't hedging. It's necessary. Any fixed, predictable rule can be learned and
exploited. If you always raise strong hands and fold weak ones, your opponent reads you
perfectly. Mixing makes you unreadable. The solution to a card game is a recipe that includes its
own dice rolls.

The particular way of playing we're after is one where **neither player can improve by
unilaterally changing their plan**. Both are already doing the best they can given what the
other is doing. Nobody has a reason to move.

## What the solver does

The method is called CFR — counterfactual regret minimisation. The intuition is simple and you
can carry it in your head:

> **Play the game against yourself, over and over. After each pass, look back at every decision
> and ask: "what would I have earned if I'd played differently there?" Whatever you wish you had
> done, lean a little further toward next time.**

That accumulated "I wish I'd done that" is the regret. Actions you regret not taking get more
weight; actions that would have gone badly get less. Repeat.

That's it. No opponent model, no training data, no neural network, no human games. The program
starts out playing at random and bootstraps itself by regretting.

Two standard choices: regrets are clipped so they never go negative, and the players update in
alternation. Both speed up convergence; nothing conceptual rides on either. We ran 1,000 passes,
each a complete walk through the game — just under an hour.

## The number that proves it worked

Here's the trap. A program playing itself and gradually agreeing with itself is *exactly* what
you'd expect to see if it were subtly broken. Self-consistency proves nothing. Two wrong players
can agree perfectly.

So we measure something else entirely: **exploitability.**

> Hand your complete strategy to an opponent. Let them study it — every situation, every
> probability, the whole thing, in advance. Now let them design the single best counter-strategy
> against it. **How much do they win?**

That's exploitability. It is a worst-case number, and it is not self-referential — a separate
procedure goes looking for your weaknesses. A big number means your strategy has a hole in it.
**Zero means unbeatable**: nobody profits from you even knowing everything you will do.

**Ours reached 0.0006 units per hand — about 0.031% of the starting pot.** A perfect adversary
holding our complete playbook wins essentially nothing. That is what makes this a solution rather
than a program's opinion. It fell smoothly the whole way — 4.7% of the pot at 50 passes, 0.57% at
200, 0.031% at 1,000 — with no plateau and no suspicious collapse.

## Why the average, and not the final answer

The strategy we report is not the last one the program produced. It is the **average of every
strategy it tried along the way.**

The guarantee attaches to the average, not the latest version. Theory says the average converges
to unbeatable play, with its exploitability bounded by accumulated regret. The final version has
no such promise — it might be good, or it might have swung somewhere odd on the last pass.

We logged both, and they are genuinely different (a test asserts they haven't become the same
object). The final version actually scored *better* late in the run — 0.00032 against 0.00061.
We still report the average, because "did better this time" is not a guarantee, and having a
guarantee is the entire point.

## How we know it isn't cheating

Two checks, and they're independent of each other.

**Check one: count the blind player's decisions.**

This is the tell for the peeking trap, and it is almost embarrassingly simple. Count how many
distinct betting situations each player actually encounters under the equilibrium:

| | betting situations reached |
|---|---:|
| Blind player | **235** |
| Seen player | **913,825** |

That ratio is the signature of having done it right. The blind player's count is tiny because it
*cannot* depend on cards — there are only so many public betting positions, and that's the whole
count. **Had the program been peeking, this number would be in the hundreds of thousands too**,
because it would have had a separate entry for every hand. You can diagnose the single most
dangerous bug in the project by reading one number and seeing that it's small.

**Check two: solve a stripped-down version by plain arithmetic.**

We built a variant where the betting cap is 1, so the second player never acts. That removes the
game theory entirely — it's now a decision problem you can compute with ordinary arithmetic: for
each hand count how often it wins, work out what looking is worth, take the better option. No
solver, no equilibrium, no iteration. Then we pointed the full CFR solver at the same variant.

| | value |
|---|---:|
| Computed directly by arithmetic | **−0.49077534** |
| CFR solver's answer | **−0.49077730** |
| Difference | **0.00000196** |

They agree to about two millionths. This tests the payoff calculations, the blind aggregation,
the per-hand play, the show-rule legality and the exploitability machinery at once, against an
answer that owes nothing to the solver.

> **A small precision note.** The brief for this document described this as matching "to six
> decimal places". Strictly the two agree through the fifth decimal and differ in the sixth
> (−0.490775 vs −0.490777). The honest phrasing is "agree to within two millionths". Worth
> getting right if you're quoting it out loud.

A third check: at a true equilibrium, what each player can win by playing perfectly should equal
the value of the game. Measured, the two sides come out at −0.0394 and +0.0407 against a game
value of −0.0398 — off by about the exploitability itself, exactly as they should be.

---

# What the solver found

Kept short — there's a fuller write-up in [RESULTS_PHASE2.md](RESULTS_PHASE2.md).

**Position decides everything.** The first player stays blind **99.997%** of the time. The second
player looks **99.5%** of the time. Neither decision is close, neither is mixed. Whether you stay
blind is essentially determined by which seat you're in.

**Staying blind is not gambling — it's postponing a decision cheaply.** You can always look
later, and looking now doubles the price of the very bet you were about to make. Staying blind on
turn 1 costs almost nothing and keeps the option alive. The first player's looking is concentrated
at their *second* turn, right after the opponent has acted and revealed something. A deferral,
not a gamble.

**The blind player's exclusive show right is almost never used** — about 1 time in 100,000. This
contradicts the frozen spec, which predicted the right would be central and warned that a
strategy not using it is "almost certainly an implementation bug". There's a clean reason: the
hands that most want to force a showdown are the hands most worth looking at first, and looking
forfeits the right. At equilibrium *seen* players demand showdowns 10–100× more often. The
report argues this is a real finding, not a bug, and recommends softening the prediction rather
than changing any rule.

**Stack size is irrelevant above a threshold.** The most anyone can put in during one hand is 33
units, so a stack of 34 and a stack of 100 are *literally the same game*. Verified rather than
asserted: stacks of 34, 40, 50, 75 and 100 give results identical to all 16 digits.

**The blind/seen split only appears when hands run long enough.** This is what justifies having
checked the cap. At caps of 2 and 4 actions, equilibrium has **nobody looking at all** — both
players stay blind the whole hand. The split appears only once the cap reaches 6. So "first
player blind, second sighted" is not a fixed property of Teen Patti; it *emerges* once there is
enough betting room for information to be worth paying double for.

---

# Things I should be able to explain if asked

- **An information set is a bundle of situations a player cannot tell apart, and a strategy has
  to give the same answer for every situation in the bundle** — because if it didn't, you'd need
  to know which one you were in, and you don't.

- **A blind player hasn't looked, so situations differing only in their own cards are one
  information set** — which is why they have 728 decision situations where a seen player has over
  two million.

- **The peeking trap: if the program lets a blind player's strategy depend on their actual cards,
  it doesn't crash or warn — it converges beautifully to a confident, worthless answer** that
  describes a player who peeks and pretends they didn't, and it corrupts exactly the question the
  paper is asking.

- **Grouping hands is a lie told to the solver — free if the hands are truly interchangeable,
  silently damaging if not** — so we grouped by suit symmetry (22,100 → 1,755, provably lossless)
  and never by strength, because 60% of strength classes mix hands with different real odds.

- **K♠K♥7♦ and K♠K♥7♠ are the same strength but not the same situation**, because the cards you
  hold are cards your opponent cannot have — and concentrating two spades in your hand leaves
  *more* opponent flushes alive, not fewer, costing exactly 11 hands.

- **Exploitability is how much an opponent could win if you handed them your entire strategy in
  advance; zero means unbeatable, and ours is 0.031% of the pot** — which is the number that
  makes this a solution rather than a program agreeing with itself.

- **We report the average of every strategy tried, not the final one**, because the mathematical
  guarantee attaches to the average and the final iterate has no such promise.

- **The blind player has 235 betting situations against the seen player's 913,825, and that gap
  is the proof we weren't peeking** — a cheating implementation would have had hundreds of
  thousands on both sides.

---

# Accuracy notes

Checked against `RULES.md`, `PHASE1.md`, `PHASE2.md` and `RESULTS_PHASE2.md`; the card examples
were re-run against the code while writing. Three things I did not smooth over:

1. **The card-removal direction is inverted in `RULES.md` §5.3**, and in the brief for this
   document. K♠K♥7♠ holds two spades, not three, and performs *worse* than K♠K♥7♦, not better.
   Phase 1 caught this and recommends amending the parenthetical; the conclusion is unaffected.

2. **"Matched to six decimal places" is slightly generous.** The agreement is within
   1.96 × 10⁻⁶ — five decimal places, differing in the sixth.

3. **A discrepancy in `PHASE1.md` §7 I could not resolve.** Its table gives 2♣2♦3♥ an equity of
   0.7403; the code gives 0.74080. The A♠K♥J♦ figure (0.7428) matches. The finding holds either
   way, since 0.7408 is still below 0.7428 — but I don't know which number is the typo.

One labelling note: 728 / 2,256,930 are the *structural* totals from Phase 1 — every decision
situation that exists. 235 / 913,825 are those *actually reached* under the equilibrium, from the
Phase 2 dominance analysis. Both pairs are real and measure slightly different things.

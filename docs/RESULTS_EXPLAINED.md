# TeenPattiBench Results, Explained in Plain English

**Who this is for.** Anyone who wants to understand what we measured and what we
found, without needing game theory, machine learning, or Teen Patti expertise.
Every term is defined the first time it appears. No prior reading required.

**Scored:** 14 September 2026. Three language models, 2,000 questions each.
**Companion documents:** [EXPLAINER.md](EXPLAINER.md) explains how the answer key
was built; [PHASE4.md](PHASE4.md) is the technical report on the first model.

---

## The one-paragraph version

We solved a card game exactly — meaning we computed, with mathematical proof, the
best possible way to play it. We then turned that solution into a 2,000-question
exam and gave the exam to three AI language models. All three scored close to what
you would get by ignoring the questions entirely and repeating a single word 2,000
times. On the part of the exam we most cared about — deciding *when to look at your
own cards* — all three did **worse than that**, and they did so for an identical
reason: every one of them answered "look" to essentially every question, so their
score is not a measure of their judgement at all. It is a measure of how the right
answers happen to be distributed.

---

## 1. The game, in sixty seconds

**Teen Patti** is a three-card betting game played widely in India. Two players
each put 1 unit into the pot (the **boot**, a compulsory ante), are dealt three
cards each, and then take turns betting until someone gives up or the hands are
compared.

The rule that makes this game scientifically interesting has no equivalent in
poker:

> **You are allowed to bet without looking at your own cards, and if you do, you
> pay half price.**

A player who has not looked is **blind**. A player who has looked is **seen**. You
may choose to look at the start of any of your own turns, but the moment you do:

- your costs double, permanently, for the rest of the hand;
- you lose a special right that only blind players have (explained below);
- you can never go back to being blind.

So the game offers you a genuine trade: **information costs money, and ignorance
comes with a discount**. Nobody had ever worked out the correct way to make that
trade. That is what this project did.

**The other blind-only right.** A blind player may demand an immediate showdown
(called a **show**) at any time. A seen player is *forbidden* from demanding one
against a blind opponent. In the old English version of the game the saying is
"you cannot see a blind man." So staying blind buys you two things: a 50% discount,
and sole control over when the hand ends.

### The five things you can do

| Action | What it means | Everyday equivalent |
|---|---|---|
| **chaal** | Match the current bet and continue | "call" |
| **raise** | Double the stake | "raise" |
| **pack** | Give up the hand and lose what you have already put in | "fold" |
| **show** | Pay a fee to force both hands to be compared right now | "call the showdown" |
| **see** | Look at your own cards (only at the start of your turn, permanent) | *no poker equivalent* |
| **stay-blind** | Deliberately decline to look, and keep the discount | *no poker equivalent* |

The last two are the research subject. Nothing in poker, bridge, or any other
studied card game asks a player to *choose to remain ignorant*.

---

## 2. Where the "right answers" come from

This is the part people are most sceptical about, so it is worth being precise.

**Nobody hand-wrote the answers.** There is no panel of expert players. The right
answer to every question was **computed**, by a program that walked through *every
situation the game can reach*, measured how much it regretted each choice it had
made, adjusted, and repeated — 1,000 times over. The technique is called **CFR+**
(Counterfactual Regret Minimisation), the same family of method used to solve
heads-up poker. The computation took 55 minutes.

Worth stressing: **nothing here is simulated or sampled.** The program does not
play random hands and average the results. It accounts for all 407 million possible
deals exactly, by arithmetic, on every one of its 1,000 passes.

### How do we know the computed answer is actually right?

By measuring a quantity called **exploitability**.

> **Exploitability** = how much money a perfect opponent could win from you, per
> hand, if you handed them your entire strategy in advance and let them design the
> ideal counter-attack.
>
> A strategy that cannot be beaten at all has exploitability **zero**.

Our solution's exploitability is **0.0006 units per hand — 0.031% of the pot.**

In plain terms: even an opponent who knows our complete strategy, has unlimited
computing power, and plays flawlessly against it, would win about three-hundredths
of one percent of a pot per hand. That is the mathematical guarantee that our
answer key is right. A human expert's answer key comes with no such guarantee, and
no way to check one.

This number is not estimated by simulation either. It is computed exactly, by
examining every one of the 2,014 decision points in the game's 5,929-node tree.

### Where the answers physically live

Every question in the exam file carries its own answer. Here is a real one:

```
id                  : eval-000014
legal_actions       : ['see', 'stay-blind']
correct_action      : stay-blind        <-- this is what gets graded
correct_amount      : 0
solver_distribution : {'see': 0.271, 'stay-blind': 0.729}
```

The last line is the solver's full opinion: it says that in this exact situation,
the optimal play is to decline to look about 73% of the time and look about 27% of
the time. We grade against the more likely option (**stay-blind**), but we keep the
whole distribution in the file so nothing is hidden.

**We re-verified this.** Every one of the 1,800 graded answers was recomputed from
the original solver file in under a second, and all 1,800 matched. The grading
program was also tested by having it grade the solver against its own answer key —
it scored 100.00%, as it must.

---

## 3. The exam

**2,000 questions.** Each one describes a real situation from the game in plain
English — pot size, betting history, whose turn it is, what each action costs, and
(if you are a seen player) what cards you hold. It ends with the instruction
*"Do not explain your answer. Your optimal action is:"*.

The 2,000 split into four groups:

| Group | Count | The question being asked |
|---|---:|---|
| **Seen betting** | 1,500 | You have looked at your cards. Bet, fold, raise, or force a showdown? |
| **Conversion ("look")** | 162 | You are blind. Do you look at your cards now, or stay blind and keep the discount? |
| **Blind betting** | 138 | You are blind and staying blind. What is your bet? |
| **Mixed** | 200 | Situations where even the solver has no single best answer — graded differently (see §4) |

Only the first three (**1,800 questions**) count toward the headline score.

### An important structural fact

There are only **428 blind situations in the entire game**. Not 428 in our sample —
428 in existence. This is because a blind player, by definition, does not know
their own cards, so their situation is described entirely by the public betting
history. A seen player's situation is described by the history *and* their hand,
which multiplies the possibilities into the hundreds of thousands.

So the blind questions are not a random sample. They are an **exhaustive
enumeration** — we took 300 of the 428 that exist and gave the remaining 128 to the
training set. There is no sampling error on the thing we care most about.

---

## 4. Every metric, defined

### Action Accuracy (AA) — the headline number

> **The percentage of questions where the model chose the same action the solver
> did.**

Simple as that. 1,800 graded questions; how many did you get right.

Two deliberate strictnesses:

- If a model produces something that cannot be read as a legal action, it counts as
  **wrong**, not as "skipped". Silently dropping unreadable answers would flatter a
  model that refuses to follow instructions.
- If a model fails to answer a question at all, the grader **stops with an error**
  rather than scoring that question zero. A prediction file that is quietly missing
  500 answers would otherwise report a plausible-looking score.

The grader is tolerant about *format*, though. "chaal", "CHAAL", and "I would chaal
here." all count as the same answer. This was tested: prose-wrapped answers parse
at 100%.

### Exact Match (EM) — did you also get the amount right?

> **The percentage of questions where the model chose the right action *and* the
> right amount of money to put in.**

EM can never exceed AA. **In these three runs, EM is meaningless** — the models
were only asked for an action word and never supplied an amount, so the grader has
nothing to check and EM collapses to exactly AA. Wherever you see EM equal to AA in
this document, it is not a second, corroborating measurement. It is the same number
printed twice.

### Total Variation Distance (TVD) — for the questions with no single right answer

For 200 of the 2,000 questions, the solver genuinely mixes: it might say "raise 40%
of the time, chaal 35%, show 25%". There is no single correct answer to grade
against, so scoring them with Action Accuracy would be dishonest.

> **TVD measures how far the model's *probabilities* are from the solver's
> probabilities.** 0 means identical. 1 means completely unrelated.

**Not measured in these runs.** TVD requires the model to output probabilities, and
these runs only output a single word. All 200 mixed questions are excluded.

### Unreadable answers

Diagnostic only, not a score: how many completions could not be parsed into any
legal action. **All three models: zero.** Every answer was a legal action for its
own question.

### The baseline — the most important idea in this document

> **A baseline is the score you would get from a policy so stupid it does not read
> the question. The simplest one: pick one word and answer it every single time.**

This matters enormously. Suppose 60% of the right answers on some part of the exam
happen to be "chaal". Then a model that answers "chaal" to everything scores 60% —
and it knows nothing whatsoever. A score is only meaningful *compared to the best
score a parrot could get on the same questions*.

The best parrot on the full 1,800 graded questions scores **25.94%** (always
answering "chaal"). We deliberately designed the exam to make parrots fail: the
answer options were balanced so that no single-word policy does well. For
comparison, PokerBench — the poker benchmark this project is modelled on — found
that "fold every time" scored close to **90%** on their first draft before they
rebalanced it.

**But the overall baseline is the wrong yardstick for individual sections.** Each
section of the exam has its own mix of right answers, so each has its own parrot
score:

| Section of the exam | Best parrot | Which word |
|---|---:|---|
| All 1,800 graded | 25.94% | "chaal" |
| Seen betting (1,500) | 25.00% | "chaal" |
| **Conversion (162)** | **62.96%** | **"stay-blind"** |
| **Blind betting (138)** | **66.67%** | **"chaal"** |

Notice that the conversion and blind-betting sections have *very high* parrot
scores. Any result on those sections must be compared against 62.96% and 66.67%,
not against 25.94%. Failing to do this is the single easiest way to misread these
results — and it turns a catastrophic failure into what looks like a success.

---

## 5. What was tested

Three open-weight language models, all run under **identical conditions**:

| | Qwen3-8B | Qwen3-4B | Llama-3.2-3B |
|---|---|---|---|
| Parameters | 8.19 billion | 4.02 billion | 3.21 billion |
| Numerical precision | float16 | float16 | float16 |
| Compressed (quantised)? | No | No | No |
| Decoding | Greedy (deterministic) | Greedy | Greedy |
| Max answer length | 16 tokens | 16 tokens | 16 tokens |
| "Thinking" mode | Off | Off | Off |
| Hardware | Tesla T4 | Tesla T4 | Tesla T4 |

All three received the identical system prompt (verified byte-for-byte against the
repository's copy), answered all 2,000 questions, produced no duplicates, and gave
a legal action for every single question.

*"Greedy" means the model always picks its single most likely next word rather than
sampling randomly — so these runs are reproducible, not lucky or unlucky.*

---

## 6. The headline results

| | Qwen3-8B | Qwen3-4B | Llama-3.2-3B | Best parrot |
|---|---:|---:|---:|---:|
| **Action Accuracy** | 32.94% | **33.11%** | 28.06% | 25.94% |
| Exact Match | 32.94%\* | 33.11%\* | 28.06%\* | — |
| Unreadable answers | 0 | 0 | 0 | — |
| **Margin over parrot** | **+7.0** | **+7.2** | **+2.1** | — |

\* *Identical to AA because no amounts were supplied — not a second measurement.*

Two immediate observations.

**Nobody is doing well.** The best result is 33.11%, which is 7.2 percentage points
above a policy of typing one word 1,800 times. Two out of three answers are wrong.

**Size barely helps, and is not monotone.** Qwen3-4B (4 billion parameters) very
slightly *outscored* Qwen3-8B (8 billion). A 0.17-point gap on 1,800 questions is
well within noise — the honest reading is that **doubling the model changed
nothing**. Llama-3.2-3B is meaningfully worse, but it differs in training data and
model family, not just size, so this is not a clean size comparison either.

---

## 7. Each section against its own parrot

This is the table that actually matters.

| Section | n | Parrot | Qwen3-8B | Qwen3-4B | Llama-3.2-3B |
|---|---:|---:|---:|---:|---:|
| All graded | 1,800 | 25.94% | 32.94% **(+7.0)** | 33.11% **(+7.2)** | 28.06% **(+2.1)** |
| Seen betting | 1,500 | 25.00% | 30.40% **(+5.4)** | 31.47% **(+6.5)** | 25.33% **(+0.3)** |
| Conversion | 162 | 62.96% | 37.65% **(−25.3)** | 37.04% **(−25.9)** | 37.65% **(−25.3)** |
| Blind betting | 138 | 66.67% | 55.07% **(−11.6)** | 46.38% **(−20.3)** | 46.38% **(−20.3)** |

**Every model is far below the parrot on both blind sections.** On conversion — the
question this entire project was built to ask — all three are roughly 25 points
worse than answering a single word forever.

Llama-3.2-3B on seen betting is +0.3 points over the parrot. On 1,500 questions
that is indistinguishable from having learned nothing at all about betting.

> **A warning about combining sections.** If you merge the two blind sections into
> one "blind" number, you get 45.67% against a combined parrot of 34.00%, which
> looks like a **+11.7 success**. It is an artefact of merging two groups with
> different answer mixes. Both halves are failures. This combined figure should
> never be quoted on its own.

---

## 8. The central finding: the models are not deciding anything

Here is the conversion section broken out by *which of your turns it is* — turn 1 is
your first opportunity to look at your cards, turn 4 is your last.

| Your turn | Questions | Right answer is "see" | Qwen3-8B says "see" | Qwen3-4B says "see" | Llama says "see" |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 75.0% | 50.0% | 100% | 100% |
| 2 | 19 | 68.4% | **100%** | **100%** | **100%** |
| 3 | 53 | 50.9% | **100%** | **100%** | **100%** |
| 4 | 86 | 19.8% | **98.8%** | **100%** | **98.8%** |

And here is what that produces as a score:

| Your turn | Right answer is "see" | Qwen3-8B score | Qwen3-4B score | Llama score |
|---:|---:|---:|---:|---:|
| 1 | 75.0% | 75.0% | 75.0% | 75.0% |
| 2 | 68.4% | 68.4% | 68.4% | 68.4% |
| 3 | 50.9% | 50.9% | 50.9% | 50.9% |
| 4 | 19.8% | 20.9% | 19.8% | 20.9% |

**Look at those two columns.** The score is the same number as the frequency of the
right answer, to the decimal, for every model at every turn.

That is not a coincidence and it is not partial skill. It is the arithmetic
signature of a model that always gives the same answer: if you answer "see" to
every question, you score exactly as often as "see" happens to be correct. All
three models answered "see" to essentially every conversion question — 159, 162 and
161 times out of 162.

**Consequences to be clear about:**

1. The apparent decline from 68% at turn 2 down to 21% at turn 4 is **not the model
   getting worse**. The model never changes. It is the *correct answer* shifting
   from "see" to "stay-blind" while the model stands still.
2. Any single combined conversion number — 37.65% — would look like partial
   competence. It is nothing of the kind. This is exactly why the grading program
   refuses to compute one.
3. Turn 1 has only 4 questions because **only 5 exist in the whole game**. It is a
   control, not a test. Report it, do not interpret it.

### Why this specific failure was predicted

The solver's optimal conversion pattern across your four turns is:

> **don't look, look, don't look, look**

Probability of looking: 0.00003, then 0.79, then 0.06, then 0.93. It goes up, then
back down, then up again. Nothing in poker resembles this, so there is no text in
any model's training data describing it. The project's hypothesis was that models
would do relatively well on ordinary betting — which resembles poker, about which
enormous amounts has been written — and fail on conversion timing, where there is
nothing to copy from.

**The failure half of the prediction is confirmed, emphatically. The success half
is not.** The models are not "good at betting and bad at conversion." They are weak
everywhere and structurally absent on conversion.

---

## 9. The folding problem

**Pack** means fold — give up the hand. It is the correct answer on **375 of the
1,500 seen-betting questions (25.0%)**.

How often did each model say it?

| | Times "pack" was given, out of 2,000 |
|---|---:|
| Qwen3-8B | **0** |
| Qwen3-4B | **0** |
| Llama-3.2-3B | **1,307** |

The two Qwen models **never fold, once, in 2,000 opportunities**. That put a hard
ceiling of 75% on their seen-betting score before the exam began; they scored 30.4%
and 31.5%, so they achieved about 40% of what was even reachable.

Llama has the opposite pathology: it folds on 65% of all questions. To be fair to
it, every one of those folds was at least *legal* — it never offered "pack" where
pack was not on the menu.

Neither is a preference. A hard zero across 2,000 chances, or a 65% rate, are
structural — and the two have **different causes**, which §10a works out.

Full answer distributions across all 2,000 questions:

| | chaal | raise | show | pack | see | stay-blind |
|---|---:|---:|---:|---:|---:|---:|
| **Correct answers** | 580 | 396 | 421 | 441 | 60 | 102 |
| Qwen3-8B | 969 | 703 | 166 | **0** | 159 | 3 |
| Qwen3-4B | 903 | 908 | 27 | **0** | 162 | 0 |
| Llama-3.2-3B | 531 | 0 | 0 | **1,307** | 161 | 1 |

Llama never once said "raise" or "show". Its entire working vocabulary across 2,000
questions was four words — and one of those, "stay-blind", it used exactly once.

---

## 10. Are the models even reading their own cards?

Here is a clean test. Take the questions that share an identical betting situation
but deal the player different cards. Same pot, same history, same everything —
only the hand differs. A player who reads their cards should sometimes answer
differently. A player who ignores them will answer the same word every time.

Among situations with at least 6 different hands tested:

| | Situations answered identically regardless of the cards held |
|---|---:|
| **The solver** (correct play) | **11 / 74** (15%) |
| Qwen3-8B | **40 / 74** (54%) |
| Qwen3-4B | **65 / 74** (88%) |
| Llama-3.2-3B | **72 / 74** (97%) |

Llama gives the same answer whether it holds the best hand in the deck or the
worst, in 97% of situations. This explains its fold rate directly: it is not
folding because its cards are bad, it is folding regardless.

It goes part of the way toward explaining the Qwen models' zero folds too.
**Folding is the only action that requires knowing your hand is bad.** Every other
action — call, raise, force a showdown — has some defensible rationale without
looking. A model that does not really consult its cards will therefore produce
*no* folds rather than merely few, and forfeit the 25% of questions where folding
is right.

But that cannot be the whole story, because Llama folds constantly without reading
its cards either. The next section separates the two.

---


### A tempting conclusion, and why we checked it

Accuracy by hand strength (bucket 0 = weakest hands, 24 = strongest) appears to
show the Qwen models improving as their hand gets better — Qwen3-8B goes from 11%
on the weakest hands to 46% on the strongest. That looks like genuine card
sensitivity.

**It is not.** It is the folding bias in disguise. Folding is correct far more often
with weak hands (55.6% of the weakest bucket, 8.2% of the strongest), so a model
that never folds automatically scores badly on weak hands and better on strong ones
without reading anything.

To separate the two, we removed every question where folding was the right answer
and re-measured:

| Hand strength → | weakest | | | | strongest |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 44.0% | 32.3% | 40.3% | 39.7% | 44.6% |
| Qwen3-4B | 67.0% | 40.1% | 39.9% | 35.9% | 42.8% |
| Llama-3.2-3B | 16.5% | 16.2% | 11.8% | 12.5% | 10.8% |
| *(best parrot in each band)* | *54%* | *47%* | *43%* | *46%* | *45%* |

**The gradient vanishes.** With the folding effect removed, no model improves as its
hand gets stronger, and every model sits at or below the parrot score in every
band. The apparent card sensitivity was an artefact. The honest conclusion is that
none of these three models meaningfully uses the cards it was dealt.
## 10a. Why the two "never" results have different causes

Every question lists its legal actions in the same fixed order — `pack` first at
every betting question, `raise` and `show` never earlier than third:

```
   947 questions offer:  pack, chaal, raise, show
   383 questions offer:  pack, chaal, raise
   381 questions offer:  pack, chaal, show
   127 questions offer:  pack, chaal
   162 questions offer:  see, stay-blind        (the look questions)
```

### Llama: it never reads past the second item on the menu

| | Menu positions it ever selects, across 1,838 betting questions |
|---|---|
| Qwen3-8B | 1, 2, 3 |
| Qwen3-4B | 1, 2 |
| **Llama-3.2-3B** | **0, 1 — never 2 or 3** |

`raise` and `show` **only ever appear at position 2 or 3**. So Llama's "never
raises, never shows" is not an opinion about raising. It is that the model never
selects anything past the second item, and those two actions live nowhere else.

What it picks within the first two slots is set by the menu, not by the game:

| Menu offered | Questions | Llama folds |
|---|---:|---:|
| pack, chaal | 127 | **0.0%** |
| pack, chaal, show | 381 | 34.1% |
| pack, chaal, raise | 383 | 72.1% |
| pack, chaal, raise, show | 947 | **95.1%** |

And certainly not by its cards — it folds *slightly more often* holding the best
hands in the deck:

| Hand strength | Llama folds | Correct to fold |
|---|---:|---:|
| weakest | 67.7% | 59.2% |
| middle | 76.8% | 27.1% |
| strongest | 75.2% | 9.0% |

### Qwen: this one really is about the action, not its position

If the Qwen models were simply ignoring the first item on the list, they would
ignore it at the **look** questions too — where the first item is `see`, not
`pack`. They do the exact opposite:

| | Picks the first listed item at *look* questions | at *betting* questions |
|---|---:|---:|
| Qwen3-8B | 98.1% | **0.0%** |
| Qwen3-4B | 100.0% | **0.0%** |
| Llama-3.2-3B | 99.4% | 71.1% |

Same slot on the page, opposite behaviour. **Qwen is avoiding the act of folding,
not the position it happens to be printed in.** Why it does so is not settled by
this data; the card-blindness above is the leading explanation, with a general
reluctance to choose the "give up" option a plausible second.

### What this costs each model before it answers anything

| | Words it ever uses | Best possible score | Actual | Share of what was reachable |
|---|---|---:|---:|---:|
| Qwen3-8B | chaal, raise, show, see, stay-blind | 79.17% | 32.94% | 41.6% |
| Qwen3-4B | chaal, raise, show, see | 73.50% | 33.11% | 45.0% |
| Llama-3.2-3B | pack, chaal, see, stay-blind | **55.78%** | 28.06% | 50.3% |

**48.6% of betting questions have their correct answer at menu position 2 or
later** — unreachable for Llama by construction.

### An open problem with the exam itself

Every question in this dataset lists the actions in the same order. That was not a
deliberate choice; it is simply the order the program happens to produce them in.
For Llama it means part of its score measures **sensitivity to how the options were
laid out** rather than understanding of the game, and as the exam currently stands
we cannot tell those two apart.

The fix is small: shuffle the order of the listed actions from question to question
and re-run. If Llama's raise and show counts stay at zero, the effect is real and
the exam is measuring what it should. If they appear, part of the present Llama
result is an artefact and the whole dataset needs re-ordering before publication.
This should be treated as a required check, not an optional one.

---

---

## 11. Do the models agree with each other?

If three models were all converging on some genuine understanding of the game, they
should agree with each other more than chance allows.

| | Agreement |
|---|---:|
| Qwen3-8B vs Qwen3-4B | 65.20% |
| Qwen3-8B vs Llama-3.2-3B | **28.70%** |
| Qwen3-4B vs Llama-3.2-3B | **27.30%** |

The two Qwen models — same family, same training data, different sizes — agree
about two-thirds of the time. Either Qwen against Llama agrees barely more than
random guessing would. They are not converging on anything. They have different
arbitrary habits.

---

## 12. What can and cannot be concluded

### Supported by this data

- All three models perform close to trivial on this benchmark, with the best at
  +7.2 points over a one-word policy.
- All three fail the conversion question completely, and in an identical way: by
  giving one constant answer that is not a decision at all.
- Two of three never fold; the third folds almost reflexively. None demonstrably
  reads its own cards once the folding bias is controlled for.
- Model size, within this range, made no measurable difference.

### Not supported by this data

- **Nothing about large models.** The biggest model here is 8 billion parameters,
  small by current standards. These results say nothing about frontier systems.
- **Nothing about whether the models *could* do better if prompted differently.**
  These were single-shot answers with no reasoning step, capped at 16 tokens, with
  "thinking" mode off. Earlier exploratory work found that appending the single
  sentence *"Your hand is weak and you are very likely beaten"* to a prompt flipped
  a model to fold on 5 out of 5 questions where folding was correct. Handed the
  conclusion, it acts on it; asked to derive it from three named cards, it does not.
  Whether chain-of-thought reasoning closes this gap is untested.
- **Nothing about Exact Match or distribution accuracy**, neither of which was
  measured because the runs did not emit amounts or probabilities.

### Known limitations

1. **Turn 1 of the conversion section has 4 questions.** That is not a small
   sample by choice — only 5 exist in the entire game. Treat it as a control.
2. **The seat difference is unexplained.** Qwen3-8B scores 44.94% playing first and
   25.45% playing second; Llama shows the reverse. Player 2's situations sit deeper
   in the betting tree, so this is confounded with question difficulty and has not
   been investigated.
3. **These files replaced an earlier set.** An earlier run of Qwen3-4B at 4-bit
   compression agreed with its own uncompressed version on only 62.5% of questions.
   The three runs reported here are all uncompressed and directly comparable, but
   the lesson stands: **numerical precision changes results substantially, and must
   always be reported.**
4. **Every question lists its actions in the same fixed order** (§10a). For
   Llama-3.2-3B this is not a detail: the model never selects past the second item
   on the list, which alone accounts for its zero raises and zero shows. Until the
   exam is re-run with shuffled action orders, part of that model's score reflects
   page layout rather than card play. This is the most important outstanding check
   on the benchmark itself.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Action Accuracy (AA)** | Percentage of questions where the model picked the solver's action. The headline score. |
| **baseline / parrot score** | What you would score by ignoring the question and answering one fixed word every time. The bar any real result must clear. |
| **blind** | A player who has not looked at their own cards. Pays half price. |
| **boot** | The compulsory 1-unit ante both players post before the deal. |
| **CFR+** | The self-play algorithm used to compute the optimal strategy. |
| **chaal** | Match the current bet and continue. Equivalent to "call". |
| **conversion** | The decision to stop being blind and look at your cards. The project's main research target. |
| **Exact Match (EM)** | Right action *and* right amount. Collapses to AA when no amount is supplied. |
| **exploitability** | How much a perfect opponent could win against your strategy if they knew it completely. Zero means unbeatable. Ours is 0.031% of a pot. |
| **greedy decoding** | The model always picks its most likely next word rather than sampling. Makes runs reproducible. |
| **mixed question** | A question where the solver itself has no single best answer. Graded by distribution distance, not accuracy. |
| **pack** | Give up the hand. Equivalent to "fold". |
| **quantisation** | Compressing a model to use less memory, at some cost to accuracy. None of these three runs used it. |
| **raise** | Double the stake. |
| **seen** | A player who has looked at their own cards. Pays full price, and loses the right to demand a show against a blind opponent. |
| **show** | Pay a fee to force an immediate comparison of hands. Only a blind player may demand one against a blind opponent. |
| **stay-blind** | Deliberately decline to look at your cards, keeping the discount. |
| **stake** | The current unit price of a betting action. |
| **TVD (total variation distance)** | How far the model's probabilities are from the solver's. 0 = identical, 1 = unrelated. Not measured in these runs. |
| **turn index** | Which of your own turns this is — 1st, 2nd, 3rd or 4th. Conversion results must always be split by this. |

---

## 14. Reproducing these numbers

The prediction files use the field name `answer`; the grader reads `action`. They
must be remapped, and the metadata header line dropped, before scoring:

```bash
# one-off: strip the metadata line and rename the field
python - <<'PY'
import json
for m in ("qwen3_8b", "qwen3_4b", "llama3.2_3b"):
    src, dst = f"results/preds_{m}_kaggle.jsonl", f"runs/preds_{m}_norm.jsonl"
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            r = json.loads(line)
            if "id" not in r:            # metadata header
                continue
            r["action"] = r["answer"]
            g.write(json.dumps(r) + "\n")
PY

# score
python src/metrics.py --eval data/teenpattibench_eval.jsonl \
                      --predictions runs/preds_qwen3_8b_norm.jsonl \
                      --json runs/report_qwen3_8b.json
```

Confirm the grader itself is working before trusting any score — this grades the
solver against its own answer key and must print 100%:

```bash
python src/metrics.py --self-test
```

**Two notes for future runs.** Emitting the field as `action` rather than `answer`
would remove the remapping step entirely. Adding `amount` and `distribution`
alongside it would make Exact Match and TVD real measurements rather than
placeholders.

---

## 15. The bottom line

We built an exam whose answers are provably correct to within 0.031% of a pot, and
which is deliberately designed so that guessing fails. Three language models took
it. The best scored 33.11% against a guessing baseline of 25.94%.

On the one section built to test something no model could have read about
beforehand — knowing *when* to buy information — all three did substantially worse
than guessing, and all three failed in exactly the same way: not by choosing badly,
but by not choosing at all.

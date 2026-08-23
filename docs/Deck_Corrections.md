# Status deck — slide-by-slide corrections

**Read first.** `Project_Status_Deck.pptx` in the project tree is not a PowerPoint
file. It is a plain-text markitdown dump (4.9 KB, UTF-8, opens with `## Slide 1`)
saved under a `.pptx` extension. The real deck is not in the project. So this is a
patch spec against the dump's content, applied wherever the real deck lives. If the
real deck is unrecoverable, say so and it can be rebuilt as a genuine 7-slide `.pptx`.

Seven slides. Four need edits. Slides 2 and 3 are clean.

---

## Slide 1 — subtitle line

**Replace:**
> Status meeting · 9 findings · 31 notebooks · 53-page report

**With:**
> Status meeting · 9 findings · 8 falsified explanations · 61-page report

Notebook count dropped rather than corrected — the deck says 31, the README says 29,
and neither has been reconciled. Put it back once the zip is counted.

---

## Slide 4 — footer line

**Replace:**
> The honest catch: hub width forces a choice. Narrow (512) and components transfer;
> wide (1024+) and transfer collapses while reconstruction improves.

**With:**
> The honest catch, and it is unexplained. Zero-shot transfer peaks at 512 (0.489)
> and falls to 0.312 at 768 — a −0.177 step against +0.015 before it. Reconstruction
> plateaus; it cannot cliff by construction. Both explanations tested, both falsified.

Three errors in the original: the cliff is at **512→768**, not "1024+"; transfer falls
to 0.312, not to chance; and the line implies a settled trade-off when the mechanism
is open.

**Also on this slide:** "92.9–96.5%" for zero-shot transfer, while slides 3 and 7 say
"93–96%" and the report's within-family band is 93.8–96.5%. Pick one and use it in all
three places. The report's own phrasing is *93 to 96 per cent*.

**Consider adding** — this slide is the natural home for G7's answer to the obvious
question, and it is currently missing from the deck entirely:

> Why does a global linear map work at all, when what encoders share is local?
> It doesn't carry the local structure — it relocates it. On healthy spaces the hub is
> NEUTRAL on neighbourhood agreement (+0.010, n=6). Transfer runs on the global
> component: real, thin, and exactly why the hub works AND why it misses the ceiling.

---

## Slide 5 — the whole argument needs reversing

This is the slide most exposed in a viva. It rests the local-vs-global verdict entirely
on the ratio-to-chance curve, which C.13.9 retracts as evidence. An examiner who has
read the report can quote it against the deck.

**Replace the caption:**
> neighbourhood overlap as a multiple of chance — every one of the 21 pairs falls monotonically

**With:**
> raw ratio to chance — but this curve is NOT the evidence: it is bounded above by N/k,
> so it falls even under perfect agreement (C.13.9, a correction to our own C.13.5)

**Then replace the k-table** with the chance-corrected numbers, which tell a sharper story:

| | k = 1 | k = 10 | k = 50 | k = 100 | k = 500 |
|---|---|---|---|---|---|
| bge – SBERT | 0.645 | 0.771 | **0.787** | 0.741 | 0.644 |
| img_base – img_large | 0.518 | 0.695 | **0.709** | 0.652 | 0.445 |
| img_base – txt_bge *(cross-modal)* | 0.141 | 0.360 | **0.507** | 0.449 | 0.268 |
| img_base – txt_gpt2 *(cross-modal)* | 0.046 | 0.149 | **0.209** | 0.204 | 0.131 |

Caption: *chance-corrected agreement (κ form), bounded in [0,1] at every k. Agreement
does not fall monotonically — it PEAKS at k ≈ 20–50 for every pair measured.*

**Replace the "Which Platonic claim survives" evidence line** so the verdict rests on
CKNNA, which is not subject to the artifact:

> SUPPORTED — models are different CHARTS of one shared manifold. Evidence: CKNNA
> (the literature's own metric), 21 of 21 pairs higher locally than at the global
> limit; image→GPT-2 falls 0.917 at k=10 to 0.169 at the limit. Two further
> independent lines agree, and Gröger et al. (ICML 2026) report the same pattern on
> 204 model pairs after permutation calibration.

**Land-on line if asked about the change:** *the shared structure is
neighbourhood-sized — not point-sized and not global. That is a sharper claim than the
one it replaces, and it came out of correcting my own evidence.*

---

## Slide 6 — box 1

**Replace:**
> **Four explanations tested and FALSIFIED**
> Collapse depressing shape correlation · the hub discarding directions · reach
> predicting writability · more encoders improving transfer.

**With:**
> **Eight explanations tested and FALSIFIED — two of them mine**
> Collapse depressing shape correlation · the hub discarding directions · reach
> predicting writability · more encoders improving transfer · the ratio-to-chance
> curve as evidence of locality (my own, retracted) · the width cliff as
> local-structure loss · the width cliff as spectral amplification (my own preferred
> account, retracted) · four further cliff statistics, all four failing the same
> discontinuity standard.

The "two of them mine" is the strongest thing on the slide. Do not bury it.

---

## Slide 7 — the OPEN column is wrong

Two of the three items listed as open were closed by G6.

**Replace the OPEN block:**
> Cross-ARCHITECTURE: a convolutional encoder. G4 accepts it as a one-line change — the strongest remaining test
> CCA spectrum: shared vs private directions per pair — not attempted anywhere in the project
> Procrustes across all 21 pairs (have it for one)

**With:**
> The width cliff: real, reproduced, and UNEXPLAINED after SIX tested explanations
> Benchmark B1 — experts vs one large model, pre-registered in Sources_and_Benchmarks but not run; needs a result or an explicit deferral
> A hub rebuilt from surviving caches reproduces the spectrum shape to 0.71% but runs 7.8 points low on transfer; the original head protocol was not preserved

**And in the DONE block**, replace the counts and add what G6 closed:

> 9 findings, 7 experiment series, 61-page report
> 7 explanations tested and recorded FALSE — two of them mine
> Adapter shipped: Core ML, 0.79 MB, iOS 16+
> Cross-lineage transfer measured (SigLIP 94.2%)
> Manifold characterised: intrinsic dim 11.2–19.9 inside 768–2048 ambient (<3%);
> Procrustes 21 pairs 0.194 vs ridge 0.667; CCA cross-modal shares 6 of 64 directions
> 47-check certification, no hard failures

Carry the Procrustes caveat in the notes: those values follow a common-PCA reduction
and are **not** comparable to Experiment B's raw 0.038.


---

## New since the sweep — four results to fold in

Run in one session against a hub rebuilt from the surviving caches. The rebuild
passed a spectrum-shape gate at 0.71%, with a global scale factor of 0.846 that
whitening cancels exactly. The rebuilt *head* protocol is not the original: it
runs ~7.8 points low, measured against SigLIP's published figure. So absolute
values below are not comparable to the report; orderings and ratios are.

**Cross-architecture — closes the last C.13.2 gap.**
ConvNeXt-base (IN-22k) 87.5% of native, SigLIP 2 86.4% measured identically,
SigLIP's published figure 94.2%. A convnet sits at or above an encoder already
known to be in-band. Architecture does not bound the claim; architecture and
objective remain confounded. Slides 4 and 7.

**The cliff — four more explanations falsified.**
Accumulated noise energy, noise/signal direction ratio, condition number, and
shared-direction fraction all score D of 0.62–0.80 against transfer's 4.03.
Six explanations tested, six falsified. The shared-direction fraction falls
1.00 → 0.12 across the sweep — a real mechanism, but smooth. Slide 6.

**Pair ordering is not a PCA artifact.**
Measured raw with a similarity transform, the ordering matches kNN overlap,
CKNNA, CCA and rank correlation exactly: bge–SBERT top at +0.738, GPT-2 pairs
bottom at ~0.06. A similarity transform recovers 42% of a full linear map
(0.281 against ridge's 0.667). Not comparable in *level* to Experiment B's
rotation-only 0.038 — fitting a scale is a more permissive family — so the
C.13.7 caveat is rephrased, not removed.

**The healthy-pairs null is now bounded.**
Cluster-bootstrap 95% CI −0.038 to +0.024, minimum detectable effect 0.034
against a pre-registered SESOI of 0.050. Six pairs from four encoders, so the
interval resamples encoders, not pairs. Rebuilt −0.010 against original +0.010:
two protocols straddling zero. Worth a line on slide 4 — it is the only place
the deck can say "demonstrated null" rather than "no effect found".

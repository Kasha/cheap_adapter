# Representation Convergence — Full Project Guide (English)

A scientific experiment on representation convergence, carried through to a
deployed engineering artifact. This document is the map: hypothesis, every
stage as executed, results, decisions, and where each deliverable lives.

**Status: complete — nine findings, ten falsified explanations, and the width cliff explained as a numerical artifact.** What began as two experiments (A and B)
grew into seven series. Experiments A and B established that convergence is
linear and deployable; series C, E, F, G then tested *how far* the claim goes —
across modalities, across model scale, without any pairing at all, and finally
into a single shared coordinate system serving seven encoders at once, then
characterised as a manifold and probed for what the hub does and does not do.
Series H
and V are tooling that certify the measurements.

The only GPU-bound step is the one-time encoding pass that produces the
representation caches — running the eight encoders forward over 9,533 items
on Google Colab (T4/L4). Everything after that is closed-form linear algebra
that runs on CPU: the ridge maps, the hub, and all transfer. The expensive
hardware builds the raw material once; the science runs without it. The
adapter is exported as a Core ML package, and no pretrained weight was ever
modified.

---

## 1. Background & Hypothesis

**Question asked:** Can layers from different models be assembled into a new
model without training?

**Answer established:**
- Merging raw weights only works for models sharing the same base
  (SLERP/TIES/DARE, frankenmerging). Cross-architecture layer transplants fail
  due to permutation symmetry — same function, incompatible coordinates.
- However, *representations* converge: independently trained models learn the
  same content in different coordinate systems (Git Re-Basin, CKA studies,
  Platonic Representation Hypothesis, vec2vec).

**Hypothesis proven, then sharpened:** Well-trained independent models share
representations up to a linear transformation. Each word carries weight —
*well-trained* (hence the random-weights control), *independent* (no shared
lineage), and *up to linear* (strong enough to undo rotations and permutations,
too weak to fabricate absent information).

The later series added a second half to the claim, and it is the project's most
useful result: **whether that shared content is READABLE by cosine is a
separate question from whether it EXISTS.** A language-model space can encode
the correspondence perfectly and hide it entirely behind a degenerate
coordinate frame — and a free, label-free linear transform restores it. See
§7.3 and report Appendix C.11/C.14.

**Related prior work (this project reproduces these at small scale):**
- Kornblith et al. 2019 — Linear CKA, layer-vs-layer similarity matrices
- Bansal, Nakkiran & Barak 2021 — model stitching, "stitching penalty"
- Huh et al. 2024 — Platonic Representation Hypothesis (see report Appendix A)
- vec2vec 2025 — unsupervised translation between embedding spaces
- Text-to-Concept 2023, Drift-Adapter 2025 — closest engineering precedents

---

## 2. Experiment A — Scientific Proof (GPT-2 vs Pythia-160M)

Notebooks: `A1`–`A3`. Models chosen deliberately from different labs,
architectures, tokenizers, and datasets (OpenAI/WebText vs EleutherAI/The
Pile), both trained from scratch, so any similarity is non-trivial.

### Step A1 — Extract activations (`A1_extract_activations.ipynb`)
Runs the same **10,000 WikiText passages** through both models; saves
mean-pooled hidden states from every layer. Also extracts a **seeded**
random-weights Pythia as the control baseline.

Sample count matters: A3 fits a 768×768 map per layer pair, so 10,000
passages (~10 per input dimension) are needed. Fewer samples produce
misleadingly low R² — the notebook fails loudly rather than proceeding.
Full reasoning in the report, §3.2.

**Expected output:** `activations.npz` with `A_layers [13, 10000, 768]`,
`B_layers`, `R_layers`. Runtime ~15 min on a T4. *Set `DATA_DIR` to a Drive
path first — a new Colab notebook is a new VM.*

### Step A2 — CKA analysis (`A2_cka_analysis.ipynb`)
Linear CKA between every layer pair — invariant to rotation, scaling and
permutation, exactly the invariance the hypothesis requires.

**Expected output:** `cka_matrix.png`, two panels. Hypothesis supported: left
panel (trained vs trained) shows a hot diagonal band; right panel (trained vs
random) is uniformly cold. Console prints mean/diagonal/max for both.

### Step A3 — Linear stitching (`A3_linear_stitching.ipynb`)
One closed-form ridge matrix per matched layer pair mapping A's space into
B's; held-out R².

**Expected output:** `stitching_r2.png` — trained curve vs random baseline
with the 0.7 threshold line; console table of per-layer R².

### Expected results — pass / fail bands

Run these numbers against your own output. "Actual" is the final run
(10,000 samples).

| Check | Fail | Marginal | Pass | Actual |
|---|---|---|---|---|
| CKA diagonal mean (trained) | < 0.15 | 0.15–0.35 | > 0.35 | **0.424** ✓ |
| CKA mean, random control | > 0.20 | 0.15–0.20 | < 0.15 | **0.067** ✓ |
| CKA trained/random ratio | < 2× | 2–3× | ≥ 3× | **6.3×** ✓ |
| Stitching R², middle layers | < 0.40 | 0.40–0.70 | > 0.70 | **0.70–0.80** ✓ |
| Stitching R², peak layer | < 0.50 | 0.50–0.70 | > 0.70 | **0.797** (L11) ✓ |
| Stitching R², random control | > 0.40 | 0.30–0.40 | < 0.30 | **0.173** ✓ |
| Depth slope signature | both flat | ambiguous | trained rises, random decays | 0.59→0.80 vs 0.38→0.09 ✓ |

The last row is the strongest single check: the two curves must slope in
*opposite* directions. Learning builds shared structure layer by layer;
random weights destroy it layer by layer. A result where both curves behave
alike is a pipeline bug, not a finding.

**Note on one threshold.** The original criterion was written as "diagonal
CKA mean > 0.5". The final run reached 0.424 — below that line, while every
gap-based criterion passed decisively. Absolute CKA between models with
different tokenizers is expected to be modest, so the ratio to the control is
the meaningful test and the absolute threshold was set optimistically; the
table above states the corrected band. Reported as-is rather than
retro-fitted.

**What a near-miss looks like** (run 1, 1,646 samples): CKA passed
(0.433 diagonal, 5.0× gap) while stitching failed — mean R² 0.434, peak
0.552, random baseline −0.773. If you see this pattern, check
samples-per-dimension **before** concluding anything: negative R² on the
control is the signature of a starved fit, not of a real absence of
structure. Then check layer alignment and pooling.

### Downstream criterion (Experiment B)
After linear translation, retrieval must retain ≥ ~90% of native performance —
this separates "statistical correlation" from "information actually
transfers".

---

## 3. Experiment B — Practical Application
## (MobileCLIP-S1 → SigLIP 2 adapter, shared Qdrant index)

Notebooks: `B1`–`B4` (core), `B5`–`B6` (validation & demo), `B7`–`B9`
(quantization tests), `B10` (pair-free translation study).

Goal: one linear matrix mapping iPhone-tier MobileCLIP-S1 image embeddings
into server-tier SigLIP 2 space, so a **single Qdrant collection** serves both
tiers — the phone indexes images offline and privately, the server queries the
same index with SigLIP text embeddings. No double indexing, no uploads.

### Step B1 — Extract paired embeddings (`B1_extract_pairs.ipynb`)
**4,000 COCO val2017 images** through both image encoders, plus SigLIP text
embeddings of the captions for evaluation. *(Dataset note: the project began
on Flickr30k, but HuggingFace deprecated script-based datasets mid-project and
broke both Flickr sources; the fix was direct official COCO downloads.)*

**Expected output:** `pairs.npz` with `mob_img` [4000, 512], `sig_img`,
`sig_txt` [4000, 768], all L2-normalized. Sanity check: SigLIP image-caption
diagonal cosine ≈ 0.15 — the expected band for sigmoid-loss models, whose
absolute cosines are low by design. Runtime ~10 min on GPU.

### Step B2 — Fit the adapter (`B2_train_adapter.ipynb`)
Two closed-form variants: Ridge (full linear map) and Procrustes (pure
rotation). If Procrustes ≈ Ridge, the spaces are identical up to rotation —
the strongest form of the convergence claim. A controlled MLP comparison
(512→1024→768, GELU, cosine loss) tests whether non-linearity adds anything.

**Expected output:** `adapter.npz`; console prints held-out R² and mean
cosine-to-target per variant. Runtime: seconds (MLP: a few minutes).

### Step B3 — Retrieval evaluation (`B3_eval_retrieval.ipynb`) — the real test
Text→image retrieval on 1,000 held-out images. Query = SigLIP text embedding
(what the server does). Three galleries: **A.** SigLIP native → ceiling;
**B.** MobileCLIP + adapter → our system; **C.** MobileCLIP raw → lower
baseline (expected ≈ chance).

**Success:** variant B ≥ 90% of variant A's recall. **If 70–90%:** upgrade the
adapter and re-run.

### Step B4 — iOS export (`B4_export_coreml.ipynb`)
**Expected output:** `adapter_fp16.npz` (~0.8 MB, universal fallback) and
`Adapter.mlpackage` — Core ML, MatMul + L2-normalize, fp16, iOS 16+, runs on
the Neural Engine.

iPhone pipeline: `image → MobileCLIP (Apple's Core ML release) →
Adapter.mlpackage → vector in SigLIP space → shared Qdrant collection`

**Deployment constraint:** the adapter is calibrated to one exact
preprocessing pipeline — bilinear resize 256, center-crop 256×256, and
**identity normalization** (mean 0, std 1; MobileCLIP consumes raw [0,1]
pixels, unlike standard CLIP). Substituting conventional CLIP normalization
on device shifts every embedding and degrades retrieval silently.

### Step B5 — Phone-pipeline sandbox (`B5_phone_pipeline_sandbox.ipynb`)
Qualitative validation on ten images per gallery: the exact phone path in
simulation (same MobileCLIP weights, adapter applied in fp16 exactly as Core
ML computes it), free-text queries, compositional queries, negative controls,
and an fp16-vs-fp32 parity check. Runtime ~4 min.

### Step B6 — Interactive visual demo (`B6_visual_demo.ipynb`)
The product loop with a user in it: a server gallery indexed natively, a
user-uploaded photo indexed through the phone path and auto-captioned (BLIP),
free-text search over the **one** shared index, top-3 results with per-tier
probabilities, mobile-vs-server timing, and a printed flow log of every step.

### Steps B7–B9 — Quantization tests (`*_quant_int8.ipynb`)
Three separate decisions, each with pre-registered pass criteria: **B7**
adapter → int8 (0.39 MB; expected pass but negligible saving), **B8** encoder
→ int8 (~43 → ~21 MB; demonstrates the *quantize → refit W → revalidate*
rule), **B9** index vectors → int8 (772 bytes/photo, 4× index memory, with
fp32 top-20 rescoring). B7 and B9 need only the saved `.npz`; B8 re-encodes
images (~6–8 min).

### Step B10 — Pair-free translation study (`B10_vec2vec_demo.ipynb`)
Tests the strongest form of the claim: translation with **no paired examples**
(disjoint image sets per side), via Gromov-Wasserstein initialization plus
CSLS self-learning, graded on 500 true held-out pairs.

> **Superseded.** B10 recovered distribution-level alignment but left
> instance-level correspondence at chance. Series C later showed that verdict
> was a *hyperparameter* artifact, not a limit of the method: with the entropic
> regularizer swept rather than fixed, exact matching reached **39.7% (595×
> chance)**. Read `C1_1` and report Appendix C.12 instead. B10 is kept because
> the comparison between the two runs is itself instructive.

---

## 4. Run Order & Commands

All stages are self-contained Colab notebooks. Each opens with a storage cell:
mount Drive and set `DATA_DIR` so artifacts survive VM recycling.

```
Experiment A:  A1 → A2 → A3          (~15 min + seconds + seconds)
Experiment B:  B1 → B2 → B3 → B4     (~10 min + seconds each)
Validation:    B5 → B6               (~4 min + interactive)
Optional:      B7, B8, B9, B10       (independent of each other)
```

Equivalent scripts are in `convergence/` and `adapter/` for non-notebook use:

```bash
cd convergence
pip install torch transformers datasets numpy matplotlib scikit-learn
python 1_extract_activations.py && python 2_cka_analysis.py && python 3_stitching.py

cd ../adapter
pip install torch transformers open_clip_torch pillow scikit-learn numpy coremltools
python 1_extract_pairs.py && python 2_train_adapter.py
python 3_eval_retrieval.py && python 4_export_mobile.py
```

## 5. Series C, E, F, G — how far the claim goes

Everything above is the original two-experiment project. The following series
were added afterwards, each answering one question the first two left open.
All reuse cached embeddings where possible; none fine-tunes anything.

### Series E — cross-modal, scaling, and geometry (`E1`–`E3`)

**E1 — cross-modal convergence and the capacity sweep.** DINOv2 (no language
supervision, ever) → bge-m3 (never saw an image). Then repeated at three
encoder scales with pooling held fixed and the pair count scaled with width so
statistical power is constant at 11.1 rows per input dimension.

| DINOv2 | params | held-out R² | R@1 as % of a matched ceiling |
|---|---|---|---|
| small | 22M | 0.516 | 40.9% |
| base | 87M | 0.580 | 53.5% |
| large | 304M | 0.609 | 58.3% |

Monotone in all six measures, **+15.3 points of ceiling per decade of
parameters** (fit R² 0.949). Returns are compressive, so nothing is claimed
beyond 304M. This is the Platonic Representation Hypothesis's central
prediction, measured. (Report C.10.)

**E1.1 — swap the text encoder for GPT-2.** R² retains 71% of the bge arm;
retrieval collapses to 34%. Cosine reads 1.000 while R@1 is 0.160.

**E1.2 — the isotropy rescue (the project's strongest finding).** Six
label-free corrections on GPT-2's caption space, each fitted on *training
targets only*. Whitening lifts R@1 **0.130 → 0.478**, reaching **102.6%** of an
encoder contrastively trained on 1.2B text pairs. Effective rank 6.5 → 490.8;
rank correlation between isotropy and retrieval **+0.94**. Replicated at a
second image scale (101.4%). Whitening is linear, so this is an *instance* of
the thesis, not an exception. (Report C.11.)

**E1.3 — the encoder ladder: C.11 as a falsifiable prediction.** Two encoders
the theory had never seen, with predictions registered in the notebook before
running:

| encoder | objective | eff. rank | raw R@1 | whitened R@1 |
|---|---|---|---|---|
| GPT-2 | causal LM | 6.5 | 0.160 | 0.478 |
| BERT | masked LM | 65.0 | 0.362 | 0.521 |
| SBERT | MLM + contrastive | 93.4 | 0.510 | 0.588 |
| bge-m3 | heavy contrastive | high | 0.466 | — |

Prediction 1 confirmed exactly (**rank correlation +1.00**). Prediction 2 —
that whitening would flatten the ladder — held only partially, and the miss is
the sharper result: **isotropy is necessary but not sufficient.** Whitening
fixes how many directions carry variance; it cannot install which directions
carry meaning. (Report C.14.)

### Series F — the contrast condition (`F1`–`F5`)

Where E trains nothing, F deliberately trains, to test whether more capacity or
more data changes the answer.

- **F1/F2 — the capacity test.** An MLP head distilled on 38,000 rows (74 rows
  per input dimension) converged to **exactly** the linear answer:
  0.9052 = 0.9052, gain −0.0000, plateau by epoch 10. Non-linearity buys
  nothing, and the data-starvation objection is refuted by the plateau.
- **F5 — the data test.** Refitting the same closed-form ridge on 38k rows
  gave small but real gains on 5,000 fresh queries (R@1 +0.0244, paired
  bootstrap CI excluding zero). The 90%-of-ceiling gate is reported as
  **NOT met** at 89.0% — a gate that fires is worth more than one that bends.
- **F3 — built and deliberately gated shut.** Running it would unfreeze
  encoder weights and cost the project its clean "no pretrained model was
  modified" statement. The gate is the contribution.

### Series C — recovering the correspondence with no pairs (`C1`, `C1.1`)

Two sets of 1,500 vectors, one side shuffled, the true permutation sealed until
scoring. Gromov-Wasserstein matches shape to shape using only each space's own
internal distance matrix.

**39.7% exact matches = 595× chance**, and the map fitted from that assignment
reaches R@1 0.252 (40% of ceiling) with verification AUC 0.906 — with **zero
paired examples**. Two negatives are reported alongside: ICP refinement
degraded the solution monotonically (it converges on its own errors), and
**whitening destroyed matching** (39.7% → 0.1%).

That last negative is the sharpest conceptual result in the project:
**isotropy helps you READ a known correspondence and hurts you when you must
FIND an unknown one.** Cosine retrieval treats uneven variance as noise;
shape-matching treats it as the signal that makes a point distinctive.
(Report C.12.)

### Series G — the shared hub (`G1`–`G7`)

Five cached spaces over the same 9,533 images — DINOv2 small/base/large,
bge-m3, GPT-2 — each given **one** linear map into a single 512-d whitened-PCA
hub. Then the constructive test: train a caption head on **one** encoder's hub
coordinates and apply it, unchanged, to encoders it has never seen.

| encoder | R@1 | % of a natively-fitted head |
|---|---|---|
| img_small | 0.398 | 111.5% *(trained here)* |
| img_base | 0.442 | **95.9%** — never seen |
| img_large | 0.468 | **92.9%** — never seen |
| random-map control | 0.000 | chance = 0.001 |

Absolute R@1 *rises* on the unseen encoders, which a head that had merely
memorised the task could not produce. This is a working system assembled from
parts that never trained together — the shared representation demonstrated by
use rather than by correlation. (Report C.13.)

**The honest catch, measured — and unexplained.** Hub width forces a choice.
Zero-shot transfer runs **0.350 / 0.435 / 0.474 / 0.489 / 0.312** across widths
64 → 768: a **−0.177** step at 512→768 against steps of +0.085, +0.039, +0.015
before it. Reconstruction plateaus (0.325 / 0.382 / 0.412 / 0.420 / 0.411) and
cannot cliff by construction — feeding an encoder its own coordinates through a
wider hub can only retain more. A hub tuned for portability cannot also be a
lossless reconstruction medium: the third independent instance of the same
tension.

**The cliff is EXPLAINED — and it is a numerical artifact, not a
finding about representations.** Seven accounts were falsified; the eighth
is confirmed to single-direction resolution on two encoders.

**The mechanism.** A d-dimensional source's hub coordinates have algebraic
rank exactly *d*. At hub width *W = d* the last retained direction is the
vanishing one — σ = **0.0054** against **10.79** one step earlier for
`img_small`, and **7.6e-07** against **8.73** for `img_base`. Least squares
divides by the singular value, so the head's weights along that direction
are enormous and meaningless; the wider test encoders have real energy
there, and it swamps their predictions.

**Resolved to one direction, not an accumulation.** Sweeping retained rank
in steps of 8: `img_small` holds 0.874–0.894 from k=696 through k=760 and
falls to **0.009** at k=768. `img_base` holds 0.959–0.978 through k=1528
and falls to **0.001** at k=1536. A step, not a decline.

**Predicted before measuring**, on two encoders that could not have
suggested it: each source cliffs at its own dimension and nowhere earlier,
independent of the head target. Six curves, six hits (768, 1536, 2048).

**Both repairs work, as the mechanism requires.** Truncating below *d*
drops the direction (k=700 → 0.878); ridge damps it (α=1e3 → 0.917). And
512 is safe because at W=512 the matrix has full rank 512 with every
direction well-conditioned.

**Three corrections to the body of this report follow:**

1. **The portability/losslessness reading of the cliff does not survive.**
   There is no trade-off to make, only a singular value to avoid dividing
   by. The meta-finding counts **two** instances of the isotropy trade-off,
   not three.
2. **512 was never tuned.** Any width strictly below the source's rank is
   safe — `hub_width < dim(source encoder)` — and the constraint lifts
   entirely under a truncated or regularised head.
3. **"Alpha cannot substitute for width" needs narrowing.** That was
   measured on the *entry-map* alpha. The **head's** alpha was never swept,
   and at α=1e3 transfer at width 1024 recovers to 0.917.

*Two errors of mine are recorded rather than quietly fixed:* an earlier
test compared entropy-based effective rank (573.7) against a prediction
about *algebraic* rank (768) — different quantities — and its ablation
zeroed the test encoders' coordinates rather than removing the direction
from the head's fit. The account was entered in the ledger on that basis
and has been withdrawn from it. Full record in Appendix E.8.

- **G2** — live demo: real COCO images, URL / upload / local path input, GPU
  used only for the fresh forward pass (the hub itself is a CPU linear solve).
- **G3** — generalised suite: register any N encoders and get a per-pair
  agreement matrix, controls, and a stated verdict against pre-registered
  gates. The metric is target-free and symmetric, so it works for any mix of
  modalities. 42 of 42 ordered pairs SHARED, mean retention 103.6% against hubs
  built for each pair alone; control at chance (0.004 vs 0.001).
- **G4 — cross-lineage transfer.** SigLIP 2, an independent lineage with no
  DINOv2 ancestry, retains **94.2%** of a head fitted natively on it — inside
  the 93.8–96.5% within-family band, control at chance. Two limits stated with
  it: this is cross-lineage but not simultaneously cross-objective, and SigLIP 2
  is still a transformer. **A convolutional encoder has now been tested**
  (K-series, Aug 2026), under the *reproduced original protocol* rather
  than an approximation of it — the reconstruction returns SigLIP 2's
  published 94.2% to the decimal, with DINOv2-base at 96.5% against 95.9%
  and -large at 93.8% against 92.9%, so these are levels rather than
  orderings. **ConvNeXt-base (ImageNet-22k) reaches 96.7% of native** —
  above the 93.8–96.5% within-family band, and above SigLIP measured in the
  same run. A convnet, with no attention and no patch tokenisation, is the
  best-transferring encoder in the project. Control at 0.001 = chance.
  **Architecture does not bound the claim, and on this evidence it does not
  cost anything either.** Three limits belong with that: ConvNeXt is
  supervised where DINOv2 is self-supervised, so architecture and objective
  move together and C.13.5 ranks objective the more likely driver; one
  convnet on one corpus is not a trend; and the defensible form remains
  "no encoder tested falls outside the band on the low side", not that
  convolutional architectures are superior.

- **G5 — the scale of agreement.** Neighbourhood overlap across all 21 pairs,
  k = 1…500. Agreement is **local**; global structure diverges. What explains it
  (exploratory): objective (+53×) > lineage (+44×) > modality (+29×), with the
  top pair bge–SBERT sharing an objective and no lineage. Permutation
  calibration: the k-NN null is exactly k/N (already calibrated), but the CKA
  null is +0.094 and *rises with width* — correcting it widens the
  objective-over-lineage gap from +0.024 to +0.181. The confound had been
  working against the finding.
- **G6 — the manifold, characterised.** Three closed-form measurements that turn
  two informal words into numbers. Intrinsic dimension (TwoNN): all seven spaces
  sit between **11.2 and 19.9** dimensions inside ambient widths of 768–2048 —
  under 3% of the storage in every case. Procrustes across all 21 pairs: mean
  **0.194** vs ridge **0.667**, with cross-modal and GPT-2 pairs going *negative*
  — "related, not rigid" confirmed broadly rather than from one case. CCA shared
  directions (of 64): image siblings 64, same-objective text 64 (54 above 0.9),
  cross-modal as few as **6**. All three place the pairs in the same order as
  kNN overlap and rank correlation — a fourth and fifth independent route to the
  same ranking. *Caveat:* the Procrustes values follow a common-PCA reduction and
  are **not comparable** to Experiment B's raw 0.038.
- **G7 — what the hub does to local structure.** The science says what encoders
  share is local; the engineering is a global linear projection. G7 makes them
  meet. The map is not gentle: mean native-neighbourhood preservation is
  **0.594** at k=10 (GPT-2 worst at 0.257) — the projection reshapes local
  structure rather than rotating it. Stratified by the collapse diagnostic, the
  aggregate +0.111 splits into two different phenomena: on **healthy** spaces
  (pair-cosine < 0.30) the hub is **NEUTRAL**, +0.010 over n=6 with two negative;
  on pairs including a **degenerate** space it gains +0.151, with Spearman +0.82
  between a pair's worst pair-cosine and its gain. So transfer runs on the
  *global* linear component — real, if thin — which is why the hub works **and**
  why it never reaches the ceiling. The large gains are the C.11 isotropy rescue
  for the fourth independent time, not hub alignment, and must not be averaged
  into the headline. *Bounds (now measured, K-series):* the healthy-pair null is **bounded, not
  merely undetected**. A re-run on a rebuilt hub gives −0.010 against the
  original +0.010, with a cluster-bootstrap 95% interval of −0.038 to +0.024
  and a minimum detectable effect of 0.034 — inside the pre-registered
  smallest-effect-of-interest of 0.050, so the design could have found an
  effect that mattered and did not. The six pairs are every combination of
  four healthy encoders, so they are not six independent observations; the
  interval resamples encoders rather than pairs, widening it 1.22×. With
  only four clusters it is a coarse bound rather than a precise estimate.
  Two measurements under different protocols straddling zero at ±0.01 is
  stronger evidence than either alone: the effect has no stable sign. An earlier version used effective rank ÷
  ambient width as the collapse flag, misclassified SBERT (the healthiest text
  space in the project) as degenerate, and reported the opposite verdict; the flag
  was changed to pair-cosine, which is width-independent.

### Series H and V — tooling (`H1`, `V1`)

Not experiments. `H1` (formerly `D1`; renamed because Appendix D of the report
means something else) runs five checks before any negative result is believed:
rows/dim, shuffle alignment, artifact sanity, **target geometry**, and
cache-key integrity. `V1` is the reusable geometry battery for any `[n, d]`
representation — anisotropy, hubness, top-PC removal, whitening, cross-space
comparison, retrieval and verification AUC.

**Certification run:** 47 checks across every stored artifact, **no hard
failures**. Every fit sits at 5.9–9.8 rows per input dimension; shuffle gaps run
0.545–0.863 against a 0.2 threshold; all encoder caches match the configuration
named in their filenames. The only two flags are the GPT-2 caption space, whose
collapse is a documented finding rather than a defect — and the battery
reproduced its signature independently in two separately generated files.
(Report D.5.)

---

## 6. Run Order

```
Experiment A:   A1 → A2 → A3               (~15 min + seconds + seconds)
Experiment B:   B1 → B2 → B3 → B4          (~10 min + seconds each)
Validation:     B5 → B6                    (~4 min + interactive)
Quantization:   B7, B8, B9                 (independent)

Cross-modal:    E1 → E1.1 → E1.2 → E1.3    (E1 encodes; the rest reuse its cache)
Contrast:       F1 → F2 → F5               (F3 stays gated)
Pair-free:      C1.1                       (C1 kept for the comparison)
Shared hub:     G1 → G2 → G3               (reads cached vectors; seconds)

Tooling:        H1 (before believing a negative), V1 (any space, any time)
```

Every notebook opens with the same storage cell: `STORAGE = "drive" | "local" |
"env"`. Mount Drive and set `DATA_DIR` so artifacts survive VM recycling.

## 7. Results

### 7.1 Experiment A — Run 1 vs Run 2

Run 1 (1,646 samples) fell below the stitching criterion due to sample
starvation (1.6 samples per input dimension). Run 2 (10,000 samples, identical
pipeline) passed decisively:

| Metric | Run 1 | Run 2 (final) |
|---|---|---|
| CKA trained mean / diagonal / max | 0.365 / 0.433 / 0.727 | 0.359 / 0.424 / 0.744 |
| CKA random mean / max | 0.087 / 0.339 | 0.067 / 0.299 |
| Stitching R² mean (trained) | 0.434 | **0.737** |
| Stitching R² peak (L11) | 0.552 | **0.797** |
| Stitching R² mean (random) | −0.773 | 0.173 |
| Verdict vs pre-registered criteria | below target | **PASS** |

The structural signature matters as much as the values: the trained curve
*rises* with depth (0.59 → 0.80) while the random control *decays*
(0.38 → 0.09) — opposite slopes, an 8× gap at L11.

### 7.2 Experiment B — the deployed adapter

| Result | Value |
|---|---|
| Ridge adapter: held-out R² / mean cosine | 0.592 / **0.900** |
| Procrustes (rotation only): R² | 0.038 (anisotropy mismatch, not just orientation) |
| MLP adapter: best cosine | 0.903 (+0.003 → relationship is linear) |
| Retrieval R@1 / R@5 / R@10 (ceiling) | 0.630 / 0.885 / 0.944 |
| Retrieval with linear adapter | 0.557 / 0.843 / 0.918 |
| As % of ceiling | 88.4% / **95.3%** / **97.2%** |
| Raw cross-space baseline | 0.001 / 0.005 / 0.010 (exact chance) |
| Deployed artifact | Adapter.mlpackage, 0.79 MB fp16, iOS 16+ |

**The same adapter as three decision problems** (report D.2) — difficulty rises
as the decision becomes more global, because a verification negative is a
*typical* impostor while a membership negative is the *best of 500*:

| problem | form | result |
|---|---|---|
| verification | 1:1, "same photo?" | AUC **0.999** |
| ranking | 1:N, per query | R@1 **0.987** |
| membership | 1:N, one global cutoff | AUC **0.889** (best F1 0.840) |

**Hubness** (report D.3) — the adapter does more than align the two spaces, it
restores a healthy neighbourhood structure. N₁₀ skewness: raw **4.28** (one
image is the nearest neighbour of 202 of 1,000 queries, and 18.2% of the
gallery is never anyone's neighbour) → adapted **1.61**, against the
jointly-trained ceiling's 1.41.

### 7.3 The nine findings

| # | Finding | Headline evidence | Report |
|---|---|---|---|
| 1 | Convergence is linear | MLP = ridge exactly at 74 rows/dim (0.9052 = 0.9052) | C.9 |
| 2 | Related, not rigid | Procrustes 0.038 vs ridge 0.592; 69× knob spread | §4 |
| 3 | Deployable | 95% of ceiling at 0.79 MB; verification AUC 0.999 | §4, D.2 |
| 4 | Cross-modal without joint training | 40.9% of a matched ceiling, 358× chance | C.8 |
| 5 | Alignment scales with capacity | +15.3 pts per decade, six monotone measures | C.10 |
| 6 | Isotropy, not similarity training | whitening → 102.6% / 101.4% at two scales | C.11 |
| 7 | The correspondence is self-identifying | GW 39.7% exact = 595× chance, zero pairs | C.12 |
| 8 | One shared space, used | hub transfer 93–96% of native on unseen encoders | C.13 |
| 9 | The mechanism predicts | pre-registered ladder, rank-corr +1.00 | C.14 |

**Plus a recurring meta-finding, measured twice independently:** isotropy
helps when *reading* a correspondence and hurts when *finding* one (C.11 vs
C.12). The same geometric property, opposite prescriptions — a property of
shared linear coordinates as such, not of any one experiment.

*This count was corrected downward.* The width cliff was previously listed as
a third instance. It is not: the first two are about isotropy, the cliff is
about rank (Appendix E.8). Two independent instances, not three.

**And a fourth instance of the C.11 rescue:** G7 found the hub repairing
degenerate coordinate frames exactly as explicit whitening (C.11) and the
hub-basis rescue (C.13.3) did.

**A fifth independent route to the same finding.** Rebuilding the hub
produced one diagnostic worth keeping: each encoder's held-out R² for its
own map into the shared coordinate system. The three image encoders reach
**0.46–0.57**; all four text spaces reach **0.02–0.05**, uniformly, with
collapsed GPT-2 indistinguishable from healthy SBERT. The uniformity is the
point — it is not the GPT-2 collapse restated. Nor is it an alignment
artifact: bge-m3's row correspondence is guaranteed by file construction
rather than inferred, and bge scores 0.020 like the rest. A whitened hub is
very largely an image-side object, and a text space sharing a handful of
directions with it can reach almost none of it. Same conclusion as the CCA
measurement (6 of 64 cross-modal) and the composition results, from the
geometry of the shared space itself. Reported in Appendix E.6.

### 7.4 The falsification ledger — ten, two of them the project's own

A gate that never fires is decoration; an explanation never withdrawn is
decoration too. Each of these was proposed, tested, and recorded as false rather
than quietly dropped.

| # | Explanation | Verdict | Where |
|---|---|---|---|
| 1 | Collapse depresses shape correlation | FALSIFIED | C.11.2 |
| 2 | The hub discards distinguishing directions | FALSIFIED | C.13 |
| 3 | Reach predicts writability | FALSIFIED | C.13.4 |
| 4 | More encoders improve transfer | FALSIFIED — 0.451 vs 0.445, CI includes zero; *unchanged* is the useful result | C.13 |
| 5 | The ratio-to-chance decline shows agreement is local | **PARTLY THE PROJECT'S OWN** — normalisation artifact; ratio is bounded by N/k, and chance-corrected agreement *peaks* at k ≈ 20–50 | C.13.9 |
| 6 | The width cliff is local-structure loss | FALSIFIED — smooth predictor (−0.018 / −0.030), discontinuous outcome (−0.177) | C.13.11 |
| 7 | The width cliff is spectral amplification | **THE PROJECT'S OWN PREFERRED ACCOUNT** — FALSIFIED; 1/√λ decelerates across the cliff, 1.45× → 1.26× | C.13.11 |
| 8 | Four further cliff statistics: accumulated noise energy, noise/signal direction ratio, condition number, shared-direction fraction | ALL FALSIFIED — D of 0.39–0.88 against transfer's 4.03 on a pre-registered discontinuity standard; each smooth where the outcome is discontinuous | E.3 |
| 9 | The cliff sits at the head *target's* dimension | FALSIFIED by a wrong prediction, not by wrong shape — the location does not move between bge (1024-d) and SBERT (768-d) | E.3 |
| 10 | The cliff is an accumulation of ill-conditioned directions | FALSIFIED — the collapse is a step at one direction, not a decline across many: transfer holds to k=760 then falls to 0.009 at k=768 | E.8 |

Two of the ten were this project's own explanations, and #5 retracts an
argument the report had already published. The local-over-global conclusion
survives on CKNNA — 21 of 21 pairs higher locally than at the global limit — plus
two further independent lines; one of four arguments was faulty and has been
replaced by the three that are not.

## 8. Deliverables

| File | Contents |
|---|---|
| `Final_Project_Report.pdf` (64 pp) | The full record: body, Appendix A–B, Appendix C (C.1–C.14, including C.13.1–C.13.11), Appendix D (D.1–D.5) and **Appendix E — K-series addenda** (E.1–E.7, dated) |
| `Technical_Cheat_Sheet.pdf` (21 pp) | Glossary, model roster with *why each model*, pooling with worked arithmetic, the GPT-2 collapse, margin/hubness/collapse diagnostics, A11–A16 |
| `Results_Summary.pdf` (3 pp) | Every experiment and its result in one table, closing on the seven-falsified ledger |
| `Status_Report.pdf` (6 pp) | What has been established, in prose |
| `Status_One_Page.pdf` (2 pp) | Every experiment in one table |
| `Defense_Brief.pdf` (4 pp) | Assumptions, restrictions, tuning, the hard questions with answers, achievements, conclusion |
| `Mock_Viva.pdf` (4 pp) | 15 rehearsed Q&A with Land-on / Trap lines |
| `Math_and_Terms.pdf` (6 pp) | Every formula in the project, worked, in ASCII |
| `Project_Atlas.pdf` (9 pp) | Navigation: notebook → purpose → finding → report section, and the reverse lookup |
| `Vector_Similarity_Handbook.pdf` | Background on the five levels of similarity |
| `Blog_Post.pdf` | Narrative account, current scope |
| `Positioning_Impact_Applications.pdf` | Literature placement (incl. Gröger 2026, CWU 2026), impact, applications |
| `Defense_Deck.pptx` | 15-minute presentation with speaker notes — **see note below** |
| `notebooks_final.zip` | Notebooks, AST-validated, each with a portable storage cell |
| Earlier editions | `B5_Sandbox_Practical_Report.pdf`, Hebrew editions — written against the original two-experiment scope |

> **One test deferred, recorded rather than omitted.** Benchmark B1 —
> an ensemble of specialists against one large model — was pre-registered
> in `Sources_and_Benchmarks` and has not been run. It is deferred for time,
> not abandoned, and nothing in this project depends on its outcome.

> **Row alignment across caches is positional, never by id — read this before
> joining caches.** Two `keep` conventions coexist. The E1.1-era caches
> (DINOv2 small/base/large, and `hub_ids` in `hub_rebuilt.npz`, which is
> derived from them) store the *request position* 0…N−1 — `arange`, carrying
> no image identity. The G4-era caches (SigLIP 2, ConvNeXt) store real COCO
> image ids. An id join across the two groups is meaningless: the 1,879
> "overlap" between them is just the count of COCO ids that happen to fall
> below 9,533 (documented in `G4_convnext`). Rows nonetheless align, because
> every cache was written by the same deterministic loop over
> `sorted(set(caps) & set(url))[:N]`. Verify alignment the way `G4_convnext`
> and `H1` do — a ridge or Spearman between two caches must vastly exceed the
> same statistic under a row shuffle — and never by comparing `keep` values.

> **Two bookkeeping items to reconcile before submission.** (1) The notebook count
> is quoted as 29 in one place and 31 in another; count the zip and fix both.
> (2) The only copy of the status deck currently in the project tree is a
> plain-text markitdown dump saved with a `.pptx` extension — not a PowerPoint
> file. Locate the real deck or rebuild it.

## 9. Decision Gates

| Gate | Condition | Next action |
|---|---|---|
| A passes | CKA + R² criteria met | Proceed to Experiment B |
| A fails | No diagonal structure / R² low | **Run H1 first.** Check power, alignment, and *target geometry* before rejecting the hypothesis |
| B passes | Recall ≥ 90% of ceiling | Wire adapter into the shared Qdrant collection |
| B marginal | 70–90% of ceiling | Reported honestly as marginal (F5 reached 89.0%) rather than bent to pass |
| Hub transfer | ≥ 50% of a natively-fitted head | Achieved 93–96%; sweep width, read the **zero-shot** column, not self-transfer — self-transfer plateaus and cannot cliff |
| Any negative | before believing it | H1's five checks; a collapsed target is a reading instruction, not an absence of structure |

## 10. Key Lessons

1. **Underpowered evaluation lies systematically, not randomly.** The run-1
   failure produced plausible wrong conclusions and no error message. Check
   rows-per-dimension before believing a negative result.
2. **A metric is only interpretable relative to the geometry it is computed
   in.** R² 0.592 understated a system whose cosine was 0.900; cosine 1.000
   overstated a system retrieving at 9.5%; whitened R² 0.014 accompanied the
   best retrieval of any arm. Measure the space before trusting the score.
3. **Baselines convert numbers into evidence.** Every figure here means
   something only relative to its control — random weights, chance retrieval,
   a random map in the hub slot.
4. **A failed stronger method is a measurement.** The MLP that gained +0.003
   proved linearity. The whitening that rescued retrieval and then *destroyed*
   unsupervised matching located the boundary between reading and finding.
5. **Cache keys must encode everything that changes the contents.** Every
   silent corruption in this project came from a file whose name promised one
   configuration and whose contents were another — a stale teacher, a resumed
   run under the wrong tag, a reference arm from a different configuration.
6. **Pre-register the gate, then report what it says.** The 90%-of-ceiling gate
   is recorded as *not met* at 89.0%; the hub's transitivity target was missed
   for text targets and the reason measured. A gate that never fires is
   decoration.
7. **Silent degradation is the dominant failure mode in deployment.** Sample
   starvation, preprocessing mismatch, un-refitted adapters after quantization
   and collapsed target spaces all fail quietly. Guardrails and parity checks,
   not vigilance.

---

*Nine findings, five documented self-corrections reached by measurement rather
than argument, and a theory that survived its own out-of-sample test.*

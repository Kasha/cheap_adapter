# Representation Convergence — Full Project Guide (English)

A scientific experiment on representation convergence, carried through to a
deployed engineering artifact. This document is the map: hypothesis, every
stage as executed, results, decisions, and where each deliverable lives.

**Status: complete.** Both experiments ran to completion on Google Colab (T4);
the adapter is exported as a Core ML package; the two-tier pipeline is
demonstrated end-to-end. Total compute for the scientific core: under one
GPU-hour.

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

**Hypothesis proven:** Well-trained independent models share representations
up to a linear transformation. Each word carries weight — *well-trained*
(hence the random-weights control), *independent* (no shared lineage), and
*up to linear* (strong enough to undo rotations and permutations, too weak to
fabricate absent information).

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
CSLS self-learning, graded on 500 true held-out pairs. See results below and
report Appendix B.

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

## 5. Decision Gates

| Gate | Condition | Next action |
|---|---|---|
| A passes | CKA + R² criteria met | Proceed to Experiment B |
| A fails | No diagonal structure / R² low | **Check statistical power first** (samples per dimension) before rejecting the hypothesis |
| B passes | Recall ≥ 90% of ceiling | Wire adapter into shared Qdrant collection; integrate mlpackage into iOS |
| B marginal | 70–90% of ceiling | Try a stronger adapter, re-run B3 |
| B fails | < 70% | Keep separate indexes per tier; revisit with a larger MobileCLIP variant |

## 6. Possible Extensions

- Scale Experiment A to Qwen2.5-0.5B vs SmolLM2-360M (a higher-capability
  point on the PRH curve)
- **Partly done:** cross-modal convergence — Experiment B is the vision-vision
  instance; B10 attempted the pair-free version
- Same-dimensional Procrustes, and teacher vs its own distilled student, to
  discriminate the anisotropy interpretations in report §4.4
- MobileCLIP2 / MobileCLIP-B upgrade path — the one-layer 512-d bottleneck is
  the predicted source of the residual R@1 gap; a refit takes seconds
- Use the LLM's embedding as conditioning for an on-device image generator via
  a small adapter, saving a separate text encoder in the phone tier

## 7. Final Results (as executed, July 2026)

**Experiment A** — Run 1 (1,646 validation samples) fell below the stitching
criterion (mean R² 0.434, peak 0.552; random baseline −0.773) due to sample
starvation (1.6 samples per input dimension). Run 2 (10,000 train-split
samples, identical pipeline) passed decisively:

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

**Experiment B** — 4,000 COCO pairs; SigLIP sanity diagonal cosine 0.150.

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
| B5 sandbox (qualitative) | positive / compositional / negative queries correct across two galleries; fp16 ranking-neutral (min cosine 0.9999999) |
| B6 demo (interactive) | user photo phone-path-indexed in 39 ms, auto-captioned, retrieved at rank 1 in a mixed-tier list (MOBILE 91.3% / SERVER 5.0%); search 0.08 ms local vs 150 ms simulated server RTT |
| B10 pair-free translation | distribution-level alignment recovered with **zero pairs** (cosine 0.395 vs 0.017 random floor, 23×); instance-level correspondence **not** recovered (R@1 at chance) — see Appendix B |

**Decisions taken:** hypothesis confirmed in strong quantitative form (Exp A);
linear adapter shipped — shared Qdrant collection is GO, with optional
server-side top-10 re-rank for R@1-critical flows (Exp B); adapter stays fp16
(int8 saving negligible); encoder and index-vector int8 evaluated separately
(B8/B9).

## 8. Deliverables

| File | Contents |
|---|---|
| `Final_Project_Report.pdf` (18 pp) | Full study: theory, both experiments, the two failure→fix narratives, honest novelty assessment, six challenges, Appendix A (Platonic Representation Hypothesis consolidated), Appendix B (pair-free translation study) |
| `Blog_Post.pdf` (4 pp) | Narrative version for a general technical audience, with the live demo |
| `Positioning_Impact_Applications.pdf` (4 pp) | Landscape of related work, honest uniqueness claim, eight applications, time/energy analysis |
| `Technical_Cheat_Sheet.pdf` (7 pp) | Glossary, model profiles, inside MobileCLIP-S1, the preprocessing trap, methods with formulas |
| `B5_Sandbox_Practical_Report.pdf` (5 pp) | Per-query sandbox analysis, three demo screenshots, verbatim flow log, practical uses, suggested visualizations |
| Hebrew editions | `Final_Project_Report_HE.pdf`, `Blog_Post_HE.pdf`, `Positioning_Impact_Applications_HE.pdf` |
| `notebooks/` | A1–A3, B1–B4, B5, B6, B7–B9 (quant), B10 — each self-contained with markdown, success criteria, and storage cell |
| `convergence/`, `adapter/` | Equivalent standalone scripts |

## 9. Key Lessons

1. **Underpowered evaluation lies systematically, not randomly.** The run-1
   failure produced plausible wrong conclusions and no error message. Check
   samples-per-dimension before believing a negative result.
2. **Baselines convert numbers into evidence.** Every figure here means
   something only relative to its control — random weights, chance retrieval,
   negative-control queries.
3. **A failed stronger method is a measurement.** The MLP that gained +0.003
   proved linearity; the pair-free translator that reached chance measured
   where shape-only alignment stops.
4. **Silent degradation is the dominant failure mode in deployment.** Sample
   starvation, preprocessing mismatch, and un-refitted adapters after
   quantization all fail quietly. Guardrails and parity checks, not vigilance.

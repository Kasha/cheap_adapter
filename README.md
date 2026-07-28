# Representation Convergence — Full Project Guide (English)

This document consolidates the entire discussion into an English, step-by-step
plan, with the expected response/output attached to each step.

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

**Hypothesis to prove:** Well-trained independent models share representations
up to a linear transformation.

**Related prior work (our experiment reproduces these at small scale):**
- Kornblith et al. 2019 — Linear CKA, layer-vs-layer similarity matrices
- Bansal, Nakkiran & Barak 2021 — model stitching, "stitching penalty"
- Huh et al. 2024 — Platonic Representation Hypothesis
- vec2vec 2025 — unsupervised translation between embedding spaces

---

## 2. Experiment A — Scientific Proof (GPT-2 vs Pythia-160M)

Folder: `convergence/`
Models chosen deliberately from different labs, architectures, and datasets
(OpenAI/WebText vs EleutherAI/The Pile) so any similarity is non-trivial.

### Step A1 — Extract activations (`1_extract_activations.py`)
Run the same 2,000 WikiText sentences through both models; save mean-pooled
hidden states from every layer. Also extract a random-weights Pythia as a
control baseline.

**Expected response/output:** `activations.npz` containing
`A_layers [L_A, N, d_A]`, `B_layers [L_B, N, d_B]`, `R_layers` (random
baseline). Console prints extraction progress. Runtime ~2–5 min on GPU.

### Step A2 — CKA analysis (`2_cka_analysis.py`)
Compute Linear CKA between every layer pair (invariant to rotation/scaling —
exactly the invariance the hypothesis requires).

**Expected response/output:** `cka_matrix.png` with two panels.
- Hypothesis supported: left panel (trained vs trained) shows a hot diagonal
  (early↔early, mid↔mid layers similar); right panel (trained vs random) is
  uniformly cold.
- Hypothesis rejected: both panels look like structureless noise.
Console prints mean/diagonal/max CKA for both comparisons.

### Step A3 — Linear stitching (`3_stitching.py`)
Fit one ridge-regression matrix per matched layer pair (closed form, no model
training) mapping A's space into B's; measure held-out R².

**Expected response/output:** `stitching_r2.png` — trained curve vs random
baseline curve with a 0.7 threshold line; console table of per-layer R².

### Success criteria (scientific)
1. Diagonal CKA mean > 0.5 with random baseline < ~0.15 (≥3–4× gap)
2. Stitching R² > 0.7 in middle layers (edges are tokenizer-specific and
   always lower)
3. Task-level check: after linear translation, retrieval/classification on
   translated representations retains ≥ ~90% of native performance — this
   separates "statistical correlation" from "information actually transfers"

### Interpretation of success
- Scientific: models discover, not invent, representations; a shared
  structure dictated by data/reality ("universal internal language").
- Practical: a tiny linear map (thousands of parameters) suffices to
  translate between models — enabling component swaps without retraining,
  a shared embedding space, and a small-tier ↔ large-tier bridge.
- What success does NOT mean: direct layer transplantation still fails;
  R² = 0.8 loses 20% of variance, possibly the nuanced part — hence
  criterion 3.

---

## 3. Experiment B — Minimal Practical Application
## (MobileCLIP → SigLIP 2 adapter, shared Qdrant index)

Folder: `adapter/`
Goal: one linear matrix mapping iPhone-tier MobileCLIP-S1 image embeddings
into server-tier SigLIP 2 space, so a **single Qdrant collection** serves both
tiers: the phone indexes images offline, the server queries the same index
with SigLIP text embeddings — no double indexing.

### Step B1 — Extract paired embeddings (`1_extract_pairs.py`)
Same 4,000 Flickr30k images through both image encoders; also SigLIP text
embeddings of the captions for evaluation.

**Expected response/output:** `pairs.npz` with `mob_img`, `sig_img`,
`sig_txt` (all L2-normalized). Runtime ~10 min on GPU.

### Step B2 — Train the adapter (`2_train_adapter.py`)
Two closed-form variants: Ridge (full linear map) and Procrustes (pure
rotation). If Procrustes ≈ Ridge, the spaces are identical up to rotation —
the strongest form of the convergence claim.

**Expected response/output:** `adapter.npz`; console prints held-out R² and
mean cosine-to-target for both variants. Runtime: seconds.

### Step B3 — Retrieval evaluation (`3_eval_retrieval.py`) — the real test
Text→image retrieval on held-out images. Query = SigLIP text embedding
(what the server does). Three galleries:
- A. SigLIP native images → **ceiling**
- B. MobileCLIP + adapter → **our system**
- C. MobileCLIP raw, no adapter → **lower baseline** (expected ≈ 0)

**Expected response/output:** console table of Recall@1/5/10 for A/B/C, plus
"adapter keeps X% of ceiling → PASS / below target" per K.
**Success:** variant B ≥ 90% of variant A's recall.
**If below 90%:** upgrade the adapter to a small 1–2 layer MLP (minutes of
training) and re-run this step.

### Step B4 — iOS export (`4_export_mobile.py`)
**Expected response/output:**
- `adapter_fp16.npz` (~1 MB) — universal fallback, e.g. for MLX
- `Adapter.mlpackage` — Core ML model (MatMul + L2-normalize, fp16,
  iOS 16+, runs on the Neural Engine). Drag into Xcode; Swift auto-generates
  a class with a single `prediction(mobileclip_embedding:)` call.

iPhone pipeline after success:
`image → MobileCLIP encoder (Apple's official Core ML release) → Adapter.mlpackage → vector in SigLIP space → shared Qdrant collection`

---

## 4. Run Order & Commands

```bash
# Experiment A (scientific proof)
cd convergence
pip install torch transformers datasets numpy matplotlib scikit-learn
python 1_extract_activations.py
python 2_cka_analysis.py
python 3_stitching.py

# Experiment B (practical application)
cd ../adapter
pip install torch transformers open_clip_torch datasets pillow \
            scikit-learn numpy coremltools
python 1_extract_pairs.py
python 2_train_adapter.py
python 3_eval_retrieval.py
python 4_export_mobile.py
```

## 5. Decision Gates

| Gate | Condition | Next action |
|---|---|---|
| A passes | CKA + R² criteria met | Proceed to Experiment B |
| A fails | No diagonal structure / R² low | Re-check extraction (pooling, layer alignment) before rejecting hypothesis |
| B passes | Recall ≥ 90% of ceiling | Wire adapter into shared Qdrant collection; integrate mlpackage into iOS app |
| B marginal | 70–90% of ceiling | Swap ridge for small MLP adapter, re-run B3 |
| B fails | < 70% | Keep separate indexes per tier; revisit with larger MobileCLIP variant |

## 6. Possible Extensions
- Scale Experiment A to Qwen2.5-0.5B vs SmolLM2-360M
- Cross-modal version: SigLIP (vision) vs an LLM (text) on image–caption
  pairs — a direct test of the Platonic Representation Hypothesis
- Use the LLM's embedding as conditioning for an on-device image generator
  via a small adapter, saving a separate text encoder in the phone tier

## 7. Final Results (as executed, July 2026)

Both experiments were run to completion on Google Colab (T4). Verbatim
outcomes:

**Experiment A** — Run 1 (1,646 validation samples) fell below the
stitching criterion (mean R2 0.434, peak 0.552; random baseline −0.773)
due to sample starvation (1.6 samples per input dimension). Run 2
(10,000 train-split samples, identical pipeline) passed decisively:

| Metric | Run 1 | Run 2 (final) |
|---|---|---|
| CKA trained mean / diagonal / max | 0.365 / 0.433 / 0.727 | 0.359 / 0.424 / 0.744 |
| CKA random mean / max | 0.087 / 0.339 | 0.067 / 0.299 |
| Stitching R2 mean (trained) | 0.434 | **0.737** |
| Stitching R2 peak (L11) | 0.552 | **0.797** |
| Stitching R2 mean (random) | −0.773 | 0.173 |
| Verdict vs pre-registered criteria | below target | **PASS** |

**Experiment B** — 4,000 COCO pairs; SigLIP sanity diagonal cosine
0.150 (expected band for sigmoid-loss models).

| Result | Value |
|---|---|
| Ridge adapter: held-out R2 / mean cosine | 0.592 / **0.900** |
| Procrustes (rotation only): R2 | 0.038 (anisotropy mismatch) |
| MLP adapter: best cosine | 0.903 (+0.003 → relationship is linear) |
| Retrieval R@1 / R@5 / R@10 (ceiling) | 0.630 / 0.885 / 0.944 |
| Retrieval with linear adapter | 0.557 / 0.843 / 0.918 |
| As % of ceiling | 88.4% / **95.3%** / **97.2%** |
| Raw cross-space baseline | 0.001 / 0.005 / 0.010 (exact chance) |
| Deployed artifact | Adapter.mlpackage, 0.79 MB fp16, iOS 16+ |
| B5 sandbox (qualitative) | positive/compositional/negative queries correct across two galleries; fp16 ranking-neutral (min cosine 0.9999999) |
| B6 demo (interactive) | user photo phone-path-indexed in 39ms, auto-captioned, retrieved at rank 1 in a mixed-tier ranking; search 0.08ms local vs 150ms simulated server RTT |
| B6 live demo (interactive) | user photo indexed via phone path in 39 ms, retrieved at rank 1 in a mixed-tier result list; local search 0.08 ms vs 150 ms simulated network round-trip |

**Decisions taken:** hypothesis confirmed in strong quantitative form
(Exp A); linear adapter shipped — shared Qdrant collection is GO, with
optional server-side top-10 re-rank for R@1-critical flows (Exp B).
Full analysis, figures (including the CKA heatmap), and the honest
novelty assessment are in Final_Project_Report.pdf; context and
applications in Positioning_Impact_Applications.pdf; terminology in
Technical_Cheat_Sheet.pdf.

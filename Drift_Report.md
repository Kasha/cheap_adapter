# Derived-document drift sweep vs `Final_Project_Report.pdf` (61 pp)

Systematic check of every derived document against the report as source of truth.
Twelve documents scanned. Findings sorted by severity.

Report state as of this sweep: 61 pages, C.13 running to C.13.11, nine findings,
seven falsified explanations (two the project's own).

---

## Severity 1 — asserts an explanation the report has FALSIFIED

C.13.11 falsified spectral amplification as the cause of the width cliff:
the amplification factor `1/√λ` rises smoothly (0.25 → 0.35 → 0.51 → 0.74 → 0.93)
and its growth rate *decelerates* across the cliff (1.45× per step before, 1.26×
across it). The cliff is a discontinuity; a smooth predictor cannot explain it.
The cliff is currently UNEXPLAINED.

| # | Document | Location | Text as written | Problem |
|---|---|---|---|---|
| **1.1** | `Defense_Brief.pdf` | Q: *"Why did the hub break at 1024 dimensions?"* | "Whitened PCA divides by singular value; past ~512 it amplifies noise-dominated tail directions, which take over cosine geometry." | A prepared viva answer giving a falsified explanation with confidence. **Highest-risk item in the sweep.** Also: the cliff is 512→768, not 1024 — the question's own premise is wrong. |
| **1.2** | `Status_Report.pdf` | p.5, hub paragraph | "a transfer cliff between 512 and 768 hub dimensions (whitened PCA amplifies low-variance noise directions past it)" | Parenthetical states the falsified cause as established fact. |
| **1.3** | `Technical_Cheat_Sheet.pdf` | §A-collapse, "Two causes of a collapse" | "the hub's own transfer cliff past 512 dimensions in Section C.13, where widening the hub amplified noise directions and destroyed transfer" | Uses the falsified explanation as the project's worked example of *pipeline-induced* collapse. The example still holds; the stated mechanism does not. |
| **1.4** | `README.md` | §5 Series G, "The honest catch" | "transfer collapses to chance while reconstruction improves, because whitened PCA amplifies noise-dominated tail directions" | Falsified cause, plus two number errors: transfer falls to **0.312**, not chance (0.001); cliff is at **512→768**, not "1024+". |

**Correct replacement text** (use verbatim, all four):

> The cliff is real and reproduced — zero-shot transfer runs 0.350 / 0.435 / 0.474
> / 0.489 / 0.312 across hub widths 64 → 768, a −0.177 step at 512→768 against
> steps of +0.085, +0.039, +0.015 before it. Reconstruction plateaus and cannot
> cliff by construction. Both candidate explanations were tested and both failed:
> local-structure loss (neighbourhood preservation moves −0.018 and cross-encoder
> overlap −0.030 across the cliff, against a transfer drop six times larger, and
> both fall smoothly across the whole sweep) and spectral amplification (the
> amplification factor decelerates exactly where transfer collapses). The cliff is
> currently unexplained, and is reported as such. One boundary: only the
> amplification applied to the *last* admitted direction was tested — a different
> spectral statistic could still succeed.

---

## Severity 2 — presents evidence the report has RETRACTED

C.13.9 corrected C.13.5: ratio-to-chance is bounded above by N/k, so a pair with
*perfect* agreement at every scale would still show a falling ratio. Part of the
observed decline is the shrinking ceiling. Under the chance-corrected κ form the
curve **rises to a peak at k ≈ 20–50 and falls after** — it is not monotone.
The local-over-global conclusion stands, but on CKNNA (21/21 pairs higher locally
than at the limit), not on this curve.

| # | Document | Location | Text as written | Problem |
|---|---|---|---|---|
| **2.1** | `Project_Status_Deck` slide 5 | caption under the k-table | "neighbourhood overlap as a multiple of chance — every one of the 21 pairs falls monotonically" | The whole slide rests the local-vs-global verdict on the retracted curve. An examiner who read C.13.9 can quote the report against the deck. |
| **2.2** | `Technical_Cheat_Sheet.pdf` | Table A14a caption | "every one of the 21 pairs falls monotonically. None holds flat." | Same. |
| **2.3** | `Status_One_Page.pdf` | G5 row + closing line | "Falls monotonically for every pair: 967x at k=1 → 2.3x at k=500" / "Agreement is LOCAL (967x → 2.3x…)" | Same, twice. |
| **2.4** | `Results_Summary.pdf` | G5 scale-of-agreement row | "ratio-to-chance falls for all 21 pairs (967x at k=1 → 2.3x at k=500): agreement is LOCAL" | Partly mitigated — the adjacent G5 CKNNA row carries the surviving argument — but the row still leads with the retracted one. |

**Correct replacement text:**

> Agreement is local. The evidence is CKNNA (Huh et al. 2024): 21 of 21 pairs score
> higher locally than at the global limit, cross-modal falling furthest
> (image→GPT-2, 0.917 at k=10 against 0.169 at the limit). The raw ratio-to-chance
> curve is *not* the evidence — it is bounded above by N/k, so it falls even under
> perfect agreement (C.13.9). Chance-corrected, agreement peaks at k ≈ 20–50: the
> shared structure is neighbourhood-sized, not point-sized and not global.

Note this is a *sharper* claim than the one it replaces. Worth leading with in the viva.

---

## Severity 3 — stale counts

| Item | Claimed | Actual | Where |
|---|---|---|---|
| Report length | 42 pp | **61 pp** | `README.md` deliverables table |
| Report length | 38-page | **61 pp** | `Defense_Brief.pdf` header line |
| Report length | 52 pp | **61 pp** | `Results_Summary.pdf` p.1 |
| Report length | 25 pp | **61 pp** | `Status_Report.pdf` closing nav line |
| Report length | 53-page | **61 pp** | Deck slides 1 and 7 |
| Falsified ledger | "Four explanations tested and FALSIFIED" | **seven** (two the project's own) | Deck slide 6 |
| Notebooks | 29 | reconcile — deck says 31 | `README.md` |
| Notebooks | 31 | reconcile — README says 29 | Deck slides 1, 7 |
| `Technical_Cheat_Sheet` | 15 pp | **21 pp** | `README.md` |
| `Project_Atlas` | 3 pp | **9 pp** | `README.md` |
| `Defense_Deck.pptx` | 13 slides | **7** in the surviving dump | `README.md` |

Report length appears in five documents with five different values, none correct.
It is the single most-repeated stale fact in the set.

---

## Severity 4 — open items now closed

Deck slide 7 lists as OPEN two tests G6 has since completed:

| Deck says | Reality |
|---|---|
| "CCA spectrum: shared vs private directions per pair — **not attempted anywhere in the project**" | Done. C.13.7 Table C12k: image siblings 64/64 shared; bge–SBERT 64 (54 above 0.9); cross-modal img_base–txt_bge **6 of 64**. Orders exactly as every other measure. |
| "Procrustes across all 21 pairs (have it for one)" | Done. C.13.7: mean **0.194** vs ridge **0.667** across 21 pairs; cross-modal and GPT-2 pairs go *negative*. Caveat to carry: measured after common-PCA reduction, so **not comparable** to Experiment B's raw 0.038. |

The only genuinely open test on that slide is ConvNeXt in G4.

---

## Severity 5 — missing content, not wrong content

| Document | Missing |
|---|---|
| `README.md` | Series G described as G1–G3 only. No G4 (cross-lineage), G5 (scale of agreement), G6 (manifold characterisation), G7 (hub local structure + width cliff). No C.13.7–C.13.11. No falsification ledger. Deliverables table omits `Results_Summary`, `Mock_Viva`, `Math_and_Terms`, `Vector_Similarity_Handbook`. |
| `README.md` | Lists `Blog_Post.pdf` and `Positioning_Impact_Applications.pdf` under "Earlier editions — written against the original two-experiment scope." Both were rebuilt against current scope. |
| Deck | No G6, no G7, no C.13.9 correction, no falsification ledger at seven. |

---

## Clean — no drift found

- **`Math_and_Terms.pdf`** — the amplification table is present *and correctly framed*: "That is the proposed mechanism for the hub width cliff — and testing it is how it came to be falsified, since the rise is smooth while the cliff is a discontinuity." This is the only document that gets the cliff right.
- **`Mock_Viva.pdf`** — Q "You have an unexplained result — the width cliff. Isn't that a hole?" is current and correctly argued, including the deceleration point and "I applied the same standard to my own preferred explanation."
- **`Project_Atlas.pdf`** — pages checked are current.
- **`Blog_Post.pdf`**, **`Positioning_Impact_Applications.pdf`** — rebuilt this session, no stale cliff or monotonicity claims found.

---

## Pattern

Three of the four Severity-1 hits and all four Severity-2 hits share one shape:
**the derived document kept the explanation after the report kept only the
observation.** Corrections that *remove* a mechanism propagate worst, because the
derived text reads as still-fluent — nothing looks broken.

Two documents were immune, and for the same reason: `Math_and_Terms` and
`Mock_Viva` both state the falsification *as the content*, so there was nothing
to go stale. Documents that carry mechanisms as claims drift; documents that carry
them as tested-and-failed do not.

Cheapest structural fix: every derived document that states a mechanism should
state its status inline — "(tested, falsified, C.13.11)" — rather than asserting
the mechanism and relying on the reader to know.

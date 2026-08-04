# ==========================================================
# A3-ter — self-contained. Paste as ONE cell after A3-bis.
# ==========================================================
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
MODEL_A  = "gpt2"
MODEL_B  = "EleutherAI/pythia-160m"
MAX_LEN  = 64                      # must match what A1 used

data = np.load(str(DATA_DIR / "activations.npz"))
A, B = data["A_layers"], data["B_layers"]
N = A.shape[1]
print(f"loaded A {A.shape}, B {B.shape}")


# ---------- helpers ----------
def reload_sentences(n, min_chars=100):
    """Mirror of A1's load_sentences: same configs, filters, dedup, order.
    activations.npz does not store the text, so it must be re-derived, and
    row i here must be the same passage as row i of A/B."""
    for config in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
        try:
            ds = load_dataset("Salesforce/wikitext", config, split="train",
                              streaming=True)
            seen, out = set(), []
            for row in ds:
                t = row["text"].strip()
                if len(t) < min_chars or t.startswith("="):
                    continue
                if t in seen:
                    continue
                seen.add(t)
                out.append(t)
                if len(out) >= n:
                    break
            if len(out) >= n:
                print(f"corpus: {config}  ({len(out)} passages)")
                return out
            print(f"{config} yielded only {len(out)} - trying next")
        except Exception as e:
            print(f"{config} unavailable ({type(e).__name__}) - trying next")
    raise RuntimeError("could not re-derive the sentence list")


def length_r2(M, lengths):
    """Held-out R2 for predicting sequence length from the pooled vector."""
    Xtr, Xte, ytr, yte = train_test_split(M, lengths, test_size=0.25,
                                          random_state=0)
    return r2_score(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))


def stitch_r2_mo(X, Y, multioutput, alpha=1.0):
    """A3's stitch_r2 with a selectable multioutput mode."""
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25,
                                          random_state=0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    reg = Ridge(alpha=alpha).fit(Xtr, Ytr)
    return r2_score(Yte, reg.predict(Xte), multioutput=multioutput)


# ---------- TEST 1: length signal at layer 0 ----------
sentences = reload_sentences(N)
assert len(sentences) == N, "row count mismatch - alignment broken"

tok_a = AutoTokenizer.from_pretrained(MODEL_A)
tok_b = AutoTokenizer.from_pretrained(MODEL_B)
len_a = np.array([min(len(tok_a.encode(s)), MAX_LEN) for s in sentences])
len_b = np.array([min(len(tok_b.encode(s)), MAX_LEN) for s in sentences])
print(f"\ntoken lengths: GPT-2 mean {len_a.mean():.1f}, "
      f"Pythia mean {len_b.mean():.1f}, cap {MAX_LEN}")
print(f"fraction hitting the cap: GPT-2 {(len_a == MAX_LEN).mean():.2f}, "
      f"Pythia {(len_b == MAX_LEN).mean():.2f}")

print("\nTEST 1 - length recoverable from the pooled layer-0 vector:")
ra = length_r2(A[0], len_a)
rb = length_r2(B[0], len_b)
print(f"  GPT-2  L0: R2 = {ra:.3f}")
print(f"  Pythia L0: R2 = {rb:.3f}")
print("  expectation: GPT-2 high (learned absolute positional embeddings "
      "are added at layer 0),")
print("               Pythia low (rotary encoding is applied inside "
      "attention, not at layer 0)")
print("  VERDICT:", "supports the positional explanation"
      if ra - rb > 0.15 else "does NOT support it - look elsewhere")

# ---------- TEST 2: variance weighting in the middle layers ----------
print("\nTEST 2 - does the middle-layer gap shrink without variance "
      "weighting?")
gaps = {}
for i in (4, 5, 6):
    for mo in ("variance_weighted", "uniform_average"):
        f = stitch_r2_mo(A[i], B[i], mo)
        r = stitch_r2_mo(B[i], A[i], mo)
        gaps.setdefault(mo, []).append(f - r)
        print(f"  L{i}  {mo:18s}  A->B {f:.3f}   B->A {r:.3f}   "
              f"gap {f - r:+.3f}")
vw = float(np.mean(np.abs(gaps["variance_weighted"])))
ua = float(np.mean(np.abs(gaps["uniform_average"])))
print(f"\n  mean |gap|: variance_weighted {vw:.3f} -> uniform_average "
      f"{ua:.3f}")
print("  VERDICT:", "variance weighting explains most of the asymmetry"
      if ua < vw * 0.6 else "the asymmetry survives - not a weighting "
      "artifact")

# ---------- OPTIONAL: anisotropy directly ----------
def eff_dim(M):
    ev = np.clip(np.linalg.eigvalsh(np.cov(M.T.astype(np.float64))), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())

print("\nOPTIONAL - effective dimensionality (of 768; lower = more "
      "anisotropic):")
for i in (0, 4, 8, 12):
    print(f"  L{i:2d}:  GPT-2 {eff_dim(A[i]):6.1f}    "
          f"Pythia {eff_dim(B[i]):6.1f}")

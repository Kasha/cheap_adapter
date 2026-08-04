# ==========================================================
# Real-data geometry plots — paste as ONE cell (needs
# pairs.npz + adapter.npz). Produces geometry_real.png
# ==========================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
TEAL, BLUE, GRAY, RED, NAVY = ("#0f766e", "#1a5276", "#95a5a6",
                               "#c0392b", "#1a1a2e")
N_SHOW = 60                      # points to plot (all are used for stats)

pairs = np.load(DATA_DIR / "pairs.npz")
ad = np.load(DATA_DIR / "adapter.npz")
te = ad["eval_idx"]              # held-out images only
W = ad["W_ridge"].astype(np.float32)

def l2(V):
    return V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)

sig = pairs["sig_img"][te].astype(np.float32)          # [n, 768]
mob = pairs["mob_img"][te].astype(np.float32)          # [n, 512]
adapted = l2(mob @ W)                                   # [n, 768]

# raw MobileCLIP zero-padded to 768 — exactly what B3's floor gallery does
raw = np.zeros_like(sig)
raw[:, :512] = mob

# ---- the statistics this figure claims to show (measured, not assumed) ----
cos_adapted = float((adapted * sig).sum(1).mean())
cos_raw = float((raw * l2(sig)).sum(1).mean())
print(f"MEASURED on {len(te)} held-out images:")
print(f"  adapted -> target mean cosine : {cos_adapted:.3f}")
print(f"  raw     -> target mean cosine : {cos_raw:.3f}")

def project(*arrays):
    """One shared 2-D PCA over the union so panels are comparable."""
    U = np.vstack(arrays).astype(np.float64)
    U = U - U.mean(0)
    _, _, Vt = np.linalg.svd(U, full_matrices=False)
    P = U @ Vt[:2].T
    out, i = [], 0
    for a in arrays:
        out.append(P[i:i + len(a)]); i += len(a)
    return out

idx = np.arange(N_SHOW)          # first N_SHOW held-out images
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

for ax, other, col, lab, title, note in [
    (axA, raw[idx], RED, "phone (raw MobileCLIP, zero-padded)",
     "A · Without the adapter",
     f"same-image cosine = {cos_raw:.3f}"),
    (axB, adapted[idx], TEAL, "phone (adapted: W\u00b7v)",
     "B · With the adapter",
     f"same-image cosine = {cos_adapted:.3f}")]:
    pm, ps = project(other, sig[idx])
    ax.scatter(*ps.T, s=28, c=BLUE, label="server (SigLIP native)", zorder=3)
    ax.scatter(*pm.T, s=28, c=col, marker="s", label=lab, zorder=3)
    for i in range(0, N_SHOW, 2):        # join each image to its twin
        ax.plot([pm[i, 0], ps[i, 0]], [pm[i, 1], ps[i, 1]], c=GRAY,
                lw=0.7, alpha=0.7, zorder=1)
    ax.set_title(f"{title}\n{note}   (real held-out embeddings)",
                 fontsize=10.5, color=NAVY)
    ax.legend(fontsize=7.6, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(str(DATA_DIR / "geometry_real.png"), dpi=150,
            bbox_inches="tight")
plt.show()
print("\nsaved geometry_real.png")
print("grey lines join each image's phone vector to its own server vector")

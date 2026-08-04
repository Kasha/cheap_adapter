"""Similarity geometry of the shared index: before adapter, after adapter,
and the query cone. Synthetic vectors are constructed to reproduce the
project's MEASURED statistics (raw cross-space cosine ~0.00, adapted
cosine 0.900, SigLIP image-image spread), so the geometry shown is
quantitatively faithful even though the points are illustrative.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

TEAL, BLUE, GRAY, RED, NAVY = ("#0f766e", "#1a5276", "#95a5a6",
                               "#c0392b", "#1a1a2e")
plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})
rng = np.random.default_rng(7)
N = 60


def l2(V):
    return V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)


# ---- build vectors with the project's measured statistics -------------
# latent "concepts" -> SigLIP native image vectors (768-d)
Z = rng.standard_normal((N, 24))
sig = l2(np.tanh(Z @ rng.standard_normal((24, 768))))

# adapted mobile vectors: cosine 0.900 to their SigLIP twin (measured B2)
perp = rng.standard_normal((N, 768))
perp -= (perp * sig).sum(1, keepdims=True) * sig      # make orthogonal
perp = l2(perp)
COS = 0.900
mob_adapted = l2(COS * sig + np.sqrt(1 - COS ** 2) * perp)

# raw mobile vectors: 512-d encoder output zero-padded to 768 -> lands in
# a different subspace, cross-space cosine ~ 0 (measured: chance retrieval)
raw512 = l2(rng.standard_normal((N, 512)))
mob_raw = np.zeros((N, 768))
mob_raw[:, :512] = raw512 * 0.55
mob_raw = l2(mob_raw + 0.02 * rng.standard_normal((N, 768)))

print("check | adapted-to-target cosine:",
      round(float((mob_adapted * sig).sum(1).mean()), 3))
print("check | raw-to-target cosine    :",
      round(float((mob_raw * sig).sum(1).mean()), 3))


def project(*arrays):
    """Shared 2-D PCA over the union, so panels are comparable."""
    U = np.vstack(arrays)
    U = U - U.mean(0)
    _, _, Vt = np.linalg.svd(U, full_matrices=False)
    P = U @ Vt[:2].T
    out, i = [], 0
    for a in arrays:
        out.append(P[i:i + len(a)])
        i += len(a)
    return out


fig = plt.figure(figsize=(13.6, 4.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.08], wspace=0.22)

# ================= PANEL A — before the adapter =================
axA = fig.add_subplot(gs[0])
pa_mob, pa_sig = project(mob_raw, sig)
axA.scatter(*pa_sig.T, s=26, c=BLUE, label="server (SigLIP native)",
            zorder=3)
axA.scatter(*pa_mob.T, s=26, c=RED, marker="s",
            label="phone (raw MobileCLIP)", zorder=3)
for i in range(0, N, 6):                       # same-image pairs
    axA.plot([pa_mob[i, 0], pa_sig[i, 0]], [pa_mob[i, 1], pa_sig[i, 1]],
             c=GRAY, lw=0.7, alpha=0.65, zorder=1)
axA.set_title("A · Without the adapter\ntwo disjoint regions",
              fontsize=10.5, color=NAVY)
axA.legend(fontsize=7.2, loc="upper center", frameon=False,
           bbox_to_anchor=(0.5, -0.04))
axA.text(0.5, 1.0,
         "same-image pairs joined by grey lines:\n"
         "cosine ≈ 0.00  ·  R@1 = 0.001 (chance)",
         transform=axA.transAxes, ha="center", va="top", fontsize=7.6,
         color=RED, linespacing=1.4)
axA.set_xticks([]); axA.set_yticks([])

# ================= PANEL B — after the adapter =================
axB = fig.add_subplot(gs[1])
pb_mob, pb_sig = project(mob_adapted, sig)
axB.scatter(*pb_sig.T, s=26, c=BLUE, label="server (SigLIP native)",
            zorder=3)
axB.scatter(*pb_mob.T, s=26, c=TEAL, marker="s",
            label="phone (adapted: W·v)", zorder=3)
for i in range(0, N, 3):
    axB.plot([pb_mob[i, 0], pb_sig[i, 0]], [pb_mob[i, 1], pb_sig[i, 1]],
             c=GRAY, lw=0.7, alpha=0.8, zorder=1)
axB.set_title("B · With the 0.79 MB adapter\none interleaved space",
              fontsize=10.5, color=NAVY)
axB.legend(fontsize=7.2, loc="upper center", frameon=False,
           bbox_to_anchor=(0.5, -0.04))
axB.text(0.5, 1.0,
         "each phone vector lands on its twin:\n"
         "cosine = 0.900  ·  R@5 = 95% of ceiling",
         transform=axB.transAxes, ha="center", va="top", fontsize=7.6,
         color=TEAL, linespacing=1.4)
axB.set_xticks([]); axB.set_yticks([])

# ================= PANEL C — the query cone =================
axC = fig.add_subplot(gs[2])
axC.set_aspect("equal")
axC.axis("off")

# unit circle = the surface all vectors live on after L2-normalization
circ = plt.Circle((0, 0), 1, fill=False, ec="#cfd6d8", lw=1.2)
axC.add_patch(circ)
axC.text(0, -1.24, "unit sphere — every indexed vector is L2-normalized,\n"
                   "so similarity is purely angular",
         ha="center", fontsize=7.4, color=GRAY, linespacing=1.4)

# query direction
qa = np.deg2rad(90)
axC.annotate("", xy=(np.cos(qa), np.sin(qa)), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color=NAVY))
axC.text(0.04, 1.08, 'query:  "singing red bird"\n(SigLIP text tower)',
         ha="center", fontsize=8, color=NAVY, linespacing=1.3)

# top-k cone
half = np.deg2rad(30)
th = np.linspace(qa - half, qa + half, 60)
axC.fill(np.concatenate([[0], np.cos(th)]),
         np.concatenate([[0], np.sin(th)]), color="#f6e58d", alpha=0.35,
         zorder=0)
axC.text(0, 0.38, "top-k cone", ha="center", fontsize=7.4, color="#8a7a1f")

# the three retrieved results, at angles matching their measured ranking
results = [(12, "MOBILE", TEAL, "P=91.3%"),
           (26, "SERVER", BLUE, "P=5.0%"),
           (-24, "MOBILE", TEAL, "P=1.0%")]
for k, (deg, tier, col, plab) in enumerate(results, 1):
    a = qa + np.deg2rad(deg)
    x, y = np.cos(a), np.sin(a)
    axC.scatter([x], [y], s=70, c=col, zorder=4,
                marker="s" if tier == "MOBILE" else "o")
    axC.plot([0, x], [0, y], c=col, lw=1.0, alpha=0.55, zorder=2)
    lx, ly = 1.16 * x, 1.16 * y
    axC.text(lx, ly, f"{k}. {tier}\n{plab}", ha="center", va="center",
             fontsize=7.2, color=col, linespacing=1.3)

# non-retrieved vectors scattered outside the cone
other = np.deg2rad([-70, -110, -150, 160, 200, 235, 265, 300, 330])
for a in other:
    col = TEAL if rng.random() < 0.4 else BLUE
    axC.scatter([np.cos(a)], [np.sin(a)], s=34, c=GRAY, alpha=0.55,
                zorder=3, marker="s" if col == TEAL else "o")

axC.set_title("C · One query, one ranking\nboth tiers in the same cone",
              fontsize=10.5, color=NAVY, pad=18)
axC.set_xlim(-1.55, 1.55); axC.set_ylim(-1.5, 1.45)

plt.savefig("/home/claude/work/similarity_geometry.png")
print("figure written")

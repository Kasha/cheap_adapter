"""Fixed-layout versions of the four discussion figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

NAVY, TEAL, BLUE, GRAY, AMBER, RED = ("#1a1a2e", "#0f766e", "#1a5276",
                                      "#95a5a6", "#b45309", "#c0392b")
plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})
rng = np.random.default_rng(0)

# ================================================================
# FIG 1 (v2): geometry vocabulary
# ================================================================
fig = plt.figure(figsize=(13.6, 11.2))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.05, 0.80],
                      hspace=0.50, wspace=0.32)

# ---- A: permutation symmetry ----
ax = fig.add_subplot(gs[0, 0:2])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.14)
ax.text(0.5, 1.10, "Permutation symmetry — why raw weights cannot be "
        "compared", ha="center", fontsize=11, color=NAVY)

def net(x0, order, col, label):
    yin = np.linspace(0.26, 0.66, 4)
    yh = np.linspace(0.12, 0.80, 6)
    yout = np.linspace(0.32, 0.60, 3)
    for y in yin:
        ax.add_patch(plt.Circle((x0, y), 0.011, fc=GRAY, ec="none"))
    for k, y in enumerate(yh):
        ax.add_patch(plt.Circle((x0 + 0.10, y), 0.015, fc=col, ec="none"))
        ax.text(x0 + 0.10, y, str(order[k]), ha="center", va="center",
                fontsize=6, color="white")
    for y in yout:
        ax.add_patch(plt.Circle((x0 + 0.20, y), 0.011, fc=NAVY, ec="none"))
    for a in yin:
        for b in yh:
            ax.plot([x0, x0 + 0.10], [a, b], c=col, lw=0.3, alpha=0.35)
    for a in yh:
        for b in yout:
            ax.plot([x0 + 0.10, x0 + 0.20], [a, b], c=NAVY, lw=0.3,
                    alpha=0.3)
    ax.text(x0 + 0.10, 0.92, label, ha="center", fontsize=7.8, color=col,
            linespacing=1.3)

net(0.03, [1, 2, 3, 4, 5, 6], BLUE, "network A")
net(0.42, [4, 1, 6, 2, 5, 3], TEAL,
    "network A'  (hidden units\npermuted + rewired)")
ax.annotate("", xy=(0.40, 0.46), xytext=(0.27, 0.46),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=AMBER))
ax.text(0.335, 0.52, "permute", ha="center", fontsize=7.4, color=AMBER)

ax.add_patch(mp.FancyBboxPatch((0.68, 0.06), 0.31, 0.80,
                               boxstyle="round,pad=0.02", fc="#f7f8fa",
                               ec="#ccd"))
ax.text(0.835, 0.78, "measured", ha="center", fontsize=8.4, color=NAVY,
        weight="bold")
ax.text(0.835, 0.58, "max |out_A − out_A'| = 0.0\n(bit-identical function)",
        ha="center", va="center", fontsize=8, color=TEAL, linespacing=1.6)
ax.text(0.835, 0.36, "cosine(W1, W1') = −0.125\n(unrelated weights)",
        ha="center", va="center", fontsize=8, color=RED, linespacing=1.6)
ax.text(0.835, 0.16, "1000-neuron layer:\n1000! ≈ 10^2568 equivalents",
        ha="center", va="center", fontsize=7.2, color=GRAY,
        linespacing=1.5)

# ---- B: L2 / cosine ----
ax = fig.add_subplot(gs[0, 2])
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-0.4, 2.7); ax.set_ylim(-1.15, 2.4)
ax.text(1.15, 2.28, "L2 normalization → dot = cosine", ha="center",
        fontsize=10, color=NAVY)
circ = plt.Circle((0, 0), 1, fill=False, ec="#cfd6d8", lw=1.2)
ax.add_patch(circ)
for r, deg, col, lab in [(2.4, 35, BLUE, "u (long)"),
                         (0.8, 62, TEAL, "v (short)")]:
    a = np.deg2rad(deg)
    ax.annotate("", xy=(r * np.cos(a), r * np.sin(a)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=col,
                                alpha=0.45))
    ax.annotate("", xy=(np.cos(a), np.sin(a)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col))
    ax.text(1.18 * np.cos(a), 1.18 * np.sin(a) + 0.06, lab, fontsize=7.4,
            color=col)
th = np.linspace(np.deg2rad(35), np.deg2rad(62), 30)
ax.plot(0.42 * np.cos(th), 0.42 * np.sin(th), c=NAVY, lw=1.2)
ax.text(0.52, 0.30, "θ", fontsize=10, color=NAVY)
ax.text(1.15, -0.55,
        "raw dot = 31.92   (|u|=83.1, |v|=10.8)\n"
        "cosine  =  0.0355\n"
        "dot after L2 = 0.0355  ← identical",
        ha="center", fontsize=7.6, color="#333", linespacing=1.8)

# ---- C: anisotropy spectra ----
ax = fig.add_subplot(gs[1, 0])
d = 64
X = rng.standard_normal((4000, d))
Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
scale = np.exp(np.linspace(0.9, -0.9, d))
Ya = (X @ Q) * scale
ev_x = np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1]
ev_y = np.sort(np.linalg.eigvalsh(np.cov(Ya.T)))[::-1]
ax.plot(ev_x, c=BLUE, lw=1.8, label="isotropic (eff. dim 62.9)")
ax.plot(ev_y, c=RED, lw=1.8, label="anisotropic (eff. dim 32.9)")
ax.set_xlabel("principal direction", fontsize=8)
ax.set_ylabel("variance (eigenvalue)", fontsize=8)
ax.set_title("Anisotropy = uneven variance across directions",
             fontsize=9.6, color=NAVY)
ax.legend(fontsize=6.9, frameon=False)
ax.tick_params(labelsize=7)

# ---- D: procrustes vs ridge ----
ax = fig.add_subplot(gs[1, 1])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Which map undoes which difference", fontsize=9.6,
             color=NAVY)
ax.text(0.42, 0.86, "Procrustes\n(rotation)", ha="center", fontsize=7.6,
        color=NAVY, linespacing=1.3)
ax.text(0.80, 0.86, "Ridge\n(full linear)", ha="center", fontsize=7.6,
        color=NAVY, linespacing=1.3)
rows = [("differ by\nROTATION only", "1.000", TEAL),
        ("rotation +\nANISOTROPY", "0.780", RED)]
for k, (lab, p, pc) in enumerate(rows):
    y = 0.60 - k * 0.30
    ax.text(0.02, y, lab, fontsize=7.6, color="#333", va="center",
            linespacing=1.35)
    ax.add_patch(mp.FancyBboxPatch((0.33, y - 0.08), 0.17, 0.16,
                                   boxstyle="round,pad=0.01", fc=pc,
                                   ec="none"))
    ax.text(0.415, y, p, ha="center", va="center", fontsize=9.5,
            color="white", weight="bold")
    ax.add_patch(mp.FancyBboxPatch((0.71, y - 0.08), 0.17, 0.16,
                                   boxstyle="round,pad=0.01", fc=TEAL,
                                   ec="none"))
    ax.text(0.795, y, "1.000", ha="center", va="center", fontsize=9.5,
            color="white", weight="bold")
ax.text(0.5, 0.035, "a rotation has no parameters left to rescale axes",
        ha="center", fontsize=7.3, color=RED)

# ---- E: SVD decomposition ----
ax = fig.add_subplot(gs[1, 2])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Any linear map = rotate · scale · rotate (SVD)",
             fontsize=9.6, color=NAVY)
parts = [("ROTATE  (V\u1d40)", "spins axes", TEAL, 0.76),
         ("SCALE  (S)", "fixes anisotropy mismatch", AMBER, 0.56),
         ("ROTATE  (U)", "into the target frame", TEAL, 0.36)]
for name, sub, col, y in parts:
    ax.add_patch(mp.FancyBboxPatch((0.05, y - 0.07), 0.90, 0.14,
                                   boxstyle="round,pad=0.012", fc=col,
                                   ec="none"))
    ax.text(0.10, y, name, fontsize=8, color="white", weight="bold",
            va="center")
    ax.text(0.52, y, sub, fontsize=7.2, color="white", va="center")
ax.text(0.5, 0.13,
        "Procrustes keeps only the rotations (discards S);\n"
        "ridge keeps all three — hence 0.038 vs 0.592 measured.",
        ha="center", fontsize=7.3, color="#333", linespacing=1.5)

# ---- F: bottom cards ----
ax = fig.add_subplot(gs[2, :])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.96, "Where each term shows up in the measured results",
        ha="center", fontsize=10.5, color=NAVY)
cards = [
    ("EMBEDDING / REPRESENTATION", BLUE,
     "768-d LLM hidden states (Exp A)\n512-d MobileCLIP image vectors\n"
     "768-d SigLIP image + text vectors"),
    ("COSINE + L2 NORM", TEAL,
     "Adapter: R\u00b2 = 0.592 yet cosine = 0.900.\n"
     "R\u00b2 penalizes magnitude; L2 discards it;\n"
     "the index only asks about angle."),
    ("ANISOTROPY (measured)", RED,
     "Procrustes R\u00b2 = 0.038\nRidge R\u00b2 = 0.592\n"
     "The spaces differ in spread,\nnot merely orientation."),
]
for k, (title, col, body) in enumerate(cards):
    x = 0.005 + k * 0.335
    ax.add_patch(mp.FancyBboxPatch((x, 0.04), 0.315, 0.78,
                                   boxstyle="round,pad=0.012", fc="white",
                                   ec=col, lw=1.3))
    ax.text(x + 0.157, 0.70, title, ha="center", fontsize=8.2, color=col,
            weight="bold")
    ax.text(x + 0.157, 0.38, body, ha="center", va="center", fontsize=7.3,
            color="#333", linespacing=1.6)

plt.savefig("/home/claude/work/geometry_vocabulary_v2.png")
plt.close()

# ================================================================
# FIG 2 (v2): similarity geometry (panel C label fix)
# ================================================================
def l2(V):
    return V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)

N = 60
Z = rng.standard_normal((N, 24))
sig = l2(np.tanh(Z @ rng.standard_normal((24, 768))))
perp = rng.standard_normal((N, 768))
perp -= (perp * sig).sum(1, keepdims=True) * sig
perp = l2(perp)
mob_ad = l2(0.9 * sig + np.sqrt(1 - 0.81) * perp)
raw512 = l2(rng.standard_normal((N, 512)))
mob_raw = np.zeros((N, 768)); mob_raw[:, :512] = raw512 * 0.55
mob_raw = l2(mob_raw + 0.02 * rng.standard_normal((N, 768)))

def project(*arrays):
    U = np.vstack(arrays); U = U - U.mean(0)
    _, _, Vt = np.linalg.svd(U, full_matrices=False)
    P = U @ Vt[:2].T
    out, i = [], 0
    for a in arrays:
        out.append(P[i:i + len(a)]); i += len(a)
    return out

fig = plt.figure(figsize=(13.8, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.12], wspace=0.24)

axA = fig.add_subplot(gs[0])
pa_m, pa_s = project(mob_raw, sig)
axA.scatter(*pa_s.T, s=26, c=BLUE, label="server (SigLIP native)",
            zorder=3)
axA.scatter(*pa_m.T, s=26, c=RED, marker="s",
            label="phone (raw MobileCLIP)", zorder=3)
for i in range(0, N, 6):
    axA.plot([pa_m[i, 0], pa_s[i, 0]], [pa_m[i, 1], pa_s[i, 1]], c=GRAY,
             lw=0.7, alpha=0.65, zorder=1)
axA.set_title("A · Without the adapter: two disjoint regions\n"
              "same-image cosine ≈ 0.00 · R@1 = 0.001 (chance)",
              fontsize=9.6, color=NAVY)
axA.legend(fontsize=7.2, loc="upper center", frameon=False,
           bbox_to_anchor=(0.5, -0.02))
axA.set_xticks([]); axA.set_yticks([])

axB = fig.add_subplot(gs[1])
pb_m, pb_s = project(mob_ad, sig)
axB.scatter(*pb_s.T, s=26, c=BLUE, label="server (SigLIP native)",
            zorder=3)
axB.scatter(*pb_m.T, s=26, c=TEAL, marker="s",
            label="phone (adapted: W·v)", zorder=3)
for i in range(0, N, 3):
    axB.plot([pb_m[i, 0], pb_s[i, 0]], [pb_m[i, 1], pb_s[i, 1]], c=GRAY,
             lw=0.7, alpha=0.8, zorder=1)
axB.set_title("B · With the 0.79 MB adapter: one space\n"
              "cosine = 0.900 · R@5 = 95% of ceiling",
              fontsize=9.6, color=NAVY)
axB.legend(fontsize=7.2, loc="upper center", frameon=False,
           bbox_to_anchor=(0.5, -0.02))
axB.set_xticks([]); axB.set_yticks([])

axC = fig.add_subplot(gs[2])
axC.set_aspect("equal"); axC.axis("off")
axC.set_xlim(-1.75, 1.75); axC.set_ylim(-1.62, 1.85)
axC.set_title("C · One query, one ranking — both tiers in one cone",
              fontsize=9.6, color=NAVY, pad=4)
axC.add_patch(plt.Circle((0, 0), 1, fill=False, ec="#cfd6d8", lw=1.2))
qa = np.deg2rad(90)
axC.annotate("", xy=(np.cos(qa), np.sin(qa)), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color=NAVY))
axC.text(0, 1.62, 'query: "singing red bird"  (SigLIP text tower)',
         ha="center", fontsize=7.6, color=NAVY)
half = np.deg2rad(30)
th = np.linspace(qa - half, qa + half, 60)
axC.fill(np.concatenate([[0], np.cos(th)]),
         np.concatenate([[0], np.sin(th)]), color="#f6e58d", alpha=0.35,
         zorder=0)
axC.text(0, 0.34, "top-k cone", ha="center", fontsize=7.2,
         color="#8a7a1f")
results = [(14, "MOBILE", TEAL, "P=91.3%", 1.42),
           (34, "SERVER", BLUE, "P=5.0%", 1.46),
           (-26, "MOBILE", TEAL, "P=1.0%", 1.42)]
for k, (deg, tier, col, plab, rr) in enumerate(results, 1):
    a = qa + np.deg2rad(deg)
    x, y = np.cos(a), np.sin(a)
    axC.scatter([x], [y], s=70, c=col, zorder=4,
                marker="s" if tier == "MOBILE" else "o")
    axC.plot([0, x], [0, y], c=col, lw=1.0, alpha=0.55, zorder=2)
    axC.text(rr * x, rr * y, f"{k}. {tier}\n{plab}", ha="center",
             va="center", fontsize=7.0, color=col, linespacing=1.3)
for a in np.deg2rad([-70, -110, -150, 160, 205, 240, 270, 300, 330]):
    axC.scatter([np.cos(a)], [np.sin(a)], s=32, c=GRAY, alpha=0.55,
                zorder=3, marker="s" if rng.random() < 0.4 else "o")
axC.text(0, -1.45, "unit sphere: every vector L2-normalized →\n"
                   "similarity is purely angular",
         ha="center", fontsize=7.2, color=GRAY, linespacing=1.4)

plt.savefig("/home/claude/work/similarity_geometry_v2.png")
plt.close()

# ================================================================
# FIG 3 (v2): pooling (annotation placement fix)
# ================================================================
fig = plt.figure(figsize=(13.4, 9.8))
gs = fig.add_gridspec(3, 3, height_ratios=[1.02, 1.0, 0.92],
                      hspace=0.60, wspace=0.34)

ax = fig.add_subplot(gs[0, :])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("From one passage to one row of the saved array",
             fontsize=11.5, color=NAVY, pad=6)

def fbox(x, y, w, h, txt, fc, tc="white", fs=7.8):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.008", fc=fc,
                                   ec="none"))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.4)

def farr(x1, x2, y, lab="", fs=7):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NAVY))
    if lab:
        ax.text((x1 + x2) / 2, y + 0.075, lab, ha="center", fontsize=fs,
                color=NAVY, linespacing=1.3)

Y = 0.40
fbox(0.005, Y, 0.13, 0.30, 'passage\n"The bird sang\nat dawn…"',
     "#5d6d7e", fs=7.4)
farr(0.14, 0.18, Y + 0.15, "tokenize\ntrunc 64")
fbox(0.185, Y, 0.12, 0.30, "token IDs\n[T], T ≤ 64", GRAY, fs=7.4)
farr(0.31, 0.35, Y + 0.15, "forward\n(no grads)")
fbox(0.355, Y - 0.08, 0.20, 0.46,
     "hidden states\n13 tensors, each [T, 768]\none vector per TOKEN\n"
     "per LAYER", BLUE, fs=7.6)
farr(0.56, 0.605, Y + 0.15, "masked\nmean over T")
fbox(0.61, Y, 0.155, 0.30, "pooled\n13 × [768]\none vector\nper layer",
     TEAL, fs=7.6)
farr(0.77, 0.815, Y + 0.15, "stack N")
fbox(0.82, Y - 0.04, 0.175, 0.38, "A_layers\n[13, 10000, 768]\n399 MB",
     NAVY, fs=7.8)
ax.text(0.455, 0.14,
        "T varies per passage → not a rectangle; and no token-level\n"
        "correspondence exists across two different tokenizers",
        ha="center", fontsize=7.2, color=BLUE, linespacing=1.4)
ax.text(0.69, 0.14, "fixed size — T is gone", ha="center", fontsize=7.2,
        color=TEAL)

# masked mean panel
ax = fig.add_subplot(gs[1, 0])
ax.set_title("Masked mean, concretely", fontsize=9.8, color=NAVY)
H = rng.standard_normal((7, 6)) * 0.6
mask = np.array([1, 1, 1, 1, 0, 0, 0])
ax.imshow(H * mask[:, None], cmap="coolwarm", vmin=-1.4, vmax=1.4,
          aspect="auto")
for i in range(7):
    ax.text(-1.2, i, "real" if mask[i] else "PAD", ha="right",
            va="center", fontsize=7, color=NAVY if mask[i] else RED)
ax.set_xticks(range(6), [f"d{i}" for i in range(6)], fontsize=6.5)
ax.set_yticks(range(7), [f"t{i}" for i in range(7)], fontsize=6.5)
ax.set_xlabel("Σ rows ÷ 4 real tokens = one [768] vector\n"
              "pads contribute 0 and are excluded from the divisor",
              fontsize=7.2, color=TEAL, linespacing=1.5)

# order-loss panel
ax = fig.add_subplot(gs[1, 1])
layers = [0, 1, 4, 8, 12]
no_pos = [1.0000, 0.9242, 0.4549, 0.1432, 0.0485]
with_pos = [1.0000, 0.9385, 0.5653, 0.3210, 0.2466]
ax.plot(layers, with_pos, "o-", c=TEAL, lw=1.8, ms=5,
        label="with positional enc.")
ax.plot(layers, no_pos, "s--", c=GRAY, lw=1.5, ms=4,
        label="without pos. enc.")
ax.axhline(1.0, c=RED, lw=1.0, ls=":", alpha=0.8)
ax.scatter([0], [1.0], s=90, facecolors="none", edgecolors=RED, lw=1.6,
           zorder=5)
ax.annotate("layer 0: EXACTLY 1.0\n(a theorem, not a toy result)",
            xy=(0, 1.0), xytext=(3.2, 0.86), fontsize=6.9, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
ax.set_xlabel("layer", fontsize=8)
ax.set_ylabel('cos("dog bites man",\n"man bites dog")', fontsize=7.2)
ax.set_title("Word order after pooling (toy causal model)",
             fontsize=9.4, color=NAVY)
ax.legend(fontsize=6.8, frameon=False, loc="upper right")
ax.set_ylim(-0.05, 1.14)
ax.tick_params(labelsize=7)

# one-directional bias panel
ax = fig.add_subplot(gs[1, 2])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Why the bias is one-directional", fontsize=9.8, color=NAVY)
ax.add_patch(mp.FancyBboxPatch((0.02, 0.54), 0.96, 0.36,
                               boxstyle="round,pad=0.02", fc="#eef6f5",
                               ec="#b6d9d4"))
ax.text(0.5, 0.72, "Pooling applies the SAME lossy step to\nboth models, "
        "and blurring can only DESTROY\ndistinctions — never invent "
        "shared ones.", ha="center", va="center", fontsize=7.8,
        color="#14544c", linespacing=1.6)
ax.add_patch(mp.FancyBboxPatch((0.02, 0.08), 0.96, 0.36,
                               boxstyle="round,pad=0.02", fc="#fdf6e3",
                               ec="#d5c9a0"))
ax.text(0.5, 0.26, "Every similarity number in Experiment A\nis therefore "
        "a LOWER BOUND:\ntrue convergence ≥ measured.", ha="center",
        va="center", fontsize=7.8, color="#6b5a2a", linespacing=1.6)

# bottom: three reasons
ax = fig.add_subplot(gs[2, :])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.97, "Why pool at all — three reasons, one decisive",
        ha="center", fontsize=10.3, color=NAVY)
cards = [
    ("1 · ALIGNMENT (decisive)", TEAL,
     "Different tokenizers → different token\nsequences of different "
     "lengths. Token-to-token\ncorrespondence does not exist; "
     "passage-to-\npassage does. Pooling moves the comparison\nto the "
     "level where alignment is real."),
    ("2 · FIXED SHAPE", BLUE,
     "T varies per passage, so per-token states\ncannot form a "
     "rectangle. CKA and ridge\nboth need [N, 768] matrices with\n"
     "row i = passage i in every model."),
    ("3 · SIZE", GRAY,
     "pooled:    3 × 399 MB = 1.20 GB\nper-token: 3 × 25.6 GB = 76.7 GB\n"
     "(T = 64 → 64× more)\n1.2 GB fits Drive and loads in seconds."),
]
for k, (title, col, body) in enumerate(cards):
    x = 0.005 + k * 0.335
    ax.add_patch(mp.FancyBboxPatch((x, 0.03), 0.315, 0.82,
                                   boxstyle="round,pad=0.012", fc="white",
                                   ec=col, lw=1.3))
    ax.text(x + 0.157, 0.74, title, ha="center", fontsize=8.3, color=col,
            weight="bold")
    ax.text(x + 0.157, 0.40, body, ha="center", va="center", fontsize=7.2,
            color="#333", linespacing=1.6)

plt.savefig("/home/claude/work/pooling_explained_v2.png")
plt.close()

# ================================================================
# FIG 4 (v2): layer correspondence (text wrap fix)
# ================================================================
fig = plt.figure(figsize=(13.2, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.12], hspace=0.46,
                      wspace=0.30)

def stack2(ax, x, n, w, color, label):
    top, bot = 0.90, 0.08
    h = (top - bot) / n
    ys = []
    for i in range(n):
        y = bot + i * h
        ax.add_patch(mp.Rectangle((x, y + h * 0.12), w, h * 0.76,
                                  fc=color, ec="none", alpha=0.9))
        ys.append(y + h * 0.5)
    ax.text(x + w / 2, top + 0.05, label, ha="center", fontsize=8.4,
            color=color, weight="bold", linespacing=1.2)
    return ys

titles = [("Equal depth (this project)\nindex = proportional: no choice",
           None),
          ("Unequal depth · PROPORTIONAL\nthe mapping is an ASSUMPTION",
           AMBER),
          ("Unequal depth · DATA-DRIVEN\ncorrespondence is a FINDING",
           TEAL)]
for col_i, (title, lc) in enumerate(titles):
    ax = fig.add_subplot(gs[0, col_i])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.10)
    ax.text(0.5, 1.06, title, ha="center", fontsize=9.2, color=NAVY,
            linespacing=1.35)
    if col_i == 0:
        ya = stack2(ax, 0.08, 12, 0.15, BLUE, "GPT-2\n12")
        yb = stack2(ax, 0.77, 12, 0.15, TEAL, "Pythia\n12")
        for i in range(12):
            ax.plot([0.23, 0.77], [ya[i], yb[i]], c=GRAY, lw=0.9,
                    alpha=0.75)
    else:
        ya = stack2(ax, 0.08, 12, 0.15, BLUE, "A\n12")
        yb = stack2(ax, 0.77, 24, 0.15, TEAL, "B\n24")
        r2_ = np.random.default_rng(3)
        for i in range(12):
            j = int(round(i * 23 / 11))
            if col_i == 2:
                j = int(np.clip(j + r2_.integers(-3, 4), 0, 23))
            ax.plot([0.23, 0.77], [ya[i], yb[j]], c=lc, lw=0.95,
                    alpha=0.85)

ax = fig.add_subplot(gs[1, 0:2])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def dbox(x, y, w, h, txt, fc, tc="white", fs=8):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.012", fc=fc,
                                   ec="none"))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.4)

dbox(0.01, 0.66, 0.21, 0.22, "compute CKA for ALL\nlayer pairs\n"
     "(rectangular matrix)", NAVY)
dbox(0.30, 0.66, 0.19, 0.22, "equal depth?", "#5d6d7e")
ax.annotate("", xy=(0.30, 0.77), xytext=(0.22, 0.77),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NAVY))
dbox(0.57, 0.80, 0.42, 0.15,
     "YES → read the diagonal (no mapping choice exists)", BLUE, fs=7.8)
dbox(0.57, 0.60, 0.42, 0.15,
     "NO → take each row's argmax: measured, not assumed", TEAL, fs=7.8)
ax.annotate("", xy=(0.57, 0.875), xytext=(0.49, 0.79),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=BLUE))
ax.annotate("", xy=(0.57, 0.675), xytext=(0.49, 0.745),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=TEAL))
dbox(0.30, 0.34, 0.44, 0.16,
     "AVOID: pre-committing to one proportional mapping\n"
     "and reporting only those cells", AMBER, fs=7.6)
ax.annotate("", xy=(0.40, 0.50), xytext=(0.40, 0.66),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=AMBER))
ax.text(0.80, 0.415, "a researcher degree of freedom:\ndifferent "
        "defensible mappings\ngive different diagonals", fontsize=7.2,
        color=AMBER, va="center", linespacing=1.4)
ax.text(0.01, 0.20, "Ranking:  data-driven (argmax)  ≥  index (when "
        "depths match)", fontsize=8.6, color=NAVY, weight="bold")
ax.text(0.115, 0.125, ">  assumed proportional (when they do not)",
        fontsize=8.6, color=NAVY, weight="bold")
ax.text(0.01, 0.028,
        "In this project the hot diagonal is a RESULT, not a "
        "construction: all 13×13 pairs were computed and the diagonal "
        "emerged.", fontsize=7.6, color="#444")

ax = fig.add_subplot(gs[1, 2])
L = 13
base = np.random.default_rng(11).random((L, L)) * 0.12
i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
M = np.clip(base + 0.62 * np.exp(-((i - j) ** 2) / 6.0), 0, 1)
M[0] *= 0.55; M[:, 0] *= 0.55; M[-1] *= 0.7; M[:, -1] *= 0.7
im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, origin="lower")
ax.plot(np.arange(L), M.argmax(1), c="#39d0c6", lw=1.4, ls="--",
        label="row argmax")
ax.set_xlabel("Pythia layer", fontsize=8)
ax.set_ylabel("GPT-2 layer", fontsize=8)
ax.set_title("The full matrix (illustrative)", fontsize=9.4, color=NAVY)
ax.legend(fontsize=7, loc="lower right", frameon=False,
          labelcolor="#39d0c6")
ax.tick_params(labelsize=7)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
    labelsize=7)

plt.savefig("/home/claude/work/layer_correspondence_v2.png")
plt.close()
print("all four v2 figures written")

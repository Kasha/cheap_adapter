"""Magnitude vs direction, and predicted-vs-true — with computed numbers."""
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
MONO = {"family": "DejaVu Sans Mono"}

fig = plt.figure(figsize=(13.6, 10.6))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)

# ---------------- Panel A: magnitude vs direction ----------------
ax = fig.add_subplot(gs[0, 0])
ax.set_aspect("equal")
ax.set_title("A · Magnitude = length, direction = angle",
             fontsize=11, color=NAVY)
for vx, vy, col, lw in [(6, 8, "#9CC3D5", 5), (3, 4, BLUE, 3)]:
    ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=col))
ax.text(3.15, 3.6, "(3, 4)\n|v| = \u221a(9+16) = 5", fontsize=9,
        color=BLUE, **MONO)
ax.text(6.15, 7.6, "(6, 8)\n|v| = 10", fontsize=9, color="#6FA3BC",
        **MONO)
ax.text(0.6, 8.8, "same direction,\ndifferent magnitude",
        fontsize=9.5, color=NAVY, linespacing=1.5)
ax.set_xlim(-0.5, 10); ax.set_ylim(-0.5, 10)
ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
ax.grid(alpha=0.25)

# ---------------- Panel B: L2 normalization ----------------
ax = fig.add_subplot(gs[0, 1])
ax.set_aspect("equal")
ax.set_title("B · L2 normalization: divide by length \u2192 unit sphere",
             fontsize=11, color=NAVY)
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(th), np.sin(th), c="#cfd6d8", lw=1.4)
vecs = [((3, 4), BLUE), ((6, 8), "#6FA3BC"), ((-1.2, 0.9), TEAL),
        ((0.4, -1.6), AMBER)]
for (vx, vy), col in vecs:
    n = np.hypot(vx, vy)
    ax.annotate("", xy=(vx / 3.2, vy / 3.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color=col,
                                alpha=0.35))
    ax.annotate("", xy=(vx / n, vy / n), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=col))
ax.text(0.0, -1.55,
        "v\u0302 = v / |v|      \u2192  every |v\u0302| = 1\n"
        "(3,4)/5 = (0.6, 0.8)      (6,8)/10 = (0.6, 0.8)  \u2190 identical",
        ha="center", fontsize=9, color="#333", **MONO)
ax.text(0.0, 1.72, "after normalization, only ANGLE can differ",
        ha="center", fontsize=9.5, color=TEAL)
ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.0, 2.05)
ax.axis("off")

# ---------------- Panel C: predicted vs true ----------------
ax = fig.add_subplot(gs[1, 0])
ax.set_aspect("equal")
ax.set_title("C · One held-out image: TRUE vs PREDICTED",
             fontsize=11, color=NAVY)
y_true = np.array([0.6, 0.8])                    # SigLIP's own vector
u_pred = 0.85 * y_true                           # right direction, short
ax.annotate("", xy=y_true, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", lw=3, color=BLUE))
ax.annotate("", xy=u_pred, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", lw=3, color=AMBER))
ax.plot([u_pred[0], y_true[0]], [u_pred[1], y_true[1]], c=RED, lw=2,
        ls="--")
ax.text(0.62, 0.83, "y  (TRUE)\nSigLIP's own embedding\nof this photo",
        fontsize=8.6, color=BLUE, linespacing=1.4)
ax.text(0.30, 0.32, "u = v\u00b7W\n(PREDICTED)\nfrom the phone",
        fontsize=8.6, color=AMBER, linespacing=1.4)
ax.text(0.66, 0.60, "residual\ny \u2212 u", fontsize=8.4, color=RED)
th2 = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(th2), np.sin(th2), c="#e3e8eb", lw=1)
ax.set_xlim(-0.1, 1.35); ax.set_ylim(-0.1, 1.25)
ax.axis("off")
ax.text(0.02, -0.06,
        "y = (0.60, 0.80)   u = (0.51, 0.68)\n"
        "same direction \u2192 cos = 1.000\n"
        "but |u| = 0.85 \u2192 squared error = 0.0225\n"
        "R\u00b2 punished \u00b7 cosine perfect",
        fontsize=9, color="#333", **MONO, va="top")

# ---------------- Panel D: the full flow with both metrics ----------------
ax = fig.add_subplot(gs[1, 1])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("D · Where each metric lives in the pipeline",
             fontsize=11, color=NAVY)

def bx(x, y, w, h, txt, fc, tc="white", fs=8):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.008", fc=fc,
                                   ec="none"))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=tc, linespacing=1.45)

bx(0.00, 0.72, 0.20, 0.18, "phone vector\nv [512]", TEAL)
bx(0.28, 0.72, 0.20, 0.18, "PREDICTED\nu = v\u00b7W [768]", AMBER)
bx(0.28, 0.44, 0.20, 0.18, "TRUE\ny = SigLIP(img)\n[768]", BLUE)
ax.annotate("", xy=(0.28, 0.81), xytext=(0.20, 0.81),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NAVY))
ax.annotate("", xy=(0.55, 0.66), xytext=(0.49, 0.78),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color=RED))
ax.annotate("", xy=(0.55, 0.60), xytext=(0.49, 0.56),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color=RED))
bx(0.56, 0.53, 0.42, 0.20,
   "R\u00b2 = 0.592   compares u to y\nNUMBER BY NUMBER, all 768\n"
   "punishes length + direction errors", "#FBF0EF", tc=RED, fs=7.8)

ax.annotate("", xy=(0.38, 0.36), xytext=(0.38, 0.44),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NAVY))
bx(0.28, 0.16, 0.20, 0.18,
   "NORMALIZE\nu\u0302, y\u0302  (length 1)", GRAY)
ax.annotate("", xy=(0.56, 0.25), xytext=(0.49, 0.25),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NAVY))
bx(0.56, 0.16, 0.42, 0.20,
   "cosine = 0.900   compares u\u0302 to y\u0302\nANGLE ONLY \u2014 "
   "length already removed\nthis is what the index computes", "#EFF6F2",
   tc=TEAL, fs=7.8)
ax.text(0.5, 0.045,
        "same u, same y \u2014 two questions:  with magnitude (R\u00b2) "
        "or without it (cosine).\nThe search index normalizes first, so "
        "only the second question matters in production.",
        ha="center", fontsize=8.6, color=NAVY, linespacing=1.6)

plt.savefig("/home/claude/work/magnitude_explained.png")
print("figure written")

# verify the panel-C arithmetic honestly
y = np.array([0.6, 0.8]); u = 0.85*y
print("check: |y|=", round(float(np.linalg.norm(y)),3),
      " |u|=", round(float(np.linalg.norm(u)),3),
      " cos=", round(float(u@y/(np.linalg.norm(u)*np.linalg.norm(y))),3),
      " sq.err=", round(float(((y-u)**2).sum()),4))

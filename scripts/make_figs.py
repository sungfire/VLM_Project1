"""Figures for the report: pipeline diagram, accuracy-token plot, token histograms."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_eval import load_jsonl  # noqa: E402

S = json.load(open("report/stats.json"))
R = S["runs"]
os.makedirs("report/figs", exist_ok=True)
C = {"base": "#7a7a7a", "tile": "#b07aa1", "dev": "#e8a33d", "s1": "#2a6fdb", "s2": "#18a058"}

# ---------------------------------------------------------------- 1. pipeline
fig, ax = plt.subplots(figsize=(10, 3.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)


def arrow(x1, y1, x2, y2, t="", dx=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.9))
    if t:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + 1.2, t, ha="center", fontsize=6.5)


box(0.5, 11, 14, 8, "4032x4032 image\n+ question + options", "#f2f2f2")
box(19, 10, 21, 10, "[Call 1] Joint probe\nglobal view, 1024 tok\n-> letter (p1) then bbox_2d\n(stops after letter if confident)", "#dbe8fb")
box(46, 21, 15, 7, "Answer = argmax p1", "#e3f4e8")
box(46, 9, 15, 8.5, "[Call 2] text-only\ntype classifier\n(0 visual tokens)", "#fdf0d9")
box(67, 17.5, 32, 9.5, "Type A (local detail): [Call 3] crop around bbox\n(x2 expand, min side 25%), <=1024 tok\nS1: up-scale <=2x", "#dbe8fb")
box(67, 2, 32, 10.5, "Type B (relation / counting / layout):\nS1: [Call 3] global view 4096 tok\nS2: keep probe answer (no call)", "#dbe8fb")
arrow(15, 15, 19, 15)
arrow(40, 17, 46, 24.5, "max p1 >= tau", dx=-4.5)
arrow(40, 13, 46, 13, "else")
arrow(61, 14.5, 67, 22, "P(A) >= 0.3", dx=-4.0)
arrow(61, 11.5, 67, 7.5, "else")
ax.text(83, 0.0, "final = argmax(alpha log p1 + log p_esc)", ha="center", fontsize=7, style="italic")
plt.tight_layout(); plt.savefig("report/figs/pipeline.png", dpi=220); plt.close()

# ---------------------------------------------------------------- 2. accuracy vs tokens
fig, ax = plt.subplots(figsize=(6.4, 3.9))
gl = [R[f"global_{t}"] for t in (256, 512, 1024, 2048, 4096)]
ax.plot([g["mean_visual"] for g in gl], [g["acc"] for g in gl], "o-", c=C["base"], label="Global resize (baseline)")
tl = [R[k] for k in ("tiling_2048", "tiling_4096") if k in R]
if tl:
    ax.plot([g["mean_visual"] for g in tl], [g["acc"] for g in tl], "s", c=C["tile"], ms=7, label="Global + 2x2 tiles (baseline)")
t2, t1 = S["tau_sweep_2k"], S["tau_sweep_4k"]
ax.plot([r["tok"] for r in t2], [r["acc"] for r in t2], "--", c=C["s2"], lw=1, label="S2 policy, tau sweep (offline)")
ax.plot([r["tok"] for r in t1], [r["acc"] for r in t1], "--", c=C["s1"], lw=1, label="S1 policy, tau sweep (offline)")
ax.plot(R["success2_2k"]["mean_visual"], R["success2_2k"]["acc"], "*", c=C["s2"], ms=15, label="Success 2 (final)")
ax.plot(R["success1_4k"]["mean_visual"], R["success1_4k"]["acc"], "*", c=C["s1"], ms=15, label="Success 1 (final)")
for b, t, c in [(2048, 68, C["s2"]), (4096, 71, C["s1"])]:
    ax.axvline(b, c=c, lw=0.6, ls=":")
    ax.plot([0, b], [t, t], c=c, lw=0.6, ls=":")
ax.set_xlabel("Mean cumulative visual tokens per instance"); ax.set_ylabel("HR-Bench 4K accuracy (%)")
ax.set_xlim(0, 4400); ax.set_ylim(52, 77); ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("report/figs/acc_tokens.png", dpi=220); plt.close()

# ---------------------------------------------------------------- 3. token histograms
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, name, col, b in [(axes[0], "success1_4k", C["s1"], 4096), (axes[1], "success2_2k", C["s2"], 2048)]:
    toks = [r["n_visual"] for r in load_jsonl(f"results/{name}.jsonl")]
    ax.hist(toks, bins=range(0, 5400, 200), color=col, alpha=0.85)
    m = sum(toks) / len(toks)
    ax.axvline(m, c="k", lw=1); ax.axvline(b, c="r", lw=0.8, ls="--")
    ax.set_title(f"{'Success 1' if name.startswith('success1') else 'Success 2'}: mean {m:.0f} (budget {b})", fontsize=8)
    ax.set_xlabel("visual tokens per instance", fontsize=8); ax.tick_params(labelsize=7)
axes[0].set_ylabel("# instances", fontsize=8)
plt.tight_layout(); plt.savefig("report/figs/token_hist.png", dpi=220); plt.close()
print("figs done")

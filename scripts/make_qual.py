"""Qualitative figure: full image with probe box (red) and crop region (yellow), plus the crop."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp
from PIL import Image
from src.run_eval import load_jsonl
from src.data import load_hrbench
from src.imaging import box_rel_to_abs
I = {i.idx: i for i in load_hrbench()}
rows = {r["idx"]: r for r in load_jsonl("results/success2_2k.jsonl")}
cases = [int(x) for x in sys.argv[2:]]
fig, axes = plt.subplots(2, len(cases), figsize=(3.2 * len(cases), 6.2))
for k, idx in enumerate(cases):
    r, it = rows[idx], I[idx]
    img = Image.open(it.image_path).convert("RGB"); w, h = img.size
    p = r["calls"][0]; z = next(c for c in r["calls"] if c["tag"] == "single")
    ax = axes[0, k]; ax.imshow(img.resize((w // 6, h // 6))); ax.axis("off")
    for b in p["boxes"][:1]:
        x1, y1, x2, y2 = [v / 6 for v in box_rel_to_abs(b, w, h)]
        ax.add_patch(mp.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
    c = [v / 6 for v in z["crop_box"]]
    ax.add_patch(mp.Rectangle((c[0], c[1]), c[2] - c[0], c[3] - c[1], fill=False, ec="yellow", lw=1.5, ls="--"))
    L = "ABCD"; p1 = L[p["probs"].index(max(p["probs"]))]
    ax.set_title(f"#{idx}: {it.question[:48]}{'...' if len(it.question) > 48 else ''}", fontsize=7)
    ax2 = axes[1, k]; ax2.imshow(img.crop(z["crop_box"])); ax2.axis("off")
    ok = "correct" if r["correct"] else "wrong"
    ax2.set_title(f"GT {r['answer']}={it.options[r['answer']][:18]} | probe {p1} ({max(p['probs']):.2f})\n"
                  f"zoom {L[z['probs'].index(max(z['probs']))]} -> final {r['pred']} ({ok})", fontsize=7)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=200)

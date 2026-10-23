"""HR-Bench 4K loading.

The parquet holds 800 rows = 200 base questions x 4 cyclic option orderings.
Rows 4g..4g+3 share the same image, so images are decoded once into
data/images/<group>.jpg. The `category` and `answer` columns are used only for
scoring/analysis, never at inference time.
"""
import base64
import os
from dataclasses import dataclass

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "data", "hr_bench_4k.parquet")
IMG_DIR = os.path.join(ROOT, "data", "images")

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Instance:
    idx: int
    group: int
    question: str
    options: dict  # {"A": ..., "B": ..., "C": ..., "D": ...}
    answer: str  # scoring only
    category: str  # scoring only ("single" / "cross")
    image_path: str


def load_hrbench(parquet=PARQUET):
    df = pd.read_parquet(parquet)
    os.makedirs(IMG_DIR, exist_ok=True)
    out = []
    for i, r in df.iterrows():
        g = int(r["index"]) // 4
        path = os.path.join(IMG_DIR, f"{g:03d}.jpg")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(base64.b64decode(r["image"]))
        out.append(Instance(
            idx=int(r["index"]), group=g, question=r["question"],
            options={k: r[k] for k in "ABCD"}, answer=r["answer"],
            category=r["category"], image_path=path,
        ))
    return out


class ImageCache:
    """Keeps the most recently decoded image (consecutive rows share an image).
    Pure file-I/O caching; no model outputs are shared between instances."""

    def __init__(self):
        self.path, self.img = None, None

    def get(self, path):
        if path != self.path:
            self.img = Image.open(path).convert("RGB")
            self.path = path
        return self.img

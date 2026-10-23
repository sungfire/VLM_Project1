"""Deterministic resize / crop helpers.

Qwen3-VL: patch 16, 2x2 spatial merge -> one visual token per 32x32 pixels.
All images are resized here to exact multiples of 32 (within the processor's
min/max pixel range) so the official processor keeps them unchanged and the
token count is exactly (W/32)*(H/32).
"""
import math

from PIL import Image

FACTOR = 32
MIN_TOKENS = 64  # processor shortest_edge=65536 px -> 64 tokens


def tokens_of(w, h):
    return (w // FACTOR) * (h // FACTOR)


def fit_grid(w, h, max_tokens, max_scale=1.0):
    """Largest (tw, th) token grid with tw*th <= max_tokens, aspect ~ w/h and
    scale <= max_scale (relative to the source pixel size)."""
    max_tokens = max(max_tokens, MIN_TOKENS)
    cap_w = max(1, round(w * max_scale / FACTOR))
    cap_h = max(1, round(h * max_scale / FACTOR))
    s = math.sqrt(max_tokens / ((w / FACTOR) * (h / FACTOR)))
    tw = max(1, min(cap_w, math.floor(w / FACTOR * s)))
    th = max(1, min(cap_h, math.floor(h / FACTOR * s)))
    # greedily grow the dimension that is most under its target aspect
    while True:
        grew = False
        for dw, dh in sorted([(1, 0), (0, 1)], key=lambda d: (tw / th) / (w / h) if d[0] else (w / h) / (tw / th)):
            if (tw + dw) * (th + dh) <= max_tokens and tw + dw <= cap_w and th + dh <= cap_h:
                tw, th = tw + dw, th + dh
                grew = True
                break
        if not grew:
            break
    while tw * th < MIN_TOKENS:  # processor would upscale anyway; do it explicitly
        if tw <= th:
            tw += 1
        else:
            th += 1
    return tw, th


def resize_to_tokens(img, max_tokens, max_scale=1.0):
    tw, th = fit_grid(img.width, img.height, max_tokens, max_scale)
    return img.resize((tw * FACTOR, th * FACTOR), Image.LANCZOS)


def box_rel_to_abs(box, w, h, scale=1000.0):
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (x1 / scale * w, y1 / scale * h, x2 / scale * w, y2 / scale * h)


def expand_box(box, w, h, factor=1.0, min_side=0, square=False):
    """Expand box (abs px) about its center by `factor`, enforce min side,
    shift to stay inside the image."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(x2 - x1, 1) * factor, max(y2 - y1, 1) * factor
    bw, bh = max(bw, min_side), max(bh, min_side)
    if square:
        bw = bh = max(bw, bh)
    bw, bh = min(bw, w), min(bh, h)
    x1 = min(max(cx - bw / 2, 0), w - bw)
    y1 = min(max(cy - bh / 2, 0), h - bh)
    return (int(x1), int(y1), int(x1 + bw), int(y1 + bh))


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def tile_boxes(w, h, rows, cols, overlap=0.0):
    out = []
    tw, th = w / cols, h / rows
    ow, oh = tw * overlap, th * overlap
    for r in range(rows):
        for c in range(cols):
            out.append((int(max(0, c * tw - ow)), int(max(0, r * th - oh)),
                        int(min(w, (c + 1) * tw + ow)), int(min(h, (r + 1) * th + oh))))
    return out

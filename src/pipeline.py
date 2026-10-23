"""Inference strategies. Each returns {"pred": letter, "calls": [call logs], ...}.

All visual tokens of every Qwen call are logged in `calls` and summed per
instance by the runner (an image sent twice is counted twice).
"""
import json
import math
import re

from .imaging import (box_rel_to_abs, expand_box, resize_to_tokens, tile_boxes,
                      union_box)
from .prompts import GROUND_PROMPT, joint_prompt, mc_prompt, type_prompt
from .qwen import LETTERS


def argmax_letter(probs):
    return LETTERS[max(range(4), key=lambda i: probs[i])]


# ---------------------------------------------------------------- baselines
def global_view_call(q, inst, img, tokens, max_scale=1.0, tag="global"):
    im = resize_to_tokens(img, tokens, max_scale)
    return q.call([im], mc_prompt(inst.question, inst.options), tag=tag)


def run_global(q, inst, img, tokens, max_scale=1.0):
    r = global_view_call(q, inst, img, tokens, max_scale)
    return {"pred": argmax_letter(r["probs"]), "calls": [r]}


def run_tiling(q, inst, img, thumb_tokens, rows, cols, tile_tokens, overlap=0.0):
    """Low-res global view + uniform rows x cols tiles in one multi-image call."""
    w, h = img.size
    ims = [resize_to_tokens(img, thumb_tokens)] if thumb_tokens else []
    for b in tile_boxes(w, h, rows, cols, overlap):
        ims.append(resize_to_tokens(img.crop(b), tile_tokens))
    prefix = ("The first image is the full image; the following images are tiles of it "
              f"in row-major order ({rows}x{cols} grid).\n" if thumb_tokens else
              f"The images are tiles of one large image in row-major order ({rows}x{cols} grid).\n")
    r = q.call(ims, mc_prompt(inst.question, inst.options, prefix), tag="tiling")
    return {"pred": argmax_letter(r["probs"]), "calls": [r]}


# ---------------------------------------------------------------- cascade
_BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]')


def parse_boxes(text):
    boxes = []
    for m in _BOX_RE.finditer(text):
        b = [float(v) for v in m.groups()]
        b = [min(max(v, 0.0), 1000.0) for v in b]
        if b[2] - b[0] > 1 and b[3] - b[1] > 1:
            boxes.append(b)
    return boxes


def locate(q, img, cfg, inst):
    im = resize_to_tokens(img, cfg["locate_tokens"])
    prompt = GROUND_PROMPT.format(question=inst.question)
    r = q.call([im], prompt, mode="generate", max_new_tokens=cfg.get("locate_max_new", 160), tag="locate")
    r["boxes"] = parse_boxes(r["text"])
    return r


def make_crop_box(img, rel_boxes, cfg):
    w, h = img.size
    if not rel_boxes:
        return None
    boxes = [box_rel_to_abs(b, w, h) for b in rel_boxes[: cfg.get("max_boxes", 4)]]
    b = union_box(boxes)
    min_side = cfg.get("min_side_frac", 0.25) * min(w, h)
    return expand_box(b, w, h, factor=cfg.get("expand", 2.0), min_side=min_side)


def zoom(q, inst, img, crop_box, cfg, tag="zoom"):
    crop = img.crop(crop_box)
    cim = resize_to_tokens(crop, cfg["zoom_tokens"], cfg.get("zoom_max_scale", 1.0))
    ims, prefix = [], ""
    if cfg.get("thumb_tokens", 0):
        ims.append(resize_to_tokens(img, cfg["thumb_tokens"]))
        w, h = img.size
        rel = [round(crop_box[0] / w * 1000), round(crop_box[1] / h * 1000),
               round(crop_box[2] / w * 1000), round(crop_box[3] / h * 1000)]
        prefix = ("The first image is the full image. The second image is a zoomed-in view of "
                  f"region {rel} (x1, y1, x2, y2 in 0-1000 coordinates) of the full image.\n")
    elif cfg.get("zoom_hint", True):
        prefix = "This image is a zoomed-in crop of a larger image.\n"
    ims.append(cim)
    r = q.call(ims, mc_prompt(inst.question, inst.options, prefix), tag=tag)
    r["crop_box"] = list(crop_box)
    return r


def fuse(p1, p3, alpha):
    s = [alpha * math.log(max(a, 1e-9)) + math.log(max(b, 1e-9)) for a, b in zip(p1, p3)]
    return LETTERS[max(range(4), key=lambda i: s[i])]


def run_cascade(q, inst, img, cfg):
    """Probe (global) -> [if unsure] Locate (grounding) -> Zoom (crop) -> fuse.
    cfg["tau"] = None runs all stages for every instance (for offline policy
    simulation); otherwise stops after the probe when max p1 >= tau."""
    calls = []
    tau = cfg.get("tau")
    if cfg.get("locate_mode", "separate") == "joint":
        # probe + locate in a single call: letter first, then evidence bbox
        im = resize_to_tokens(img, cfg["probe_tokens"])
        p = q.call([im], joint_prompt(inst.question, inst.options), mode="joint",
                   max_new_tokens=cfg.get("locate_max_new", 160), tag="probe", stop_tau=tau)
        p["boxes"] = parse_boxes(p["text"])
        calls.append(p)
        p1, boxes = p["probs"], p["boxes"]
        if tau is not None and max(p1) >= tau:
            return {"pred": argmax_letter(p1), "calls": calls, "stage": "probe"}
    else:
        p = run_global(q, inst, img, cfg["probe_tokens"])["calls"][0]
        p["tag"] = "probe"
        calls.append(p)
        p1 = p["probs"]
        if tau is not None and max(p1) >= tau:
            return {"pred": argmax_letter(p1), "calls": calls, "stage": "probe"}
        loc = locate(q, img, cfg, inst)
        calls.append(loc)
        boxes = loc["boxes"]
    box = make_crop_box(img, boxes, cfg)
    if box is None:  # grounding failed -> keep probe answer
        return {"pred": argmax_letter(p1), "calls": calls, "stage": "locate_fail"}
    z = zoom(q, inst, img, box, cfg)
    calls.append(z)
    pred = fuse(p1, z["probs"], cfg.get("alpha", 0.0))
    return {"pred": pred, "calls": calls, "stage": "zoom"}


def escalate(q, inst, img, boxes, v, tag):
    """One escalation observation: either a larger global view or a zoomed crop
    around the probe's evidence box (optionally with a thumbnail)."""
    if v.get("kind") == "global":
        return global_view_call(q, inst, img, v["tokens"], tag=tag)
    box = make_crop_box(img, boxes, v)
    return zoom(q, inst, img, box, v, tag=tag) if box is not None else None


def run_routed(q, inst, img, cfg):
    """Final method.
    1. Joint probe (one call, global view of probe_tokens): option letter first
       (A-D distribution p1) then evidence bbox. Stops after the letter token if
       max p1 >= tau.
    2. Otherwise a text-only call (0 visual tokens) predicts whether the question
       needs one local detail (type A) or multi-object/layout evidence (type B).
    3. Type A -> `single` escalation (zoomed crop); type B -> `relational`
       escalation (or none). Final answer = argmax(alpha*log p1 + log p_esc).
    """
    calls = []
    tau = cfg["tau"]
    im = resize_to_tokens(img, cfg["probe_tokens"])
    p = q.call([im], joint_prompt(inst.question, inst.options), mode="joint",
               max_new_tokens=cfg.get("locate_max_new", 160), tag="probe", stop_tau=tau)
    p["boxes"] = parse_boxes(p["text"])
    calls.append(p)
    p1 = p["probs"]
    if max(p1) >= tau:
        return {"pred": argmax_letter(p1), "calls": calls, "stage": "probe"}
    ty = q.binary_call(type_prompt(inst.question, inst.options), tag="type")
    calls.append(ty)
    route = "single" if ty["p"] >= cfg["type_th"] else "relational"
    v = cfg.get(route)
    if not v:
        return {"pred": argmax_letter(p1), "calls": calls, "stage": f"{route}_keep"}
    z = escalate(q, inst, img, p["boxes"], v, tag=route)
    if z is None:
        return {"pred": argmax_letter(p1), "calls": calls, "stage": f"{route}_nobox"}
    calls.append(z)
    return {"pred": fuse(p1, z["probs"], v.get("alpha", 0.0)), "calls": calls, "stage": route}


STRATEGIES = {
    "routed": run_routed,
    "global": lambda q, inst, img, c: run_global(q, inst, img, c["tokens"], c.get("max_scale", 1.0)),
    "tiling": lambda q, inst, img, c: run_tiling(q, inst, img, c["thumb_tokens"], c["rows"], c["cols"],
                                                 c["tile_tokens"], c.get("overlap", 0.0)),
    "cascade": run_cascade,
}

"""Development tool: run several zoom-stage variants on top of an existing
force-all cascade log (re-using its logged boxes) so that zoom settings can be
compared offline with src.simulate. Only instances whose probe confidence is
below --max_conf are run (the others never escalate at any tau < max_conf).

    python -m src.zoom_sweep --log results/joint_dev.jsonl --variants configs/zoom_variants.yaml \
        --out results/joint_dev_zooms.jsonl --device cuda:0
"""
import argparse
import json
import os

import yaml

from .data import ImageCache, load_hrbench
from .pipeline import global_view_call, make_crop_box, zoom
from .qwen import Qwen
from .run_eval import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_conf", type=float, default=0.995)
    ap.add_argument("--only", nargs="*", default=None, help="variant names to run")
    args = ap.parse_args()

    variants = yaml.safe_load(open(args.variants))
    if args.only:
        variants = [v for v in variants if v["name"] in args.only]
    insts = {it.idx: it for it in load_hrbench()}
    rows = load_jsonl(args.log)
    done = {(r["idx"], r["variant"]) for r in load_jsonl(args.out)}
    q = Qwen(args.device)
    cache = ImageCache()
    with open(args.out, "a") as f:
        for r in rows:
            probe = next(c for c in r["calls"] if c["tag"] == "probe")
            if max(probe["probs"]) >= args.max_conf:
                continue
            boxes = next((c["boxes"] for c in r["calls"] if "boxes" in c), [])
            inst = insts[r["idx"]]
            img = cache.get(inst.image_path)
            for v in variants:
                if (r["idx"], v["name"]) in done:
                    continue
                if v.get("kind") == "global":
                    z = global_view_call(q, inst, img, v["tokens"], tag=v["name"])
                else:
                    box = make_crop_box(img, boxes, v)
                    if box is None:
                        continue
                    z = zoom(q, inst, img, box, v, tag=v["name"])
                f.write(json.dumps({"idx": r["idx"], "variant": v["name"], "call": z}) + "\n")
                f.flush()


if __name__ == "__main__":
    main()

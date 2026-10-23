"""Run one inference configuration over all 800 HR-Bench 4K instances.

    python -m src.run_eval --config configs/success2_2k.yaml --device cuda:0

Writes one JSON line per instance (every Qwen call with its visual tokens,
generated tokens and latency) to results/<name>.jsonl and prints a summary.
Re-running resumes from the existing file.
"""
import argparse
import json
import os
import time

import yaml

from .data import ImageCache, load_hrbench
from .pipeline import STRATEGIES
from .qwen import Qwen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(rows):
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    out = {"n": n, "acc": round(100 * acc, 2)}
    for cat in ("single", "cross"):
        sub = [r for r in rows if r["category"] == cat]
        if sub:
            out[f"acc_{cat}"] = round(100 * sum(r["correct"] for r in sub) / len(sub), 2)
    out["mean_visual"] = round(sum(r["n_visual"] for r in rows) / n, 1)
    out["max_visual"] = max(r["n_visual"] for r in rows)
    out["mean_calls"] = round(sum(r["n_calls"] for r in rows) / n, 3)
    out["mean_gen"] = round(sum(r["n_gen"] for r in rows) / n, 2)
    out["mean_latency"] = round(sum(r["latency"] for r in rows) / n, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override cfg keys: k=v (yaml-parsed)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    name = cfg["name"]
    out = args.out or os.path.join(ROOT, "results", f"{name}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    insts = load_hrbench()
    if args.limit:
        insts = insts[: args.limit]
    done = {r["idx"] for r in load_jsonl(out)}
    q = Qwen(args.device, attn=cfg.get("attn", "sdpa"))
    strat = STRATEGIES[cfg["strategy"]]
    cache = ImageCache()

    with open(out, "a") as f:
        for i, inst in enumerate(insts):
            if inst.idx in done:
                continue
            t0 = time.perf_counter()
            img = cache.get(inst.image_path)
            res = strat(q, inst, img, cfg)
            calls = res.pop("calls")
            for c in calls:
                c.pop("_attn", None)
            rec = {"idx": inst.idx, "group": inst.group, "category": inst.category,
                   "answer": inst.answer, "pred": res["pred"], "correct": res["pred"] == inst.answer,
                   "n_visual": sum(c["n_visual"] for c in calls), "n_calls": len(calls),
                   "n_gen": sum(c["n_gen"] for c in calls),
                   "latency": round(time.perf_counter() - t0, 4), **res, "calls": calls}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"[{name}] {i + 1}/{len(insts)}", flush=True)

    rows = [r for r in load_jsonl(out) if r["idx"] in {it.idx for it in insts}]
    s = summarize(rows)
    print(json.dumps({"name": name, **s}))
    with open(os.path.join(ROOT, "results", "summary.jsonl"), "a") as f:
        f.write(json.dumps({"name": name, "file": os.path.basename(out), **s}) + "\n")


if __name__ == "__main__":
    main()

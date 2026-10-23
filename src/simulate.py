"""Offline simulation of confidence-gated cascades from force-all logs.

A force-all cascade log (tau=null) contains, for every instance, the probe,
locate and zoom calls. Because the later stages do not depend on the stopping
threshold, the accuracy and exact mean visual tokens of any (tau, alpha)
policy can be computed without re-running the model. The probe stage can also
be swapped with any `global` run (e.g. a different probe resolution).

    python -m src.simulate --cascade results/cascade_dev.jsonl [--probe results/global_1024.jsonl]
"""
import argparse
import json
import math

from .run_eval import load_jsonl

LET = "ABCD"


def _lp(p):
    return [math.log(max(x, 1e-9)) for x in p]


def build(cascade_rows, probe_rows=None, zoom_tag="zoom", zooms=None):
    """zooms: optional {idx: call} from src.zoom_sweep replacing the logged zoom call."""
    probe_by = {r["idx"]: r for r in probe_rows} if probe_rows else None
    items = []
    for r in cascade_rows:
        calls = {c["tag"]: c for c in r["calls"]}
        if probe_by is not None:
            pc = probe_by[r["idx"]]["calls"][0]
        else:
            pc = calls["probe"]
        loc = calls.get("locate")
        z = zooms.get(r["idx"]) if zooms is not None else calls.get(zoom_tag)
        items.append({
            "idx": r["idx"], "category": r["category"], "answer": r["answer"],
            "p1": pc["probs"], "t1": pc["n_visual"],
            "p3": z["probs"] if z else None,
            "t2": (loc["n_visual"] if loc else 0) + (z["n_visual"] if z else 0),
        })
    return items


def policy(items, tau, alpha, use_probe=True):
    correct, tok, esc = 0, 0, 0
    per_cat = {"single": [0, 0], "cross": [0, 0]}
    for it in items:
        p1 = it["p1"]
        if use_probe and max(p1) >= tau:
            pred, t = LET[p1.index(max(p1))], it["t1"]
        else:
            esc += 1
            t = (it["t1"] if use_probe else 0) + it["t2"]
            if it["p3"] is None:
                pred = LET[p1.index(max(p1))]
            else:
                a = alpha if use_probe else 0.0
                s = [a * x + y for x, y in zip(_lp(p1), _lp(it["p3"]))]
                pred = LET[s.index(max(s))]
        ok = pred == it["answer"]
        correct += ok
        tok += t
        per_cat[it["category"]][0] += ok
        per_cat[it["category"]][1] += 1
    n = len(items)
    return {"tau": tau, "alpha": alpha, "acc": round(100 * correct / n, 2),
            "acc_single": round(100 * per_cat["single"][0] / max(per_cat["single"][1], 1), 2),
            "acc_cross": round(100 * per_cat["cross"][0] / max(per_cat["cross"][1], 1), 2),
            "mean_visual": round(tok / n, 1), "esc_rate": round(esc / n, 3)}


def sweep(items, taus=None, alphas=(0.0, 0.25, 0.5, 1.0)):
    taus = taus or [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.01]
    return [policy(items, t, a) for t in taus for a in alphas]


def best_under(results, budget):
    ok = [r for r in results if r["mean_visual"] <= budget]
    return max(ok, key=lambda r: r["acc"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--zoom_tag", default="zoom")
    ap.add_argument("--zooms", default=None, help="zoom_sweep output file")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--budgets", nargs="*", type=float, default=[2048, 4096])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    rows = load_jsonl(args.cascade)
    probe = load_jsonl(args.probe) if args.probe else None
    if probe:
        have = {r["idx"] for r in probe}
        rows = [r for r in rows if r["idx"] in have]
    zooms = None
    if args.zooms:
        zooms = {z["idx"]: z["call"] for z in load_jsonl(args.zooms) if z["variant"] == args.variant}
    items = build(rows, probe, args.zoom_tag, zooms)
    res = sweep(items)
    if args.all:
        for r in res:
            print(json.dumps(r))
    print("n =", len(items), "| zoom-only (no probe):", json.dumps(policy(items, 2, 0, use_probe=False)))
    print("probe-only:", json.dumps(policy(items, 0, 0)))
    for b in args.budgets:
        print(f"best <= {b}:", json.dumps(best_under(res, b)))


if __name__ == "__main__":
    main()

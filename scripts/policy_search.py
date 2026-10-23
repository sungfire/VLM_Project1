"""Grid-search routed cascade policies offline.

Inputs: a force-all joint log (probe probs + boxes), a zoom_sweep file with
variant results for uncertain instances, and per-question type probabilities
from the text-only classifier. Policy:
    if max p1 >= tau: answer from probe
    elif P(single-region) >= th: zoom variant S, fuse(alpha_s)
    else: relational path R (None = keep probe) with fuse(alpha_r)
"""
import argparse
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_eval import load_jsonl  # noqa: E402

LET = "ABCD"


def lp(p):
    return [math.log(max(x, 1e-9)) for x in p]


def fuse(p1, p3, a):
    s = [a * x + y for x, y in zip(lp(p1), lp(p3))]
    return LET[s.index(max(s))]


def evaluate(items, tau, th, vs, a_s, vr, a_r):
    ok = tok = 0
    cat = {"single": [0, 0], "cross": [0, 0]}
    for it in items:
        p1 = it["p1"]
        t = it["t1"]
        pred = LET[p1.index(max(p1))]
        if max(p1) < tau:
            v, a = (vs, a_s) if it["ptype"] >= th else (vr, a_r)
            z = it["z"].get(v) if v else None
            if z is not None:
                t += z["n_visual"]
                pred = fuse(p1, z["probs"], a)
        c = pred == it["answer"]
        ok += c
        tok += t
        cat[it["category"]][0] += c
        cat[it["category"]][1] += 1
    n = len(items)
    return {"acc": round(100 * ok / n, 2), "single": round(100 * cat["single"][0] / cat["single"][1], 2),
            "cross": round(100 * cat["cross"][0] / cat["cross"][1], 2), "tok": round(tok / n, 1)}


def load_items(log, zooms, types):
    zs = {}
    for z in load_jsonl(zooms):
        zs.setdefault(z["idx"], {})[z["variant"]] = z["call"]
    ty = json.load(open(types))
    items = []
    for r in load_jsonl(log):
        p = r["calls"][0]
        items.append({"idx": r["idx"], "answer": r["answer"], "category": r["category"],
                      "p1": p["probs"], "t1": p["n_visual"], "z": zs.get(r["idx"], {}),
                      "ptype": ty[str(r["group"])][0]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--zooms", required=True)
    ap.add_argument("--types", default="results/type_p1.json")
    ap.add_argument("--budget", type=float, default=2048)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--max_conf", type=float, default=0.995)
    args = ap.parse_args()
    items = load_items(args.log, args.zooms, args.types)
    variants = sorted({v for it in items for v in it["z"]})
    print("n =", len(items), "variants:", variants)
    res = []
    taus = [0.9, 0.95, 0.97, 0.98, 0.99, args.max_conf]
    for tau, th, vs, a_s, vr, a_r in itertools.product(
            taus, [0.0, 0.3, 0.5, 0.7, 1.01], variants, [0, 0.25, 0.5, 1.0],
            [None] + variants, [0, 0.5, 1.0, 2.0]):
        if vr is None and a_r != 0:
            continue
        r = evaluate(items, tau, th, vs, a_s, vr, a_r)
        if r["tok"] <= args.budget:
            res.append({**r, "tau": tau, "th": th, "vs": vs, "a_s": a_s, "vr": vr, "a_r": a_r})
    res.sort(key=lambda r: (-r["acc"], r["tok"]))
    for r in res[: args.top]:
        print(json.dumps(r))


if __name__ == "__main__":
    main()

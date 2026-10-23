"""Compute every number used in the report (writes report/stats.json)."""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_search import evaluate, load_items  # noqa: E402
from src.run_eval import load_jsonl, summarize  # noqa: E402

LET = "ABCD"
os.makedirs("report", exist_ok=True)
out = {}


def am(p):
    return LET[p.index(max(p))]


# ---------------------------------------------------------------- runs
runs = {}
for name in ["global_256", "global_512", "global_1024", "global_2048", "global_4096",
             "tiling_2048", "tiling_4096", "cascade_dev", "joint_dev", "joint_p2048_z2048",
             "success1_4k", "success2_2k"]:
    p = f"results/{name}.jsonl"
    rows = load_jsonl(p)
    if len(rows) == 800:
        runs[name] = rows
        out.setdefault("runs", {})[name] = summarize(rows)
native = load_jsonl("results/global_16384.jsonl")
out["native_partial"] = summarize(native) if native else None

# ---------------------------------------------------------------- items for simulation
ty = json.load(open("results/type_p1.json"))


def items_logged(log, with_locate=False):
    items = []
    for r in load_jsonl(log):
        calls = {c["tag"]: c for c in r["calls"]}
        p, z, loc = calls["probe"], calls.get("zoom"), calls.get("locate")
        zz = None
        if z:
            zz = dict(z)
            if with_locate and loc:
                zz["n_visual"] = z["n_visual"] + loc["n_visual"]
        items.append({"idx": r["idx"], "answer": r["answer"], "category": r["category"],
                      "p1": p["probs"], "t1": p["n_visual"], "z": {"zoom": zz} if zz else {},
                      "ptype": ty[str(r["group"])][0]})
    return items


S2 = items_logged("results/joint_dev.jsonl")
abl2 = {
    "Full (Success 2 config)": evaluate(S2, 0.999, 0.3, "zoom", 0.25, None, 0),
    "w/o early stopping (tau=1)": evaluate(S2, 1.01, 0.3, "zoom", 0.25, None, 0),
    "w/o type routing (zoom all)": evaluate(S2, 0.999, 0.0, "zoom", 0.25, None, 0),
    "w/o fusion (alpha=0)": evaluate(S2, 0.999, 0.3, "zoom", 0.0, None, 0),
    "w/o escalation (joint probe only)": evaluate(S2, 0.0, 0.3, "zoom", 0.25, None, 0),
    "zoom always, zoom answer only": evaluate(S2, 1.01, 0.0, "zoom", 0.0, None, 0),
}
SEP = items_logged("results/cascade_dev.jsonl", with_locate=True)
abl2["separate grounding call (+1024 tok)"] = evaluate(SEP, 0.999, 0.3, "zoom", 0.25, None, 0)
out["ablation_2k"] = abl2

S1 = load_items("results/joint_dev.jsonl", "results/zooms_joint_dev.jsonl", "results/type_p1.json")
abl1 = {
    "Full (Success 1 config)": evaluate(S1, 0.995, 0.3, "z1024_e2_up2", 0.25, "g4096", 0),
    "relational: keep probe": evaluate(S1, 0.995, 0.3, "z1024_e2_up2", 0.25, None, 0),
    "relational: zoom crop (no routing)": evaluate(S1, 0.995, 0.0, "z1024_e2_up2", 0.25, None, 0),
    "single: no 2x up-scaling": evaluate(S1, 0.995, 0.3, "z1024_e2", 0.25, "g4096", 0),
    "single: larger context (expand 3)": evaluate(S1, 0.995, 0.3, "z1024_e3", 0.25, "g4096", 0),
    "single: thumbnail 512 + crop": evaluate(S1, 0.995, 0.3, "t512_z1024_e2", 0.25, "g4096", 0),
    "single: crop 2048 tok": evaluate(S1, 0.995, 0.3, "z2048_e2", 0.25, "g4096", 0),
}
S1b = load_items("results/joint_p2048_z2048.jsonl", "results/zooms_joint_p2048.jsonl", "results/type_p1.json")
abl1["probe 2048 + crop 2048 (expand 3)"] = evaluate(S1b, 0.995, 0.3, "z2048_e3", 0.25, "g4096", 0)
out["ablation_4k"] = abl1

# tau sweeps for figure
taus = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]
out["tau_sweep_2k"] = [dict(evaluate(S2, t, 0.3, "zoom", 0.25, None, 0), tau=t) for t in taus]
out["tau_sweep_4k"] = [dict(evaluate(S1, t, 0.3, "z1024_e2_up2", 0.25, "g4096", 0), tau=t) for t in taus if t <= 0.995]

# ---------------------------------------------------------------- final-run analysis
for name in ["success1_4k", "success2_2k"]:
    rows = runs[name]
    a = {}
    toks = [r["n_visual"] for r in rows]
    lat = sorted(r["latency"] for r in rows)
    a["tokens"] = {"mean": statistics.mean(toks), "median": statistics.median(toks), "min": min(toks), "max": max(toks),
                   "hist": dict(collections.Counter(toks).most_common(8))}
    a["latency"] = {"mean": statistics.mean(lat), "p50": lat[len(lat) // 2], "p95": lat[int(0.95 * len(lat))]}
    a["gen_tokens_mean"] = statistics.mean(r["n_gen"] for r in rows)
    a["calls_dist"] = dict(collections.Counter(r["n_calls"] for r in rows))
    # per call type
    pc = collections.defaultdict(lambda: {"n": 0, "vis": 0, "gen": 0, "lat": 0.0, "inp": 0})
    for r in rows:
        for c in r["calls"]:
            d = pc[c["tag"]]
            d["n"] += 1; d["vis"] += c["n_visual"]; d["gen"] += c["n_gen"]; d["lat"] += c["latency"]; d["inp"] += c["n_input"]
    a["per_call"] = {k: {"count": v["n"], "mean_visual": v["vis"] / v["n"], "mean_input": v["inp"] / v["n"],
                         "mean_gen": v["gen"] / v["n"], "mean_latency": v["lat"] / v["n"]} for k, v in pc.items()}
    # stage breakdown and failure taxonomy
    st = collections.defaultdict(lambda: [0, 0])
    tax = collections.Counter()
    route_vs_cat = collections.Counter()
    for r in rows:
        st[r["stage"]][0] += r["correct"]; st[r["stage"]][1] += 1
        p1ok = am(r["calls"][0]["probs"]) == r["answer"]
        if r["stage"] == "probe":
            tax["stopped_correct" if r["correct"] else "premature_stop(wrong & confident)"] += 1
        else:
            tax[{(True, True): "escalated: kept correct", (True, False): "escalated: harmful (probe right -> final wrong)",
                 (False, True): "escalated: fixed (probe wrong -> final right)",
                 (False, False): "escalated: still wrong"}[(p1ok, r["correct"])]] += 1
            ty_call = next((c for c in r["calls"] if c["tag"] == "type"), None)
            if ty_call:
                route_vs_cat[(r["category"], "single-route" if ty_call["p"] >= 0.3 else "relational-route")] += 1
        cat_stage = r["category"]
        st[f"{cat_stage}:{r['stage']}"][0] += r["correct"]; st[f"{cat_stage}:{r['stage']}"][1] += 1
    a["stages"] = {k: {"correct": v[0], "n": v[1]} for k, v in st.items()}
    a["taxonomy"] = dict(tax)
    a["route_vs_category"] = {f"{k[0]}|{k[1]}": v for k, v in route_vs_cat.items()}
    out[f"final_{name}"] = a

# type classifier vs label (analysis only)
tp = {th: (sum(1 for p, c in ty.values() if p >= th and c == "single"), sum(1 for p, c in ty.values() if p >= th and c == "cross")) for th in (0.3, 0.5)}
out["type_classifier"] = tp

json.dump(out, open("report/stats.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k.startswith("ablation")}, indent=1))
for n in ["final_success1_4k", "final_success2_2k"]:
    print(n, json.dumps({k: out[n][k] for k in ["tokens", "latency", "calls_dist", "taxonomy", "route_vs_category", "per_call"]}, indent=0, default=str))

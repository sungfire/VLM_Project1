"""Summarize every results/*.jsonl that has all 800 instances."""
import sys, json, glob, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_eval import load_jsonl, summarize
for p in sorted(glob.glob('results/*.jsonl')):
    if p.endswith('summary.jsonl') or os.path.basename(p).startswith('zooms_'): continue
    rows = load_jsonl(p)
    s = summarize(rows) if rows else {}
    print(f"{os.path.basename(p):40s} " + " ".join(f"{k}={v}" for k, v in s.items()))

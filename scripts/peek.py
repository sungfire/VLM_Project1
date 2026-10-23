import sys, json
sys.path.insert(0, '.')
from src.run_eval import load_jsonl, summarize
for p in sys.argv[1:]:
    rows = load_jsonl(p)
    if rows: print(p, json.dumps(summarize(rows)))

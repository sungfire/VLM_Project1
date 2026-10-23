# Efficient High-Resolution VQA on HR-Bench 4K (Qwen3-VL-2B-Instruct, frozen)

Training-free inference procedure that answers HR-Bench 4K multiple-choice questions
under a mean cumulative visual-token budget, using only the original
`Qwen/Qwen3-VL-2B-Instruct` checkpoint and its official processor.

## Results (800 instances = 200 questions x 4 option orderings)

| Config | Mean visual tokens | Accuracy | Single | Cross | Mean Qwen calls | Mean gen. tokens | Mean latency |
|---|---|---|---|---|---|---|---|
| **Success 1** (`configs/success1_4k.yaml`) | 2730.2 (<= 4096) | **74.00%** | 88.00 | 60.00 | 2.34 | 35.9 | 1.95 s |
| **Success 2** (`configs/success2_2k.yaml`) | 1354.4 (<= 2048) | **71.88%** | 88.75 | 55.00 | 2.10 | 38.1 | 1.88 s |
| Baseline: global resize 4096 tokens | 4066.4 | 70.50% | 81.75 | 59.25 | 1 | 1 | 1.75 s |
| Baseline: global resize 2048 tokens | 2028.2 | 66.38% | 78.00 | 54.75 | 1 | 1 | 0.65 s |
| Baseline: global resize 1024 tokens | 1020.2 | 63.00% | 71.25 | 54.75 | 1 | 1 | 0.43 s |
| Baseline: global 1024 + 2x2 tiles (768 each) | 4040.4 | 62.25% | 73.50 | 51.00 | 1 | 1 | 1.16 s |
| Baseline: global 512 + 2x2 tiles (384 each) | 2025.5 | 56.38% | 65.00 | 47.75 | 1 | 1 | 0.72 s |

Environment: 1x NVIDIA RTX 3090 (24 GB) per run, bf16, SDPA attention, batch size 1,
greedy decoding (`do_sample=False, num_beams=1`), non-thinking Instruct model.

## Method (routed confidence-gated zoom)

1. **Joint probe** (1 call): the whole image resized to <= 1024 visual tokens (1024x1024 px
   for the common 4032x4032 images). The prompt asks for the option letter first and then
   the bounding box (`bbox_2d`, 0-1000 relative coordinates) of the evidence region. The A-D
   distribution `p1` is read from the logits of the first generated token. If
   `max p1 >= tau`, a stopping criterion ends generation right after that token and `p1`
   gives the answer.
2. **Type routing** (text-only call, 0 visual tokens): Qwen classifies the question as
   (A) one local detail vs (B) multi-object relation / counting / layout.
3. **Escalation** (1 call):
   - type A: crop of the original image around the probe's box (box expanded 2x, min side
     25% of the short image side), resized to <= 1024 tokens (up-scaling up to 2x allowed in
     Success 1);
   - type B: Success 1 re-observes the full image at 4096 tokens; Success 2 keeps the probe answer.
4. **Fusion**: answer = `argmax(alpha * log p1 + log p_escalation)`.

Every image of every call is counted (`<|image_pad|>` tokens = `prod(image_grid_thw)/4`;
images are pre-resized to multiples of 32 px so the processor does not resize them again).
The dataset `category`/`answer` columns are used only for scoring.

## Reproduce

```bash
# 1. environment (Python 3.11)
pip install uv && uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv/bin/python -r requirements.txt

# 2. model + data (HF cache, data/hr_bench_4k.parquet)
.venv/bin/python scripts/download.py

# 3. final runs (each writes results/<name>.jsonl and prints a summary)
.venv/bin/python -m src.run_eval --config configs/success1_4k.yaml --device cuda:0
.venv/bin/python -m src.run_eval --config configs/success2_2k.yaml --device cuda:0

# baselines
.venv/bin/python -m src.run_eval --config configs/baselines/global.yaml --set name=global_2048 tokens=2048
scripts/run_global_sweep.sh cuda:0 256 512 1024 2048 4096

.venv/bin/python -m src.run_eval --config configs/baselines/tiling.yaml                     # tiling_4096
.venv/bin/python -m src.run_eval --config configs/baselines/tiling.yaml --set name=tiling_2048 thumb_tokens=512 tile_tokens=384

# summary table of all results/*.jsonl
.venv/bin/python scripts/table.py

# report (needs python-docx): stats/ablations -> figures -> report/project1_report.docx
.venv/bin/python scripts/report_stats.py && .venv/bin/python scripts/make_figs.py
.venv/bin/python scripts/make_qual.py report/figs/qual.png 252 344 88 148 && .venv/bin/python scripts/build_report.py
```

Each line of `results/<name>.jsonl` holds one instance: prediction, correctness, total
visual tokens, number of calls, generated tokens, end-to-end latency, the stage at which
it stopped, and the full list of Qwen calls (image sizes, `image_grid_thw`, visual / input /
generated tokens, generated text, A-D probabilities, boxes, crop box, latency).

## Code layout

| File | Purpose |
|---|---|
| `src/data.py` | HR-Bench 4K loader (800 instances, images extracted to `data/images/`) |
| `src/qwen.py` | Frozen Qwen3-VL wrapper with per-call accounting; choice / joint / generate / text-only modes |
| `src/imaging.py` | Token-budget resize (multiples of 32 px), box conversion, crop expansion, tiling |
| `src/prompts.py` | All prompts |
| `src/pipeline.py` | Strategies: `global`, `tiling`, `cascade` (dev), `routed` (final) |
| `src/run_eval.py` | Runs a config over all 800 instances |
| `src/simulate.py`, `src/zoom_sweep.py`, `scripts/policy_search.py` | Development: force-all logs -> offline policy / threshold search |
| `configs/` | Final configs, baselines, development configs |

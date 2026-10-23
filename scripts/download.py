"""Download the frozen model and HR-Bench 4K (parquet) into the HF cache / data/."""
import os
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
snapshot_download("Qwen/Qwen3-VL-2B-Instruct")
hf_hub_download("DreamMr/HR-Bench", "hr_bench_4k.parquet", repo_type="dataset",
                local_dir=os.path.join(ROOT, "data"))

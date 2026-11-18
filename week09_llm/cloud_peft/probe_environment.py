"""Read-only cloud preflight; downloads metadata and one streamed dataset record, no model weights."""
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path
import torch
from datasets import load_dataset
from huggingface_hub import HfApi

def main():
    info = {"python": platform.python_version(), "platform": platform.platform(), "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda, "gpus": [], "storage": shutil.disk_usage(Path.cwd())._asdict(), "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate")}, "credential_presence": {k: bool(os.environ.get(k)) for k in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME", "HF_TOKEN", "AWS_ACCESS_KEY_ID")}}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        info["gpus"].append({"name": torch.cuda.get_device_name(i), "free_bytes": free, "total_bytes": total})
    api = HfApi()
    for kind, repo in (("dataset", "codeparrot/codeparrot-clean"), ("model", "unsloth/Qwen3-8B-Base-bnb-4bit")):
        try:
            details = api.repo_info(repo, repo_type=kind, files_metadata=True)
            info[kind] = {"id": repo, "revision": details.sha, "weight_bytes": sum(f.size or 0 for f in details.siblings if f.rfilename.endswith(".safetensors"))}
        except Exception as exc:
            info[kind] = {"error": repr(exc)}
    try:
        revision = info["dataset"]["revision"]
        stream = load_dataset("codeparrot/codeparrot-clean", split="train", revision=revision, streaming=True)
        row = next(iter(stream))
        processed = "\n".join(line[:512] for line in row["content"].split("\n"))
        info["dataset_access"] = {"status": "VERIFIED", "split": "train", "fields": sorted(row), "path": row.get("path"), "original_characters": len(row["content"]), "processed_characters": len(processed), "lines": len(row["content"].split("\n")), "max_processed_line": max(map(len, processed.split("\n")))}
    except Exception as exc:
        info["dataset_access"] = {"status": "BLOCKED", "error": repr(exc)}
    info["full_8b_gpu_experiment"] = "BLOCKED" if not info["cuda_available"] else "NOT RUN"
    target = Path(__file__).parent / "evidence/environment.json"
    target.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()

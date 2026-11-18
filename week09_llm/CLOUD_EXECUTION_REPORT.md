# Cloud execution report

Overall: **PARTIAL — GPU experiment BLOCKED**. Executed on Codex Cloud Linux, 2026-11-18. No local Mac execution and no paid GPU job was started.

## Actual environment and access

Python 3.12.14; PyTorch 2.15.0+cpu; Transformers 4.57.6; PEFT 0.18.1; bitsandbytes 0.49.0; Accelerate 1.12.0; datasets 4.4.1. `torch.cuda.is_available()` is false, CUDA runtime is null, GPU count is zero, and neither nvidia-smi nor /dev/nvidia* is present. CUDA quantization compatibility is **NOT RUN**; real Transformers quantizer dispatch with the native configuration passed on CPU.

The filesystem has 8,417,406,976 bytes total and 6,235,013,120 bytes free at preflight. The base model's safetensors alone occupy 6,073,558,092 bytes. This environment cannot accommodate a safe full model cache and experiment artifacts; the notebook requires 15 GiB free storage and at least 13 GiB free CUDA VRAM before loading. These are conservative admission checks, not a guarantee that an 8B run fits. Actual GPU feasibility requires the 512-token smoke run.

KAGGLE_API_TOKEN, KAGGLE_KEY, KAGGLE_USERNAME, HF_TOKEN and AWS_ACCESS_KEY_ID were absent; standard Kaggle credential directories and /workspace/.aws had no credentials. No Kaggle/GPU connector was exposed. The available Hugging Face Jobs listing tool was attempted read-only and returned `MCP tool hf_jobs was not returned by tools/list` (UNAVAILABLE). No authenticated, already-authorized GPU execution route was established. GitHub authentication is available and used only for source delivery.

**VERIFIED dataset access:** streamed one real record from `codeparrot/codeparrot-clean`, split `train`, revision `35a59fb025bc0a102f7d96eac09d145b896d487b`. First record: `examples/asyncio/websocket/echo/client_coroutines.py`, 2,044 characters, 64 lines, maximum processed line 79 characters. This is an access probe, not a completed experiment dataset manifest. Model metadata access is verified at revision `e3276524f7f4ce7df7b65dd9fcb7552a7f2d6de9`; no pretrained model weights were downloaded.

## Actual commands

Run from `/workspace/Practical-DeepLearning`:

```bash
python -m venv /workspace/peft-venv
/workspace/peft-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/workspace/peft-venv/bin/pip install transformers==4.57.6 peft==0.18.1 datasets==4.4.1 accelerate==1.12.0 bitsandbytes==0.49.0 nbformat pytest
OMP_NUM_THREADS=2 /workspace/peft-venv/bin/python -m pytest week09_llm/cloud_peft/tests -q
HF_HOME=/workspace/hf-cache OMP_NUM_THREADS=2 /workspace/peft-venv/bin/python week09_llm/cloud_peft/probe_environment.py
/workspace/peft-venv/bin/python week09_llm/cloud_peft/run_notebook.py --output-root /workspace/cloud-runs
```

The initial global pip attempt failed on the read-only user installation directory; the isolated workspace virtual environment succeeded. The first test run had 9 passes and one test referencing the old notebook; that reference was fixed. Final test run: **10 passed in 3.93s**. Notebook execution reached the cloud CUDA guard and stopped with `BLOCKED: Linux cloud CUDA GPU required. Never run training on a local Mac.` Actual logs and machine-readable measurements are in [cloud_peft/evidence](cloud_peft/evidence).

## Minimal remaining execution

Provide an already-authorized Linux CUDA cloud runtime with sufficient quota and storage (e.g. a Kaggle GPU session with Internet enabled). The notebook is standalone; upload `Practice_peft_cloud.ipynb`, retain `SMOKE_ONLY=True`, and Run All. Existing cloud PyTorch must be CUDA-enabled. Alternatively clone this branch on that runtime and execute:

```bash
# From the repository root on a Linux cloud GPU, with provider CUDA PyTorch installed:
python -m pip install -r week09_llm/cloud_peft/requirements.txt
python week09_llm/cloud_peft/run_notebook.py --output-root /kaggle/working
# Only after smoke results.json is successful; command runs in a fresh process:
python week09_llm/cloud_peft/run_notebook.py --full --smoke-results /kaggle/working/python_custom_lora_cloud_smoke/results.json --output-root /kaggle/working
```

For another Linux cloud runtime substitute a writable cloud output directory. In interactive notebook use, restart the kernel before the full run, set `SMOKE_ONLY=False`, and set the absolute smoke results path. Full training starts from a fresh base; two smoke updates are not counted toward the 500 updates. The gate requires matching sources, helper hash, environment, GPU and hyperparameters. A failed/OOM smoke must be resolved before full execution. Preserve failed logs; do not label them completed.

## Adapter files and retrieval

**Final trained 8B adapters: NOT AVAILABLE.** No final-task optimizer updates ran. The CPU smoke uses temporary synthetic test checkpoints, deliberately excluded from source control and not distributed as coursework weights.

After successful GPU execution retrieve the complete `python_custom_lora_cloud/` output directory: `adapter_final.pt`, periodic `adapter_step_*.pt`, `results.json`, `run_config.json`, `training_log.jsonl`, `data_manifest.json`, `token_blocks.pt`, tokenizer files, `baseline.json`, `comparison.html`, and `coursework_report.md`. In Kaggle use Save Version / Save & Run All and download Output. Keep the smoke directory and logs too. Preserve all checkpoints outside the Git repository.

Reload in the same compatible cloud environment: read metadata/config from `torch.load(path, map_location="cpu", weights_only=True)`, construct `ExperimentConfig(**payload["metadata"]["config"])`, call `load_cloud_base(config, payload["metadata"]["model_revision"])`, `inject_custom_lora(model, config.rank)`, then `load_custom_lora(model, path, config.base_model, payload["metadata"]["model_revision"])`. Helpers are in `cloud_peft/peft_utils.py`. The executed notebook performs the actual logits and identical-generation reload checks before writing success. These are custom adapter-only inference files, not HF PEFT adapters or exact optimizer-resume checkpoints.

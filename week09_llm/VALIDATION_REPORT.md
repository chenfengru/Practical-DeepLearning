# Validation report

Overall: **PARTIAL — final GPU coursework BLOCKED**.

| Check | Actual result | Status |
|---|---|---|
| Entire teacher notebook read and snapshot retained | 31 original cells, section order tested | VERIFIED |
| Historical outputs retained and originals untouched | All legacy cell outputs/counts match original archive | VERIFIED |
| Notebook schema and all code-cell syntax | nbformat validation and compile; inactive execution loads no model | VERIFIED |
| Original LoRA numerical grading assertions | Forward and A/B gradient assertions passed | VERIFIED CPU |
| Prompt embedding replacement and gradients | Shapes, prompt equality, frozen original embeddings passed | VERIFIED CPU |
| Native NF4 configuration dispatch | Real Transformers quantizer created with FP16 compute, NF4 retained | VERIFIED CPU |
| Preprocessing regression | Each long line cropped separately; file content after line one retained; dedup/repository split/512-token packing passed | VERIFIED CPU |
| Real random tiny Qwen3 integration | 2 optimizer updates, accumulation 4, 12/12 B matrices updated | VERIFIED CPU ONLY |
| Tiny checkpoint reload | 24 adapter tensors; exact logits and greedy token equality | VERIFIED CPU ONLY |
| Tiny integration probe token NLL | 3.498637915 → 3.498585343 on 14 predicted tokens | VERIFIED CPU ONLY |
| Missing/corrupt/mismatched smoke evidence | Full-run gate rejects invalid evidence | VERIFIED CPU |
| Dataset access | One actual canonical record streamed at pinned revision | VERIFIED |
| Notebook real execution attempt | Stopped at CUDA guard, before model download | BLOCKED |
| 8B CUDA NF4 loading, 512-token backward and optimizer smoke | No GPU available | NOT RUN |
| Final 500 updates, generation, evaluation, adapter reload | 0 updates; no final adapter | NOT RUN |

Final command: `OMP_NUM_THREADS=2 /workspace/peft-venv/bin/python -m pytest week09_llm/cloud_peft/tests -q` → **10 passed in 3.93s**. Actual measurements and logs are in `cloud_peft/evidence/`; `cpu_smoke.json` explicitly labels the synthetic model and CPU device. The tiny NLL probe reuses the synthetic training blocks and is not a held-out generalization metric. Synthetic gate fixtures test rejection behavior only and are not GPU evidence. CPU smoke checkpoints live in pytest temporary directories and are excluded from Git.

Original historical results retained: prompt losses 0.0567 and 0.0857, passing custom LoRA tests, 108 injected attention modules, successful gradient check, 100 example steps, 216 saved tensors, 20.32 MB checkpoint. These are historical notebook records; this run did not retrieve or reload that historical checkpoint.

The CPU PyTorch build has no CUDA runtime. CUDA package/kernel compatibility, full-model VRAM fit, real 512-token GPU smoke, final metrics and trained weights remain unverified. Disk is also insufficient for safe 8B execution. No already-authorized GPU service was established; Hugging Face Jobs listing returned UNAVAILABLE and Kaggle credentials were absent. The minimal remaining step is to run the supplied smoke and full commands on an authorized Linux cloud CUDA runtime with sufficient storage, then retrieve results and weights. Do not mark the coursework complete until all GPU checks pass.

# Validation report

Branch: `codex/week09-peft-kaggle-fix`. Scope: Week 09 PEFT assignment only. Original double-extension notebook and unrelated work remain unchanged. No credentials, model weights or datasets were added to Git.

## Confirmed remote failure and root-cause repair

User-provided Kaggle evidence: Python 3.13, PyTorch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, bitsandbytes 0.50.2, Tesla T4 x2. Execution failed **before training** with `AttributeError: 'BitsAndBytesConfig' object has no attribute 'get'` in model loading. No new final-task metrics or weights resulted from that attempt.

The prior helper replaced `model_config.quantization_config` with a BitsAndBytesConfig object. Transformers 5.20.2's `get_hf_quantizer` calls `supports_quant_method` on this field before merging/converting it, and that function expects `.get()`. The repair retains a copied JSON dictionary from AutoConfig, preserves native NF4/storage/double-quantization metadata, changes compute dtype to FP16 for T4, and uses `dtype=` without an additional quantization_config argument. No dependency change or runtime monkey-patch was introduced. A local regression reproduces the old exception and exercises the corrected real Transformers dispatch/merge path without weights; only its CUDA environment validation is stubbed in the test.

The notebook now defaults to `RUN_LEGACY_EXPERIMENTS=False`, `RUN_FINAL_EXPERIMENT=True`, `SMOKE_ONLY=True`. Full training requires a completed two-update GPU smoke artifact with matching helper SHA256, immutable model/dataset revisions, Python/CUDA/package versions, GPU, sequence length and optimizer settings. Missing, failed or incompatible evidence blocks the full run before model loading and again before training. Smoke evidence must record complete gradient coverage and both reload checks. The teacher's final closing material follows the implemented final task; section order is tested against the official `practice_peft.ipynb`.

## Requirement audit

| Requirement | Status and evidence |
| --- | --- |
| Entire source notebook and original final instructions inspected | Complete; original final task required Python data, 512-character crops, custom attention LoRA plus an extra adapter type, batch ≥4 via accumulation 4, 500 updates, before/after generations |
| Preserve earlier results | Complete; original file untouched; historical outputs/execution counts copied exactly and gated off by default |
| Custom prompt loss ≤0.1 | Recorded pass: 0.0567; not retrained |
| PEFT prompt loss <0.1 | Recorded pass: 0.0857; not retrained |
| Original LoRA forward/loss/gradient numeric assertions | Passed locally using the final custom implementation; historical pass also retained |
| Original full-model injection and gradients | Historical pass: 108 attention modules and nonzero B gradients; fresh final 8B checks pending |
| Example 100-step training | Historical checkpoint output confirms step 100; saved Trainer cell has a stale NameError. Example not retrained |
| Explicit checkpoint format / automatic saving disabled | Implemented; v1 format and CPU round trip passed. Legacy optional Trainer uses `save_strategy="no"`; final uses no Trainer |
| Native pre-quantized configuration regression | Passed locally against Transformers 5.20.2 without downloading weights; real CUDA model load pending |
| Teacher section order and historical outputs | Tested against the official teacher notebook and original experiment archive; no outputs deleted |
| Mandatory model-load / two-update GPU gate | Implemented and failure/mismatch cases tested locally; actual GPU pass pending |
| CUDA-only Kaggle gate before any model download | Implemented and locally tested; no local 8B loading |
| Dataset selection, extension filter, crop, deduplication, disjoint evaluation | Implemented and tested synthetically; actual remote streaming unverified |
| Attention + additional MLP adapters | Implemented; tiny Qwen3 integration passed. Expected 8B count: 216 modules / 432 tensors, asserted remotely |
| 512-token blocks, effective batch size 4, 500 successful updates | Implemented and statically checked; actual final GPU run pending |
| Baseline, post-training NLL/perplexity, paired generations | Implemented; actual final metrics and generations pending |
| CUDA memory checks, FP16 T4 compute, seeded sampling, immutable revisions | Implemented; GPU memory and NF4 compute unverified remotely |
| Gradient smoke and post-save reload logits/generation checks | Tiny CPU Qwen3 passed; full-size GPU checks pending |
| Avoid accidental training continuation/overwrites | CPU guards passed; exclusive run directories/checkpoints and in-memory training marker |
| Kaggle deploy/monitor/retrieve | Unavailable: CLI/package/standard authentication absent. Exact manual procedure provided in KAGGLE_RUN.md |
| Final adapter artifact and actual final metrics | Pending Kaggle execution; never represented as available or passed |
| Coursework report | English evidence-based draft provided; notebook produces a measured report after execution |

## Local tests

Command: `python3 -m pytest week09_llm/peft_assignment/tests -q`.

Ten tests passed in the final local run (3.15 seconds), using PyTorch 2.9.1 and Transformers 5.20.2 on CPU. The tests use only a tiny Qwen3 initialized from configuration and synthetic data; they do not contact a model or dataset server. All new notebook code cells compile, notebook schema validates, historical outputs match the original, and execution with both experiment flags disabled leaves no model loaded.

```text
..........                                                               [100%]
10 passed in 3.15s
```

Coverage includes the assignment's numeric LoRA assertions, prompt batch embeddings and gradients, genuine-EOS preservation, causal alignment, repository-disjoint data selection, fixed-length packing, two CPU optimizer updates, frozen-base behavior, finite gradients, adapter round-trip matching logits and generation, and rerun/base/revision/invalid-checkpoint guards. New coverage reproduces the exact quantization error, checks native dictionary dispatch, enforces the default smoke mode, verifies teacher section order, and rejects absent/failed/incompatible smoke evidence.

## CUDA, memory and loss review

Model placement stays `device_map={"": 0}` with assertions that all parameters reside on CUDA 0; there is no post-load model move or CPU/disk offload. The second T4 remains unused. FP16 autocast and native quantized linear compute are explicit, while adapters and normalization preparation use FP32. Non-reentrant gradient checkpointing is enabled and training/evaluation forward passes use `use_cache=False`. The initial memory check requires 13 GiB free; snapshots include free/allocated/peak memory and the full 512-token backward smoke tests actual headroom. Evaluation releases vocabulary-sized logits between batches. True T4 peak use remains unmeasured until smoke runs.

Loss is next-token cross-entropy on shifted logits and targets over full packed 512-token blocks, retaining true EOS tokens. Micro-batch size 1 and four accumulation passes produce effective batch size 4. GradScaler, finite-loss checks and clipping protect FP16 updates; only successful optimizer updates count toward 500. Custom saving/loading remains adapter-only, and the final loop does not use Trainer or set a PEFT-loaded flag. The optional historical Trainer example has automatic saving disabled.

## Final acceptance after Kaggle

Do not mark the final task complete until the retrieved `results.json` reports `status = completed`, exactly 500 successful optimizer updates, finite baseline/after metrics, and both reload tests true. Preserve the executed notebook, training log, checkpoint and hash, pinned revisions, data manifest and token blocks. Inspect the eight paired samples and report the measured NLL change even if performance worsens. A smoke-only run or the historical quotes checkpoint is insufficient.

## Git scope

All Week 09 notebooks were initially untracked. The coursework commit includes the corrected single-extension notebook, its helper/builder/tests/reports and scoped ignore rules, plus the unchanged original archive and official teacher notebook required by the builder and provenance tests. Other Week 09 notebooks and all unrelated repository changes are excluded. No GPU outputs or checkpoint binaries are committed.

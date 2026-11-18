# Week 09: Parameter-Efficient Fine-Tuning

Overall status: **PARTIAL / GPU BLOCKED**. The teacher notebook at `yandexdataschool/Practical_DL`, branch `fall26`, was read in full before editing. Its exact fetched snapshot is preserved in `cloud_peft/evidence/teacher_original.ipynb` and hashed in the cloud notebook metadata. Original student notebooks and earlier reports remain unchanged. The six sections below retain the teacher's progression.

## 1. Fox Prompt Tuning (1 point)

**VERIFIED historical record, not rerun:** custom prompt tuning loss 0.0567, below 0.1; the saved continuation denies that the fox jumped over the dog. Historical output provenance is the original double-extension student notebook; the cloud notebook retains these outputs behind `RUN_LEGACY_EXPERIMENTS=False`. They are not new sequential execution evidence.

## 2. Hugging Face PEFT

**VERIFIED historical record, not rerun:** HF prompt tuning loss 0.0857, below 0.1, with a saved fox continuation. Library prompt tuning is distinct from the final task's custom LoRA implementation.

## 3. Custom LoRA (1 point)

The custom adapter implements `base(x) + x @ A @ B`, with Kaiming-initialized A and zero B, following the assignment's unscaled numerical convention. Frozen base layers stay in the autograd graph so gradients propagate through the entire network. FP32 adapter arithmetic is outside CUDA autocast and its output is cast to the base output dtype. **VERIFIED now on cloud CPU:** the teacher's original numerical forward/backward assertions passed, including loss 1.3711389 and gradient-sum target checks (-0.60158 and 0.9931, tolerance 1e-4).

## 4. LoRA injection and gradient verification

**VERIFIED historical record:** 108 Q/K/V attention adapter modules and a successful gradient check. The final design adds gate/up/down MLP adapters, satisfying the extra-adapter requirement: expected 216 modules / 432 tensors for 36 layers. **VERIFIED now on a tiny random two-layer Qwen3:** 12 custom modules received finite gradients, all 12 B matrices changed after optimization, and base weights remained frozen. Full 8B injection and CUDA gradients are **NOT RUN**.

Quantized loading preserves the checkpoint's native bitsandbytes dictionary and changes compute dtype to FP16, without redundant quantization configuration or calling `.to()` on the dispatched model. K-bit preparation uses non-reentrant gradient checkpointing and disables the KV cache during training. GPU compatibility remains unverified.

## 5. Training example

**VERIFIED historical record, not rerun:** previous quotes example reached 100 steps; checkpoint output records 216 tensors and 20.32 MB. The earlier saved example contains stale execution output and does not prove clean Run All execution. This example does not satisfy final Python-code training. The optional Trainer example disables automatic saves; the final custom loop saves explicit adapter-only files and never relies on Trainer's PEFT serialization detection.

## 6. Final Python code generation fine-tuning (3 points)

**Implementation prepared; GPU execution BLOCKED.** Corrected two conflicts in the inherited final-task implementation: changed `codeparrot/codeparrot-clean-valid` to the user-required canonical `codeparrot/codeparrot-clean`; replaced cropping the entire file to 512 characters with retaining the first 512 characters of **every line**. The canonical dataset exposes only split `train`; its README links a separate smaller validation repository. To retain the exact requested identity, this version streams the canonical dataset and makes repository-disjoint training/evaluation subsets. It does not claim a native validation split.

Select .py files; remove duplicate processed content; use a seeded repository hash split; target 2,048 training and 128 held-out files. Pack EOS-separated code into 512-token sequences. Record file identities, processed/original character counts and line counts, token blocks, immutable revisions and manifest hashes. Smoke uses 32/16 files, two updates and shorter generations. Full training uses custom rank-8 attention plus MLP LoRA, micro-batch 1, accumulation 4 (effective batch 4), AdamW at 2e-4 with 25 warmup updates, and 500 **successful** optimizer updates. FP16 overflow-skipped attempts do not count. Parameter changes must be asserted before final saving.

Evaluate token-weighted held-out next-token negative log-likelihood and perplexity before and after; use identical prompts and greedy decoding settings for Python continuations. Syntax parsing is descriptive and does not establish functional correctness. Full training is gated on real GPU forward/backward, two optimizer updates, checkpoint save/reload, finite evaluation and identical reload generations in a matching environment.

**Actual final-task results:** 0 completed updates; training loss, held-out loss, before/after generations, GPU memory and final adapter are **NOT RUN / unavailable**. One real canonical dataset record was successfully streamed. Cloud CPU integration results (two synthetic updates and exact reload equality) establish basic code behavior only. They are not a pretrained 8B result or a 500-step run. Completion requires the actual GPU smoke, full 500-step experiment, final evaluation and reload checks described in CLOUD_EXECUTION_REPORT.md.

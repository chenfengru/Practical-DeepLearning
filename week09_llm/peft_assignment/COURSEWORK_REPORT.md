# Week 09: Parameter-Efficient Fine-Tuning

## Completed preliminary experiments

The saved notebook records successful custom prompt tuning (final next-token loss **0.0567**) and Hugging Face PEFT prompt tuning (**0.0857**). Both are below the required 0.1 threshold, and recorded continuations deny that the fox jumped over the dog. The custom LoRA numerical unit test and gradient check passed. Attention Q/K/V adapters were attached to all 36 transformer blocks, producing **108 modules** and **5,308,416 trainable parameters**. The historical checkpoint cell reports **100 example training steps**, **216 saved tensors**, and **20.32 MB**. These are preserved historical outputs, not newly repeated GPU measurements. The saved example Trainer cell also contains a stale `NameError`; its output alone does not prove a clean sequential run.

## Final experiment design

The final task uses `unsloth/Qwen3-8B-Base-bnb-4bit` with a frozen NF4 backbone and our own rank-8 LoRA implementation, rather than the library's LoRA wrapper. Adapters cover Q/K/V attention projections and gate/up/down MLP projections, satisfying the additional-adapter requirement: **216 modules, 432 adapter tensors**. The convention follows the assignment's numerical test, adding `x @ A @ B` without an additional scaling factor.

Python file prefixes come from `codeparrot/codeparrot-clean-valid`, the smaller validation repository linked by the assignment dataset. Each selected record is cropped to 512 characters. Repository-disjoint subsets contain 2,048 training and 128 evaluation files, with duplicate cropped text excluded. Prefixes are separated by EOS and packed into 512-token sequences. Training uses micro-batch size 1, gradient accumulation 4, learning rate 0.0002 with 25 warm-up updates, and 500 successful optimizer updates. Kaggle CUDA 0 uses FP16 compute and FP32 adapters; the second T4 is unused.

Before and after adaptation, the same held-out sequences are evaluated using token-weighted next-token negative log-likelihood and perplexity. Greedy continuations use eight fixed prompts: empty, `import`, `from`, `while`, `try`, `if`, `for`, and `torch`. An escaped HTML table displays the paired samples. Python syntax parsing is descriptive only: truncated fragments may fail, and syntax success does not demonstrate functional correctness.

## Model-loading correction

The first remote final-task attempt failed before training on Python 3.13 / PyTorch 2.11.0+cu130 / Transformers 5.20.2 / PEFT 0.21.2 / bitsandbytes 0.50.2 with Tesla T4 x2. The loader had replaced the checkpoint's native quantization dictionary with a `BitsAndBytesConfig` object. Transformers' pre-quantized dispatch called `.get()` on that object and raised `AttributeError`. The corrected loader keeps the dictionary, preserves the NF4 format, and overrides only compute dtype to FP16 for T4. It passes `dtype=torch.float16` and no redundant explicit quantization config. No dependency upgrade, downgrade or monkey-patch is required.

The uploaded notebook defaults to an actual model-load and two-update GPU smoke experiment. The 500-step run requires matching successful smoke evidence, including finite gradients and checkpoint reload checks. Earlier experiments remain gated off, their outputs preserved, and the teacher's section order retained.

## Checkpoint correction and validation

The final task uses a custom optimization loop, avoiding Transformers' treatment of the hand-written adapter as a PEFT model. Automatic Trainer checkpointing is disabled in the optional legacy example. Custom checkpoints save adapter tensors, module specifications, immutable source revisions, hyperparameters and the completed step count, without the base model. Loading checks names, shapes, dtypes and finite values; GPU execution will additionally verify matching logits and identical greedy generations after restoration. Checkpoints are for inference and do not claim exact optimizer continuation.

Local verification passed the original LoRA numerical assertions, prompt embedding/gradient checks, a tiny random Qwen3 training and checkpoint round trip, preprocessing checks and notebook structure/syntax checks. No 8B model was loaded locally.

## Results and limitations

**The attempted final GPU execution failed at model loading; no final-task training steps completed in that attempt. The corrected smoke and final 500-step experiments have not yet been executed.** Kaggle CLI authentication is unavailable in the local environment, and the Mac has no CUDA. Final held-out NLL, perplexity, training loss, runtime, GPU peak memory and generation comparison are therefore **unmeasured**, and a final-task adapter checkpoint is **not yet available**. The notebook is prepared to generate these artifacts and a replacement English report populated with actual metrics after a successful Kaggle run. The earlier 100-step quotes example does not satisfy the final Python training requirement.

The experiment is limited to a small held-out subset and does not test functional code correctness. Quantized CUDA operations are not guaranteed bitwise deterministic across environments. Report improvements only after inspecting measured before/after metrics and generations; an improvement is not assumed.

# Kaggle execution and checkpoint loading

## Prepared files

- Upload `week09_llm/Practice_peft_kaggle.ipynb` (single extension). It embeds its helper code; no repository attachment is required.
- `Practice_peft_kaggle.ipynb.ipynb` is the untouched original, including recorded results.
- `peft_utils.py` is the same helper implementation embedded in the upload notebook. `build_notebook.py` regenerates that notebook while preserving historical outputs.

The local environment has no Kaggle executable, Python Kaggle package, standard credential file, or Kaggle token/key environment variables. Therefore no remote job has been submitted and no final GPU metrics/checkpoint have been produced in this editing session.

## Exact execution procedure

1. Open [Kaggle Notebooks](https://www.kaggle.com/code), create a **private** notebook, and use **File → Import Notebook → Upload** to import `Practice_peft_kaggle.ipynb`.
2. In notebook **Settings / Session options**, set **Accelerator → GPU T4 x2**, enable **Internet**, and select a recent default PyTorch image. Start a fresh session so CUDA 0 has at least 13 GiB free. GPU eligibility may require Kaggle account verification.
3. Leave `RUN_LEGACY_EXPERIMENTS = False` and `RUN_FINAL_EXPERIMENT = True`. The earlier experiments retain their saved outputs and will be skipped by Run All. The original notebook remains an evidence archive; no earlier task needs retraining.
4. Required first run (the notebook default): leave `SMOKE_ONLY = True`; leave `EXPERIMENT_NAME = "python_custom_lora_v2"`. Use **Save Version → Save & Run All** with `SMOKE_ONLY=True`. This performs a real model load and only two optimizer updates, using the distinct output directory `/kaggle/working/python_custom_lora_v2_smoke`. Require both `MODEL_LOAD_PASSED` and `GPU_SMOKE_PASSED` in the execution log. Check the full 512-token backward test, all 216 adapter gradients, both checkpoint reload tests, and `training.global_step = 2` in `results.json`. Download the executed notebook, logs and complete smoke output directory, then attach that directory to the full-run notebook as a **private Kaggle Dataset**. A smoke run does **not** complete the assignment. It writes `status="smoke_only"` and `gpu_smoke_passed=true`; these fields are set only after its actual updates and reload assertions finish.
5. **Only after the smoke passes**, restart the session to release the smoke model. Set `SMOKE_RESULTS_PATH` to the attached smoke `results.json`, for example `/kaggle/input/YOUR-SMOKE-DATASET/python_custom_lora_v2_smoke/results.json` (use the actual path shown in Input). Set `SMOKE_ONLY = False`, keep the legacy flag false, and use **Save Version → Save & Run All**. The notebook validates the completed two-step evidence before loading the full-run model and again before optimization; missing, failed or incompatible evidence blocks the run. Keep the same source revisions, helper code, installed package versions and GPU type. This trains a fresh base for 500 successful optimizer updates and saves to `/kaggle/working/python_custom_lora_v2`. Keep the notebook private.
6. Watch the saved-version logs: baseline metrics, 216 adapter modules, gradient coverage, step 1 and every 25 updates, then held-out metrics and reload tests. The loop uses one GPU, not DataParallel/DDP. T4 x2 exposes two separate devices; this implementation intentionally leaves CUDA 1 unused.
7. When the version finishes, download its **executed notebook** and all files in the output directory. Verify `results.json` contains `status: "completed"`, `training.global_step: 500`, and both checkpoint reload flags `true`. Do not report a completed run from a smoke result or partial log.

Do not click Run All repeatedly within a live session. The exclusive output-directory creation and `TRAINING_STARTED.json` guard deliberately reject reruns. To deliberately repeat an experiment, restart the session and choose a new experiment name. Never rerun training on already trained weights. These are inference-only checkpoints; exact optimizer continuation is unsupported.

Dependencies are installed **only if missing**, without a blanket upgrade. The notebook records actual versions and immutable model/dataset revisions. If an installed Transformers is older than 4.56, explicitly install a compatible version on Kaggle and restart; do not install GPU packages or the model on the Mac. The `dtype=` model-loading API requires Transformers >=4.56; the confirmed Kaggle version 5.20.2 already qualifies. No dependency version change is needed for this fix. Quantized kernels are not guaranteed bitwise reproducible across hardware or package versions.

## Quantization failure and fix

The previous loader assigned a `BitsAndBytesConfig` object to `model_config.quantization_config`. In the installed Transformers 5.20.2, `get_hf_quantizer` passes that field to `supports_quant_method`, which calls `.get()` before converting it. This reproduces `AttributeError: 'BitsAndBytesConfig' object has no attribute 'get'` without any weights.

The fixed loader retains the checkpoint's native JSON dictionary and its NF4, double-quantization, storage and exclusion settings. It changes only `bnb_4bit_compute_dtype` to `float16` for T4, then calls `from_pretrained(config=model_config, dtype=torch.float16, device_map={"": 0}, ...)` with **no explicit `quantization_config` argument**. Transformers performs its normal pre-quantized dispatch. The teacher's original example supplies an explicit quantization config alongside this pre-quantized checkpoint; the new loader avoids that redundant combination. There is no monkey-patch and no requantization or dependency downgrade.

Local regression testing runs the actual Transformers dispatch/merge code with only the GPU environment validator stubbed. It does not construct the 8B model or download weights. This regression pass does **not** establish that the actual CUDA model load succeeds; that remains the first remote smoke check.

## Expected outputs

| File | Purpose |
| --- | --- |
| `adapter_final.pt` | Custom LoRA v1, 432 tensors, base revision, rank, target module specifications, step count; no base weights |
| `adapter_step_0002.pt` | Two-update smoke snapshot (smoke mode only) |
| `adapter_step_0100.pt` … `adapter_step_0500.pt` | Periodic inference-only adapter snapshots |
| `tokenizer/` | Tokenizer configuration for generation |
| `run_config.json` | Hyperparameters, immutable revisions, package versions, seed, GPU strategy |
| `data_manifest.json` | Cropped text hashes, repository/file identities, disjoint partitions |
| `token_blocks.pt` | The exact small tokenized training/evaluation partitions |
| `baseline.json` | Baseline token NLL, perplexity, syntax counts, eight continuations |
| `training_log.jsonl` | Actual per-update loss, learning rate, norm and elapsed time |
| `results.json` | Final metrics, before/after samples, smoke/reload checks, CUDA memory, checkpoint hash |
| `comparison.html` | HTML-escaped side-by-side generations |
| `coursework_report.md` | Concise English report filled with actual measured metrics |
| `TRAINING_STARTED.json` | Prevents implicit continuation or accidental duplicate training |

The historical 216-tensor, 20.32 MB checkpoint belongs to the **attention-only quotes example**, not the Python final task. Its saved output records existence on a previous remote session; the binary is not present locally. If you still have that session, download it before restarting. The final checkpoint adds MLP adapters and therefore has 432 tensors and a different size. Do not pass these custom files to `PeftModel.from_pretrained` or `model.save_pretrained`.

## Load and evaluate later on Kaggle (no training)

Start a **fresh** Kaggle GPU session. Attach the downloaded output directory as a private Kaggle Dataset. Paste/run the notebook's `custom-lora-utilities` definitions cell, then run the following, replacing only the dataset directory. Alternatively, upload `peft_utils.py` as a private notebook input and import its functions. No earlier experiments or training cells should run.

```python
artifact_dir = Path("/kaggle/input/YOUR-PRIVATE-DATASET/python_custom_lora_v2")
saved_run = json.loads((artifact_dir / "run_config.json").read_text())
config = ExperimentConfig(**saved_run["config"])
metadata = saved_run["metadata"]
require_kaggle_cuda()
seed_everything(config.seed)
model, tokenizer = load_kaggle_base(config, metadata["model_revision"])
modules = inject_custom_lora(model, config.rank)
loaded_metadata = load_custom_lora(model, artifact_dir / "adapter_final.pt", config.base_model, metadata["model_revision"])
assert loaded_metadata["global_step"] == 500
blocks = torch.load(artifact_dir / "token_blocks.pt", map_location="cpu", weights_only=True)
metrics = evaluate_loss(model, blocks["eval"], "cuda:0")
samples = generate_samples(model, tokenizer, "cuda:0", max_new_tokens=config.max_new_tokens)
print(json.dumps({"metrics": metrics, "samples": samples}, indent=2))
```

The actual mounted directory depends on how the output was packaged; select the folder containing `run_config.json` in Kaggle's Input panel. The loader validates checkpoint format, tensor names, dimensions, FP32 dtype, finite values, base name and revision before changing adapter tensors. Use checkpoints from this notebook that you trust. It never loads the 8B model on a local machine.

The older quotes example's flat format is not automatically accepted as v1: it has fewer targets and lacks pinned revisions. Preserve it as evidence; do not silently substitute it for final-task weights. The optional legacy example now uses `save_strategy="no"` and the explicit custom saver, avoiding Transformers' incompatible automatic PEFT saving. Its saved error output is historical and deliberately retained.

## Local verification

From the repository root:

```sh
python3 week09_llm/peft_assignment/build_notebook.py
python3 -m pytest week09_llm/peft_assignment/tests -q
```

These tests construct a tiny random Qwen3 **from configuration** and download no model weights or datasets. They cannot establish that NF4 kernels, T4 memory use, the 8B gradients, dataset streaming or final generation quality work remotely.

## Sources checked during implementation

- [CodeParrot clean dataset](https://huggingface.co/datasets/codeparrot/codeparrot-clean) and its linked [smaller validation repository](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid). The latter exposes split `train`; a repository-disjoint partition supplies this experiment's own held-out evaluation.
- [Published model config](https://huggingface.co/unsloth/Qwen3-8B-Base-bnb-4bit/blob/main/config.json), including the original BF16 compute setting overridden with FP16 for T4.
- [Hugging Face bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes): quantized backbones are adapted by training extra parameters.
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) for importing notebooks, selecting accelerators and saved runs.

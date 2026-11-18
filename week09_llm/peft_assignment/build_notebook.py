"""Build the upload-ready notebook without altering the original experiment archive."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Practice_peft_kaggle.ipynb.ipynb'
DESTINATION = ROOT / 'Practice_peft_kaggle.ipynb'


def cell(kind, source, label):
    result = {'cell_type': kind, 'id': label, 'metadata': {}, 'source': source.strip('\n').splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


original = json.loads(SOURCE.read_text())
helpers = (ROOT / 'peft_assignment/peft_utils.py').read_text()
cells = [cell('markdown', '''# Week 09 PEFT: Qwen3 Python code generation on Kaggle

Completed experiment outputs below are historical evidence, preserved from `Practice_peft_kaggle.ipynb.ipynb`. Default **Run All skips every historical experiment** and executes only the final task on a fresh base model. Do not enable the legacy flag unless you deliberately want to reproduce earlier work.

Recorded: custom prompt loss **0.0567**; HF prompt loss **0.0857**; custom LoRA unit and gradient tests passed; **108** attention LoRA modules; example run **100** steps; example adapter checkpoint **216 tensors / 20.32 MB**. Its checkpoint output confirms step 100, although the currently saved example Trainer cell has a stale `NameError` output. No final-task GPU results have been measured yet.

Final task: use the assignment's smaller CodeParrot validation repository, form disjoint training/evaluation subsets, crop each dataset record to 512 characters, pack 512-token sequences, attach our own attention and MLP LoRA, train 500 successful optimizer updates with effective batch size 4, and compare identical prompts and held-out token loss. The result is a coursework experiment, not a functional code benchmark.

Use Kaggle Settings → Accelerator → GPU T4 x2; Internet on. This notebook uses **one T4**, CUDA 0, with FP16 compute and FP32 adapters. The second T4 is intentionally unused. No 8B downloads or training are permitted locally. The default is a two-update GPU smoke run. The full run requires matching successful smoke evidence. New cells have no fabricated outputs.
''', 'overview'), cell('code', '''RUN_LEGACY_EXPERIMENTS = False
RUN_FINAL_EXPERIMENT = True
SMOKE_ONLY = True  # Default: actual model load + two GPU updates + checkpoint reload checks.
SMOKE_RESULTS_PATH = "/kaggle/working/python_custom_lora_v2_smoke/results.json"  # For full runs, set this to the attached successful smoke artifact.
EXPERIMENT_NAME = "python_custom_lora_v2"  # Use a new name for an intentionally new fresh run.
assert not (RUN_LEGACY_EXPERIMENTS and RUN_FINAL_EXPERIMENT), "Use a separate fresh session for legacy experiments."
''', 'execution-controls'), cell('code', helpers, 'custom-lora-utilities'), cell('markdown', '''## Preserved earlier experiments

All old code is gated by `RUN_LEGACY_EXPERIMENTS`. Outputs remain exactly as recorded and do not indicate execution of the edited notebook. The optional old Trainer example disables automatic saving; the final task does not use Trainer or masquerade as a PEFT model. The original double-extension notebook remains untouched.
''', 'historical-evidence')]

historical_tail = None
for index, item in enumerate(original['cells']):
    archived = copy.deepcopy(item)
    archived['id'] = f'legacy-{index:02d}'
    archived['metadata']['week09_historical_output'] = True
    if index == 33:
        historical_tail = archived
        continue  # Keep the teacher's closing material after the completed final task.
    if index in (31, 32):
        continue  # Replace the unfinished final-task cells below.
    if archived['cell_type'] == 'code':
        source = ''.join(archived['source'])
        if index == 1:
            source = source.replace('%pip install -q -U peft bitsandbytes accelerate datasets', '# Install missing packages using the final environment cell; no blanket upgrades.')
        if index == 15:
            source = source.replace('assert isinstance(model.model.embed_tokens, nn.Embedding), "please reload the model"', 'if isinstance(model.model.embed_tokens, WordEmbeddingsWithLearnedPrompts):\n    model.model.embed_tokens = model.model.embed_tokens.original_word_embeddings\nassert isinstance(model.model.embed_tokens, nn.Embedding)')
        if index == 28:
            source = source.replace('# silence a warning from HF trainer', '# Legacy Trainer quantization gate only; custom final loop needs no PEFT flag.')
            source = source.replace('logging_steps=1, output_dir=', 'save_strategy="no", logging_steps=1, output_dir=').replace('report_to="tensorboard"', 'report_to="none"')
        if index == 29:
            source = '''# Explicit custom checkpoint format; never use save_pretrained for this adapter.
assert trainer.state.global_step == 100, "Save the already completed example; do not retrain it."
save_path = "/kaggle/working/week09_example_custom_lora_v1.pt"
assert len(custom_adapter_state(model)) == 216
checkpoint = save_custom_lora(model, save_path, {"base_model": model_name, "rank": 8, "global_step": 100, "experiment": "historical_quotes_example"})
print(checkpoint)
'''
        archived['source'] = ('if RUN_LEGACY_EXPERIMENTS:\n' + ''.join('    ' + line + '\n' for line in source.splitlines())).splitlines(keepends=True)
    cells.append(archived)

cells.extend([
cell('markdown', '''## Final assignment — smoke first, then a fresh full experiment

Run the following cells in order with `SMOKE_ONLY=True` first. To start 500 steps later, attach its successful `results.json` as a private Kaggle input, set `SMOKE_RESULTS_PATH` to that file, and set `SMOKE_ONLY=False`. Full training is blocked without matching GPU smoke evidence. The output directory is exclusive: rerunning a training cell or reusing a run name raises an error. Weights-only periodic adapters support inference, **not exact training continuation**; optimizer, scaler, RNG and data cursor states are intentionally not advertised as resume checkpoints. To evaluate a retrieved checkpoint later, follow `peft_assignment/KAGGLE_RUN.md`.

The upstream smaller repository is `codeparrot/codeparrot-clean-valid`, exposed as split `train`. We subdivide it by repository hash, exclude duplicate cropped text, and save the selected file identities and token blocks. The 512-character crop means each dataset row/file prefix, followed by packing several prefixes into exact 512-token blocks. Both sides use identical preprocessing.
''', 'final-method'),
cell('code', '''if RUN_FINAL_EXPERIMENT:
    require_kaggle_cuda()
    import importlib.util
    import subprocess
    import sys
    # Install missing dependencies only; retain the Kaggle image's existing versions.
    requirements = {"transformers": "transformers>=4.56,<6", "peft": "peft>=0.15,<1", "bitsandbytes": "bitsandbytes>=0.45.5,<1", "accelerate": "accelerate>=1,<2", "datasets": "datasets>=3,<6", "huggingface_hub": "huggingface_hub>=0.30,<2"}
    missing = [requirement for module, requirement in requirements.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    import importlib.metadata
    from packaging.version import Version
    assert Version(importlib.metadata.version("transformers")) >= Version("4.56"), "The dtype= loading API used here needs Transformers >=4.56. Install a compatible version explicitly and restart."
    assert Version(torch.__version__.split("+")[0]) >= Version("2.3"), "Use a recent Kaggle PyTorch image."
    print({package: importlib.metadata.version(package) for package in ["torch", *requirements]})
''', 'final-environment'),
cell('code', '''if RUN_FINAL_EXPERIMENT:
    from huggingface_hub import HfApi
    import datasets
    config = ExperimentConfig()
    if SMOKE_ONLY:
        config.max_steps = 2
        config.save_steps = 2
        config.train_files = 32
        config.eval_files = 16
        config.max_new_tokens = 16
    seed_everything(config.seed)
    run_name = EXPERIMENT_NAME + ("_smoke" if SMOKE_ONLY else "")
    output_dir = Path("/kaggle/working") / run_name
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    memory_before = memory_snapshot("before_model_load")
    assert memory_before["devices"][0]["free_gib"] >= 13, "Free CUDA 0 by restarting the Kaggle session."
    api = HfApi()
    # Resolve immutable source revisions before downloading any model weights.
    model_revision = api.model_info(config.base_model).sha
    dataset_revision = api.dataset_info(config.dataset).sha
    environment = {package: importlib.metadata.version(package) for package in ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "huggingface_hub"]}
    environment.update({"python": sys.version.split()[0], "cuda": str(torch.version.cuda)})
    metadata = {"base_model": config.base_model, "model_revision": model_revision, "dataset": config.dataset, "dataset_revision": dataset_revision, "seed": config.seed, "environment": environment, "smoke_only": SMOKE_ONLY, "gpu_strategy": "single_process_cuda_0", "execution_device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0), "helper_sha256": HELPER_SOURCE_SHA256}
    if not SMOKE_ONLY:
        metadata["smoke_gate"] = validate_smoke_gate(SMOKE_RESULTS_PATH, config, metadata)
    output_dir.mkdir(exist_ok=False)  # Protect previous artifacts and accidental reruns.
    (output_dir / "run_config.json").write_text(json.dumps({"config": asdict(config), "metadata": metadata}, indent=2))
    stream = datasets.load_dataset(config.dataset, split="train", revision=dataset_revision, streaming=True)
    train_records, eval_records = select_python_examples(stream, config)
    model, tokenizer = load_kaggle_base(config, model_revision)
    model_load_passed = True  # Only reached after real NF4 load, FP16 and CUDA placement assertions.
    print("MODEL_LOAD_PASSED: native NF4 checkpoint on CUDA 0, FP16 compute", flush=True)
    device = torch.device("cuda:0")
    train_blocks = pack_examples(train_records, tokenizer, config.sequence_length)
    eval_blocks = pack_examples(eval_records, tokenizer, config.sequence_length)
    assert all(len(block) == 512 for block in train_blocks + eval_blocks)
    manifest = {"train": [{key: value for key, value in row.items() if key != "text"} for row in train_records], "eval": [{key: value for key, value in row.items() if key != "text"} for row in eval_records]}
    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    metadata["data_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    torch.save({"train": train_blocks, "eval": eval_blocks}, output_dir / "token_blocks.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    print({"train_files": len(train_records), "eval_files": len(eval_records), "train_sequences": len(train_blocks), "eval_sequences": len(eval_blocks), "sequence_length": config.sequence_length})
    memory_loaded = memory_snapshot("base_model_loaded")
''', 'final-data-and-base'),
cell('code', '''if RUN_FINAL_EXPERIMENT:
    # Capture the untouched base on exactly the same held-out sequences and prompts.
    baseline_metrics = evaluate_loss(model, eval_blocks, device)
    before_samples = generate_samples(model, tokenizer, device, max_new_tokens=config.max_new_tokens)
    (output_dir / "baseline.json").write_text(json.dumps({"metrics": baseline_metrics, "syntax": syntax_summary(before_samples), "samples": before_samples}, indent=2))
    base_probe_metrics = evaluate_loss(model, eval_blocks[:2], device)
    lora_modules = inject_custom_lora(model, config.rank)
    assert len(lora_modules) == 216  # 36 decoder layers x (3 attention + 3 MLP).
    assert len(custom_adapter_state(model)) == 432
    zero_adapter_metrics = evaluate_loss(model, eval_blocks[:2], device)
    assert abs(zero_adapter_metrics["token_nll"] - base_probe_metrics["token_nll"]) < 1e-5, "Zero-initialized LoRA changed base loss"
    probe_batch = collate_blocks([eval_blocks[0]], device)
    assert all(torch.count_nonzero(module.adapter_B).item() == 0 for _, module in lora_modules)
    smoke_metrics = gradient_smoke_test(model, train_blocks[0], device)
    print({"baseline": baseline_metrics, "gradient_smoke": smoke_metrics, "lora_modules": len(lora_modules), "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad)})
    memory_smoke = memory_snapshot("after_full_512_token_backward_smoke")
''', 'final-baseline-and-smoke'),
cell('code', '''if RUN_FINAL_EXPERIMENT:
    # 500 successful optimizer updates; micro-batch 1 x accumulation 4 = batch 4.
    # No Trainer, automatic PEFT checkpointing, or implicit continuation.
    if not SMOKE_ONLY:
        metadata["smoke_gate"] = validate_smoke_gate(SMOKE_RESULTS_PATH, config, metadata)
    training_metrics = train_custom_lora(model, train_blocks, device, config, output_dir, metadata)
    assert training_metrics["global_step"] == config.max_steps
    checkpoint_info = save_custom_lora(model, output_dir / "adapter_final.pt", {**metadata, "global_step": training_metrics["global_step"], "config": asdict(config)})
    print(checkpoint_info)
''', 'final-training'),
cell('code', '''if RUN_FINAL_EXPERIMENT:
    after_metrics = evaluate_loss(model, eval_blocks, device)
    after_samples = generate_samples(model, tokenizer, device, max_new_tokens=config.max_new_tokens)
    # Destructive-in-memory reload smoke: zero adapters, restore from disk, compare.
    model.eval()
    with torch.no_grad(), autocast_for(device):
        logits_before_reload = model(**probe_batch, use_cache=False).logits[:, -1].float().cpu()
    with torch.no_grad():
        for _, module in lora_modules:
            module.adapter_A.zero_()
            module.adapter_B.zero_()
    restored_metadata = load_custom_lora(model, output_dir / "adapter_final.pt", config.base_model, model_revision)
    with torch.no_grad(), autocast_for(device):
        logits_after_reload = model(**probe_batch, use_cache=False).logits[:, -1].float().cpu()
    assert torch.allclose(logits_before_reload, logits_after_reload, atol=1e-4, rtol=1e-4), "Checkpoint reload changed logits"
    assert any(torch.count_nonzero(module.adapter_B).item() for _, module in lora_modules)
    reload_samples = generate_samples(model, tokenizer, device, max_new_tokens=config.max_new_tokens)
    assert reload_samples == after_samples, "Reloaded greedy generations must match"
    memory_final = memory_snapshot("after_training_and_reload")
    results = {"status": "smoke_only" if SMOKE_ONLY else "completed", "metadata": metadata, "config": asdict(config), "model_load_passed": model_load_passed, "gpu_smoke_passed": bool(SMOKE_ONLY and training_metrics["global_step"] == 2), "baseline": baseline_metrics, "after": after_metrics, "nll_change": after_metrics["token_nll"] - baseline_metrics["token_nll"], "syntax_before": syntax_summary(before_samples), "syntax_after": syntax_summary(after_samples), "training": {key: value for key, value in training_metrics.items() if key != "history"}, "gradient_smoke": smoke_metrics, "checkpoint_reload_logits_passed": True, "checkpoint_reload_generations_passed": True, "checkpoint": checkpoint_info, "before_samples": before_samples, "after_samples": after_samples, "memory": [memory_before, memory_loaded, memory_smoke, memory_final]}
    if SMOKE_ONLY:
        validate_path = output_dir / "smoke_candidate.json"
        validate_path.write_text(json.dumps(results, indent=2))
        validate_smoke_gate(validate_path, config, metadata)
        validate_path.unlink()
        print("GPU_SMOKE_PASSED: two updates and adapter reload checks completed", flush=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({key: results[key] for key in ["status", "baseline", "after", "nll_change", "training", "checkpoint_reload_logits_passed"]}, indent=2))
''', 'final-evaluation-and-reload'),
cell('code', '''if RUN_FINAL_EXPERIMENT:
    from html import escape
    from IPython.display import HTML, display
    rows = ["<tr><th>Prompt</th><th>Before</th><th>After</th></tr>"]
    for prompt in PROMPTS:
        rows.append(f"<tr><td><pre>{escape(prompt or '(empty)')}</pre></td><td><pre>{escape(before_samples[prompt])}</pre></td><td><pre>{escape(after_samples[prompt])}</pre></td></tr>")
    comparison_html = '<table border="1" style="table-layout:fixed;width:100%;white-space:pre-wrap">' + "".join(rows) + "</table>"
    (output_dir / "comparison.html").write_text(comparison_html)
    display(HTML(comparison_html))
    improvement = baseline_metrics["token_nll"] - after_metrics["token_nll"]
    report = f"""# Week 09 PEFT: Python code generation

Run status: {results['status']}. Model: {config.base_model}, revision {model_revision}.
Dataset: {config.dataset}, revision {dataset_revision}. Repository-disjoint subsets: {len(train_records)} training and {len(eval_records)} evaluation files. Each file prefix was cropped to {config.max_characters} characters and packed into {config.sequence_length}-token sequences.

Our custom, unscaled rank-{config.rank} LoRA adapters were applied to Q/K/V attention projections and gate/up/down MLP projections ({len(lora_modules)} modules). Only adapters were optimized, using FP16 autocast with FP32 adapters, a frozen NF4 backbone, micro-batch 1, and {config.gradient_accumulation_steps} accumulation steps. Completed optimizer updates: {training_metrics['global_step']}.

Held-out token NLL: {baseline_metrics['token_nll']:.6f} before, {after_metrics['token_nll']:.6f} after. Perplexity: {baseline_metrics['perplexity']:.4f} before, {after_metrics['perplexity']:.4f} after. NLL improvement (positive is better): {improvement:.6f}. Training mean loss: {training_metrics['mean_training_loss']:.6f}. Runtime: {training_metrics['elapsed_seconds']:.1f} seconds.

Greedy continuations for eight fixed prompts are in comparison.html. Parseable nonempty samples: {syntax_summary(before_samples)['parseable_nonempty_samples']}/8 before and {syntax_summary(after_samples)['parseable_nonempty_samples']}/8 after. Syntax parsing is only descriptive: truncated continuations or Python 2 code may fail, and no functional correctness is asserted.

Adapter-only saving/loading passed matching-logit and identical-generation checks. Full numeric details, GPU memory measurements, environment versions, and immutable source revisions are in results.json. This small experiment does not establish generalization beyond the selected held-out subset; numerical determinism of quantized CUDA kernels is not guaranteed.
"""
    (output_dir / "coursework_report.md").write_text(report)
    print(report)
    print("Download all files from", output_dir, "using Kaggle Save Version → Save & Run All → Output.")
''', 'final-comparison-and-report'),
cell('markdown', '''## Completion checks

A full assignment run requires `SMOKE_ONLY = False`, `results.json` status `completed`, `training.global_step = 500`, 216 LoRA modules / 432 adapter tensors, finite held-out losses before and after, gradient coverage, and both checkpoint reload checks. Improvement is reported from measured values, never assumed. Download `adapter_final.pt`, tokenizer files, run config, token blocks, file manifest, logs, metrics, comparison HTML, report, and the executed notebook. A weights-only checkpoint cannot exactly resume optimization.

References: [CodeParrot dataset](https://huggingface.co/datasets/codeparrot/codeparrot-clean), [smaller validation repository](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid), [model configuration](https://huggingface.co/unsloth/Qwen3-8B-Base-bnb-4bit/blob/main/config.json), [Hugging Face bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes), [Kaggle notebooks](https://www.kaggle.com/docs/notebooks).
''', 'final-checklist')])
if historical_tail is not None:
    cells.append(historical_tail)
helper_hash = hashlib.sha256(helpers.encode()).hexdigest()
controls = next(item for item in cells if item.get('id') == 'execution-controls')
controls['source'].append(f'\nHELPER_SOURCE_SHA256 = {helper_hash!r}\n')
notebook = {'cells': cells, 'metadata': copy.deepcopy(original['metadata']), 'nbformat': 4, 'nbformat_minor': 5}
notebook['metadata'].pop('widgets', None)
notebook['metadata'].pop('colab', None)
notebook['metadata']['week09'] = {'archive': SOURCE.name, 'archive_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'helpers_sha256': hashlib.sha256(helpers.encode()).hexdigest(), 'default_execution': 'final_gpu_smoke_only'}
DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(DESTINATION)

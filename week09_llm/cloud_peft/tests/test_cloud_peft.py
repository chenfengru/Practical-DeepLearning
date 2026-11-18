"""Local tests: synthetic weights only, no CUDA or network/model downloads."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.quantizers.auto import AutoHfQuantizer
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
from transformers import BitsAndBytesConfig

ASSIGNMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT))
import peft_utils as utils


def tiny_model():
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=64, attention_dropout=0.0)
    return Qwen3ForCausalLM(config)


def test_native_quantization_config_and_real_quantizer_dispatch():
    native = {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True}
    config = Qwen3Config(quantization_config=copy.deepcopy(native))
    utils.configure_native_quantization(config)
    assert native["bnb_4bit_compute_dtype"] == "bfloat16"
    assert config.quantization_config == {**native, "bnb_4bit_compute_dtype": "float16"}
    quantizer = AutoHfQuantizer.from_config(config.quantization_config, pre_quantized=True)
    assert quantizer.pre_quantized
    assert quantizer.quantization_config.bnb_4bit_compute_dtype == torch.float16
    assert quantizer.quantization_config.bnb_4bit_quant_type == "nf4"


def test_original_grading_assertions_and_prompt_embeddings():
    archive = json.loads((ASSIGNMENT.parent / "Practice_peft_kaggle.ipynb.ipynb").read_text())
    scope = {"torch": torch, "nn": nn, "F": F, "LoRALayer": utils.CustomLoRALayer}
    exec("".join(archive["cells"][21]["source"]), scope)
    exec("".join(archive["cells"][8]["source"]), scope)
    scope["tokenizer"] = SimpleNamespace(pad_token_id=0)
    embeddings = nn.Embedding(20, 8).requires_grad_(False)
    layer = scope["WordEmbeddingsWithLearnedPrompts"](embeddings, 3)
    ids = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 0, 3, 4]])
    output = layer(ids)
    assert output.shape == (2, 5, 8)
    assert torch.equal(output[:, 3:], embeddings(ids[:, 3:]))
    assert torch.equal(output[:, :3], layer.learnable_prompts.expand(2, -1, -1))
    output.sum().backward()
    assert layer.learnable_prompts.grad is not None
    assert embeddings.weight.grad is None


def test_real_tiny_qwen_training_checkpoint_round_trip_and_rerun_guards(tmp_path):
    utils.seed_everything(42)
    model = tiny_model()
    base_state = copy.deepcopy(model.state_dict())
    blocks = [[1, 4, 7, 9, 2, 3, 6, 8], [1, 5, 2, 8, 9, 3, 6, 7]]
    original = utils.evaluate_loss(model, blocks, "cpu")
    modules = utils.inject_custom_lora(model, rank=2)
    assert len(modules) == 12
    assert utils.evaluate_loss(model, blocks, "cpu") == original
    smoke = utils.gradient_smoke_test(model, blocks[0], "cpu")
    assert smoke["modules_with_finite_nonzero_B_gradients"] == 12
    config = utils.ExperimentConfig(max_steps=2, save_steps=1, gradient_accumulation_steps=4, rank=2)
    metadata = {"base_model": "tiny-test", "model_revision": "test-revision"}
    result = utils.train_custom_lora(model, blocks, "cpu", config, tmp_path, metadata)
    assert result["global_step"] == 2 and len(result["history"]) == 2
    changed = sum(torch.count_nonzero(m.adapter_B).item() > 0 for _, m in modules)
    assert changed == len(modules)
    assert result["skipped_overflow_steps"] == 0
    assert utils.evaluate_loss(model, blocks, "cpu")["token_nll"] < original["token_nll"]
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in base_state.items() if ".self_attn." not in name and ".mlp." not in name)
    for _, module in modules:
        assert not module.module.weight.requires_grad
    saved = utils.save_custom_lora(model, tmp_path / "final.pt", metadata)
    assert saved["tensors"] == 24
    restored = tiny_model()
    restored.load_state_dict(base_state)
    utils.inject_custom_lora(restored, rank=2)
    utils.load_custom_lora(restored, tmp_path / "final.pt", "tiny-test", "test-revision")
    assert utils.evaluate_loss(restored, blocks, "cpu") == utils.evaluate_loss(model, blocks, "cpu")
    batch = utils.collate_blocks([blocks[0]], "cpu")
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model(**batch).logits, restored(**batch).logits)
        assert torch.equal(model.generate(**batch, max_new_tokens=3, do_sample=False, pad_token_id=0), restored.generate(**batch, max_new_tokens=3, do_sample=False, pad_token_id=0))
    (ASSIGNMENT / "evidence/cpu_smoke.json").write_text(json.dumps({"status": "VERIFIED CPU ONLY", "model": "random tiny Qwen3; not pretrained 8B", "device": "cpu", "baseline": original, "after": utils.evaluate_loss(model, blocks, "cpu"), "training": result, "modules": len(modules), "modules_with_updated_B": changed, "checkpoint_tensors": saved["tensors"], "reload_logits_equal": True, "reload_generations_equal": True}, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="already started"):
        utils.train_custom_lora(model, blocks, "cpu", config, tmp_path, metadata)
    with pytest.raises(FileExistsError):
        utils.save_custom_lora(model, tmp_path / "final.pt", metadata)
    with pytest.raises(RuntimeError, match="already attached"):
        utils.inject_custom_lora(model)
    with pytest.raises(ValueError, match="base model mismatch"):
        utils.load_custom_lora(model, tmp_path / "final.pt", "wrong-model")
    with pytest.raises(ValueError, match="revision mismatch"):
        utils.load_custom_lora(model, tmp_path / "final.pt", expected_revision="wrong-revision")
    payload = torch.load(tmp_path / "final.pt", weights_only=True)
    payload["state_dict"].pop(next(iter(payload["state_dict"])))
    torch.save(payload, tmp_path / "corrupt.pt")
    with pytest.raises(ValueError, match="names do not match"):
        utils.load_custom_lora(model, tmp_path / "corrupt.pt")


def test_dataset_deduplication_repo_split_crop_and_packing():
    stream = [{"repo_name": f"repo-{index}", "path": f"file-{index}.py", "content": f"print({index})\n" + "x" * 600} for index in range(1000)]
    stream.insert(0, {"repo_name": "not-python", "path": "readme.md", "content": "ignore"})
    stream.insert(2, stream[1].copy())
    config = utils.ExperimentConfig(train_files=8, eval_files=4, max_scanned=1000)
    train, evaluation = utils.select_python_examples(stream, config)
    assert len(train) == 8 and len(evaluation) == 4
    assert all(len(row["text"].split("\n")[1]) == 512 and row["text"].startswith("print(") for row in train + evaluation)
    assert len({row["sha256"] for row in train + evaluation}) == 12
    assert {row["repo_name"] for row in train}.isdisjoint(row["repo_name"] for row in evaluation)
    assert (train, evaluation) == utils.select_python_examples(stream, config)
    class Tokenizer:
        eos_token_id = 31

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(char) % 30 for char in text]}
    blocks = utils.pack_examples(train, Tokenizer(), 512)
    assert all(len(block) == 512 for block in blocks)
    assert any(31 in block for block in blocks)
    with pytest.raises(RuntimeError, match="Insufficient"):
        utils.select_python_examples([], config)


def test_causal_alignment_and_eos_preservation():
    ids = torch.tensor([[1, 2, 0, 3]])
    logits = torch.zeros(1, 4, 4)
    for index, target in enumerate(ids[0, 1:]):
        logits[0, index, target] = 20
    assert utils.causal_loss(logits, ids).item() < 1e-5
    batch = utils.collate_blocks([[1, 2, 0, 3]], "cpu")
    assert torch.all(batch["attention_mask"] == 1)


def test_cuda_guard_precedes_any_hub_access():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cloud CUDA"):
            utils.load_cloud_base(utils.ExperimentConfig(), "no-network")


def test_notebook_schema_syntax_archive_preservation_and_inactive_execution():
    path = ASSIGNMENT.parent / "Practice_peft_cloud.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    archive_path = ASSIGNMENT.parent / "Practice_peft_kaggle.ipynb.ipynb"
    archive = json.loads(archive_path.read_text())
    assert notebook.metadata.week09.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert notebook.metadata.week09.helpers_sha256 == hashlib.sha256((ASSIGNMENT / "peft_utils.py").read_bytes()).hexdigest()
    assert next(item.source for item in notebook.cells if item.id == "custom-lora-utilities") == (ASSIGNMENT / "peft_utils.py").read_text().strip("\n")
    scope = {}
    for item in notebook.cells:
        if item.cell_type != "code":
            continue
        compile(item.source, f"{path.name}:{item.id}", "exec")
        if item.id.startswith("legacy-"):
            index = int(item.id.split("-")[1])
            assert item.outputs == archive["cells"][index].get("outputs", [])
            assert item.execution_count == archive["cells"][index].get("execution_count")
        exec(item.source, scope)
        if item.id == "execution-controls":
            scope["RUN_LEGACY_EXPERIMENTS"] = False
            scope["RUN_FINAL_EXPERIMENT"] = False
    assert "model" not in scope
    training_source = next(item.source for item in notebook.cells if item.id == "final-training")
    assert "trainer.train(" not in training_source
    assert utils.ExperimentConfig().max_steps == 500
    assert utils.ExperimentConfig().gradient_accumulation_steps == 4


def test_syntax_metric_does_not_execute_generated_code():
    result = utils.syntax_summary({"valid": "import os", "invalid": "for", "empty": "", "danger": "raise RuntimeError('never executed')"})
    assert result["parseable_nonempty_samples"] == 2


def test_gpu_smoke_gate_rejects_missing_failed_and_incompatible_evidence(tmp_path):
    # Synthetic evidence tests gate behavior only; it is never saved as GPU results.
    config = utils.ExperimentConfig()
    metadata = {"base_model": config.base_model, "model_revision": "revision", "dataset": config.dataset, "dataset_revision": "data-revision", "dataset_split": "train", "helper_sha256": "helper-hash", "environment": {"torch": "test"}, "gpu_name": "Tesla T4", "execution_device": "cuda:0"}
    evidence = {"status": "smoke_only", "model_load_passed": True, "gpu_smoke_passed": True, "checkpoint_reload_logits_passed": True, "checkpoint_reload_generations_passed": True, "training": {"global_step": 2}, "metadata": metadata, "config": utils.asdict(config), "gradient_smoke": {"modules_with_finite_nonzero_B_gradients": 216}, "checkpoint": {"tensors": 432}, "baseline": {"token_nll": 2.0}, "after": {"token_nll": 1.9}}
    path = tmp_path / "synthetic-evidence.json"
    with pytest.raises(RuntimeError, match="run SMOKE_ONLY=True"):
        utils.validate_smoke_gate(path, config, metadata)
    path.write_text(json.dumps(evidence))
    assert utils.validate_smoke_gate(path, config, metadata)["global_step"] == 2
    for key, value in [("gpu_smoke_passed", False), ("training", {"global_step": 0}), ("gradient_smoke", {"modules_with_finite_nonzero_B_gradients": 108}), ("checkpoint", {"tensors": 216}), ("after", {"token_nll": float("nan")})]:
        path.write_text(json.dumps({**evidence, key: value}))
        with pytest.raises(RuntimeError, match="Full training blocked"):
            utils.validate_smoke_gate(path, config, metadata)
    path.write_text(json.dumps(evidence))
    for key in ("model_revision", "helper_sha256", "gpu_name", "environment"):
        with pytest.raises(RuntimeError, match="mismatch"):
            utils.validate_smoke_gate(path, config, {**metadata, key: "changed"})
    with pytest.raises(RuntimeError, match="hyperparameter mismatch"):
        utils.validate_smoke_gate(path, utils.ExperimentConfig(rank=4), metadata)


def test_teacher_section_order_and_default_smoke_path():
    notebook = nbformat.read(ASSIGNMENT.parent / "Practice_peft_cloud.ipynb", as_version=4)
    teacher = nbformat.read(ASSIGNMENT / "evidence/teacher_original.ipynb", as_version=4)
    headings = [item.source.splitlines()[0] for item in teacher.cells if item.cell_type == "markdown" and item.source.startswith("### ")][1:]
    markdown = [item.source for item in notebook.cells if item.cell_type == "markdown"]
    positions = [next(index for index, text in enumerate(markdown) if text.startswith(heading)) for heading in headings]
    assert positions == sorted(positions)
    assert "If you reach this:" in markdown[-1]
    controls = next(item.source for item in notebook.cells if item.id == "execution-controls")
    scope = {}
    exec(controls, scope)
    assert scope["RUN_LEGACY_EXPERIMENTS"] is False
    assert scope["RUN_FINAL_EXPERIMENT"] is True
    assert scope["SMOKE_ONLY"] is True
    setup = next(item.source for item in notebook.cells if item.id == "final-data-and-base")
    assert setup.index("validate_smoke_gate(") < setup.index("load_cloud_base(")
    training = next(item.source for item in notebook.cells if item.id == "final-training")
    assert training.index("validate_smoke_gate(") < training.index("train_custom_lora(")

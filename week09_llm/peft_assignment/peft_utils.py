"""CPU-testable custom LoRA and Kaggle-only QLoRA experiment utilities.

The notebook embeds this file verbatim; it does not need a repository upload.
No model or dataset is downloaded at import time.
"""

import ast
import hashlib
import json
import math
import os
import random
import time
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ExperimentConfig:
    base_model: str = "unsloth/Qwen3-8B-Base-bnb-4bit"
    dataset: str = "codeparrot/codeparrot-clean-valid"
    seed: int = 42
    rank: int = 8
    max_characters: int = 512
    sequence_length: int = 512
    train_files: int = 2048
    eval_files: int = 128
    max_scanned: int = 30000
    max_steps: int = 500
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 25
    save_steps: int = 100
    max_new_tokens: int = 128


TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
PROMPTS = ("", "import", "from", "while", "try", "if", "for", "torch")


class CustomLoRALayer(nn.Module):
    """Original assignment convention: y = base(x) + x A B, without scaling."""

    def __init__(self, module, rank):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.module = module
        self.module.requires_grad_(False)
        self.adapter_A = nn.Parameter(torch.empty(module.in_features, rank, device=module.weight.device, dtype=torch.float32))
        self.adapter_B = nn.Parameter(torch.zeros(rank, module.out_features, device=module.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.adapter_A, a=math.sqrt(5))

    def forward(self, inputs):
        base = self.module(inputs)
        # Keep adapter arithmetic in FP32 even inside FP16 autocast (T4).
        context = torch.autocast("cuda", enabled=False) if inputs.is_cuda else nullcontext()
        with context:
            delta = inputs.to(device=self.adapter_A.device, dtype=self.adapter_A.dtype) @ self.adapter_A @ self.adapter_B
        return base + delta.to(device=base.device, dtype=base.dtype)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Quantized GPU kernels are not guaranteed bitwise deterministic.
    torch.use_deterministic_algorithms(True, warn_only=True)


def require_kaggle_cuda():
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Run the final experiment on Kaggle with GPU T4 x2 and Internet enabled. No local model download is allowed.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This notebook uses a single CUDA process; do not launch it with torchrun.")


def memory_snapshot(label):
    require_kaggle_cuda()
    result = {"label": label, "devices": []}
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        result["devices"].append({"index": index, "name": torch.cuda.get_device_name(index), "free_gib": free / 2**30, "total_gib": total / 2**30, "allocated_gib": torch.cuda.memory_allocated(index) / 2**30, "peak_gib": torch.cuda.max_memory_allocated(index) / 2**30})
    print(json.dumps(result), flush=True)
    return result


def validate_smoke_gate(path, config, metadata):
    """Require completed GPU evidence matching this code, environment and sources.

    This validates a previously retrieved results.json, never creates a pass.
    CPU tests use synthetic evidence explicitly and do not imply GPU execution.
    """
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("Full training blocked: run SMOKE_ONLY=True and attach its successful results.json first")
    evidence = json.loads(path.read_text())
    checks = ("model_load_passed", "gpu_smoke_passed", "checkpoint_reload_logits_passed", "checkpoint_reload_generations_passed")
    if evidence.get("status") != "smoke_only" or not all(evidence.get(key) is True for key in checks):
        raise RuntimeError("Full training blocked: GPU model-load, two-update and reload smoke checks must all pass")
    if evidence.get("training", {}).get("global_step") != 2:
        raise RuntimeError("Full training blocked: smoke must complete exactly two successful optimizer updates")
    previous = evidence.get("metadata", {})
    for key in ("base_model", "model_revision", "dataset", "dataset_revision", "helper_sha256", "environment", "gpu_name", "execution_device"):
        if key not in metadata or previous.get(key) != metadata[key]:
            raise RuntimeError(f"Full training blocked: smoke metadata mismatch for {key}; rerun smoke on this environment/code")
    if previous.get("execution_device") != "cuda:0":
        raise RuntimeError("Full training blocked: smoke evidence must come from CUDA 0")
    reduced = {"max_steps", "save_steps", "train_files", "eval_files", "max_new_tokens"}
    for key, value in asdict(config).items():
        if key not in reduced and evidence.get("config", {}).get(key) != value:
            raise RuntimeError(f"Full training blocked: smoke hyperparameter mismatch for {key}")
    if evidence.get("gradient_smoke", {}).get("modules_with_finite_nonzero_B_gradients") != 216:
        raise RuntimeError("Full training blocked: incomplete adapter gradient coverage")
    if evidence.get("checkpoint", {}).get("tensors") != 432:
        raise RuntimeError("Full training blocked: incorrect smoke adapter tensor count")
    for stage in ("baseline", "after"):
        value = evidence.get(stage, {}).get("token_nll")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError("Full training blocked: missing finite smoke evaluation metrics")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "global_step": 2}


def inject_custom_lora(model, rank=8):
    if any(isinstance(module, CustomLoRALayer) for module in model.modules()):
        raise RuntimeError("Adapters are already attached; do not overwrite trained weights.")
    model.requires_grad_(False)
    for layer in model.model.layers:
        for target in TARGETS:
            component, name = target.split(".")
            parent = getattr(layer, component)
            setattr(parent, name, CustomLoRALayer(getattr(parent, name), rank))
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, CustomLoRALayer)]
    assert len(modules) == 6 * len(model.model.layers)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert len(trainable) == 2 * len(modules)
    assert all(name.endswith((".adapter_A", ".adapter_B")) for name in trainable)
    return modules


def custom_adapter_state(model):
    # Also recognizes the original notebook's LoRALayer, without a PEFT flag.
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if name.endswith((".adapter_A", ".adapter_B"))}


def save_custom_lora(model, path, metadata):
    state = custom_adapter_state(model)
    if not state or not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Cannot save empty or nonfinite adapter weights")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserving existing checkpoint: {path}")
    modules = []
    for name, module in model.named_modules():
        if hasattr(module, "adapter_A") and hasattr(module, "adapter_B"):
            modules.append({"name": name, "rank": module.adapter_A.shape[1], "in_features": module.adapter_A.shape[0], "out_features": module.adapter_B.shape[1]})
    payload = {"format": "week09-custom-lora", "format_version": 1, "convention": "base(x) + x @ A @ B; scale=1", "modules": modules, "metadata": metadata, "state_dict": state, "resume_supported": False}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "tensors": len(state), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_custom_lora(model, path, expected_base=None, expected_revision=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "week09-custom-lora" or payload.get("format_version") != 1:
        raise ValueError("Expected explicit week09-custom-lora v1 checkpoint; legacy flat checkpoints must be migrated explicitly")
    if expected_base and payload["metadata"].get("base_model") != expected_base:
        raise ValueError("Checkpoint base model mismatch")
    if expected_revision and payload["metadata"].get("model_revision") != expected_revision:
        raise ValueError("Checkpoint base revision mismatch")
    current = custom_adapter_state(model)
    state = payload["state_dict"]
    if set(state) != set(current):
        raise ValueError("Adapter tensor names do not match; inject the same targets and rank first")
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if value.shape != current[name].shape or value.dtype != current[name].dtype or not torch.isfinite(value).all().item():
            raise ValueError(f"Invalid adapter tensor: {name}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))
    return payload["metadata"]


def select_python_examples(stream, config):
    train, evaluation, seen = [], [], set()
    for scanned, example in enumerate(stream, 1):
        path = example.get("path", "")
        text = example.get("content", "")[:config.max_characters]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if path.endswith(".py") and text.strip() and digest not in seen:
            # Split by repository to reduce leakage between files from one project.
            repository = example.get("repo_name") or path
            bucket = int(hashlib.sha256(f"{config.seed}:{repository}".encode()).hexdigest(), 16) % 10
            destination, limit = (evaluation, config.eval_files) if bucket == 0 else (train, config.train_files)
            if len(destination) < limit:
                destination.append({"text": text, "sha256": digest, "path": path, "repo_name": repository})
                seen.add(digest)
        if len(train) == config.train_files and len(evaluation) == config.eval_files:
            break
        if scanned >= config.max_scanned:
            break
    if len(train) != config.train_files or len(evaluation) != config.eval_files:
        raise RuntimeError(f"Insufficient unique Python files within scan limit: train={len(train)}, eval={len(evaluation)}")
    assert {row["sha256"] for row in train}.isdisjoint(row["sha256"] for row in evaluation)
    assert {row["repo_name"] for row in train}.isdisjoint(row["repo_name"] for row in evaluation)
    return train, evaluation


def pack_examples(records, tokenizer, sequence_length):
    tokens = []
    for row in records:
        tokens.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        tokens.append(tokenizer.eos_token_id)
    # Fixed 512-token blocks, packing multiple short files with EOS boundaries.
    blocks = [tokens[start:start + sequence_length] for start in range(0, len(tokens) - sequence_length + 1, sequence_length)]
    if not blocks:
        raise ValueError("Not enough tokens for one full sequence")
    return blocks


def collate_blocks(blocks, device):
    ids = torch.tensor(blocks, dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def causal_loss(logits, input_ids):
    # EOS is genuine data, never masked by matching its token ID to PAD.
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))


def autocast_for(device):
    return torch.autocast("cuda", dtype=torch.float16) if torch.device(device).type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate_loss(model, blocks, device):
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    for block in blocks:
        batch = collate_blocks([block], device)
        with autocast_for(device):
            output = model(**batch, use_cache=False)
            loss = causal_loss(output.logits, batch["input_ids"])
        count = len(block) - 1
        total += loss.item() * count
        tokens += count
        del output, loss  # Do not retain vocabulary-sized logits for the next batch.
    model.train(was_training)
    nll = total / tokens
    assert math.isfinite(nll)
    return {"token_nll": nll, "perplexity": math.exp(nll) if nll < 700 else None, "predicted_tokens": tokens, "sequences": len(blocks)}


@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts=PROMPTS, max_new_tokens=128):
    was_training = model.training
    model.eval()
    results = {}
    for prompt in prompts:
        if prompt:
            batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        else:
            start = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            batch = collate_blocks([[start]], device)
        with autocast_for(device):
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        results[prompt] = prompt + tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    model.train(was_training)
    return results


def syntax_summary(samples):
    # Truncated continuations can fail ast.parse even if they resemble Python.
    parsed = 0
    for sample in samples.values():
        try:
            ast.parse(sample)
            parsed += bool(sample.strip())
        except (SyntaxError, ValueError):
            pass
    return {"parseable_nonempty_samples": parsed, "total_samples": len(samples), "limitation": "Syntax only; truncated Python and Python 2 can fail. No functional correctness claim."}


def gradient_smoke_test(model, block, device):
    """Backward only: leaves weights and optimizer state untouched."""
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        batch = collate_blocks([block], device)
        with autocast_for(device):
            output = model(**batch, use_cache=False)
            loss = causal_loss(output.logits, batch["input_ids"])
        assert torch.isfinite(loss).item()
        loss.backward()
        adapters = [(name, module) for name, module in model.named_modules() if isinstance(module, CustomLoRALayer)]
        assert adapters
        for name, module in adapters:
            assert module.adapter_A.grad is not None and torch.isfinite(module.adapter_A.grad).all().item(), name
            assert module.adapter_B.grad is not None and torch.isfinite(module.adapter_B.grad).all().item(), name
            assert module.adapter_B.grad.norm().item() > 0, name
        assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.endswith((".adapter_A", ".adapter_B")))
        return {"loss": loss.item(), "modules_with_finite_nonzero_B_gradients": len(adapters)}
    finally:
        model.zero_grad(set_to_none=True)


def train_custom_lora(model, blocks, device, config, output_dir, metadata):
    """Fresh run only. Periodic files are inference weights, not resume checkpoints."""
    if getattr(model, "_week09_training_started", False):
        raise RuntimeError("Training already started on these weights; reload a fresh base for a new run")
    if config.max_steps < 1 or config.gradient_accumulation_steps < 1 or config.save_steps < 1:
        raise ValueError("Training steps, accumulation, and save interval must be positive")
    marker = Path(output_dir) / "TRAINING_STARTED.json"
    with marker.open("x") as file:
        json.dump({"config": asdict(config), "continuation": "unsupported: reload a fresh base and use a new output directory"}, file)
    model._week09_training_started = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(parameter.dtype == torch.float32 for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda")
    generator = random.Random(config.seed)
    order = list(range(len(blocks)))
    generator.shuffle(order)
    cursor, history, completed, skipped = 0, [], 0, 0
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while completed < config.max_steps:
        average_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            if cursor == len(order):
                generator.shuffle(order)
                cursor = 0
            batch = collate_blocks([blocks[order[cursor]]], device)
            cursor += 1
            with autocast_for(device):
                output = model(**batch, use_cache=False)
                loss = causal_loss(output.logits, batch["input_ids"])
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite training loss")
            average_loss += loss.item() / config.gradient_accumulation_steps
            scaler.scale(loss / config.gradient_accumulation_steps).backward()
            del output, loss
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(trainable, 1.0)
        lr = config.learning_rate * min((completed + 1) / max(config.warmup_steps, 1), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() < previous_scale:
            skipped += 1
            if skipped > 20:
                raise FloatingPointError("Too many FP16 overflow steps; the run is incomplete")
            continue
        if not torch.isfinite(grad_norm).item():
            raise FloatingPointError("Nonfinite gradient norm")
        completed += 1
        entry = {"step": completed, "loss": average_loss, "learning_rate": lr, "grad_norm": float(grad_norm), "elapsed_seconds": time.monotonic() - started}
        history.append(entry)
        with (Path(output_dir) / "training_log.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        if completed == 1 or completed % 25 == 0:
            print(json.dumps(entry), flush=True)
        if completed % config.save_steps == 0:
            save_custom_lora(model, Path(output_dir) / f"adapter_step_{completed:04d}.pt", {**metadata, "global_step": completed, "config": asdict(config)})
    return {"global_step": completed, "skipped_overflow_steps": skipped, "mean_training_loss": sum(item["loss"] for item in history) / len(history), "elapsed_seconds": time.monotonic() - started, "history": history}


def configure_native_quantization(model_config):
    """Retain the checkpoint's JSON config for Transformers' pre-quantized path.

    Transformers 5.20.2 calls supports_quant_method(...).get() before it
    constructs BitsAndBytesConfig. Assigning that object here breaks dispatch.
    T4 requires an FP16 compute override; this does not requantize the weights.
    """
    native = getattr(model_config, "quantization_config", None)
    if not isinstance(native, dict):
        raise TypeError("Expected the pre-quantized checkpoint's native quantization dictionary")
    if native.get("quant_method") != "bitsandbytes" or not native.get("load_in_4bit", native.get("_load_in_4bit", False)):
        raise ValueError("Expected a native 4-bit bitsandbytes checkpoint")
    model_config.quantization_config = deepcopy(native)
    model_config.quantization_config["bnb_4bit_compute_dtype"] = "float16"
    return model_config


def load_kaggle_base(config, revision):
    require_kaggle_cuda()  # Must precede every from_pretrained call.
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import prepare_model_for_kbit_training

    model_config = configure_native_quantization(AutoConfig.from_pretrained(config.base_model, revision=revision))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    # No explicit quantization_config argument: the checkpoint is already NF4.
    model = AutoModelForCausalLM.from_pretrained(config.base_model, revision=revision, config=model_config, device_map={"": 0}, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    assert model.is_loaded_in_4bit
    assert all(parameter.device.type == "cuda" and parameter.device.index == 0 for parameter in model.parameters())
    quantized_modules = [module for module in model.modules() if hasattr(module, "compute_dtype")]
    assert quantized_modules and all(module.compute_dtype == torch.float16 for module in quantized_modules)
    return model, tokenizer

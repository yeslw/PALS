import argparse
import csv
import gzip
import inspect
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.data import C4_TRAIN_PATH, get_loaders, set_seed
from lib.layerwrapper import WrappedGPT, get_effective_layer_weight, prune_effective_layer_weight, set_effective_layer_weight
from lib.prune import find_layers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--probe_sparsity", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    return parser.parse_args()


def normalize_device(device_like):
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, int):
        return torch.device(f"cuda:{device_like}")
    if isinstance(device_like, str):
        if device_like == "disk":
            raise RuntimeError("disk offload is not supported by this script")
        return torch.device(device_like)
    raise TypeError(f"unsupported device specifier: {device_like}")


def get_module_device(module, fallback_device):
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return fallback_device


def infer_model_label(model_name_or_path):
    stripped = model_name_or_path.rstrip("/")
    base = os.path.basename(stripped)
    return base if base else stripped


def build_c4_sample(text, seqlen, tokenizer, rng):
    trainenc = tokenizer(text, return_tensors="pt")
    if trainenc.input_ids.shape[1] <= seqlen:
        return None

    start = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    end = start + seqlen
    inp = trainenc.input_ids[:, start:end]
    tar = inp.clone()
    tar[:, :-1] = -100
    return inp, tar


def extract_c4_text(line):
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="ignore")
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    text = record.get("text")
    if not isinstance(text, str) or not text:
        return None
    return text


def build_c4_local_calibration_loader(nsamples, seed, seqlen, tokenizer):
    rng = random.Random(seed)
    trainloader = []

    if C4_TRAIN_PATH.endswith(".gz"):
        with gzip.open(C4_TRAIN_PATH, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                text = extract_c4_text(line)
                if text is None:
                    continue
                sample = build_c4_sample(text, seqlen, tokenizer, rng)
                if sample is None:
                    continue
                trainloader.append(sample)
                if len(trainloader) >= nsamples:
                    break
    else:
        file_size = os.path.getsize(C4_TRAIN_PATH)
        max_attempts = max(nsamples * 128, 4096)

        with open(C4_TRAIN_PATH, "rb") as handle:
            attempts = 0
            while len(trainloader) < nsamples and attempts < max_attempts:
                attempts += 1
                offset = rng.randrange(file_size)
                handle.seek(offset)
                if offset != 0:
                    handle.readline()
                line = handle.readline()
                if not line:
                    handle.seek(0)
                    line = handle.readline()
                text = extract_c4_text(line)
                if text is None:
                    continue
                sample = build_c4_sample(text, seqlen, tokenizer, rng)
                if sample is None:
                    continue
                trainloader.append(sample)

        if len(trainloader) < nsamples:
            with open(C4_TRAIN_PATH, "rt", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    text = extract_c4_text(line)
                    if text is None:
                        continue
                    sample = build_c4_sample(text, seqlen, tokenizer, rng)
                    if sample is None:
                        continue
                    trainloader.append(sample)
                    if len(trainloader) >= nsamples:
                        break

    if len(trainloader) != nsamples:
        raise RuntimeError(
            f"unable to build {nsamples} c4 calibration samples from local fallback; got {len(trainloader)}"
        )

    return trainloader, None


def load_calibration_data(dataset_name, nsamples, seed, seqlen, tokenizer):
    try:
        return get_loaders(dataset_name, nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
    except NotImplementedError as exc:
        if "c4" in dataset_name and "LocalFileSystem" in str(exc):
            print("falling back to direct local C4 calibration loader due to datasets LocalFileSystem cache issue")
            return build_c4_local_calibration_loader(nsamples, seed, seqlen, tokenizer)
        raise


def get_model_info(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return {
            "blocks": model.model.layers,
            "block_prefixes": ["model.layers"],
            "embed_keys": ["model.embed_tokens"],
            "embed_modules": [getattr(model.model, "embed_tokens", None)],
            "kind": "llama_like",
        }
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return {
            "blocks": model.model.decoder.layers,
            "block_prefixes": ["model.decoder.layers", "model.layers"],
            "embed_keys": ["model.decoder.embed_tokens", "model.embed_tokens"],
            "embed_modules": [getattr(model.model.decoder, "embed_tokens", None), getattr(model.model, "embed_tokens", None)],
            "kind": "opt_like",
        }
    raise RuntimeError("unsupported model structure: expected model.model.layers or model.model.decoder.layers")


def get_embed_device(model, model_info, fallback_device):
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for key in model_info["embed_keys"]:
            if key in device_map:
                return normalize_device(device_map[key])
    for module in model_info["embed_modules"]:
        if module is not None:
            return get_module_device(module, fallback_device)
    return fallback_device


def get_block_device(model, model_info, layer_idx, fallback_device):
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for prefix in model_info["block_prefixes"]:
            key = f"{prefix}.{layer_idx}"
            if key in device_map:
                return normalize_device(device_map[key])
        suffix = f".layers.{layer_idx}"
        for key, value in device_map.items():
            if key.endswith(suffix):
                return normalize_device(value)
    return get_module_device(model_info["blocks"][layer_idx], fallback_device)


def move_value_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_value_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_value_to_device(item, device) for key, item in value.items()}
    return value


def clone_value_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(clone_value_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [clone_value_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_value_to_cpu(item) for key, item in value.items()}
    return value


def supported_block_arg_names(block):
    return set(inspect.signature(block.forward).parameters.keys())


def prepare_block_kwargs(block, cached_kwargs, device, allowed_names):
    prepared = {}
    for key, value in cached_kwargs.items():
        if key in allowed_names and value is not None:
            prepared[key] = move_value_to_device(value, device)
    return prepared


def block_forward(block, hidden_states, block_kwargs):
    output = block(hidden_states, **block_kwargs)
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


@torch.no_grad()
def run_block_over_samples(block, hidden_states, block_kwargs):
    outputs = torch.empty_like(hidden_states, device=hidden_states.device)
    for sample_idx in range(hidden_states.shape[0]):
        outputs[sample_idx] = block_forward(block, hidden_states[sample_idx : sample_idx + 1], block_kwargs).squeeze(0)
    return outputs


@torch.no_grad()
def capture_first_block_inputs(model, blocks, dataloader, nsamples, seqlen, hidden_size, embed_device, dtype):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    captured_inputs = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=embed_device)
    cache = {"index": 0, "kwargs": {}}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            captured_inputs[cache["index"]] = inp
            cache["index"] += 1
            if not cache["kwargs"]:
                for key, value in kwargs.items():
                    cache["kwargs"][key] = clone_value_to_cpu(value)
            raise ValueError

    original_block = blocks[0]
    blocks[0] = Catcher(original_block)
    try:
        for batch in dataloader:
            try:
                model(batch[0].to(embed_device))
            except ValueError:
                continue
    finally:
        blocks[0] = original_block
        model.config.use_cache = use_cache

    if cache["index"] != nsamples:
        raise RuntimeError(f"expected to capture {nsamples} calibration samples but got {cache['index']}")
    return captured_inputs, cache["kwargs"]


def wanda_mask_from_wrapper(layer, wrapped_layer, sparsity):
    weight = get_effective_layer_weight(layer)
    if sparsity <= 0:
        return torch.zeros_like(weight, dtype=torch.bool)
    if sparsity >= 1:
        return torch.ones_like(weight, dtype=torch.bool)
    if wrapped_layer.scaler_row is None:
        raise RuntimeError(f"missing activation statistics for layer {wrapped_layer.layer_name}")
    metric = torch.abs(weight.to(dtype=torch.float32)) * torch.sqrt(
        wrapped_layer.scaler_row.to(device=weight.device, dtype=torch.float32).reshape((1, -1))
    )
    prune_per_output = int(metric.shape[1] * sparsity)
    if prune_per_output <= 0:
        return torch.zeros_like(metric, dtype=torch.bool)
    if prune_per_output >= metric.shape[1]:
        return torch.ones_like(metric, dtype=torch.bool)
    mask = torch.zeros_like(metric, dtype=torch.bool)
    smallest = torch.topk(metric, prune_per_output, dim=-1, largest=False)[1]
    mask.scatter_(1, smallest, True)
    return mask


@torch.no_grad()
def apply_single_block_wanda_probe(block, hidden_states, block_kwargs, sparsity):
    subset = find_layers(block)
    if not subset:
        raise RuntimeError("no prunable linear layers were found inside the selected transformer block")

    backups = {name: get_effective_layer_weight(layer).detach().cpu().clone() for name, layer in subset.items()}
    wrapped_layers = {name: WrappedGPT(layer) for name, layer in subset.items()}

    def add_batch(name):
        def hook(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)
        return hook

    handles = []
    for name, layer in subset.items():
        handles.append(layer.register_forward_hook(add_batch(name)))

    try:
        _ = run_block_over_samples(block, hidden_states, block_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    for name, layer in subset.items():
        mask = wanda_mask_from_wrapper(layer, wrapped_layers[name], sparsity)
        prune_effective_layer_weight(layer, mask)

    return subset, backups


@torch.no_grad()
def restore_block_weights(subset, backups):
    for name, layer in subset.items():
        set_effective_layer_weight(layer, backups[name])


@torch.no_grad()
def collect_dense_block_outputs(blocks, block_arg_names, layer_devices, initial_inputs, cached_kwargs, cache_dtype):
    dense_outputs = []
    current = initial_inputs
    for layer_idx, block in enumerate(blocks):
        device = layer_devices[layer_idx]
        if current.device != device:
            current = current.to(device)
        block_kwargs = prepare_block_kwargs(block, cached_kwargs, device, block_arg_names[layer_idx])
        current = run_block_over_samples(block, current, block_kwargs)
        dense_outputs.append(current.detach().cpu().to(cache_dtype))
    return dense_outputs


def apply_output_pipeline(model, hidden_states):
    if hasattr(model, "model") and hasattr(model.model, "norm") and model.model.norm is not None:
        norm_module = model.model.norm
        hidden_states = hidden_states.to(get_module_device(norm_module, hidden_states.device))
        hidden_states = norm_module(hidden_states)
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        final_layer_norm = getattr(decoder, "final_layer_norm", None)
        project_out = getattr(decoder, "project_out", None)
        if final_layer_norm is not None:
            hidden_states = hidden_states.to(get_module_device(final_layer_norm, hidden_states.device))
            hidden_states = final_layer_norm(hidden_states)
        if project_out is not None:
            hidden_states = hidden_states.to(get_module_device(project_out, hidden_states.device))
            hidden_states = project_out(hidden_states)
    else:
        raise RuntimeError("unsupported output pipeline")

    output_head = model.get_output_embeddings()
    hidden_states = hidden_states.to(get_module_device(output_head, hidden_states.device))
    return output_head(hidden_states)


@torch.no_grad()
def cache_dense_logits_and_loss(model, dense_final_hidden, input_ids, cache_dtype, chunk_size=1):
    total_nll = 0.0
    total_tokens = 0
    dense_logits_cache = None

    for start in range(0, dense_final_hidden.shape[0], chunk_size):
        end = min(start + chunk_size, dense_final_hidden.shape[0])
        logits = apply_output_pipeline(model, dense_final_hidden[start:end])
        if dense_logits_cache is None:
            dense_logits_cache = torch.empty(
                (dense_final_hidden.shape[0], logits.shape[1], logits.shape[2]),
                dtype=cache_dtype,
                device="cpu",
            )
        dense_logits_cache[start:end] = logits.detach().cpu().to(cache_dtype)

        labels = input_ids[start:end].to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        total_nll += F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        ).item()
        total_tokens += shift_labels.numel()

    dense_loss = total_nll / max(total_tokens, 1)
    dense_ppl = float(math.exp(dense_loss)) if dense_loss < 20 else float("inf")
    return dense_logits_cache, dense_loss, dense_ppl


@torch.no_grad()
def evaluate_pruned_hidden(model, pruned_final_hidden, input_ids, dense_logits_cache, chunk_size=1):
    total_nll = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_positions = 0

    for start in range(0, pruned_final_hidden.shape[0], chunk_size):
        end = min(start + chunk_size, pruned_final_hidden.shape[0])
        logits = apply_output_pipeline(model, pruned_final_hidden[start:end])
        labels = input_ids[start:end].to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        total_nll += F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        ).item()
        total_tokens += shift_labels.numel()

        dense_logits = dense_logits_cache[start:end].to(logits.device).float()
        dense_log_probs = F.log_softmax(dense_logits, dim=-1)
        pruned_log_probs = F.log_softmax(logits.float(), dim=-1)
        dense_probs = dense_log_probs.exp()
        total_kl += (dense_probs * (dense_log_probs - pruned_log_probs)).sum(dim=-1).sum().item()
        total_positions += dense_logits.shape[0] * dense_logits.shape[1]

    loss = total_nll / max(total_tokens, 1)
    ppl = float(math.exp(loss)) if loss < 20 else float("inf")
    kl = total_kl / max(total_positions, 1)
    return loss, ppl, kl


def relative_l2_error(pruned_hidden, dense_hidden):
    pruned_hidden = pruned_hidden.float()
    dense_hidden = dense_hidden.float()
    numerator = torch.linalg.norm((pruned_hidden - dense_hidden).reshape(-1), ord=2)
    denominator = torch.linalg.norm(dense_hidden.reshape(-1), ord=2) + 1e-12
    return float((numerator / denominator).item())


def finite_min_max(values):
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 1.0
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if math.isclose(min_value, max_value):
        pad = 1.0 if math.isclose(min_value, 0.0) else abs(min_value) * 0.1
        min_value -= pad
        max_value += pad
    return min_value, max_value


def pearson_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denominator <= 1e-18:
        return float("nan")
    return float((x * y).sum() / denominator)


def spearman_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x_rank = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float64)
    y_rank = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(np.float64)
    return pearson_correlation(x_rank, y_rank)


def safe_float(value):
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def color_from_value(value, vmin, vmax):
    if not math.isfinite(value):
        return (220, 220, 220)
    if vmax <= vmin:
        ratio = 0.0
    else:
        ratio = (value - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    red = int(255 * ratio)
    green = int(255 * (1.0 - abs(ratio - 0.5) * 1.6))
    blue = int(255 * (1.0 - ratio))
    return (red, green, blue)


def save_heatmap(matrix, output_path):
    matrix = np.asarray(matrix, dtype=np.float64)
    rows, cols = matrix.shape
    cell = max(8, min(24, 640 // max(rows, cols, 1)))
    left = 64
    top = 36
    width = left + cols * cell + 24
    height = top + rows * cell + 36
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    finite_values = matrix[np.isfinite(matrix)]
    vmax = float(finite_values.max()) if finite_values.size else 1.0
    vmin = 0.0

    for row in range(rows):
        for col in range(cols):
            x0 = left + col * cell
            y0 = top + row * cell
            color = color_from_value(float(matrix[row, col]), vmin, vmax)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(235, 235, 235))

    tick_step = max(1, rows // 10)
    for idx in range(0, rows, tick_step):
        draw.text((8, top + idx * cell + max(cell // 4, 1)), str(idx), fill="black", font=font)
        draw.text((left + idx * cell + max(cell // 4, 1), 8), str(idx), fill="black", font=font)

    draw.text((left, height - 20), "observe layer", fill="black", font=font)
    draw.text((8, 8), "probe layer", fill="black", font=font)
    image.save(output_path)


def save_scatter(x_values, y_values, output_path, title):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    width, height = 800, 600
    left, right, top, bottom = 72, 36, 40, 56
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    xmin, xmax = finite_min_max(x_values)
    ymin, ymax = finite_min_max(y_values)

    def x_to_px(value):
        return left + (value - xmin) * (width - left - right) / max(xmax - xmin, 1e-12)

    def y_to_px(value):
        return height - bottom - (value - ymin) * (height - top - bottom) / max(ymax - ymin, 1e-12)

    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.text((left, 12), title, fill="black", font=font)
    draw.text((left, height - 22), "local_error", fill="black", font=font)
    draw.text((width - 120, top), "final_kl", fill="black", font=font)

    for x_value, y_value in zip(x_values, y_values):
        if not (math.isfinite(float(x_value)) and math.isfinite(float(y_value))):
            continue
        px = x_to_px(float(x_value))
        py = y_to_px(float(y_value))
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(40, 90, 200), outline=(40, 90, 200))

    image.save(output_path)


def save_curve(values, output_path, title, y_label):
    values = np.asarray(values, dtype=np.float64)
    width, height = 900, 420
    left, right, top, bottom = 64, 28, 40, 48
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    finite_values = values[np.isfinite(values)]
    ymin, ymax = finite_min_max(finite_values if finite_values.size else np.array([0.0, 1.0]))
    xmax = max(len(values) - 1, 1)

    def x_to_px(index):
        return left + index * (width - left - right) / xmax

    def y_to_px(value):
        return height - bottom - (value - ymin) * (height - top - bottom) / max(ymax - ymin, 1e-12)

    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.text((left, 12), title, fill="black", font=font)
    draw.text((left, height - 22), "layer_idx", fill="black", font=font)
    draw.text((width - 180, top), y_label, fill="black", font=font)

    points = []
    for idx, value in enumerate(values):
        if math.isfinite(float(value)):
            points.append((x_to_px(idx), y_to_px(float(value))))
    if len(points) >= 2:
        draw.line(points, fill=(220, 70, 70), width=2)
    for point in points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(220, 70, 70), outline=(220, 70, 70))

    tick_step = max(1, len(values) // 10)
    for idx in range(0, len(values), tick_step):
        draw.text((x_to_px(idx), height - bottom + 6), str(idx), fill="black", font=font)

    image.save(output_path)


def write_metrics_csv(metrics, output_path):
    fieldnames = ["layer_idx", "local_error", "final_kl", "final_loss_delta", "downstream_area", "amplification_ratio"]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def write_matrix_csv(matrix, output_path):
    matrix = np.asarray(matrix, dtype=np.float64)
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["probe_layer"] + [f"observe_{idx}" for idx in range(matrix.shape[1])])
        for row_idx in range(matrix.shape[0]):
            row = [row_idx]
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                row.append(value if math.isfinite(float(value)) else "nan")
            writer.writerow(row)


def build_summary(args, local_errors, final_kls, amplification_ratios):
    amplification_mean = float(np.nanmean(amplification_ratios)) if len(amplification_ratios) else float("nan")
    amplification_std = float(np.nanstd(amplification_ratios)) if len(amplification_ratios) else float("nan")
    amplification_cv = amplification_std / (amplification_mean + 1e-12) if math.isfinite(amplification_mean) else float("nan")
    return {
        "model_name": args.model_name_or_path,
        "nsamples": int(args.nsamples),
        "seqlen": int(args.seqlen),
        "probe_sparsity": float(args.probe_sparsity),
        "mean_local_error": safe_float(np.nanmean(local_errors)),
        "mean_final_kl": safe_float(np.nanmean(final_kls)),
        "spearman_local_vs_finalkl": safe_float(spearman_correlation(local_errors, final_kls)),
        "pearson_local_vs_finalkl": safe_float(pearson_correlation(local_errors, final_kls)),
        "amplification_ratio_mean": safe_float(amplification_mean),
        "amplification_ratio_std": safe_float(amplification_std),
        "amplification_ratio_cv": safe_float(amplification_cv),
    }


def load_model_and_tokenizer(args, default_device):
    model_local_only = os.path.exists(args.model_name_or_path)
    tokenizer_local_only = os.path.exists(args.tokenizer_name_or_path)
    load_dtype = torch.float16 if default_device.type == "cuda" else torch.float32
    model_kwargs = {
        "torch_dtype": load_dtype,
        "cache_dir": args.cache_dir,
        "low_cpu_mem_usage": True,
        "local_files_only": model_local_only,
    }
    if default_device.type == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if default_device.type != "cuda":
        model.to(default_device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        use_fast=False,
        cache_dir=args.cache_dir,
        local_files_only=tokenizer_local_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and args.seqlen > max_positions:
        raise ValueError(f"requested seqlen={args.seqlen} exceeds max_position_embeddings={max_positions}")

    model.seqlen = args.seqlen
    model.eval()
    return model, tokenizer


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    default_device = normalize_device(args.device)
    if default_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"loading model: {args.model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(args, default_device)
    model_info = get_model_info(model)
    blocks = model_info["blocks"]
    n_layers = len(blocks)
    cache_dtype = next(model.parameters()).dtype
    embed_device = get_embed_device(model, model_info, default_device)
    layer_devices = [get_block_device(model, model_info, idx, default_device) for idx in range(n_layers)]
    block_arg_names = [supported_block_arg_names(block) for block in blocks]

    print(f"loading calibration data: dataset={args.dataset}, nsamples={args.nsamples}, seqlen={args.seqlen}")
    dataloader, _ = load_calibration_data(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
    )
    input_ids = torch.cat([batch[0] for batch in dataloader], dim=0).cpu()

    hidden_size = model.config.hidden_size
    print("capturing dense inputs to the first transformer block")
    dense_input0, cached_kwargs = capture_first_block_inputs(
        model,
        blocks,
        dataloader,
        args.nsamples,
        args.seqlen,
        hidden_size,
        embed_device,
        cache_dtype,
    )
    dense_input0_cache = dense_input0.detach().cpu().to(cache_dtype)

    print("running dense baseline layer by layer")
    dense_outputs = collect_dense_block_outputs(
        blocks,
        block_arg_names,
        layer_devices,
        dense_input0,
        cached_kwargs,
        cache_dtype,
    )
    del dense_input0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("caching dense logits and dense loss")
    dense_final_hidden = dense_outputs[-1].to(layer_devices[-1])
    dense_logits_cache, dense_loss, dense_ppl = cache_dense_logits_and_loss(
        model,
        dense_final_hidden,
        input_ids,
        cache_dtype,
        chunk_size=1,
    )
    del dense_final_hidden
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    propagation_matrix = np.full((n_layers, n_layers), np.nan, dtype=np.float64)
    metrics = []
    local_errors = []
    final_kls = []
    amplification_ratios = []

    for probe_idx in range(n_layers):
        print(f"probing layer {probe_idx}/{n_layers - 1}")
        probe_input = dense_input0_cache if probe_idx == 0 else dense_outputs[probe_idx - 1]
        current = probe_input.to(layer_devices[probe_idx])
        probe_kwargs = prepare_block_kwargs(blocks[probe_idx], cached_kwargs, layer_devices[probe_idx], block_arg_names[probe_idx])
        subset, backups = apply_single_block_wanda_probe(blocks[probe_idx], current, probe_kwargs, args.probe_sparsity)

        try:
            for observe_idx in range(probe_idx, n_layers):
                observe_device = layer_devices[observe_idx]
                if current.device != observe_device:
                    current = current.to(observe_device)
                observe_kwargs = prepare_block_kwargs(blocks[observe_idx], cached_kwargs, observe_device, block_arg_names[observe_idx])
                current = run_block_over_samples(blocks[observe_idx], current, observe_kwargs)
                dense_hidden = dense_outputs[observe_idx].to(observe_device)
                propagation_matrix[probe_idx, observe_idx] = relative_l2_error(current, dense_hidden)

            probe_loss, probe_ppl, final_kl = evaluate_pruned_hidden(
                model,
                current,
                input_ids,
                dense_logits_cache,
                chunk_size=1,
            )
        finally:
            restore_block_weights(subset, backups)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        local_error = float(propagation_matrix[probe_idx, probe_idx])
        final_loss_delta = float(probe_loss - dense_loss)
        downstream_area = float(np.nansum(propagation_matrix[probe_idx, probe_idx:]))
        amplification_ratio = float(final_kl / (local_error + 1e-12))

        metrics.append(
            {
                "layer_idx": probe_idx,
                "local_error": local_error,
                "final_kl": float(final_kl),
                "final_loss_delta": final_loss_delta,
                "downstream_area": downstream_area,
                "amplification_ratio": amplification_ratio,
            }
        )
        local_errors.append(local_error)
        final_kls.append(float(final_kl))
        amplification_ratios.append(amplification_ratio)

        print(
            f"layer={probe_idx} local_error={local_error:.6e} final_kl={final_kl:.6e} "
            f"loss_delta={final_loss_delta:.6e} ppl_dense={dense_ppl:.6f} ppl_probe={probe_ppl:.6f}"
        )

    write_metrics_csv(metrics, os.path.join(args.save_dir, "layer_probe_metrics.csv"))
    write_matrix_csv(propagation_matrix, os.path.join(args.save_dir, "propagation_matrix.csv"))

    summary = build_summary(args, local_errors, final_kls, amplification_ratios)
    with open(os.path.join(args.save_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    save_heatmap(propagation_matrix, os.path.join(args.save_dir, "propagation_heatmap.png"))
    save_scatter(local_errors, final_kls, os.path.join(args.save_dir, "local_vs_finalkl_scatter.png"), "local error vs final KL")
    save_curve(amplification_ratios, os.path.join(args.save_dir, "amplification_ratio_curve.png"), "amplification ratio by layer", "amplification_ratio")

    print("saved outputs:")
    print(os.path.join(args.save_dir, "layer_probe_metrics.csv"))
    print(os.path.join(args.save_dir, "propagation_matrix.csv"))
    print(os.path.join(args.save_dir, "summary.json"))
    print(os.path.join(args.save_dir, "propagation_heatmap.png"))
    print(os.path.join(args.save_dir, "local_vs_finalkl_scatter.png"))
    print(os.path.join(args.save_dir, "amplification_ratio_curve.png"))


if __name__ == "__main__":
    main()

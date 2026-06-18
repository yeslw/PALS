import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_dependency_propagation import finite_min_max


ALL_RESULTS_FIELDS = [
    "source_file",
    "model_name",
    "model_label",
    "phase_label",
    "method",
    "variant",
    "method_variant",
    "global_sparsity",
    "lambda_prop",
    "seed",
    "shuffle_seed",
    "dataset",
    "nsamples",
    "seqlen",
    "stream_batch_size",
    "baseline_prune_method",
    "schedule_source",
    "ppl_wikitext2",
    "delta_vs_base_ppl",
    "actual_sparsity",
    "weighted_global_sparsity",
    "weighted_global_error",
    "schedule_min",
    "schedule_max",
    "dry_run_schedule_only",
]
GROUPED_FIELDS = [
    "method",
    "variant",
    "method_variant",
    "global_sparsity",
    "lambda_prop",
    "count",
    "ppl_wikitext2_mean",
    "ppl_wikitext2_std",
    "actual_sparsity_mean",
    "actual_sparsity_std",
    "weighted_global_sparsity_mean",
    "weighted_global_sparsity_std",
    "win_count_ppl",
    "loss_count_ppl",
    "tie_count_ppl",
    "mean_delta_ppl",
    "median_delta_ppl",
    "best_delta_ppl",
    "worst_delta_ppl",
]
BEST_FIELDS = [
    "method",
    "global_sparsity",
    "best_lambda_prop",
    "ppl_wikitext2_mean",
    "mean_delta_ppl",
    "median_delta_ppl",
    "best_delta_ppl",
    "worst_delta_ppl",
    "win_count_ppl",
    "loss_count_ppl",
    "tie_count_ppl",
    "count",
    "base_ppl_wikitext2_mean",
    "base_count",
    "best_shuffled_lambda_prop",
    "best_shuffled_mean_delta_ppl",
]
COLOR_PALETTE = [
    (30, 136, 229),
    (229, 57, 53),
    (67, 160, 71),
    (251, 140, 0),
    (142, 36, 170),
    (0, 172, 193),
    (124, 179, 66),
    (141, 110, 99),
]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()



def maybe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number



def maybe_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return int(text)



def collect_compare_files(results_root):
    matches = []
    for current_root, _, files in os.walk(results_root):
        if "compare_one_line.csv" in files:
            matches.append(os.path.join(current_root, "compare_one_line.csv"))
    matches.sort()
    return matches



def read_compare_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}, got {len(rows)}")
    row = rows[0]
    return {
        "source_file": path,
        "model_name": row.get("model_name", ""),
        "model_label": row.get("model_label", ""),
        "phase_label": row.get("phase_label", ""),
        "method": row.get("method", ""),
        "baseline_prune_method": row.get("baseline_prune_method", ""),
        "schedule_source": row.get("schedule_source", ""),
        "global_sparsity": maybe_float(row.get("global_sparsity")),
        "lambda_prop": maybe_float(row.get("lambda_prop")),
        "seed": maybe_int(row.get("seed")),
        "shuffle_seed": maybe_int(row.get("shuffle_seed")),
        "dataset": row.get("dataset", ""),
        "nsamples": maybe_int(row.get("nsamples")),
        "seqlen": maybe_int(row.get("seqlen")),
        "stream_batch_size": maybe_int(row.get("stream_batch_size")),
        "ppl_base": maybe_float(row.get("ppl_base")),
        "ppl_hybrid": maybe_float(row.get("ppl_hybrid")),
        "ppl_shuffled": maybe_float(row.get("ppl_shuffled")),
        "delta_ppl": maybe_float(row.get("delta_ppl")),
        "delta_ppl_shuffled": maybe_float(row.get("delta_ppl_shuffled")),
        "actual_sparsity_base": maybe_float(row.get("actual_sparsity_base")),
        "actual_sparsity_hybrid": maybe_float(row.get("actual_sparsity_hybrid")),
        "actual_sparsity_shuffled": maybe_float(row.get("actual_sparsity_shuffled")),
        "weighted_global_sparsity_base": maybe_float(row.get("weighted_global_sparsity_base")),
        "weighted_global_sparsity_hybrid": maybe_float(row.get("weighted_global_sparsity_hybrid")),
        "weighted_global_sparsity_shuffled": maybe_float(row.get("weighted_global_sparsity_shuffled")),
        "weighted_global_error_base": maybe_float(row.get("weighted_global_error_base")),
        "weighted_global_error_hybrid": maybe_float(row.get("weighted_global_error_hybrid")),
        "weighted_global_error_shuffled": maybe_float(row.get("weighted_global_error_shuffled")),
        "schedule_min_base": maybe_float(row.get("schedule_min_base")),
        "schedule_max_base": maybe_float(row.get("schedule_max_base")),
        "schedule_min_hybrid": maybe_float(row.get("schedule_min_hybrid")),
        "schedule_max_hybrid": maybe_float(row.get("schedule_max_hybrid")),
        "schedule_min_shuffled": maybe_float(row.get("schedule_min_shuffled")),
        "schedule_max_shuffled": maybe_float(row.get("schedule_max_shuffled")),
        "dry_run_schedule_only": row.get("dry_run_schedule_only", ""),
    }



def method_variant_name(method, variant):
    if variant == "base":
        return f"{method}_base"
    return f"{method}_prop_{variant}"



def build_base_key(row):
    return (
        row.get("model_label"),
        row.get("phase_label"),
        row.get("method"),
        row.get("global_sparsity"),
        row.get("seed"),
    )



def expand_rows(compare_rows):
    base_by_key = {}
    hybrid_rows = []
    shuffled_rows = []
    for row in compare_rows:
        common = {
            "source_file": row["source_file"],
            "model_name": row["model_name"],
            "model_label": row["model_label"],
            "phase_label": row["phase_label"],
            "method": row["method"],
            "global_sparsity": row["global_sparsity"],
            "seed": row["seed"],
            "shuffle_seed": row["shuffle_seed"],
            "dataset": row["dataset"],
            "nsamples": row["nsamples"],
            "seqlen": row["seqlen"],
            "stream_batch_size": row["stream_batch_size"],
            "baseline_prune_method": row["baseline_prune_method"],
            "schedule_source": row["schedule_source"],
            "dry_run_schedule_only": row["dry_run_schedule_only"],
        }
        base_payload = {
            **common,
            "variant": "base",
            "method_variant": method_variant_name(row["method"], "base"),
            "lambda_prop": None,
            "ppl_wikitext2": row["ppl_base"],
            "delta_vs_base_ppl": 0.0 if row["ppl_base"] is not None else None,
            "actual_sparsity": row["actual_sparsity_base"],
            "weighted_global_sparsity": row["weighted_global_sparsity_base"],
            "weighted_global_error": row["weighted_global_error_base"],
            "schedule_min": row["schedule_min_base"],
            "schedule_max": row["schedule_max_base"],
        }
        key = build_base_key(row)
        if key not in base_by_key:
            base_by_key[key] = base_payload

        hybrid_rows.append(
            {
                **common,
                "variant": "hybrid",
                "method_variant": method_variant_name(row["method"], "hybrid"),
                "lambda_prop": row["lambda_prop"],
                "ppl_wikitext2": row["ppl_hybrid"],
                "delta_vs_base_ppl": row["delta_ppl"],
                "actual_sparsity": row["actual_sparsity_hybrid"],
                "weighted_global_sparsity": row["weighted_global_sparsity_hybrid"],
                "weighted_global_error": row["weighted_global_error_hybrid"],
                "schedule_min": row["schedule_min_hybrid"],
                "schedule_max": row["schedule_max_hybrid"],
            }
        )
        shuffled_rows.append(
            {
                **common,
                "variant": "shuffled",
                "method_variant": method_variant_name(row["method"], "shuffled"),
                "lambda_prop": row["lambda_prop"],
                "ppl_wikitext2": row["ppl_shuffled"],
                "delta_vs_base_ppl": row["delta_ppl_shuffled"],
                "actual_sparsity": row["actual_sparsity_shuffled"],
                "weighted_global_sparsity": row["weighted_global_sparsity_shuffled"],
                "weighted_global_error": row["weighted_global_error_shuffled"],
                "schedule_min": row["schedule_min_shuffled"],
                "schedule_max": row["schedule_max_shuffled"],
            }
        )
    base_rows = sorted(base_by_key.values(), key=lambda item: (item["method"], item["global_sparsity"], item["seed"] or -1))
    all_rows = base_rows + hybrid_rows + shuffled_rows
    all_rows.sort(
        key=lambda item: (
            item["method"],
            item["variant"],
            float(item["global_sparsity"] or -1.0),
            -1.0 if item["lambda_prop"] is None else float(item["lambda_prop"]),
            -1 if item["seed"] is None else int(item["seed"]),
            item["source_file"],
        )
    )
    return all_rows



def summarize_metric(values):
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite))



def summarize_deltas(rows):
    deltas = np.asarray(
        [row["delta_vs_base_ppl"] for row in rows if row.get("delta_vs_base_ppl") is not None and math.isfinite(row["delta_vs_base_ppl"])],
        dtype=np.float64,
    )
    if deltas.size == 0:
        return {
            "win_count_ppl": 0,
            "loss_count_ppl": 0,
            "tie_count_ppl": 0,
            "mean_delta_ppl": None,
            "median_delta_ppl": None,
            "best_delta_ppl": None,
            "worst_delta_ppl": None,
        }
    return {
        "win_count_ppl": int(np.sum(deltas < 0.0)),
        "loss_count_ppl": int(np.sum(deltas > 0.0)),
        "tie_count_ppl": int(np.sum(deltas == 0.0)),
        "mean_delta_ppl": float(np.mean(deltas)),
        "median_delta_ppl": float(np.median(deltas)),
        "best_delta_ppl": float(np.min(deltas)),
        "worst_delta_ppl": float(np.max(deltas)),
    }



def build_grouped_summary(all_rows):
    grouped = defaultdict(list)
    for row in all_rows:
        lambda_key = None if row["variant"] == "base" else row["lambda_prop"]
        key = (row["method"], row["variant"], row["global_sparsity"], lambda_key)
        grouped[key].append(row)

    summary_rows = []
    for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1], item[2], -1.0 if item[3] is None else item[3])):
        bucket = grouped[key]
        ppl_mean, ppl_std = summarize_metric([item["ppl_wikitext2"] for item in bucket])
        actual_mean, actual_std = summarize_metric([item["actual_sparsity"] for item in bucket])
        weighted_mean_value, weighted_std_value = summarize_metric([item["weighted_global_sparsity"] for item in bucket])
        delta_payload = summarize_deltas(bucket if key[1] != "base" else [])
        summary_rows.append(
            {
                "method": key[0],
                "variant": key[1],
                "method_variant": method_variant_name(key[0], key[1]),
                "global_sparsity": key[2],
                "lambda_prop": key[3],
                "count": len(bucket),
                "ppl_wikitext2_mean": ppl_mean,
                "ppl_wikitext2_std": ppl_std,
                "actual_sparsity_mean": actual_mean,
                "actual_sparsity_std": actual_std,
                "weighted_global_sparsity_mean": weighted_mean_value,
                "weighted_global_sparsity_std": weighted_std_value,
                **delta_payload,
            }
        )
    return summary_rows



def write_csv(rows, fieldnames, output_path):
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def color_palette(index):
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]



def save_lambda_sweep_plot(grouped_rows, method, output_path):
    width, height = 980, 620
    left, right, top, bottom = 80, 260, 48, 56
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    method_rows = [row for row in grouped_rows if row["method"] == method]
    line_rows = [row for row in method_rows if row["variant"] in ("hybrid", "shuffled") and row["lambda_prop"] is not None and row["ppl_wikitext2_mean"] is not None]
    base_rows = [row for row in method_rows if row["variant"] == "base" and row["ppl_wikitext2_mean"] is not None]
    if not line_rows and not base_rows:
        draw.text((24, 24), f"{method} lambda sweep: no data", fill="black", font=font)
        image.save(output_path)
        return

    x_values = np.asarray([row["lambda_prop"] for row in line_rows], dtype=np.float64) if line_rows else np.asarray([0.0, 1.0], dtype=np.float64)
    y_values = []
    for row in line_rows:
        y_values.append(row["ppl_wikitext2_mean"])
    for row in base_rows:
        y_values.append(row["ppl_wikitext2_mean"])
    xmin, xmax = finite_min_max(x_values)
    ymin, ymax = finite_min_max(y_values)
    if math.isclose(xmin, xmax, rel_tol=0.0, abs_tol=1e-12):
        xmin -= 0.05
        xmax += 0.05
    if math.isclose(ymin, ymax, rel_tol=0.0, abs_tol=1e-12):
        ymin -= 1.0
        ymax += 1.0

    def x_to_px(value):
        return left + (value - xmin) * (width - left - right) / max(xmax - xmin, 1e-12)

    def y_to_px(value):
        return height - bottom - (value - ymin) * (height - top - bottom) / max(ymax - ymin, 1e-12)

    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.text((left, 14), f"{method} hybrid-prop lambda sweep", fill="black", font=font)
    draw.text((left, height - 22), "lambda_prop", fill="black", font=font)
    draw.text((width - 180, top), "ppl_wikitext2", fill="black", font=font)

    grouped_lines = defaultdict(list)
    for row in line_rows:
        grouped_lines[(row["variant"], row["global_sparsity"])] .append(row)

    legend_y = top
    line_index = 0
    for key in sorted(grouped_lines.keys(), key=lambda item: (item[1], item[0])):
        rows = sorted(grouped_lines[key], key=lambda item: item["lambda_prop"])
        color = color_palette(line_index)
        points = [(x_to_px(row["lambda_prop"]), y_to_px(row["ppl_wikitext2_mean"])) for row in rows]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=color)
        label = f"gs={key[1]:.2f} | {key[0]}"
        draw.rectangle((width - 230, legend_y + 3, width - 218, legend_y + 15), fill=color, outline=color)
        draw.text((width - 210, legend_y), label, fill="black", font=font)
        legend_y += 18
        line_index += 1

    for row in sorted(base_rows, key=lambda item: item["global_sparsity"]):
        color = (120, 120, 120)
        y = y_to_px(row["ppl_wikitext2_mean"])
        draw.line((left, y, width - right, y), fill=color, width=1)
        draw.text((width - 210, legend_y), f"gs={row['global_sparsity']:.2f} | base={row['ppl_wikitext2_mean']:.4f}", fill="black", font=font)
        legend_y += 18

    image.save(output_path)



def build_best_lambda_rows(grouped_rows):
    base_index = {}
    shuffled_index = {}
    for row in grouped_rows:
        key = (row["method"], row["global_sparsity"])
        if row["variant"] == "base":
            base_index[key] = row
        elif row["variant"] == "shuffled":
            best_row = shuffled_index.get(key)
            score = row.get("mean_delta_ppl")
            best_score = None if best_row is None else best_row.get("mean_delta_ppl")
            if best_row is None or (score is not None and (best_score is None or score < best_score)):
                shuffled_index[key] = row

    grouped_hybrid = defaultdict(list)
    for row in grouped_rows:
        if row["variant"] == "hybrid":
            grouped_hybrid[(row["method"], row["global_sparsity"])] .append(row)

    best_rows = []
    for key in sorted(grouped_hybrid.keys()):
        candidates = grouped_hybrid[key]
        candidates = [row for row in candidates if row.get("mean_delta_ppl") is not None]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: (row["mean_delta_ppl"], row.get("ppl_wikitext2_mean") or float("inf"), row.get("lambda_prop") or float("inf")))
        base = base_index.get(key)
        shuffled = shuffled_index.get(key)
        best_rows.append(
            {
                "method": key[0],
                "global_sparsity": key[1],
                "best_lambda_prop": best.get("lambda_prop"),
                "ppl_wikitext2_mean": best.get("ppl_wikitext2_mean"),
                "mean_delta_ppl": best.get("mean_delta_ppl"),
                "median_delta_ppl": best.get("median_delta_ppl"),
                "best_delta_ppl": best.get("best_delta_ppl"),
                "worst_delta_ppl": best.get("worst_delta_ppl"),
                "win_count_ppl": best.get("win_count_ppl"),
                "loss_count_ppl": best.get("loss_count_ppl"),
                "tie_count_ppl": best.get("tie_count_ppl"),
                "count": best.get("count"),
                "base_ppl_wikitext2_mean": None if base is None else base.get("ppl_wikitext2_mean"),
                "base_count": None if base is None else base.get("count"),
                "best_shuffled_lambda_prop": None if shuffled is None else shuffled.get("lambda_prop"),
                "best_shuffled_mean_delta_ppl": None if shuffled is None else shuffled.get("mean_delta_ppl"),
            }
        )
    return best_rows



def save_hybrid_vs_base_barplot(grouped_rows, best_rows, output_path):
    width, height = 1100, 620
    left, right, top, bottom = 72, 28, 48, 120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    base_index = {(row["method"], row["global_sparsity"]): row for row in grouped_rows if row["variant"] == "base"}
    shuffled_index = {}
    for row in grouped_rows:
        if row["variant"] != "shuffled":
            continue
        key = (row["method"], row["global_sparsity"])
        prev = shuffled_index.get(key)
        if prev is None or (
            row.get("mean_delta_ppl") is not None
            and (prev.get("mean_delta_ppl") is None or row["mean_delta_ppl"] < prev["mean_delta_ppl"])
        ):
            shuffled_index[key] = row

    bar_groups = []
    for row in best_rows:
        key = (row["method"], row["global_sparsity"])
        base = base_index.get(key)
        shuffled = shuffled_index.get(key)
        bar_groups.append(
            {
                "label": f"{row['method']}|gs={row['global_sparsity']:.2f}",
                "values": [
                    None if base is None else base.get("ppl_wikitext2_mean"),
                    row.get("ppl_wikitext2_mean"),
                    None if shuffled is None else shuffled.get("ppl_wikitext2_mean"),
                ],
            }
        )

    if not bar_groups:
        draw.text((24, 24), "hybrid vs base: no data", fill="black", font=font)
        image.save(output_path)
        return

    y_values = []
    for group in bar_groups:
        for value in group["values"]:
            if value is not None and math.isfinite(value):
                y_values.append(value)
    ymin, ymax = finite_min_max(y_values)
    ymin = min(0.0, ymin)
    if math.isclose(ymin, ymax, rel_tol=0.0, abs_tol=1e-12):
        ymax += 1.0

    def y_to_px(value):
        return height - bottom - (value - ymin) * (height - top - bottom) / max(ymax - ymin, 1e-12)

    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.text((left, 14), "base vs best hybrid vs best shuffled", fill="black", font=font)
    draw.text((width - 120, top), "ppl", fill="black", font=font)

    labels = ["base", "best_hybrid", "best_shuffled"]
    colors = [(120, 120, 120), (30, 136, 229), (229, 57, 53)]
    usable_width = width - left - right
    group_width = max(90, usable_width // max(len(bar_groups), 1))
    bar_width = max(18, group_width // 5)

    for group_idx, group in enumerate(bar_groups):
        group_x = left + group_idx * group_width + group_width // 6
        for value_idx, value in enumerate(group["values"]):
            if value is None or not math.isfinite(value):
                continue
            x0 = group_x + value_idx * (bar_width + 10)
            x1 = x0 + bar_width
            y0 = y_to_px(value)
            draw.rectangle((x0, y0, x1, height - bottom), fill=colors[value_idx], outline=colors[value_idx])
            draw.text((x0, max(top, y0 - 14)), f"{value:.3f}", fill="black", font=font)
        draw.text((group_x, height - bottom + 8), group["label"], fill="black", font=font)

    legend_y = top
    for idx, label in enumerate(labels):
        draw.rectangle((width - 190, legend_y + 3, width - 178, legend_y + 15), fill=colors[idx], outline=colors[idx])
        draw.text((width - 170, legend_y), label, fill="black", font=font)
        legend_y += 18

    image.save(output_path)



def main():
    args = parse_args()
    save_dir = args.save_dir or args.results_root
    os.makedirs(save_dir, exist_ok=True)

    compare_files = collect_compare_files(args.results_root)
    if not compare_files:
        raise RuntimeError(f"no compare_one_line.csv files found under {args.results_root}")

    compare_rows = [read_compare_csv(path) for path in compare_files]
    all_rows = expand_rows(compare_rows)
    grouped_rows = build_grouped_summary(all_rows)
    best_rows = build_best_lambda_rows(grouped_rows)

    all_results_path = os.path.join(save_dir, "all_results.csv")
    grouped_summary_path = os.path.join(save_dir, "grouped_summary.csv")
    best_lambda_path = os.path.join(save_dir, "best_lambda_by_method_and_sparsity.csv")
    write_csv(all_rows, ALL_RESULTS_FIELDS, all_results_path)
    write_csv(grouped_rows, GROUPED_FIELDS, grouped_summary_path)
    write_csv(best_rows, BEST_FIELDS, best_lambda_path)

    save_lambda_sweep_plot(grouped_rows, "owl", os.path.join(save_dir, "lambda_sweep_owl.png"))
    save_lambda_sweep_plot(grouped_rows, "dlp", os.path.join(save_dir, "lambda_sweep_dlp.png"))
    save_hybrid_vs_base_barplot(grouped_rows, best_rows, os.path.join(save_dir, "hybrid_vs_base_barplot.png"))

    print(all_results_path)
    print(grouped_summary_path)
    print(best_lambda_path)


if __name__ == "__main__":
    main()

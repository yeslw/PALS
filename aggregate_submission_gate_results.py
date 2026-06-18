import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_dependency_propagation import finite_min_max


ORDERED_METHODS = [
    "uniform",
    "local_error",
    "combo_rank",
    "downstream_area",
    "shuffled_combo_rank",
    "reversed_combo_rank",
    "shuffled_downstream_area",
    "reversed_downstream_area",
]
PAIRWISE_CONFIGS = [
    {"name": "combo_rank_vs_uniform", "challenger": "combo_rank", "baseline": "uniform"},
    {"name": "combo_rank_vs_local_error", "challenger": "combo_rank", "baseline": "local_error"},
    {"name": "combo_rank_vs_shuffled_combo_rank", "challenger": "combo_rank", "baseline": "shuffled_combo_rank"},
    {"name": "combo_rank_vs_reversed_combo_rank", "challenger": "combo_rank", "baseline": "reversed_combo_rank"},
    {"name": "downstream_area_vs_uniform", "challenger": "downstream_area", "baseline": "uniform"},
    {"name": "downstream_area_vs_local_error", "challenger": "downstream_area", "baseline": "local_error"},
    {"name": "downstream_area_vs_shuffled_downstream_area", "challenger": "downstream_area", "baseline": "shuffled_downstream_area"},
    {"name": "downstream_area_vs_reversed_downstream_area", "challenger": "downstream_area", "baseline": "reversed_downstream_area"},
]
ALL_RESULTS_FIELDS = [
    "source_file",
    "model_name",
    "model_label",
    "global_sparsity",
    "score_type",
    "base_score_type",
    "schedule_transform",
    "alpha",
    "actual_sparsity",
    "ppl_wikitext2",
    "zeroshot_avg",
    "boolq",
    "hellaswag",
    "arc_challenge",
    "seed",
    "shuffle_seed",
    "schedule_min",
    "schedule_max",
    "schedule_weighted_mean",
    "eval_zeroshot",
    "zeroshot_tasks",
]
GROUPED_FIELDS = [
    "global_sparsity",
    "score_type",
    "base_score_type",
    "schedule_transform",
    "alpha",
    "count",
    "ppl_wikitext2_mean",
    "ppl_wikitext2_std",
    "zeroshot_avg_mean",
    "zeroshot_avg_std",
    "boolq_mean",
    "boolq_std",
    "hellaswag_mean",
    "hellaswag_std",
    "arc_challenge_mean",
    "arc_challenge_std",
    "actual_sparsity_mean",
    "actual_sparsity_std",
    "schedule_weighted_mean_mean",
    "schedule_weighted_mean_std",
]
PAIRWISE_FIELDS = [
    "pair_name",
    "challenger",
    "baseline",
    "matched_count",
    "challenger_alpha_mean",
    "baseline_alpha_mean",
    "win_count_ppl",
    "loss_count_ppl",
    "mean_delta_ppl",
    "median_delta_ppl",
    "win_count_avgacc",
    "loss_count_avgacc",
    "mean_delta_avgacc",
    "worst_delta_avgacc",
    "best_delta_avgacc",
    "matched_conditions",
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


def collect_one_line_files(results_root):
    matches = []
    for current_root, _, files in os.walk(results_root):
        if "one_line_result.csv" in files:
            matches.append(os.path.join(current_root, "one_line_result.csv"))
    matches.sort()
    return matches


def read_one_line_csv(path):
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
        "global_sparsity": maybe_float(row.get("global_sparsity")),
        "score_type": row.get("score_type", ""),
        "base_score_type": row.get("base_score_type", ""),
        "schedule_transform": row.get("schedule_transform", ""),
        "alpha": maybe_float(row.get("alpha")),
        "actual_sparsity": maybe_float(row.get("actual_sparsity")),
        "ppl_wikitext2": maybe_float(row.get("ppl_wikitext2")),
        "zeroshot_avg": maybe_float(row.get("zeroshot_avg")),
        "boolq": maybe_float(row.get("boolq")),
        "hellaswag": maybe_float(row.get("hellaswag")),
        "arc_challenge": maybe_float(row.get("arc_challenge")),
        "seed": maybe_int(row.get("seed")),
        "shuffle_seed": maybe_int(row.get("shuffle_seed")),
        "schedule_min": maybe_float(row.get("schedule_min")),
        "schedule_max": maybe_float(row.get("schedule_max")),
        "schedule_weighted_mean": maybe_float(row.get("schedule_weighted_mean")),
        "eval_zeroshot": row.get("eval_zeroshot", ""),
        "zeroshot_tasks": row.get("zeroshot_tasks", ""),
    }


def summarize_metric(values):
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite))


def write_csv(rows, fieldnames, output_path):
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(payload, output_path):
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def build_grouped_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["global_sparsity"], row["score_type"], row["alpha"])
        grouped[key].append(row)

    summary_rows = []
    for key in sorted(grouped.keys()):
        bucket = grouped[key]
        ppl_mean, ppl_std = summarize_metric([item["ppl_wikitext2"] for item in bucket])
        avg_mean, avg_std = summarize_metric([item["zeroshot_avg"] for item in bucket])
        boolq_mean, boolq_std = summarize_metric([item["boolq"] for item in bucket])
        hella_mean, hella_std = summarize_metric([item["hellaswag"] for item in bucket])
        arc_mean, arc_std = summarize_metric([item["arc_challenge"] for item in bucket])
        sparsity_mean, sparsity_std = summarize_metric([item["actual_sparsity"] for item in bucket])
        weighted_mean_mean, weighted_mean_std = summarize_metric([item["schedule_weighted_mean"] for item in bucket])
        summary_rows.append(
            {
                "global_sparsity": key[0],
                "score_type": key[1],
                "base_score_type": bucket[0].get("base_score_type", ""),
                "schedule_transform": bucket[0].get("schedule_transform", ""),
                "alpha": key[2],
                "count": len(bucket),
                "ppl_wikitext2_mean": ppl_mean,
                "ppl_wikitext2_std": ppl_std,
                "zeroshot_avg_mean": avg_mean,
                "zeroshot_avg_std": avg_std,
                "boolq_mean": boolq_mean,
                "boolq_std": boolq_std,
                "hellaswag_mean": hella_mean,
                "hellaswag_std": hella_std,
                "arc_challenge_mean": arc_mean,
                "arc_challenge_std": arc_std,
                "actual_sparsity_mean": sparsity_mean,
                "actual_sparsity_std": sparsity_std,
                "schedule_weighted_mean_mean": weighted_mean_mean,
                "schedule_weighted_mean_std": weighted_mean_std,
            }
        )
    return summary_rows


def build_condition_index(rows):
    index = {}
    for row in rows:
        key = (row["model_label"], row["global_sparsity"], row["seed"], row["score_type"])
        index[key] = row
    return index


def summarize_pairwise(cfg, matches):
    deltas_ppl = [item["delta_ppl"] for item in matches if item["delta_ppl"] is not None]
    deltas_avg = [item["delta_avgacc"] for item in matches if item["delta_avgacc"] is not None]
    challenger_alphas = [item["challenger_alpha"] for item in matches if item["challenger_alpha"] is not None]
    baseline_alphas = [item["baseline_alpha"] for item in matches if item["baseline_alpha"] is not None]
    return {
        "pair_name": cfg["name"],
        "challenger": cfg["challenger"],
        "baseline": cfg["baseline"],
        "matched_count": len(matches),
        "challenger_alpha_mean": float(np.mean(challenger_alphas)) if challenger_alphas else None,
        "baseline_alpha_mean": float(np.mean(baseline_alphas)) if baseline_alphas else None,
        "win_count_ppl": int(sum(delta < 0.0 for delta in deltas_ppl)),
        "loss_count_ppl": int(sum(delta > 0.0 for delta in deltas_ppl)),
        "mean_delta_ppl": float(np.mean(deltas_ppl)) if deltas_ppl else None,
        "median_delta_ppl": float(np.median(deltas_ppl)) if deltas_ppl else None,
        "win_count_avgacc": int(sum(delta > 0.0 for delta in deltas_avg)),
        "loss_count_avgacc": int(sum(delta < 0.0 for delta in deltas_avg)),
        "mean_delta_avgacc": float(np.mean(deltas_avg)) if deltas_avg else None,
        "worst_delta_avgacc": float(np.min(deltas_avg)) if deltas_avg else None,
        "best_delta_avgacc": float(np.max(deltas_avg)) if deltas_avg else None,
        "matched_conditions": ";".join(item["condition_label"] for item in matches),
    }


def build_pairwise_comparisons(rows):
    indexed = build_condition_index(rows)
    model_gs_seed_keys = sorted({(row["model_label"], row["global_sparsity"], row["seed"]) for row in rows})
    summary_rows = []
    for cfg in PAIRWISE_CONFIGS:
        matches = []
        for model_label, global_sparsity, seed in model_gs_seed_keys:
            challenger = indexed.get((model_label, global_sparsity, seed, cfg["challenger"]))
            baseline = indexed.get((model_label, global_sparsity, seed, cfg["baseline"]))
            if challenger is None or baseline is None:
                continue
            ppl_challenger = challenger.get("ppl_wikitext2")
            ppl_baseline = baseline.get("ppl_wikitext2")
            avg_challenger = challenger.get("zeroshot_avg")
            avg_baseline = baseline.get("zeroshot_avg")
            matches.append(
                {
                    "condition_label": f"gs={global_sparsity}|seed={seed}",
                    "challenger_alpha": challenger.get("alpha"),
                    "baseline_alpha": baseline.get("alpha"),
                    "delta_ppl": None if ppl_challenger is None or ppl_baseline is None else float(ppl_challenger - ppl_baseline),
                    "delta_avgacc": None if avg_challenger is None or avg_baseline is None else float(avg_challenger - avg_baseline),
                }
            )
        summary_rows.append(summarize_pairwise(cfg, matches))
    return summary_rows


def evaluate_pair_pass(row):
    mean_delta_ppl = row.get("mean_delta_ppl")
    mean_delta_avgacc = row.get("mean_delta_avgacc")
    worst_delta_avgacc = row.get("worst_delta_avgacc")
    return {
        "matched_count": int(row.get("matched_count") or 0),
        "pass_mean_ppl": bool(mean_delta_ppl is not None and mean_delta_ppl < 0.0),
        "pass_majority_ppl": bool((row.get("win_count_ppl") or 0) > (row.get("loss_count_ppl") or 0)),
        "pass_mean_avgacc": bool(mean_delta_avgacc is not None and mean_delta_avgacc > 0.0),
        "pass_majority_avgacc": bool((row.get("win_count_avgacc") or 0) > (row.get("loss_count_avgacc") or 0)),
        "pass_no_avgacc_backlash": bool(worst_delta_avgacc is not None and worst_delta_avgacc >= 0.0),
    }


def build_pass_fail_summary(pairwise_rows):
    pair_payload = {}
    for row in pairwise_rows:
        verdict = evaluate_pair_pass(row)
        pair_payload[row["pair_name"]] = {
            "challenger": row["challenger"],
            "baseline": row["baseline"],
            "matched_count": int(row["matched_count"]),
            "mean_delta_ppl": row["mean_delta_ppl"],
            "mean_delta_avgacc": row["mean_delta_avgacc"],
            "worst_delta_avgacc": row["worst_delta_avgacc"],
            "verdict": {
                **verdict,
                "overall_pass_strict": bool(
                    verdict["matched_count"] > 0
                    and verdict["pass_mean_ppl"]
                    and verdict["pass_majority_ppl"]
                    and verdict["pass_mean_avgacc"]
                    and verdict["pass_majority_avgacc"]
                    and verdict["pass_no_avgacc_backlash"]
                ),
            },
        }

    def strict(pair_name):
        return bool(pair_payload.get(pair_name, {}).get("verdict", {}).get("overall_pass_strict", False))

    return {
        "criteria": {
            "pass_mean_ppl": "mean_delta_ppl < 0",
            "pass_majority_ppl": "win_count_ppl > loss_count_ppl",
            "pass_mean_avgacc": "mean_delta_avgacc > 0",
            "pass_majority_avgacc": "win_count_avgacc > loss_count_avgacc",
            "pass_no_avgacc_backlash": "worst_delta_avgacc >= 0",
            "overall_pass_strict": "all of the above are true and matched_count > 0",
        },
        "pairs": pair_payload,
        "high_level": {
            "combo_rank_beats_baselines": strict("combo_rank_vs_uniform") and strict("combo_rank_vs_local_error"),
            "combo_rank_position_structure_useful": strict("combo_rank_vs_shuffled_combo_rank") and strict("combo_rank_vs_reversed_combo_rank"),
            "downstream_area_beats_baselines": strict("downstream_area_vs_uniform") and strict("downstream_area_vs_local_error"),
            "downstream_area_position_structure_useful": strict("downstream_area_vs_shuffled_downstream_area") and strict("downstream_area_vs_reversed_downstream_area"),
        },
    }


def interpolate_color(color_a, color_b, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(color_a, color_b))


def placeholder_image(output_path, title, subtitle):
    image = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 24), title, fill="black", font=font)
    draw.text((24, 48), subtitle, fill="black", font=font)
    image.save(output_path)


def save_metric_heatmap(rows, metric_key, output_path, title, lower_is_better):
    conditions = sorted({(row["global_sparsity"], row["seed"]) for row in rows})
    if not rows or not conditions:
        placeholder_image(output_path, title, "no data")
        return

    matrix = []
    finite_values = []
    index = {(row["score_type"], row["global_sparsity"], row["seed"]): row for row in rows}
    for method in ORDERED_METHODS:
        row_values = []
        for global_sparsity, seed in conditions:
            cell = index.get((method, global_sparsity, seed))
            value = None if cell is None else cell.get(metric_key)
            row_values.append(value)
            if value is not None and math.isfinite(value):
                finite_values.append(value)
        matrix.append(row_values)

    width = 180 + 110 * len(conditions)
    height = 90 + 42 * len(ORDERED_METHODS)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((16, 14), title, fill="black", font=font)

    if not finite_values:
        draw.text((16, 40), "no finite values", fill="black", font=font)
        image.save(output_path)
        return

    vmin, vmax = finite_min_max(np.asarray(finite_values, dtype=np.float64))
    if math.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    left = 160
    top = 48
    cell_w = 100
    cell_h = 32
    for col_idx, (global_sparsity, seed) in enumerate(conditions):
        x0 = left + col_idx * cell_w
        draw.text((x0 + 4, top - 18), f"gs={global_sparsity}", fill="black", font=font)
        draw.text((x0 + 16, top - 6), f"seed={seed}", fill="black", font=font)

    for row_idx, method in enumerate(ORDERED_METHODS):
        y0 = top + row_idx * cell_h
        draw.text((16, y0 + 8), method, fill="black", font=font)
        for col_idx, value in enumerate(matrix[row_idx]):
            x0 = left + col_idx * cell_w
            x1 = x0 + cell_w - 4
            y1 = y0 + cell_h - 4
            if value is None or not math.isfinite(value):
                fill = (235, 235, 235)
                label = "NA"
            else:
                frac = (value - vmin) / max(vmax - vmin, 1e-12)
                good_frac = 1.0 - frac if lower_is_better else frac
                bad = (235, 90, 90)
                good = (90, 180, 110)
                fill = interpolate_color(bad, good, good_frac)
                label = f"{value:.4f}"
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(120, 120, 120))
            draw.text((x0 + 8, y0 + 10), label, fill="black", font=font)

    image.save(output_path)


def draw_zero_axis(draw, left, right, top, bottom, ymin, ymax):
    zero_y = bottom - (0.0 - ymin) * (bottom - top) / max(ymax - ymin, 1e-12)
    draw.line((left, zero_y, right, zero_y), fill=(80, 80, 80), width=1)
    return zero_y


def save_family_barplot(pairwise_rows, challenger, output_path, title):
    relevant = [row for row in pairwise_rows if row["challenger"] == challenger]
    if not relevant:
        placeholder_image(output_path, title, "no pairwise rows")
        return

    width, height = 980, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((20, 16), title, fill="black", font=font)

    panel_left = 80
    panel_right = width - 40
    panel_top_1 = 60
    panel_bottom_1 = 330
    panel_top_2 = 410
    panel_bottom_2 = 680
    bar_w = 120
    gap = 40

    ppl_values = [row["mean_delta_ppl"] for row in relevant if row["mean_delta_ppl"] is not None]
    acc_values = [row["mean_delta_avgacc"] for row in relevant if row["mean_delta_avgacc"] is not None]
    if not ppl_values:
        ppl_values = [0.0]
    if not acc_values:
        acc_values = [0.0]
    ppl_min = min(0.0, min(ppl_values))
    ppl_max = max(0.0, max(ppl_values))
    acc_min = min(0.0, min(acc_values))
    acc_max = max(0.0, max(acc_values))
    if math.isclose(ppl_min, ppl_max):
        ppl_min -= 1.0
        ppl_max += 1.0
    if math.isclose(acc_min, acc_max):
        acc_min -= 0.01
        acc_max += 0.01

    draw.text((panel_left, panel_top_1 - 24), "mean delta PPL (challenger - baseline, lower is better)", fill="black", font=font)
    draw.text((panel_left, panel_top_2 - 24), "mean delta avg acc (challenger - baseline, higher is better)", fill="black", font=font)
    draw.line((panel_left, panel_top_1, panel_left, panel_bottom_1), fill="black", width=2)
    draw.line((panel_left, panel_top_2, panel_left, panel_bottom_2), fill="black", width=2)
    draw.line((panel_left, panel_bottom_1, panel_right, panel_bottom_1), fill="black", width=2)
    draw.line((panel_left, panel_bottom_2, panel_right, panel_bottom_2), fill="black", width=2)
    zero_y_ppl = draw_zero_axis(draw, panel_left, panel_right, panel_top_1, panel_bottom_1, ppl_min, ppl_max)
    zero_y_acc = draw_zero_axis(draw, panel_left, panel_right, panel_top_2, panel_bottom_2, acc_min, acc_max)

    for idx, row in enumerate(relevant):
        baseline = row["baseline"]
        x0 = panel_left + 30 + idx * (bar_w + gap)
        x1 = x0 + bar_w

        delta_ppl = row["mean_delta_ppl"]
        if delta_ppl is not None:
            y = panel_bottom_1 - (delta_ppl - ppl_min) * (panel_bottom_1 - panel_top_1) / max(ppl_max - ppl_min, 1e-12)
            color = (90, 180, 110) if delta_ppl < 0.0 else (235, 90, 90)
            draw.rectangle((x0, min(y, zero_y_ppl), x1, max(y, zero_y_ppl)), fill=color, outline=color)
            draw.text((x0, min(y, zero_y_ppl) - 14), f"{delta_ppl:.4f}", fill="black", font=font)

        delta_acc = row["mean_delta_avgacc"]
        if delta_acc is not None:
            y = panel_bottom_2 - (delta_acc - acc_min) * (panel_bottom_2 - panel_top_2) / max(acc_max - acc_min, 1e-12)
            color = (90, 180, 110) if delta_acc > 0.0 else (235, 90, 90)
            draw.rectangle((x0, min(y, zero_y_acc), x1, max(y, zero_y_acc)), fill=color, outline=color)
            draw.text((x0, min(y, zero_y_acc) - 14), f"{delta_acc:.4f}", fill="black", font=font)

        draw.text((x0, panel_bottom_1 + 12), baseline, fill="black", font=font)
        draw.text((x0, panel_bottom_2 + 12), baseline, fill="black", font=font)

    image.save(output_path)


def read_schedule_values(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    rows.sort(key=lambda item: int(item["layer_idx"]))
    return np.asarray([float(row["assigned_sparsity"]) for row in rows], dtype=np.float64)


def find_schedule_example_paths(rows):
    schedule_map = {}
    for row in rows:
        schedule_path = os.path.join(os.path.dirname(row["source_file"]), "schedule.csv")
        if os.path.exists(schedule_path):
            key = (row["global_sparsity"], row["seed"], row["score_type"])
            schedule_map[key] = schedule_path
    conditions = sorted({(row["global_sparsity"], row["seed"]) for row in rows})
    for global_sparsity, seed in conditions:
        required = [
            (global_sparsity, seed, "combo_rank"),
            (global_sparsity, seed, "shuffled_combo_rank"),
            (global_sparsity, seed, "reversed_combo_rank"),
        ]
        if all(key in schedule_map for key in required):
            return {
                "title": f"gs={global_sparsity}, seed={seed}",
                "paths": {
                    "combo_rank": schedule_map[(global_sparsity, seed, "combo_rank")],
                    "shuffled_combo_rank": schedule_map[(global_sparsity, seed, "shuffled_combo_rank")],
                    "reversed_combo_rank": schedule_map[(global_sparsity, seed, "reversed_combo_rank")],
                },
            }
    return None


def save_schedule_examples(rows, output_path):
    example = find_schedule_example_paths(rows)
    if example is None:
        placeholder_image(output_path, "schedule_examples", "no common combo_rank/shuffled/reversed trio found")
        return

    series = {
        name: read_schedule_values(path)
        for name, path in example["paths"].items()
    }
    width, height = 980, 520
    left, right, top, bottom = 72, 220, 48, 56
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 14), f"schedule examples ({example['title']})", fill="black", font=font)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.text((left, height - 22), "layer_idx", fill="black", font=font)
    draw.text((width - 190, top), "assigned_sparsity", fill="black", font=font)

    all_values = np.concatenate(list(series.values()))
    ymin, ymax = finite_min_max(all_values)
    xmax = max(len(next(iter(series.values()))) - 1, 1)

    def x_to_px(index):
        return left + index * (width - left - right) / xmax

    def y_to_px(value):
        return height - bottom - (value - ymin) * (height - top - bottom) / max(ymax - ymin, 1e-12)

    palette = {
        "combo_rank": (220, 70, 70),
        "shuffled_combo_rank": (30, 136, 229),
        "reversed_combo_rank": (67, 160, 71),
    }
    legend_y = top + 4
    for name in ["combo_rank", "shuffled_combo_rank", "reversed_combo_rank"]:
        values = series[name]
        color = palette[name]
        points = [(x_to_px(idx), y_to_px(float(value))) for idx, value in enumerate(values)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        for px, py in points:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color, outline=color)
        draw.rectangle((width - 210, legend_y + 3, width - 198, legend_y + 15), fill=color, outline=color)
        draw.text((width - 190, legend_y), name, fill="black", font=font)
        legend_y += 20

    tick_step = max(1, len(next(iter(series.values()))) // 10)
    for idx in range(0, len(next(iter(series.values()))), tick_step):
        draw.text((x_to_px(idx), height - bottom + 6), str(idx), fill="black", font=font)

    image.save(output_path)


def main():
    args = parse_args()
    save_dir = args.save_dir or args.results_root
    os.makedirs(save_dir, exist_ok=True)

    result_files = collect_one_line_files(args.results_root)
    if not result_files:
        raise RuntimeError(f"no one_line_result.csv files found under {args.results_root}")

    all_rows = [read_one_line_csv(path) for path in result_files]
    grouped_rows = build_grouped_summary(all_rows)
    pairwise_rows = build_pairwise_comparisons(all_rows)
    pass_fail_summary = build_pass_fail_summary(pairwise_rows)

    all_results_path = os.path.join(save_dir, "all_results.csv")
    grouped_summary_path = os.path.join(save_dir, "grouped_summary.csv")
    pairwise_path = os.path.join(save_dir, "pairwise_comparison.csv")
    pass_fail_path = os.path.join(save_dir, "pass_fail_summary.json")
    ppl_heatmap_path = os.path.join(save_dir, "ppl_heatmap_by_method.png")
    acc_heatmap_path = os.path.join(save_dir, "avgacc_heatmap_by_method.png")
    combo_barplot_path = os.path.join(save_dir, "combo_rank_vs_controls_barplot.png")
    downstream_barplot_path = os.path.join(save_dir, "downstream_area_vs_controls_barplot.png")
    schedule_examples_path = os.path.join(save_dir, "schedule_examples.png")

    write_csv(all_rows, ALL_RESULTS_FIELDS, all_results_path)
    write_csv(grouped_rows, GROUPED_FIELDS, grouped_summary_path)
    write_csv(pairwise_rows, PAIRWISE_FIELDS, pairwise_path)
    write_json(pass_fail_summary, pass_fail_path)

    save_metric_heatmap(all_rows, "ppl_wikitext2", ppl_heatmap_path, "PPL heatmap by method", True)
    save_metric_heatmap(all_rows, "zeroshot_avg", acc_heatmap_path, "3-task avg accuracy heatmap by method", False)
    save_family_barplot(pairwise_rows, "combo_rank", combo_barplot_path, "combo_rank vs controls")
    save_family_barplot(pairwise_rows, "downstream_area", downstream_barplot_path, "downstream_area vs controls")
    save_schedule_examples(all_rows, schedule_examples_path)

    print("saved outputs:")
    print(all_results_path)
    print(grouped_summary_path)
    print(pairwise_path)
    print(pass_fail_path)
    print(ppl_heatmap_path)
    print(acc_heatmap_path)
    print(combo_barplot_path)
    print(downstream_barplot_path)
    print(schedule_examples_path)


if __name__ == "__main__":
    main()

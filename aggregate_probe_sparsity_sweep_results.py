import argparse
import csv
import json
import math
import os
from collections import defaultdict

SUMMARY_FIELDS = [
    "source_file",
    "model_name",
    "model_label",
    "dataset",
    "probe_sparsity",
    "probe_seed",
    "probe_nsamples",
    "probe_seqlen",
    "global_sparsity",
    "score_type",
    "base_score_type",
    "schedule_transform",
    "alpha",
    "generator_mode",
    "profile_temperature",
    "profile_beta",
    "seed",
    "shuffle_seed",
    "actual_sparsity",
    "ppl_wikitext2",
    "zeroshot_avg",
    "boolq",
    "hellaswag",
    "arc_challenge",
    "schedule_min",
    "schedule_max",
    "schedule_weighted_mean",
    "eval_zeroshot",
    "zeroshot_tasks",
    "probe_metrics_csv",
    "probe_summary_json",
    "eval_results_json",
    "one_line_result_csv",
]
GROUPED_FIELDS = [
    "probe_sparsity",
    "count",
    "ppl_wikitext2_mean",
    "ppl_wikitext2_std",
    "actual_sparsity_mean",
    "actual_sparsity_std",
    "schedule_min_mean",
    "schedule_max_mean",
    "schedule_weighted_mean_mean",
    "alpha_values",
    "generator_mode_values",
    "seed_values",
    "probe_seed_values",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def read_single_csv_row(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return row
    raise ValueError(f"csv file has no data rows: {path}")


def maybe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    if text.lower() == "nan":
        return float("nan")
    return float(text)


def maybe_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return int(text)


def serialize_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return value


def finite_values(values):
    return [value for value in values if value is not None and math.isfinite(value)]


def safe_mean(values):
    usable = finite_values(values)
    if not usable:
        return None
    return sum(usable) / len(usable)


def safe_std(values):
    usable = finite_values(values)
    if not usable:
        return None
    mean = sum(usable) / len(usable)
    variance = sum((value - mean) ** 2 for value in usable) / len(usable)
    return math.sqrt(variance)


def unique_join(values):
    items = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            items.append(text)
            seen.add(text)
    return ",".join(items)


def parse_probe_tag_to_float(tag):
    if tag is None:
        return None
    return float(str(tag).replace("p", "."))


def extract_probe_tag(path):
    if not path:
        return None
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if not part.startswith("probe_") or part == "probe_metrics":
            continue
        suffix = part[len("probe_"):]
        if not suffix:
            continue
        try:
            parse_probe_tag_to_float(suffix)
        except ValueError:
            continue
        return suffix
    return None


def extract_seed_from_probe_metrics_path(path):
    if not path:
        return None
    parts = os.path.normpath(path).split(os.sep)
    try:
        probe_metrics_idx = parts.index("probe_metrics")
    except ValueError:
        probe_metrics_idx = -1
    if probe_metrics_idx >= 0:
        for part in parts[probe_metrics_idx + 1:]:
            if part.startswith("seed_"):
                try:
                    return int(part[len("seed_"):])
                except ValueError:
                    return None
    for part in parts:
        if part.startswith("seed_"):
            try:
                return int(part[len("seed_"):])
            except ValueError:
                return None
    return None


def load_probe_summary(probe_metrics_csv):
    if not probe_metrics_csv:
        return None, {}
    probe_summary_path = os.path.join(os.path.dirname(probe_metrics_csv), "summary.json")
    if not os.path.exists(probe_summary_path):
        return probe_summary_path, {}
    return probe_summary_path, read_json(probe_summary_path)


def build_result_row(one_line_result_csv):
    one_line_row = read_single_csv_row(one_line_result_csv)
    eval_results_json = os.path.join(os.path.dirname(one_line_result_csv), "eval_results.json")
    eval_payload = read_json(eval_results_json) if os.path.exists(eval_results_json) else {}
    probe_metrics_csv = eval_payload.get("probe_metrics_csv")
    probe_summary_json, probe_summary = load_probe_summary(probe_metrics_csv)
    probe_tag = extract_probe_tag(one_line_result_csv) or extract_probe_tag(probe_metrics_csv)
    probe_sparsity = probe_summary.get("probe_sparsity")
    if probe_sparsity is None and probe_tag is not None:
        probe_sparsity = parse_probe_tag_to_float(probe_tag)
    return {
        "source_file": one_line_result_csv,
        "model_name": eval_payload.get("model_name", one_line_row.get("model_name")),
        "model_label": eval_payload.get("model_label", one_line_row.get("model_label")),
        "dataset": eval_payload.get("dataset"),
        "probe_sparsity": maybe_float(probe_sparsity),
        "probe_seed": extract_seed_from_probe_metrics_path(probe_metrics_csv),
        "probe_nsamples": maybe_int(probe_summary.get("nsamples")),
        "probe_seqlen": maybe_int(probe_summary.get("seqlen")),
        "global_sparsity": maybe_float(one_line_row.get("global_sparsity")),
        "score_type": one_line_row.get("score_type"),
        "base_score_type": one_line_row.get("base_score_type"),
        "schedule_transform": one_line_row.get("schedule_transform"),
        "alpha": maybe_float(one_line_row.get("alpha")),
        "generator_mode": one_line_row.get("generator_mode"),
        "profile_temperature": maybe_float(one_line_row.get("profile_temperature")),
        "profile_beta": maybe_float(one_line_row.get("profile_beta")),
        "seed": maybe_int(one_line_row.get("seed")),
        "shuffle_seed": maybe_int(one_line_row.get("shuffle_seed")),
        "actual_sparsity": maybe_float(one_line_row.get("actual_sparsity")),
        "ppl_wikitext2": maybe_float(one_line_row.get("ppl_wikitext2")),
        "zeroshot_avg": maybe_float(one_line_row.get("zeroshot_avg")),
        "boolq": maybe_float(one_line_row.get("boolq")),
        "hellaswag": maybe_float(one_line_row.get("hellaswag")),
        "arc_challenge": maybe_float(one_line_row.get("arc_challenge")),
        "schedule_min": maybe_float(one_line_row.get("schedule_min")),
        "schedule_max": maybe_float(one_line_row.get("schedule_max")),
        "schedule_weighted_mean": maybe_float(one_line_row.get("schedule_weighted_mean")),
        "eval_zeroshot": eval_payload.get("eval_zeroshot", one_line_row.get("eval_zeroshot")),
        "zeroshot_tasks": eval_payload.get("zeroshot_tasks", one_line_row.get("zeroshot_tasks")),
        "probe_metrics_csv": probe_metrics_csv,
        "probe_summary_json": probe_summary_json,
        "eval_results_json": eval_results_json,
        "one_line_result_csv": one_line_result_csv,
    }


def iter_result_files(results_root):
    for root, _, files in os.walk(results_root):
        if "one_line_result.csv" in files:
            yield os.path.join(root, "one_line_result.csv")


def sort_key_for_row(row):
    probe_sparsity = row.get("probe_sparsity")
    if probe_sparsity is None or not math.isfinite(probe_sparsity):
        return (1, float("inf"), row.get("source_file", ""))
    return (0, probe_sparsity, row.get("source_file", ""))


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("probe_sparsity")].append(row)
    summary_rows = []
    for probe_sparsity, items in sorted(grouped.items(), key=lambda item: (item[0] is None, float("inf") if item[0] is None else item[0])):
        summary_rows.append(
            {
                "probe_sparsity": probe_sparsity,
                "count": len(items),
                "ppl_wikitext2_mean": safe_mean([item.get("ppl_wikitext2") for item in items]),
                "ppl_wikitext2_std": safe_std([item.get("ppl_wikitext2") for item in items]),
                "actual_sparsity_mean": safe_mean([item.get("actual_sparsity") for item in items]),
                "actual_sparsity_std": safe_std([item.get("actual_sparsity") for item in items]),
                "schedule_min_mean": safe_mean([item.get("schedule_min") for item in items]),
                "schedule_max_mean": safe_mean([item.get("schedule_max") for item in items]),
                "schedule_weighted_mean_mean": safe_mean([item.get("schedule_weighted_mean") for item in items]),
                "alpha_values": unique_join(item.get("alpha") for item in items),
                "generator_mode_values": unique_join(item.get("generator_mode") for item in items),
                "seed_values": unique_join(item.get("seed") for item in items),
                "probe_seed_values": unique_join(item.get("probe_seed") for item in items),
            }
        )
    return summary_rows


def build_summary_payload(rows):
    ppl_rows = [row for row in rows if row.get("ppl_wikitext2") is not None and math.isfinite(row.get("ppl_wikitext2"))]
    summary = {
        "count": len(rows),
        "probe_sparsity_values": [row.get("probe_sparsity") for row in sorted(rows, key=sort_key_for_row)],
        "score_type_values": sorted({row.get("score_type") for row in rows if row.get("score_type")}),
        "generator_mode_values": sorted({row.get("generator_mode") for row in rows if row.get("generator_mode")}),
        "global_sparsity_values": sorted({row.get("global_sparsity") for row in rows if row.get("global_sparsity") is not None}),
        "alpha_values": sorted({row.get("alpha") for row in rows if row.get("alpha") is not None}),
        "seed_values": sorted({row.get("seed") for row in rows if row.get("seed") is not None}),
        "probe_seed_values": sorted({row.get("probe_seed") for row in rows if row.get("probe_seed") is not None}),
    }
    if ppl_rows:
        best_row = min(ppl_rows, key=lambda row: row["ppl_wikitext2"])
        summary["best_probe_sparsity_by_ppl"] = best_row.get("probe_sparsity")
        summary["best_ppl_wikitext2"] = best_row.get("ppl_wikitext2")
        summary["best_result"] = {field: serialize_value(best_row.get(field)) for field in SUMMARY_FIELDS}
    else:
        summary["best_probe_sparsity_by_ppl"] = None
        summary["best_ppl_wikitext2"] = None
        summary["best_result"] = None
    return summary


def write_csv(rows, fieldnames, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_value(row.get(field)) for field in fieldnames})


def write_json(payload, path):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    save_dir = args.save_dir or args.results_root
    os.makedirs(save_dir, exist_ok=True)

    rows = [build_result_row(path) for path in iter_result_files(args.results_root)]
    rows.sort(key=sort_key_for_row)
    grouped_rows = group_rows(rows)
    summary_payload = build_summary_payload(rows)

    write_csv(rows, SUMMARY_FIELDS, os.path.join(save_dir, "probe_sparsity_ppl.csv"))
    write_csv(grouped_rows, GROUPED_FIELDS, os.path.join(save_dir, "probe_sparsity_grouped_summary.csv"))
    write_json(summary_payload, os.path.join(save_dir, "probe_sparsity_summary.json"))


if __name__ == "__main__":
    main()

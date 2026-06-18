import argparse
import math
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from paper_experiment_utils import ensure_dir, maybe_float, read_csv_rows, write_json


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
    parser.add_argument("--paper_root", type=str, required=True)
    parser.add_argument("--tables_dir", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    return parser.parse_args()


def finite_values(values):
    output = []
    for value in values:
        number = maybe_float(value)
        if number is not None and math.isfinite(number):
            output.append(number)
    return output


def finite_min_max(values, default_min=0.0, default_max=1.0):
    valid = finite_values(values)
    if not valid:
        return default_min, default_max
    vmin = min(valid)
    vmax = max(valid)
    if vmax <= vmin:
        pad = 0.5 if vmin == 0 else abs(vmin) * 0.1
        return vmin - pad, vmax + pad
    return vmin, vmax


def load_table(path):
    if not os.path.exists(path):
        return []
    return read_csv_rows(path)


def draw_line_chart(series_map, output_path, title, x_label, y_label):
    prepared = []
    for name, points in series_map.items():
        filtered = []
        for x_value, y_value in points:
            x_number = maybe_float(x_value)
            y_number = maybe_float(y_value)
            if x_number is None or y_number is None:
                continue
            filtered.append((x_number, y_number))
        filtered.sort(key=lambda item: item[0])
        if filtered:
            prepared.append((name, filtered))
    if not prepared:
        return False

    width, height = 1100, 760
    left, right, top, bottom = 90, 260, 50, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x_values = [point[0] for _, points in prepared for point in points]
    y_values = [point[1] for _, points in prepared for point in points]
    xmin, xmax = finite_min_max(x_values)
    ymin, ymax = finite_min_max(y_values)
    yrange = max(ymax - ymin, 1e-12)
    xrange = max(xmax - xmin, 1e-12)
    ymin -= yrange * 0.05
    ymax += yrange * 0.08

    def x_to_px(value):
        return left + (value - xmin) * (width - left - right) / xrange

    def y_to_px(value):
        return height - bottom - (value - ymin) * (height - top - bottom) / max(ymax - ymin, 1e-12)

    draw.rectangle((0, 0, width - 1, height - 1), outline=(220, 220, 220))
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.text((left, 16), title, fill="black", font=font)
    draw.text((left, height - 30), x_label, fill="black", font=font)
    draw.text((width - right + 20, top), y_label, fill="black", font=font)

    tick_count = 5
    for tick_index in range(tick_count + 1):
        x_value = xmin + (xmax - xmin) * tick_index / tick_count
        y_value = ymin + (ymax - ymin) * tick_index / tick_count
        x_pos = x_to_px(x_value)
        y_pos = y_to_px(y_value)
        draw.line((x_pos, top, x_pos, height - bottom), fill=(235, 235, 235), width=1)
        draw.line((left, y_pos, width - right, y_pos), fill=(235, 235, 235), width=1)
        draw.text((x_pos - 10, height - bottom + 8), f"{x_value:.2f}", fill="black", font=font)
        draw.text((10, y_pos - 4), f"{y_value:.3f}", fill="black", font=font)

    legend_y = top + 10
    for index, (name, points) in enumerate(prepared):
        color = COLOR_PALETTE[index % len(COLOR_PALETTE)]
        previous = None
        for x_value, y_value in points:
            px = x_to_px(x_value)
            py = y_to_px(y_value)
            if previous is not None:
                draw.line((previous[0], previous[1], px, py), fill=color, width=3)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=color)
            previous = (px, py)
        legend_x = width - right + 20
        draw.line((legend_x, legend_y + 6, legend_x + 24, legend_y + 6), fill=color, width=3)
        draw.ellipse((legend_x + 8, legend_y + 2, legend_x + 16, legend_y + 10), fill=color, outline=color)
        draw.text((legend_x + 32, legend_y), name, fill="black", font=font)
        legend_y += 18

    image.save(output_path)
    return True


def draw_bar_chart(items, output_path, title, y_label):
    prepared = []
    for item in items:
        value = maybe_float(item.get("value"))
        if value is None:
            continue
        prepared.append({"label": str(item.get("label", "item")), "value": value})
    if not prepared:
        return False

    width = max(1100, 110 + 80 * len(prepared))
    height = 760
    left, right, top, bottom = 90, 40, 50, 150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    values = [item["value"] for item in prepared]
    ymin, ymax = finite_min_max(values)
    if ymin > 0:
        ymin = 0.0
    if ymax < 0:
        ymax = 0.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    def y_to_px(value):
        return height - bottom - (value - ymin) * (height - top - bottom) / max(ymax - ymin, 1e-12)

    zero_y = y_to_px(0.0)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(220, 220, 220))
    draw.line((left, zero_y, width - right, zero_y), fill="black", width=2)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.text((left, 16), title, fill="black", font=font)
    draw.text((width - 120, top), y_label, fill="black", font=font)

    tick_count = 5
    for tick_index in range(tick_count + 1):
        y_value = ymin + (ymax - ymin) * tick_index / tick_count
        y_pos = y_to_px(y_value)
        draw.line((left, y_pos, width - right, y_pos), fill=(235, 235, 235), width=1)
        draw.text((10, y_pos - 4), f"{y_value:.3f}", fill="black", font=font)

    plot_width = width - left - right
    bar_width = max(24, min(54, int(plot_width / max(len(prepared) * 1.6, 1))))
    gap = max(18, int((plot_width - bar_width * len(prepared)) / max(len(prepared) + 1, 1)))
    x_pos = left + gap
    for index, item in enumerate(prepared):
        color = COLOR_PALETTE[index % len(COLOR_PALETTE)]
        value = item["value"]
        top_y = y_to_px(max(value, 0.0))
        bottom_y = y_to_px(min(value, 0.0))
        draw.rectangle((x_pos, top_y, x_pos + bar_width, bottom_y), fill=color, outline=color)
        draw.text((x_pos - 8, height - bottom + 10), item["label"][:14], fill="black", font=font)
        draw.text((x_pos - 4, min(top_y, bottom_y) - 14), f"{value:.3f}", fill="black", font=font)
        x_pos += bar_width + gap

    image.save(output_path)
    return True


def best_rows(rows, key_fields, metric_field):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(field) for field in key_fields)].append(row)
    output = []
    for key in sorted(buckets.keys(), key=lambda item: tuple("" if value is None else str(value) for value in item)):
        best_row = None
        best_value = None
        for row in buckets[key]:
            value = maybe_float(row.get(metric_field))
            if value is None:
                continue
            if best_value is None or value < best_value:
                best_value = value
                best_row = row
        if best_row is not None:
            output.append(best_row)
    return output


def alpha_sweep_figure(tables_dir, figures_dir):
    rows = load_table(os.path.join(tables_dir, "paper_alpha_sweep_table.csv"))
    series = defaultdict(list)
    for row in rows:
        score_type = row.get("score_type")
        if score_type not in {None, "", "downstream_area"}:
            continue
        label = f"gs={row.get('global_sparsity')} {row.get('generator_mode')}"
        series[label].append((row.get("alpha"), row.get("ppl_wikitext2_mean")))
    return draw_line_chart(series, os.path.join(figures_dir, "alpha_sweep_downstream_area.png"), "alpha sweep", "alpha", "mean ppl")


def main_submission_figure(tables_dir, figures_dir):
    rows = load_table(os.path.join(tables_dir, "paper_main_table.csv"))
    selected = best_rows(rows, ["global_sparsity", "score_type", "generator_mode"], "ppl_wikitext2_mean")
    items = []
    for row in selected:
        items.append({"label": f"gs{row.get('global_sparsity')}_{row.get('score_type')}", "value": row.get("ppl_wikitext2_mean")})
    return draw_bar_chart(items, os.path.join(figures_dir, "submission_gate_best_ppl.png"), "submission gate best mean ppl", "mean ppl")


def baseline_figure(tables_dir, figures_dir):
    rows = load_table(os.path.join(tables_dir, "paper_baseline_table.csv"))
    items = [{"label": row.get("method_label"), "value": row.get("ppl_wikitext2_mean")} for row in rows]
    return draw_bar_chart(items, os.path.join(figures_dir, "baseline_comparison.png"), "baseline comparison", "mean ppl")


def controls_figure(tables_dir, figures_dir):
    rows = load_table(os.path.join(tables_dir, "paper_controls_table.csv"))
    items = []
    for row in rows:
        items.append({"label": f"gs{row.get('global_sparsity')}_{row.get('control_score_type')}", "value": row.get("mean_delta_ppl")})
    return draw_bar_chart(items, os.path.join(figures_dir, "control_delta_ppl.png"), "control deltas vs downstream_area", "delta ppl")


def mechanism_figure(tables_dir, figures_dir):
    rows = load_table(os.path.join(tables_dir, "paper_mechanism_table.csv"))
    items = []
    for row in rows:
        items.append({"label": f"{row.get('label')}_{row.get('probe_sparsity')}", "value": row.get("spearman_local_vs_finalkl_mean")})
    return draw_bar_chart(items, os.path.join(figures_dir, "mechanism_spearman.png"), "mechanism local vs final KL", "spearman")


def main():
    args = parse_args()
    tables_dir = args.tables_dir or os.path.join(args.paper_root, "tables")
    figures_dir = ensure_dir(args.figures_dir or os.path.join(args.paper_root, "figures"))

    outputs = {
        "alpha_sweep_downstream_area": alpha_sweep_figure(tables_dir, figures_dir),
        "submission_gate_best_ppl": main_submission_figure(tables_dir, figures_dir),
        "baseline_comparison": baseline_figure(tables_dir, figures_dir),
        "control_delta_ppl": controls_figure(tables_dir, figures_dir),
        "mechanism_spearman": mechanism_figure(tables_dir, figures_dir),
    }
    write_json({"paper_root": args.paper_root, "tables_dir": tables_dir, "figures_dir": figures_dir, "generated": outputs}, os.path.join(figures_dir, "figure_manifest.json"))
    for name, generated in outputs.items():
        print(f"{name}: {'ok' if generated else 'skipped'}")


if __name__ == "__main__":
    main()

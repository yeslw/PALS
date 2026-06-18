import json
import os
import sys

from paper_experiment_utils import TaskSpec, ensure_dir, float_tag, merge_dicts, slugify


def bool_text(value):
    return "true" if bool(value) else "false"


def make_common_config(config, repo_root):
    defaults = config.get("defaults", {})
    runtime = config.get("runtime", {})
    outputs = config.get("outputs", {})
    results_root = ensure_dir(outputs.get("results_root", "/2T/zhuhe/results/paper_experiments"))
    paper_root = ensure_dir(outputs.get("paper_root", "/2T/zhuhe/results/paper_outputs"))
    return {
        "repo_root": repo_root,
        "python": runtime.get("python_executable") or sys.executable,
        "results_root": results_root,
        "paper_root": paper_root,
        "model_name_or_path": defaults["model_name_or_path"],
        "tokenizer_name_or_path": defaults.get("tokenizer_name_or_path", defaults["model_name_or_path"]),
        "probe_metrics_csv": defaults.get("probe_metrics_csv"),
        "dataset": defaults.get("dataset", "c4"),
        "nsamples": int(defaults.get("nsamples", 128)),
        "seqlen": int(defaults.get("seqlen", 2048)),
        "mechanism_nsamples": int(defaults.get("mechanism_nsamples", 32)),
        "mechanism_seqlen": int(defaults.get("mechanism_seqlen", 512)),
        "probe_sparsity": float(defaults.get("probe_sparsity", 0.5)),
        "stream_batch_size": int(defaults.get("stream_batch_size", 4)),
        "profile_temperature": float(defaults.get("profile_temperature", 0.15)),
        "profile_beta": defaults.get("profile_beta"),
        "cache_dir": defaults.get("cache_dir", "/2T/zhuhe/results/llm_weights"),
        "zeroshot_tasks": defaults.get("zeroshot_tasks", "boolq,hellaswag,arc_challenge"),
        "phase_label": config.get("meta", {}).get("name", "paper_experiments"),
    }


def discover_baseline_adapters(repo_root):
    hybrid_runner = os.path.join(repo_root, "run_hybrid_prop_correction.py")
    return {
        "owl": {
            "available": os.path.exists(hybrid_runner),
            "runner": "run_hybrid_prop_correction.py" if os.path.exists(hybrid_runner) else None,
            "notes": "Uses existing hybrid propagation correction wrapper.",
        },
        "dlp": {
            "available": os.path.exists(hybrid_runner),
            "runner": "run_hybrid_prop_correction.py" if os.path.exists(hybrid_runner) else None,
            "notes": "Uses existing hybrid propagation correction wrapper.",
        },
        "alphapruning": {
            "available": False,
            "runner": None,
            "notes": "AlphaPruning implementation was not found in the current repository; placeholder only.",
        },
    }


def score_alpha_map(score_type, alpha=None, alpha_map=None):
    if isinstance(alpha_map, dict):
        if score_type in alpha_map:
            return {score_type: float(alpha_map[score_type])}
        base_type = score_type
        if score_type.startswith("shuffled_"):
            base_type = score_type[len("shuffled_"):]
        elif score_type.startswith("reversed_"):
            base_type = score_type[len("reversed_"):]
        if base_type in alpha_map:
            return {score_type: float(alpha_map[base_type])}
        if "default" in alpha_map:
            return {score_type: float(alpha_map["default"])}
    if alpha is None:
        raise ValueError(f"missing alpha for score_type={score_type}")
    return {score_type: float(alpha)}


def base_args(common):
    return [
        "--model_name_or_path",
        common["model_name_or_path"],
        "--tokenizer_name_or_path",
        common["tokenizer_name_or_path"],
        "--dataset",
        common["dataset"],
        "--cache_dir",
        common["cache_dir"],
    ]


def add_probe_metrics(common, command):
    if common.get("probe_metrics_csv"):
        command.extend(["--probe_metrics_csv", common["probe_metrics_csv"]])


def build_submission_task(common, group_name, stage, score_type, global_sparsity, seed, generator_mode, alpha_payload, output_dir, eval_zeroshot=False, debug_schedule=False, extra_metadata=None, exclusive_gpu=False):
    command = [common["python"], "run_dependency_submission_gate.py"]
    command.extend(base_args(common))
    add_probe_metrics(common, command)
    command.extend(
        [
            "--global_sparsity",
            str(global_sparsity),
            "--score_type",
            score_type,
            "--alpha_map_json",
            json.dumps(alpha_payload),
            "--seed",
            str(seed),
            "--nsamples",
            str(common["nsamples"]),
            "--seqlen",
            str(common["seqlen"]),
            "--device",
            "cuda:0",
            "--save_dir",
            output_dir,
            "--eval_zeroshot",
            bool_text(eval_zeroshot),
            "--zeroshot_tasks",
            common["zeroshot_tasks"],
            "--debug_schedule",
            bool_text(debug_schedule),
            "--generator_mode",
            generator_mode,
            "--profile_temperature",
            str(common["profile_temperature"]),
            "--stream_batch_size",
            str(common["stream_batch_size"]),
        ]
    )
    if common["profile_beta"] is not None:
        command.extend(["--profile_beta", str(common["profile_beta"])])
    metadata = {"score_type": score_type, "global_sparsity": float(global_sparsity), "seed": int(seed), "generator_mode": generator_mode}
    metadata.update(extra_metadata or {})
    task_id_parts = [slugify(group_name), float_tag(global_sparsity), str(seed), slugify(score_type), slugify(generator_mode)]
    if metadata.get("alpha") is not None:
        task_id_parts.append(float_tag(metadata["alpha"]))
    if metadata.get("label"):
        task_id_parts.append(slugify(metadata["label"]))
    if metadata.get("appendix_name"):
        task_id_parts.append(slugify(metadata["appendix_name"]))
    return TaskSpec(
        id="_".join(task_id_parts),
        group=group_name,
        stage=stage,
        name=os.path.basename(output_dir),
        command=command,
        output_dir=output_dir,
        expected_files=["schedule.csv", "eval_results.json", "one_line_result.csv"],
        metadata=metadata,
        exclusive_gpu=bool(exclusive_gpu),
    )


def build_budget_task(common, group_name, score_type, global_sparsity, alpha, seed, generator_mode, output_dir, eval_zeroshot=False, debug_schedule=False, extra_metadata=None, exclusive_gpu=False):
    command = [common["python"], "run_dependency_budget_ablation.py"]
    command.extend(base_args(common))
    add_probe_metrics(common, command)
    command.extend(
        [
            "--global_sparsity",
            str(global_sparsity),
            "--score_type",
            score_type,
            "--alpha",
            str(alpha),
            "--generator_mode",
            generator_mode,
            "--profile_temperature",
            str(common["profile_temperature"]),
            "--nsamples",
            str(common["nsamples"]),
            "--seqlen",
            str(common["seqlen"]),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--save_dir",
            output_dir,
            "--eval_zeroshot",
            bool_text(eval_zeroshot),
            "--zeroshot_tasks",
            common["zeroshot_tasks"],
            "--debug_schedule",
            bool_text(debug_schedule),
        ]
    )
    if common["profile_beta"] is not None:
        command.extend(["--profile_beta", str(common["profile_beta"])])
    metadata = {"score_type": score_type, "global_sparsity": float(global_sparsity), "alpha": float(alpha), "seed": int(seed), "generator_mode": generator_mode}
    metadata.update(extra_metadata or {})
    task_id_parts = [slugify(group_name), float_tag(global_sparsity), float_tag(alpha), str(seed), slugify(score_type), slugify(generator_mode)]
    if metadata.get("appendix_name"):
        task_id_parts.append(slugify(metadata["appendix_name"]))
    return TaskSpec(
        id="_".join(task_id_parts),
        group=group_name,
        stage=group_name,
        name=os.path.basename(output_dir),
        command=command,
        output_dir=output_dir,
        expected_files=["schedule.csv", "eval_results.json", "one_line_result.csv"],
        metadata=metadata,
        exclusive_gpu=bool(exclusive_gpu),
    )


def build_mechanism_task(common, seed, probe_sparsity, output_dir):
    command = [common["python"], "analyze_dependency_propagation.py"]
    command.extend(base_args(common))
    command.extend(
        [
            "--nsamples",
            str(common["mechanism_nsamples"]),
            "--seqlen",
            str(common["mechanism_seqlen"]),
            "--probe_sparsity",
            str(probe_sparsity),
            "--device",
            "cuda:0",
            "--save_dir",
            output_dir,
            "--seed",
            str(seed),
        ]
    )
    return TaskSpec(
        id=f"mechanism_{seed}_{float_tag(probe_sparsity)}",
        group="mechanism",
        stage="mechanism",
        name=os.path.basename(output_dir),
        command=command,
        output_dir=output_dir,
        expected_files=["summary.json", "layer_probe_metrics.csv", "propagation_matrix.csv"],
        metadata={"seed": int(seed), "probe_sparsity": float(probe_sparsity)},
    )


def build_baseline_task(common, method, global_sparsity, lambda_prop, seed, output_dir, exclusive_gpu=False):
    command = [common["python"], "run_hybrid_prop_correction.py"]
    command.extend(base_args(common))
    command.extend(
        [
            "--probe_metrics_csv",
            common["probe_metrics_csv"],
            "--method",
            method,
            "--global_sparsity",
            str(global_sparsity),
            "--lambda_prop",
            str(lambda_prop),
            "--seed",
            str(seed),
            "--nsamples",
            str(common["nsamples"]),
            "--seqlen",
            str(common["seqlen"]),
            "--stream_batch_size",
            str(common["stream_batch_size"]),
            "--device",
            "cuda:0",
            "--save_dir",
            output_dir,
            "--phase_label",
            common["phase_label"],
        ]
    )
    return TaskSpec(
        id=f"baseline_{slugify(method)}_{float_tag(global_sparsity)}_{float_tag(lambda_prop)}_{seed}",
        group="baselines",
        stage="baselines",
        name=os.path.basename(output_dir),
        command=command,
        output_dir=output_dir,
        expected_files=["compare_one_line.csv", "summary.json", "schedule_base.csv", "schedule_hybrid.csv"],
        metadata={"method": method, "global_sparsity": float(global_sparsity), "lambda_prop": float(lambda_prop), "seed": int(seed)},
        exclusive_gpu=bool(exclusive_gpu),
    )


def build_schedule_eval_task(common, name, schedule_csv, schedule_source, seed, output_dir, depends_on=None, extra_metadata=None, exclusive_gpu=False):
    command = [common["python"], "run_schedule_file_eval.py"]
    command.extend(base_args(common))
    command.extend(
        [
            "--schedule_csv",
            schedule_csv,
            "--schedule_label",
            name,
            "--schedule_source",
            schedule_source,
            "--nsamples",
            str(common["nsamples"]),
            "--seqlen",
            str(common["seqlen"]),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--stream_batch_size",
            str(common["stream_batch_size"]),
            "--save_dir",
            output_dir,
            "--eval_zeroshot",
            "true",
            "--zeroshot_tasks",
            common["zeroshot_tasks"],
        ]
    )
    metadata = {"schedule_label": name, "schedule_source": schedule_source, "seed": int(seed), "schedule_csv": schedule_csv}
    metadata.update(extra_metadata or {})
    return TaskSpec(
        id=f"accuracy_schedule_{slugify(name)}_{seed}",
        group="accuracy",
        stage="accuracy_schedule_eval",
        name=os.path.basename(output_dir),
        command=command,
        output_dir=output_dir,
        expected_files=["eval_results.json", "one_line_result.csv", "schedule_resolved.csv"],
        metadata=metadata,
        exclusive_gpu=bool(exclusive_gpu),
        depends_on=list(depends_on or []),
    )


def build_tasks(config, repo_root, selected_groups=None):
    common = make_common_config(config, repo_root)
    groups = config.get("groups", {})
    selected = None if not selected_groups else set(selected_groups)
    tasks = []
    reports = {"baseline_availability": discover_baseline_adapters(repo_root), "skipped_accuracy_items": []}
    baseline_index = {}

    def enabled(name):
        group = groups.get(name, {})
        return bool(group.get("enabled", False)) and (selected is None or name in selected)

    if enabled("mechanism"):
        group = groups["mechanism"]
        for seed in group.get("seeds", [0]):
            for probe_sparsity in group.get("probe_sparsities", [common["probe_sparsity"]]):
                output_dir = os.path.join(common["results_root"], "mechanism", f"probe_{float_tag(probe_sparsity)}", f"seed_{seed}")
                tasks.append(build_mechanism_task(common, seed, probe_sparsity, output_dir))

    if enabled("probe_to_budget"):
        group = merge_dicts({"generator_mode": "legacy", "eval_zeroshot": False, "debug_schedule": False}, groups["probe_to_budget"])
        for global_sparsity in group.get("global_sparsities", [0.7]):
            for alpha in group.get("alphas", [0.1]):
                for seed in group.get("seeds", [0]):
                    for score_type in group.get("score_types", ["uniform"]):
                        output_dir = os.path.join(common["results_root"], "probe_to_budget", f"gs_{float_tag(global_sparsity)}", f"alpha_{float_tag(alpha)}", f"seed_{seed}", score_type, group["generator_mode"])
                        tasks.append(build_budget_task(common, "probe_to_budget", score_type, global_sparsity, alpha, seed, group["generator_mode"], output_dir, group["eval_zeroshot"], group["debug_schedule"], exclusive_gpu=group.get("exclusive_gpu", False)))

    for group_name, defaults in [
        ("submission_gate", {"generator_mode": "rank_logistic", "eval_zeroshot": False, "debug_schedule": False}),
        ("controls", {"generator_mode": "rank_logistic", "eval_zeroshot": False, "debug_schedule": False}),
        ("generator_ablation", {"global_sparsities": [0.7], "eval_zeroshot": False, "debug_schedule": True}),
        ("alpha_sweep", {"generator_mode": "rank_logistic", "score_types": ["downstream_area"], "eval_zeroshot": False, "debug_schedule": False}),
    ]:
        if not enabled(group_name):
            continue
        group = merge_dicts(defaults, groups[group_name])
        generator_modes = group.get("generator_modes", [group.get("generator_mode", "rank_logistic")])
        for generator_mode in generator_modes:
            for global_sparsity in group.get("global_sparsities", [0.7]):
                for seed in group.get("seeds", [0]):
                    for score_type in group.get("score_types", ["downstream_area"]):
                        alphas = group.get("alphas") or [None]
                        for alpha in alphas:
                            alpha_payload = score_alpha_map(score_type, alpha=alpha, alpha_map=group.get("alpha_map"))
                            output_dir_parts = [common["results_root"], group_name, f"gs_{float_tag(global_sparsity)}"]
                            if alpha is not None:
                                output_dir_parts.append(f"alpha_{float_tag(alpha)}")
                            output_dir_parts.extend([f"seed_{seed}", score_type, generator_mode])
                            output_dir = os.path.join(*output_dir_parts)
                            tasks.append(build_submission_task(common, group_name, group_name, score_type, global_sparsity, seed, generator_mode, alpha_payload, output_dir, group["eval_zeroshot"], group["debug_schedule"], {"alpha": alpha_payload[score_type]}, group.get("exclusive_gpu", False)))

    if enabled("baselines"):
        group = groups["baselines"]
        for method in group.get("methods", ["owl", "dlp"]):
            if not reports["baseline_availability"].get(method, {}).get("available"):
                continue
            for global_sparsity in group.get("global_sparsities", [0.7]):
                for lambda_prop in group.get("lambda_props", [0.1]):
                    for seed in group.get("seeds", [0]):
                        output_dir = os.path.join(common["results_root"], "baselines", method, f"gs_{float_tag(global_sparsity)}", f"lambda_{float_tag(lambda_prop)}", f"seed_{seed}")
                        task = build_baseline_task(common, method, global_sparsity, lambda_prop, seed, output_dir, group.get("exclusive_gpu", False))
                        tasks.append(task)
                        baseline_index[(method, float(global_sparsity), float(lambda_prop), int(seed))] = task

    if enabled("accuracy"):
        group = groups["accuracy"]
        for run_cfg in group.get("submission_gate_runs", []):
            merged = merge_dicts({"generator_mode": "rank_logistic", "score_type": "downstream_area", "global_sparsity": 0.7}, run_cfg)
            for seed in merged.get("seeds", group.get("seeds", [0])):
                output_dir = os.path.join(common["results_root"], "accuracy", slugify(merged["name"]), f"seed_{seed}")
                alpha_payload = score_alpha_map(merged["score_type"], alpha=merged.get("alpha"), alpha_map=merged.get("alpha_map"))
                tasks.append(build_submission_task(common, "accuracy", "accuracy_submission_gate", merged["score_type"], merged["global_sparsity"], seed, merged["generator_mode"], alpha_payload, output_dir, True, merged.get("debug_schedule", False), {"label": merged["name"], "alpha": alpha_payload[merged["score_type"]]}, merged.get("exclusive_gpu", group.get("exclusive_gpu", False))))
        for run_cfg in group.get("schedule_eval_runs", []):
            merged = merge_dicts({"variant": "base", "global_sparsity": 0.7, "lambda_prop": 0.1}, run_cfg)
            for seed in merged.get("seeds", group.get("seeds", [0])):
                schedule_filename = {"base": "schedule_base.csv", "hybrid": "schedule_hybrid.csv", "shuffled": "schedule_shuffled.csv"}[merged["variant"]]
                baseline_task = baseline_index.get((merged["method"], float(merged["global_sparsity"]), float(merged["lambda_prop"]), int(seed)))
                schedule_csv = os.path.join(common["results_root"], "baselines", merged["method"], f"gs_{float_tag(merged['global_sparsity'])}", f"lambda_{float_tag(merged['lambda_prop'])}", f"seed_{seed}", schedule_filename)
                if baseline_task is None and not os.path.exists(schedule_csv):
                    reports["skipped_accuracy_items"].append({"name": merged["name"], "seed": int(seed), "reason": f"missing baseline schedule {schedule_csv}"})
                    continue
                output_dir = os.path.join(common["results_root"], "accuracy", slugify(merged["name"]), f"seed_{seed}")
                depends_on = [baseline_task.id] if baseline_task is not None else []
                tasks.append(build_schedule_eval_task(common, merged["name"], schedule_csv, f"{merged['method']}_{merged['variant']}", seed, output_dir, depends_on, {"method": merged["method"], "variant": merged["variant"], "global_sparsity": float(merged["global_sparsity"]), "lambda_prop": float(merged["lambda_prop"])}, merged.get("exclusive_gpu", group.get("exclusive_gpu", False))))

    if enabled("appendix"):
        group = groups["appendix"]
        for run_cfg in group.get("submission_gate_runs", []):
            merged = merge_dicts({"generator_mode": "rank_logistic", "eval_zeroshot": False, "debug_schedule": False}, run_cfg)
            for global_sparsity in merged.get("global_sparsities", [0.7]):
                for seed in merged.get("seeds", [0]):
                    for score_type in merged.get("score_types", ["downstream_area"]):
                        alphas = merged.get("alphas") or [merged.get("alpha")]
                        for alpha in alphas:
                            alpha_payload = score_alpha_map(score_type, alpha=alpha, alpha_map=merged.get("alpha_map"))
                            output_dir = os.path.join(common["results_root"], "appendix", slugify(merged.get("name", "submission_gate")), f"gs_{float_tag(global_sparsity)}", f"alpha_{float_tag(alpha_payload[score_type])}", f"seed_{seed}", score_type, merged["generator_mode"])
                            tasks.append(build_submission_task(common, "appendix", slugify(merged.get("name", "appendix_submission_gate")), score_type, global_sparsity, seed, merged["generator_mode"], alpha_payload, output_dir, merged["eval_zeroshot"], merged["debug_schedule"], {"appendix_name": merged.get("name", "submission_gate"), "alpha": alpha_payload[score_type]}, merged.get("exclusive_gpu", group.get("exclusive_gpu", False))))
        for run_cfg in group.get("budget_runs", []):
            merged = merge_dicts({"generator_mode": "rank_logistic", "eval_zeroshot": False, "debug_schedule": False}, run_cfg)
            for global_sparsity in merged.get("global_sparsities", [0.7]):
                for alpha in merged.get("alphas", [0.1]):
                    for seed in merged.get("seeds", [0]):
                        for score_type in merged.get("score_types", ["downstream_area"]):
                            output_dir = os.path.join(common["results_root"], "appendix", slugify(merged.get("name", "budget")), f"gs_{float_tag(global_sparsity)}", f"alpha_{float_tag(alpha)}", f"seed_{seed}", score_type, merged["generator_mode"])
                            tasks.append(build_budget_task(common, "appendix", score_type, global_sparsity, alpha, seed, merged["generator_mode"], output_dir, merged["eval_zeroshot"], merged["debug_schedule"], {"appendix_name": merged.get("name", "budget")}, merged.get("exclusive_gpu", group.get("exclusive_gpu", False))))

    return tasks, common, reports

import argparse
import errno
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

from paper_experiment_utils import (
    append_jsonl,
    command_to_text,
    detect_gpu_devices,
    ensure_dir,
    load_config,
    load_status_index,
    manifest_path,
    status_csv_path,
    status_jsonl_path,
    task_is_complete,
    write_csv,
    write_json,
    write_task_manifest,
)
from paper_task_factory import build_tasks


STATUS_FIELDS = [
    "task_id",
    "group",
    "stage",
    "name",
    "status",
    "assigned_device",
    "exit_code",
    "start_time",
    "end_time",
    "duration_sec",
    "output_dir",
    "log_path",
    "command",
    "depends_on",
    "metadata_json",
    "skip_reason",
]
SUCCESS_STATES = {"success", "skipped_completed"}
TERMINAL_STATES = SUCCESS_STATES | {"failed", "failed_missing_outputs", "blocked_dependency", "deadlock"}
FAILED_STATES = {"failed", "failed_missing_outputs"}
OOM_PATTERNS = (
    "cuda out of memory",
    "out of memory",
    "cuda oom",
    "tried to allocate",
    "cublas_status_alloc_failed",
)


def iso_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--list_tasks", action="store_true")
    parser.add_argument("--status_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rerun_failed", action="store_true")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_parallel_gpus", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--skip_postprocess", action="store_true")
    return parser.parse_args()


def parse_group_filters(values):
    groups = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                groups.append(item)
    return groups


def parse_gpu_override(value):
    if value is None:
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def read_log_tail(log_path, max_bytes=262144):
    if not log_path or not os.path.exists(log_path):
        return ""
    with open(log_path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="ignore").lower()


def log_indicates_oom(log_path):
    text = read_log_tail(log_path)
    return bool(text) and any(pattern in text for pattern in OOM_PATTERNS)


def requested_gpu_count(task, gpu_devices, oom_retry_task_ids, oom_retry_gpu_count):
    if task.exclusive_gpu:
        return len(gpu_devices)
    count = 1
    if task.id in oom_retry_task_ids:
        count = max(count, oom_retry_gpu_count)
    return max(1, min(len(gpu_devices), count))


def allocated_devices_in_use(running):
    devices = []
    for bundle in running.values():
        devices.extend(bundle.get("allocated_devices", []))
    return devices


def status_row(task, status, assigned_device, exit_code, start_time, end_time, log_path, skip_reason=""):
    duration_sec = None
    if start_time and end_time:
        duration_sec = round(max(0.0, end_time - start_time), 3)
    return {
        "task_id": task.id,
        "group": task.group,
        "stage": task.stage,
        "name": task.name,
        "status": status,
        "assigned_device": assigned_device,
        "exit_code": exit_code,
        "start_time": None if start_time is None else datetime.fromtimestamp(start_time, tz=timezone.utc).replace(microsecond=0).isoformat(),
        "end_time": None if end_time is None else datetime.fromtimestamp(end_time, tz=timezone.utc).replace(microsecond=0).isoformat(),
        "duration_sec": duration_sec,
        "output_dir": task.output_dir,
        "log_path": log_path,
        "command": command_to_text(task.command),
        "depends_on": ",".join(task.depends_on),
        "metadata_json": __import__("json").dumps(task.metadata, sort_keys=True),
        "skip_reason": skip_reason,
    }


def persist_status(results_root, row, status_rows):
    jsonl_path = status_jsonl_path(results_root)
    csv_path = status_csv_path(results_root)
    try:
        append_jsonl(jsonl_path, row)
        status_rows.append(row)
        write_csv(status_rows, STATUS_FIELDS, csv_path)
    except OSError as error:
        if getattr(error, "errno", None) == errno.ENOSPC:
            _, _, free_bytes = shutil.disk_usage(results_root)
            raise RuntimeError(
                f"no space left on device while writing orchestrator status under {results_root}; "
                f"jsonl_path={jsonl_path}; csv_path={csv_path}; free_bytes={free_bytes}. "
                f"Free space on that filesystem or move results_root to a larger mount such as /2T, then rerun."
            ) from error
        raise


def make_log_path(results_root, task):
    return os.path.join(results_root, "_orchestrator", "logs", task.group, f"{task.id}.log")


def dependency_state(task, terminal_index):
    if not task.depends_on:
        return "ready"
    seen = []
    for dependency_id in task.depends_on:
        state = terminal_index.get(dependency_id)
        if state is None:
            return "waiting"
        seen.append(state)
    if any(state not in SUCCESS_STATES for state in seen):
        return "blocked"
    return "ready"


def start_task(task, repo_root, results_root, assigned_device):
    ensure_dir(task.output_dir)
    log_path = make_log_path(results_root, task)
    ensure_dir(os.path.dirname(log_path))
    log_handle = open(log_path, "a")
    env = os.environ.copy()
    env.update(task.env)
    env["PYTHONUNBUFFERED"] = "1"
    allocated_devices = []
    if assigned_device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = assigned_device
        allocated_devices = [item.strip() for item in assigned_device.split(",") if item.strip()]
    log_handle.write(f"[{iso_now()}] START {task.id} assigned_device={assigned_device}\n")
    log_handle.write(command_to_text(task.command) + "\n\n")
    log_handle.flush()
    process = subprocess.Popen(
        task.command,
        cwd=repo_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return {
        "task": task,
        "process": process,
        "log_handle": log_handle,
        "log_path": log_path,
        "start_time": time.time(),
        "assigned_device": assigned_device,
        "allocated_devices": allocated_devices,
    }


def print_task_preview(tasks):
    print(f"expanded {len(tasks)} tasks")
    for task in tasks:
        print(f"[{task.group}] {task.id} -> {task.output_dir}")
        print(f"  depends_on={task.depends_on}")
        print(f"  command={command_to_text(task.command)}")


def run_postprocess_step(repo_root, common, command, log_path):
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w") as log_handle:
        log_handle.write(command_to_text(command) + "\n\n")
        log_handle.flush()
        started_at = time.time()
        process = subprocess.run(command, cwd=repo_root, stdout=log_handle, stderr=subprocess.STDOUT)
    return {
        "command": command,
        "log_path": log_path,
        "exit_code": int(process.returncode),
        "duration_sec": round(max(0.0, time.time() - started_at), 3),
        "paper_root": common["paper_root"],
        "results_root": common["results_root"],
    }


def run_postprocess(repo_root, args, common):
    summary = {"enabled": True, "steps": []}
    post_dir = os.path.join(common["results_root"], "_orchestrator", "postprocess")
    aggregate_script = os.path.join(repo_root, "aggregate_paper_results.py")
    figures_script = os.path.join(repo_root, "make_paper_figures.py")
    if os.path.exists(aggregate_script):
        aggregate_command = [
            common["python"],
            aggregate_script,
            "--config",
            args.config,
            "--results_root",
            common["results_root"],
            "--paper_root",
            common["paper_root"],
        ]
        aggregate_result = run_postprocess_step(repo_root, common, aggregate_command, os.path.join(post_dir, "aggregate.log"))
        aggregate_result["name"] = "aggregate_paper_results"
        summary["steps"].append(aggregate_result)
    if os.path.exists(figures_script):
        last_exit = summary["steps"][-1]["exit_code"] if summary["steps"] else 0
        if last_exit == 0:
            figure_command = [common["python"], figures_script, "--paper_root", common["paper_root"]]
            figure_result = run_postprocess_step(repo_root, common, figure_command, os.path.join(post_dir, "figures.log"))
            figure_result["name"] = "make_paper_figures"
            summary["steps"].append(figure_result)
        else:
            summary["steps"].append(
                {
                    "name": "make_paper_figures",
                    "command": [common["python"], figures_script, "--paper_root", common["paper_root"]],
                    "log_path": os.path.join(post_dir, "figures.log"),
                    "exit_code": None,
                    "duration_sec": 0.0,
                    "skipped_reason": "aggregate_paper_results failed",
                    "paper_root": common["paper_root"],
                    "results_root": common["results_root"],
                }
            )
    summary["all_success"] = all(step.get("exit_code") in {0, None} for step in summary["steps"])
    write_json(summary, os.path.join(post_dir, "postprocess_summary.json"))
    return summary


def next_ready_gpu_task(pending, terminal_index, running, free_gpus, gpu_devices, oom_retry_task_ids, oom_retry_gpu_count):
    if any(bundle["task"].exclusive_gpu for bundle in running.values()):
        return None, None
    for task in pending:
        if not task.gpu_required:
            continue
        if dependency_state(task, terminal_index) != "ready":
            continue
        needed = requested_gpu_count(task, gpu_devices, oom_retry_task_ids, oom_retry_gpu_count)
        if task.exclusive_gpu and running:
            continue
        if needed > len(free_gpus):
            continue
        return task, needed
    return None, None


def gpu_launch_assignment(task, gpu_devices, free_gpus, needed):
    if task.exclusive_gpu:
        allocated_devices = list(gpu_devices)
    else:
        allocated_devices = list(free_gpus[:needed])
    assigned_device = ",".join(allocated_devices)
    slot_key = "exclusive_gpu" if task.exclusive_gpu else f"gpu_bundle_{'_'.join(allocated_devices)}"
    return slot_key, assigned_device, allocated_devices


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    config = load_config(args.config)
    selected_groups = parse_group_filters(args.group)
    tasks, common, reports = build_tasks(config, repo_root, selected_groups)
    if args.max_tasks is not None:
        tasks = tasks[: max(0, int(args.max_tasks))]

    results_root = common["results_root"]
    ensure_dir(os.path.join(results_root, "_orchestrator"))
    write_task_manifest(tasks, manifest_path(results_root))
    write_json({"config_path": args.config, "selected_groups": selected_groups, "task_count": len(tasks), "reports": reports, "common": common}, os.path.join(results_root, "_orchestrator", "resolved_plan.json"))
    write_json(reports.get("baseline_availability", {}), os.path.join(results_root, "_orchestrator", "baseline_availability.json"))

    if args.list_tasks or args.dry_run:
        print_task_preview(tasks)
        return

    previous_index = load_status_index(status_jsonl_path(results_root))
    if args.status_only:
        completed = sum(1 for task in tasks if task_is_complete(task))
        failed_before = sum(1 for task in tasks if previous_index.get(task.id, {}).get("status") in {"failed", "failed_missing_outputs"})
        print(f"tasks={len(tasks)} completed_outputs={completed} previous_failures={failed_before}")
        return

    runtime = config.get("runtime", {})
    gpu_devices = detect_gpu_devices(parse_gpu_override(args.gpus) or runtime.get("gpu_devices"))
    max_parallel = args.max_parallel_gpus or int(runtime.get("max_parallel_gpu_tasks", len(gpu_devices)))
    if max_parallel < 1:
        raise ValueError("max_parallel_gpus must be >= 1")
    gpu_devices = gpu_devices[:max_parallel]
    oom_retry_gpu_count = max(1, min(len(gpu_devices), int(runtime.get("oom_retry_gpu_count", 1))))
    retry_oom_once = bool(runtime.get("retry_oom_once", False))
    print(f"using gpu slots: {gpu_devices}")
    if oom_retry_gpu_count > 1:
        print(f"oom retries will request {oom_retry_gpu_count} gpus per task")

    status_rows = []
    terminal_index = {}
    pending = []
    oom_retry_task_ids = set()
    if oom_retry_gpu_count > 1:
        for task in tasks:
            previous_row = previous_index.get(task.id, {})
            if previous_row.get("status") not in FAILED_STATES:
                continue
            previous_log_path = previous_row.get("log_path") or make_log_path(results_root, task)
            if log_indicates_oom(previous_log_path):
                oom_retry_task_ids.add(task.id)
        if oom_retry_task_ids:
            print(f"detected {len(oom_retry_task_ids)} prior oom failures; rerunning them with {oom_retry_gpu_count} gpus")
    for task in tasks:
        previous_status = previous_index.get(task.id, {}).get("status")
        complete = task_is_complete(task)
        if not args.force and complete and not (args.rerun_failed and previous_status in FAILED_STATES):
            row = status_row(task, "skipped_completed", "n/a", None, None, None, make_log_path(results_root, task), "expected files already exist")
            persist_status(results_root, row, status_rows)
            terminal_index[task.id] = row["status"]
        else:
            pending.append(task)

    running = {}
    cpu_running = None

    while pending or running or cpu_running is not None:
        progressed = False
        finished_slots = []
        for slot_key, bundle in list(running.items()):
            return_code = bundle["process"].poll()
            if return_code is None:
                continue
            bundle["log_handle"].write(f"\n[{iso_now()}] END {bundle['task'].id} exit_code={return_code}\n")
            bundle["log_handle"].close()
            end_time = time.time()
            if return_code == 0 and task_is_complete(bundle["task"]):
                state = "success"
            elif return_code == 0:
                state = "failed_missing_outputs"
            else:
                state = "failed"
            if (
                state == "failed"
                and retry_oom_once
                and oom_retry_gpu_count > 1
                and not bundle["task"].exclusive_gpu
                and bundle["task"].id not in oom_retry_task_ids
                and log_indicates_oom(bundle["log_path"])
            ):
                oom_retry_task_ids.add(bundle["task"].id)
                pending.insert(0, bundle["task"])
                finished_slots.append(slot_key)
                progressed = True
                print(
                    f"requeueing {bundle['task'].id} after oom on gpu {bundle['assigned_device']}; "
                    f"retrying with {oom_retry_gpu_count} gpus"
                )
                continue
            row = status_row(bundle["task"], state, bundle["assigned_device"], return_code, bundle["start_time"], end_time, bundle["log_path"])
            persist_status(results_root, row, status_rows)
            terminal_index[bundle["task"].id] = state
            finished_slots.append(slot_key)
        for slot_key in finished_slots:
            running.pop(slot_key, None)

        if cpu_running is not None:
            return_code = cpu_running["process"].poll()
            if return_code is not None:
                cpu_running["log_handle"].write(f"\n[{iso_now()}] END {cpu_running['task'].id} exit_code={return_code}\n")
                cpu_running["log_handle"].close()
                end_time = time.time()
                if return_code == 0 and task_is_complete(cpu_running["task"]):
                    state = "success"
                elif return_code == 0:
                    state = "failed_missing_outputs"
                else:
                    state = "failed"
                row = status_row(cpu_running["task"], state, "cpu", return_code, cpu_running["start_time"], end_time, cpu_running["log_path"])
                persist_status(results_root, row, status_rows)
                terminal_index[cpu_running["task"].id] = state
                cpu_running = None

        blocked_tasks = []
        still_pending = []
        for task in pending:
            dep_state = dependency_state(task, terminal_index)
            if dep_state == "blocked":
                blocked_tasks.append(task)
            else:
                still_pending.append(task)
        pending = still_pending
        for task in blocked_tasks:
            row = status_row(task, "blocked_dependency", "n/a", None, None, None, make_log_path(results_root, task), "at least one dependency failed")
            persist_status(results_root, row, status_rows)
            terminal_index[task.id] = row["status"]
            progressed = True

        free_gpus = [device for device in gpu_devices if device not in allocated_devices_in_use(running)]
        while free_gpus:
            task, needed = next_ready_gpu_task(
                pending,
                terminal_index,
                running,
                free_gpus,
                gpu_devices,
                oom_retry_task_ids,
                oom_retry_gpu_count,
            )
            if task is None:
                break
            if task not in pending:
                continue
            pending.remove(task)
            slot_key, assigned_device, allocated_devices = gpu_launch_assignment(task, gpu_devices, free_gpus, needed)
            free_gpus = [device for device in free_gpus if device not in allocated_devices]
            running[slot_key] = start_task(task, repo_root, results_root, assigned_device)
            progressed = True
            print(f"started {task.id} on gpu {assigned_device}")
            if task.exclusive_gpu:
                break

        if cpu_running is None:
            ready_cpu = [task for task in pending if not task.gpu_required and dependency_state(task, terminal_index) == "ready"]
            if ready_cpu:
                task = ready_cpu[0]
                pending.remove(task)
                cpu_running = start_task(task, repo_root, results_root, "cpu")
                progressed = True
                print(f"started {task.id} on cpu")

        if pending and not running and cpu_running is None and not progressed:
            for task in pending:
                row = status_row(task, "deadlock", "n/a", None, None, None, make_log_path(results_root, task), "no runnable task remained")
                persist_status(results_root, row, status_rows)
                terminal_index[task.id] = row["status"]
            pending = []
            break

        if pending or running or cpu_running is not None:
            time.sleep(1.0)

    counts = {}
    for row in status_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    write_json({"status_counts": counts, "task_count": len(tasks), "generated_at": iso_now()}, os.path.join(results_root, "_orchestrator", "run_summary.json"))
    print("run summary:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")

    if not args.skip_postprocess:
        postprocess_summary = run_postprocess(repo_root, args, common)
        print("postprocess summary:")
        for step in postprocess_summary.get("steps", []):
            status = step.get("exit_code")
            if status is None:
                print(f"  {step['name']}: skipped ({step.get('skipped_reason', 'n/a')})")
            else:
                print(f"  {step['name']}: exit_code={status} log={step['log_path']}")


if __name__ == "__main__":
    main()

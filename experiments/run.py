"""Record development iterations and execute frozen programs in separate processes."""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rollout import new_rollout_id


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def source_files(source, *, exclude=None):
    """Yield the code and text configuration included in a source snapshot."""
    excluded = {"data", "__pycache__", "build", "dist", "runs", "node_modules",
                "venv", "assets", "paper", "integrations"}
    extensions = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".sh"}
    for directory, folders, files in os.walk(source):
        folders[:] = sorted(name for name in folders
                            if name not in excluded and not name.startswith(".")
                            and not name.endswith(".egg-info")
                            and not (Path(directory) / name).is_symlink()
                            and Path(directory) / name != exclude)
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix in extensions or name == ".gitignore":
                yield path


def snapshot(source, destination):
    """Copy project code and text configuration, excluding datasets and environments."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True)
    for path in source_files(source, exclude=destination):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def source_hashes(source):
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files(source)}


def create_experiment(task, agent, model, criteria, *, environment, root=ROOT):
    """Define one task and agent session; model should include its version."""
    root = Path(root).resolve()
    directory = root / "experiments" / "data" / new_rollout_id()
    directory.mkdir(parents=True)
    write_json(directory / "experiment.json", {
        "task": task, "agent": agent, "model": model, "criteria": criteria,
        "environment": environment, "created_at": timestamp(), "project_root": str(root),
    })
    return directory


def start_iteration(experiment, prompt, feedback="", hypothesis=""):
    """Capture the starting code and the actual input supplied to the agent."""
    experiment = Path(experiment).resolve()
    metadata = json.loads((experiment / "experiment.json").read_text())
    previous = sorted((experiment / "iterations").glob("*/iteration.json"))
    directory = experiment / "iterations" / new_rollout_id()
    directory.mkdir(parents=True)
    write_json(directory / "iteration.json", {
        "status": "developing", "started_at": timestamp(),
        "previous_iteration": previous[-1].parent.name if previous else None,
        "hypothesis": hypothesis,
    })
    (directory / "prompt.md").write_text(prompt)
    (directory / "feedback.md").write_text(feedback)
    snapshot(metadata["project_root"], directory / "before")
    return directory


def finish_iteration(iteration, response, *, usage=None, traces=()):
    """Freeze a candidate, its agent response, measured costs, and source diff."""
    iteration = Path(iteration).resolve()
    experiment = json.loads((iteration.parents[1] / "experiment.json").read_text())
    metadata = json.loads((iteration / "iteration.json").read_text())
    before, after = iteration / "before", iteration / "source"
    snapshot(experiment["project_root"], after)
    names = sorted({p.relative_to(base) for base in (before, after)
                    for p in base.rglob("*") if p.is_file()})
    changes = []
    for name in names:
        old, new = before / name, after / name
        changes.extend(difflib.unified_diff(
            old.read_text().splitlines(keepends=True) if old.exists() else [],
            new.read_text().splitlines(keepends=True) if new.exists() else [],
            fromfile=f"before/{name}", tofile=f"source/{name}",
        ))
    (iteration / "changes.diff").write_text("".join(changes))
    (iteration / "response.md").write_text(response)
    trace_names = []
    for index, path in enumerate(traces):
        path = Path(path)
        name = f"trace_{index:03d}_{path.name}"
        shutil.copy2(path, iteration / name)
        trace_names.append(name)
    write_json(iteration / "source_sha256.json", source_hashes(after))
    metadata.update(status="frozen", finished_at=timestamp(), usage=usage or {}, traces=trace_names)
    write_json(iteration / "iteration.json", metadata)
    return iteration


def run_trial(iteration, command):
    """Run one complete workflow from its frozen snapshot and retain its exit status.

    main.run() uses CAP_ROLLOUT_ID/CAP_DATA_ROOT to associate phase logs with
    this trial. Physical outcomes are recorded separately by evaluate.py.
    """
    iteration = Path(iteration).resolve()
    experiment = json.loads((iteration.parents[1] / "experiment.json").read_text())
    trial_id = new_rollout_id()
    directory = iteration / "trials" / trial_id
    directory.mkdir(parents=True)
    data_root = Path(experiment["project_root"])
    execution = {
        "trial_id": trial_id, "environment": experiment["environment"],
        "command": list(command), "cwd": str(iteration / "source"),
        "started_at": timestamp(), "status": "running", "returncode": None,
        "rollouts": {phase: str(data_root / phase / "data" / "rollouts" / trial_id)
                     for phase in ("pick", "place")},
    }
    write_json(directory / "execution.json", execution)
    env = dict(os.environ, CAP_ROLLOUT_ID=trial_id, CAP_DATA_ROOT=str(data_root))
    started = time.monotonic()
    try:
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            source = iteration / "source"
            manifest = (iteration / "source_sha256.json").read_bytes()
            expected = json.loads(manifest)
            actual = source_hashes(source)
            if actual != expected:
                changed = sorted(name for name in actual.keys() | expected.keys()
                                 if actual.get(name) != expected.get(name))
                raise ValueError(f"Frozen source differs from source_sha256.json: {', '.join(changed)}")
            execution["source_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
            result = subprocess.run(command, cwd=source, env=env,
                                    stdout=stdout, stderr=stderr)
        execution.update(returncode=result.returncode,
                         status="completed" if result.returncode == 0 else "failed")
    except BaseException as error:
        execution.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        execution.update(finished_at=timestamp(), elapsed_s=time.monotonic() - started)
        write_json(directory / "execution.json", execution)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("create")
    for name in ("task", "agent", "model", "criteria"):
        create.add_argument(f"--{name}", required=True)
    create.add_argument("--environment", choices=("hardware", "simulation", "kinematic_demo"), required=True)
    start = commands.add_parser("start")
    start.add_argument("experiment", type=Path)
    start.add_argument("--prompt", type=Path, required=True)
    start.add_argument("--feedback", type=Path)
    start.add_argument("--hypothesis", default="")
    finish = commands.add_parser("finish")
    finish.add_argument("iteration", type=Path)
    finish.add_argument("--response", type=Path, required=True)
    finish.add_argument("--usage", type=Path)
    finish.add_argument("--trace", type=Path, action="append", default=[])
    trial = commands.add_parser("trial")
    trial.add_argument("iteration", type=Path)
    trial.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "create":
        directory = create_experiment(args.task, args.agent, args.model, args.criteria,
                                      environment=args.environment)
    elif args.action == "start":
        directory = start_iteration(args.experiment, args.prompt.read_text(),
                                    args.feedback.read_text() if args.feedback else "", args.hypothesis)
    elif args.action == "finish":
        directory = finish_iteration(args.iteration, args.response.read_text(),
                                     usage=json.loads(args.usage.read_text()) if args.usage else None,
                                     traces=args.trace)
    else:
        directory = run_trial(args.iteration, args.command)
    print(directory)
    if args.action == "trial":
        execution = json.loads((directory / "execution.json").read_text())
        sys.exit(0 if execution["status"] == "completed" else 1)


if __name__ == "__main__":
    main()

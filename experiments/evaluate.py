"""Record physical assessments from an automated check, evaluator agent, or human observer."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.run import timestamp, write_json
from rollout import new_rollout_id


def record_assessment(trial, success: bool | None, *, evaluator, measurements=None, evidence=(), notes=""):
    """Record True/False/None for success/failure/unknown, independently of exit status.

    Evidence files are copied into the assessment directory. Each assessment is
    retained; reports use the latest one. The evaluator supplies the judgment.
    """
    trial = Path(trial).resolve()
    directory = trial / "assessments" / new_rollout_id()
    directory.mkdir(parents=True)
    artifacts = []
    for index, path in enumerate(evidence):
        path = Path(path)
        name = f"{index:03d}_{path.name}"
        shutil.copy2(path, directory / name)
        artifacts.append(name)
    write_json(directory / "physical.json", {
        "success": success, "status": "unknown" if success is None else "assessed",
        "evaluated_at": timestamp(), "evaluator": evaluator,
        "measurements": measurements or {}, "evidence": artifacts, "notes": notes,
    })
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--outcome", choices=("success", "failure", "unknown"), required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    success = {"success": True, "failure": False, "unknown": None}[args.outcome]
    print(record_assessment(args.trial, success, evaluator=args.evaluator,
                            measurements=json.loads(args.measurements.read_text()) if args.measurements else None,
                            evidence=args.evidence, notes=args.notes))


if __name__ == "__main__":
    main()

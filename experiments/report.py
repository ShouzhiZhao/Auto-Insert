"""Summarize execution and physical outcomes without treating missing labels as failures."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(experiment):
    experiment = Path(experiment)
    metadata = json.loads((experiment / "experiment.json").read_text())
    trials = []
    for path in sorted(experiment.glob("iterations/*/trials/*/execution.json")):
        execution = json.loads(path.read_text())
        assessments = sorted(path.parent.glob("assessments/*/physical.json"))
        physical = json.loads(assessments[-1].read_text()) if assessments else {
            "status": "not_evaluated", "success": None,
        }
        trials.append({"iteration": path.parents[2].name, "trial_id": path.parent.name,
                       "execution_status": execution["status"], "physical": physical,
                       "assessment": str(assessments[-1].relative_to(experiment)) if assessments else None})
    counts = Counter(row["execution_status"] for row in trials)
    assessed = sum(row["physical"]["success"] is not None for row in trials)
    successes = sum(row["physical"]["success"] is True for row in trials)
    return {
        "task": metadata["task"], "environment": metadata["environment"],
        "criteria": metadata["criteria"], "trial_count": len(trials),
        "execution_counts": dict(counts), "physically_assessed": assessed,
        "physical_successes": successes, "physical_failures": assessed - successes,
        "not_evaluated": sum(row["physical"]["status"] == "not_evaluated" for row in trials),
        "unknown": sum(row["physical"]["status"] == "unknown" for row in trials),
        "physical_success_rate_on_assessed": successes / assessed if assessed else None,
        "iterations": [{"iteration": path.parent.name, **json.loads(path.read_text())}
                       for path in sorted(experiment.glob("iterations/*/iteration.json"))],
        "trials": trials,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    print(json.dumps(summarize(parser.parse_args().experiment), indent=2))

"""Exercise frozen trials, independent assessments, and failure provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from experiments.evaluate import record_assessment
from experiments.report import summarize
from experiments.run import create_experiment, finish_iteration, run_trial, start_iteration


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("main.py", "rollout.py", "AGENTS.md", "README.md", "pyproject.toml"):
            shutil.copy2(ROOT / name, self.root / name)
        for name in ("pick", "place", "experiments"):
            shutil.copytree(ROOT / name, self.root / name,
                            ignore=shutil.ignore_patterns("data", "__pycache__"))
        self.experiment = create_experiment(
            "test-insertion", "test-agent", "test-model-version", "Independent seating measurement",
            environment="kinematic_demo", root=self.root,
        )

    def freeze(self):
        iteration = start_iteration(self.experiment, "Implement the trial.",
                                    feedback="No earlier trial.", hypothesis="Test the candidate.")
        finish_iteration(iteration, "Candidate ready.")
        return iteration

    def test_iteration_records_and_runs_frozen_source(self):
        entry = self.root / "entry.py"
        entry.write_text('print("before")\n')
        ignored = self.root / "pick" / "data" / "old_source.py"
        ignored.parent.mkdir()
        ignored.write_text("historical data\n")
        iteration = start_iteration(self.experiment, "Change the output.", "Old output observed.")
        entry.write_text('print("candidate")\n')
        helper = self.root / "place" / "features.py"
        helper.write_text('"""Task-specific feature extraction."""\n')
        trace = self.experiment / "session.jsonl"
        trace.write_text('{"tool": "edit", "file": "entry.py"}\n')
        finish_iteration(iteration, "Changed the output.", usage={"input_tokens": 10}, traces=[trace])
        entry.write_text('raise RuntimeError("Live edits must not run")\n')

        trial = run_trial(iteration, [sys.executable, "entry.py"])
        self.assertEqual((trial / "stdout.log").read_text(), "candidate\n")
        self.assertEqual(read_json(trial / "execution.json")["status"], "completed")
        self.assertIn('+print("candidate")', (iteration / "changes.diff").read_text())
        self.assertFalse((iteration / "source" / "pick" / "data").exists())
        self.assertTrue((iteration / "source" / "place" / "features.py").exists())
        digest = hashlib.sha256((iteration / "source" / "entry.py").read_bytes()).hexdigest()
        self.assertEqual(read_json(iteration / "source_sha256.json")["entry.py"], digest)
        self.assertEqual(read_json(iteration / "iteration.json")["usage"], {"input_tokens": 10})
        self.assertEqual((iteration / "trace_000_session.jsonl").read_text(), trace.read_text())
        next_iteration = start_iteration(self.experiment, "Fix the next issue.", str(trial))
        self.assertEqual(read_json(next_iteration / "iteration.json")["previous_iteration"], iteration.name)

    def test_demo_completion_is_not_physical_success(self):
        (self.root / "place" / "features.py").write_text('"""Additional perception module."""\n')
        iteration = self.freeze()
        trial = run_trial(iteration, [sys.executable, "main.py"])
        execution = read_json(trial / "execution.json")
        self.assertEqual(execution["returncode"], 0, (trial / "stderr.log").read_text())
        for phase, location in execution["rollouts"].items():
            rollout = Path(location)
            self.assertEqual(rollout, self.root / phase / "data" / "rollouts" / trial.name)
            self.assertEqual(read_json(rollout / "result.json")["status"], "completed")
            self.assertTrue((self.root / phase / "data" / "calibrations" / f"{trial.name}.json").exists())
        self.assertTrue((Path(execution["rollouts"]["place"]) / "source" / "features.py").exists())
        report = summarize(self.experiment)
        self.assertEqual(report["execution_counts"], {"completed": 1})
        self.assertEqual(report["not_evaluated"], 1)
        self.assertIsNone(report["physical_success_rate_on_assessed"])

        evidence = self.experiment / "inspection.txt"
        evidence.write_text("Synthetic evaluator fixture: object not seated.\n")
        assessment = record_assessment(trial, False, evaluator="test-observer",
                                       measurements={"seating_depth_mm": 0.2}, evidence=[evidence])
        evidence.unlink()
        self.assertTrue((assessment / "000_inspection.txt").exists())
        report = summarize(self.experiment)
        self.assertEqual(report["execution_counts"], {"completed": 1})
        self.assertEqual(report["physical_failures"], 1)
        self.assertEqual(report["physical_success_rate_on_assessed"], 0.0)

        record_assessment(trial, None, evaluator="test-observer", notes="Measurement inconclusive.")
        report = summarize(self.experiment)
        self.assertEqual(report["unknown"], 1)
        self.assertEqual(report["physically_assessed"], 0)
        self.assertIsNone(report["physical_success_rate_on_assessed"])
        self.assertEqual(len(list(trial.glob("assessments/*/physical.json"))), 2)

    def test_trial_rejects_modified_frozen_source_then_runs_the_restored_snapshot(self):
        program = 'print("frozen source")\n'
        (self.root / "entry.py").write_text(program)
        for change in ("edit", "remove", "add"):
            with self.subTest(change=change):
                iteration = self.freeze()
                source = iteration / "source"
                script = source / "entry.py"
                extra = source / "extra.py"
                if change == "edit":
                    script.write_text('raise SystemExit("must not run")\n')
                elif change == "remove":
                    script.unlink()
                else:
                    extra.write_text('raise SystemExit("must not run")\n')

                with self.assertRaisesRegex(ValueError, r"Frozen source differs.*\.py"):
                    run_trial(iteration, [sys.executable, "entry.py"])

                rejected = next(iteration.glob("trials/*/execution.json"))
                result = read_json(rejected)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["returncode"])
                self.assertIn("extra.py" if change == "add" else "entry.py", result["error"]["message"])
                self.assertEqual(rejected.with_name("stderr.log").read_text(), "")
                script.write_text(program)
                extra.unlink(missing_ok=True)
                (source / "__pycache__").mkdir(exist_ok=True)
                (source / "__pycache__" / "local.pyc").touch()
                trial = run_trial(iteration, [sys.executable, "entry.py"])
                self.assertEqual((trial / "stdout.log").read_text(), "frozen source\n")
                manifest = (iteration / "source_sha256.json").read_bytes()
                self.assertEqual(read_json(trial / "execution.json")["source_manifest_sha256"],
                                 hashlib.sha256(manifest).hexdigest())

    def test_failed_workflow_keeps_phase_logs_and_trial_link(self):
        (self.root / "entry.py").write_text(
            'from main import build_demo, run\n'
            'pick, place = build_demo()\n'
            'def fail():\n'
            '    raise RuntimeError("injected gripper failure")\n'
            'pick.close = fail\n'
            'run(pick, place, label="failure-test")\n'
        )
        iteration = self.freeze()
        trial = run_trial(iteration, [sys.executable, "entry.py"])
        execution = read_json(trial / "execution.json")
        self.assertEqual(execution["status"], "failed")
        self.assertNotEqual(execution["returncode"], 0)
        self.assertIn("injected gripper failure", (trial / "stderr.log").read_text())
        self.assertEqual(read_json(Path(execution["rollouts"]["pick"]) / "result.json")["status"], "failed")
        self.assertEqual(read_json(Path(execution["rollouts"]["place"]) / "result.json")["status"], "not_started")
        self.assertEqual(summarize(self.experiment)["not_evaluated"], 1)

    def test_cli_exit_failure_and_independent_positive_assessment(self):
        (self.root / "entry.py").write_text('raise SystemExit(7)\n')
        iteration = self.freeze()
        result = subprocess.run(
            [sys.executable, "-m", "experiments.run", "trial", str(iteration), "--", sys.executable, "entry.py"],
            cwd=self.root, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        trial = Path(result.stdout.strip())
        self.assertEqual(read_json(trial / "execution.json")["returncode"], 7)
        subprocess.run(
            [sys.executable, "-m", "experiments.evaluate", str(trial), "--outcome", "success",
             "--evaluator", "synthetic-test-fixture", "--notes", "Tests independent result fields."],
            cwd=self.root, capture_output=True, text=True, check=True,
        )
        report = summarize(self.experiment)
        self.assertEqual(report["execution_counts"], {"failed": 1})
        self.assertEqual(report["physical_successes"], 1)
        self.assertEqual(report["physical_success_rate_on_assessed"], 1.0)

    def test_readme_cli_lifecycle(self):
        def cli(*arguments):
            return subprocess.check_output([sys.executable, "-m", *arguments],
                                           cwd=self.root, text=True).strip()

        experiment = Path(cli("experiments.run", "create", "--task", "demo-workflow",
                              "--agent", "manual-walkthrough", "--model", "none",
                              "--environment", "kinematic_demo", "--criteria", "No physical evaluation."))
        (experiment / "prompt.md").write_text("Run the existing kinematic workflow.\n")
        (experiment / "feedback.md").write_text("First trial.\n")
        iteration = cli("experiments.run", "start", str(experiment),
                        "--prompt", str(experiment / "prompt.md"),
                        "--feedback", str(experiment / "feedback.md"))
        (experiment / "response.md").write_text("Using the existing demo without code changes.\n")
        cli("experiments.run", "finish", iteration, "--response", str(experiment / "response.md"))
        trial = cli("experiments.run", "trial", iteration, "--", sys.executable, "main.py")
        cli("experiments.evaluate", trial, "--outcome", "unknown", "--evaluator", "test-operator")
        report = json.loads(cli("experiments.report", str(experiment)))
        self.assertEqual(report["execution_counts"], {"completed": 1})
        self.assertEqual(report["unknown"], 1)
        self.assertEqual(report["physical_successes"], 0)

    def test_launch_error_records_execution_failure(self):
        iteration = self.freeze()
        with self.assertRaises(FileNotFoundError):
            run_trial(iteration, [str(self.root / "missing-executable")])
        path = next(iteration.glob("trials/*/execution.json"))
        execution = read_json(path)
        self.assertEqual(execution["status"], "failed")
        self.assertIsNone(execution["returncode"])
        self.assertEqual(execution["error"]["type"], "FileNotFoundError")
        self.assertIn("finished_at", execution)
        self.assertEqual(summarize(self.experiment)["not_evaluated"], 1)


if __name__ == "__main__":
    unittest.main()

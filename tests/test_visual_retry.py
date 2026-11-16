"""Recoverable visual failures must reacquire frames through the real logging wrapper."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from main import build_demo, run
from pick.perception import ObservationUnavailable, PickPerception
from place.perception import PlacePerception
from rollout import RolloutLog


class VisualRetryTests(unittest.TestCase):
    def test_capture_and_detection_failures_recover_with_logging(self):
        for phase in ("pick", "place"):
            for stage in ("approach", "contact"):
                with self.subTest(phase=phase, stage=stage), tempfile.TemporaryDirectory() as directory:
                    pick, place = build_demo()
                    motion = pick if phase == "pick" else place
                    robot = motion.robot
                    cls = PickPerception if phase == "pick" else PlacePerception
                    tool, target = ("tool", "object") if phase == "pick" else ("object", "socket")
                    captures = 0

                    def capture():
                        nonlocal captures
                        captures += 1
                        if captures <= 9:
                            self.assertEqual(robot.commands, [])
                        if captures == 1:
                            return None
                        if captures == 2:
                            return np.zeros((2, 2)), None
                        if captures == 3:
                            raise ObservationUnavailable("synchronized frame unavailable")
                        return np.full((2, 2), captures), np.full((2, 2), captures)

                    def detect(image):
                        frame = int(image[0, 0])
                        if frame == 4:
                            return None
                        if frame == 5:
                            return {}
                        if frame == 6:
                            return {tool: None, target: [0, 0, 0]}
                        if frame == 7:
                            return {tool: [0, 0, 0], target: [np.nan, 0, 0]}
                        if frame == 8:
                            raise ObservationUnavailable("inference temporarily unavailable")
                        return {tool: [0, 0, 0], target: [0, 0, 0]}

                    observer = cls(SimpleNamespace(capture=capture), detect)
                    motion.perception = observer
                    with patch("time.sleep"), RolloutLog(motion, phase, data_root=directory) as log:
                        with log.stage(stage):
                            getattr(motion, stage)()
                    self.assertGreaterEqual(captures, 11)
                    self.assertIs(motion.perception, observer)
                    self.assertEqual(len(robot.commands), 3 if stage == "approach" else 0)
                    events = [json.loads(line) for line in (log.directory / "events.jsonl").read_text().splitlines()]
                    failures = [row for row in events if row["event"] == "observation_unavailable"]
                    self.assertEqual(len(failures), 2)
                    self.assertIn("inference", failures[-1]["message"])
                    self.assertEqual(json.loads((log.directory / "result.json").read_text())["status"], "completed")

    def test_nonvisual_faults_and_interrupts_are_not_retried(self):
        for phase in ("pick", "place"):
            for fault in (RuntimeError("device fault"), ValueError("invalid detector config"), KeyboardInterrupt()):
                with self.subTest(phase=phase, fault=type(fault).__name__):
                    pick, place = build_demo()
                    motion = pick if phase == "pick" else place

                    def observe():
                        raise fault

                    motion.perception = SimpleNamespace(observe=observe)
                    with patch("time.sleep"), self.assertRaises(type(fault)) as caught:
                        motion.contact()
                    self.assertIs(caught.exception, fault)
                    self.assertEqual(motion.robot.commands, [])

    def test_optional_press_check_retries_without_advancing_depth(self):
        _, place = build_demo()
        robot = place.robot
        origin = robot.pose().z
        answers = iter([False, ObservationUnavailable("occluded"), None, False, True])
        checked_depths = []

        def inserted():
            checked_depths.append(origin - robot.pose().z)
            answer = next(answers)
            if isinstance(answer, ObservationUnavailable):
                raise answer
            return answer

        with patch("time.sleep"):
            place.press(depth_mm=3, inserted=inserted)
        self.assertEqual(checked_depths, [0, 0.25, 0.25, 0.25, 0.5])
        self.assertEqual(origin - robot.pose().z, 0.5)

    def test_press_observes_final_depth_and_retries_without_another_depth_step(self):
        for accepted in (True, False):
            with self.subTest(accepted=accepted), patch("time.sleep"):
                _, place = build_demo()
                robot = place.robot
                origin = robot.pose()
                depths = []
                answers = iter([False, False, None, ObservationUnavailable("occluded"), accepted])

                def inserted():
                    depths.append(origin.z - robot.pose().z)
                    answer = next(answers)
                    if isinstance(answer, ObservationUnavailable):
                        raise answer
                    return answer

                place.press(depth_mm=0.375, inserted=inserted)

                self.assertEqual(depths, [0, 0.25, 0.375, 0.375, 0.375])
                self.assertEqual([origin.z - target.z for target, _, _ in robot.commands],
                                 [0.25, 0.375, 0.375])
                self.assertEqual(robot.pose().yaw, origin.yaw)

    def test_rank_deficient_contact_reobserves_unexplained_features(self):
        for phase_index, phase in enumerate(("pick", "place")):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                motion = build_demo()[phase_index]
                calibration = motion.calibration
                q = calibration.reference_q
                jacobian = np.eye(6, 4)
                jacobian[:, 2] = 0
                calibration = type(calibration).from_projection(q, lambda pose: jacobian @ (pose - q))
                self.assertEqual(calibration.samples["jacobian_rank"], 3)
                motion.calibration = calibration
                mismatch = np.array([0, 0, 0.5, 0, 0, 0])
                observations = iter([mismatch] * 4 + [np.zeros(6)] * 3)
                motion.perception = SimpleNamespace(observe=lambda: next(observations))

                with patch("time.sleep"), RolloutLog(motion, phase, data_root=directory) as log:
                    with log.stage("contact"):
                        motion.contact()

                self.assertEqual(list(observations), [])
                self.assertEqual(motion.robot.commands, [])
                events = [json.loads(line) for line in (log.directory / "events.jsonl").read_text().splitlines()]
                residuals = [row["residual_px_rad"] for row in events if row["event"] == "correction"]
                self.assertEqual(residuals, [(-mismatch).tolist()] * 4 + [[0.0] * 6] * 3)

    def test_full_workflow_retries_vision_and_keeps_original_result_contract(self):
        with tempfile.TemporaryDirectory() as directory, patch("time.sleep"):
            pick, place = build_demo()
            for motion in (pick, place):
                observe = motion.perception.observe
                attempts = [0]

                def flaky(observe=observe, attempts=attempts):
                    attempts[0] += 1
                    if attempts[0] in (1, 3, 5):
                        raise ObservationUnavailable("temporary frame loss")
                    return observe()

                motion.perception.observe = flaky
            result = run(pick, place, data_root=directory)
            self.assertEqual(set(result), {"pick", "place", "rollout_id"})
            self.assertEqual((result["pick"], result["place"]), ("done", "done"))
            self.assertFalse(pick.robot.closed)
            for phase in ("pick", "place"):
                path = Path(directory) / phase / "data" / "rollouts" / result["rollout_id"] / "events.jsonl"
                events = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(sum(row["event"] == "observation_unavailable" for row in events), 3)
                self.assertFalse(any(row["event"] == "workflow_timing" for row in events))


if __name__ == "__main__":
    unittest.main()

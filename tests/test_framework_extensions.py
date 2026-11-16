"""Logger forwarding, declared workspace bounds, stage diagnostics, and the disturbed demo."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from main import build_demo, run
from pick.motion import Pose, WorkspaceLimit
from rollout import RolloutLog


def events(directory, phase, rollout_id):
    path = Path(directory) / phase / "data" / "rollouts" / rollout_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


class ThreeViewPerception:
    """A replacement perception class: three views and its own camera helper."""
    def __init__(self, camera, detect):
        self.camera, self.detect = camera, detect

    def observe(self):
        self.camera.set_exposure(4.0)
        images = self.camera.capture()
        return np.concatenate([self.detect(image) for image in (*images, images[0])])[:6]


class Camera:
    exposure_ms = 1.0

    def set_exposure(self, exposure_ms):
        self.exposure_ms = exposure_ms

    def capture(self):
        return np.zeros((2, 2)), np.ones((2, 2))


class JumpRobot:
    """Arrive at every target immediately; contact force appears at contact_z_mm."""
    def __init__(self, start, contact_z_mm=None):
        self.current, self.contact_z_mm, self.commands = start, contact_z_mm, []

    def pose(self):
        return self.current

    def move(self, target, speed_mm_s, wait=True):
        self.commands.append((target, speed_mm_s, wait))
        self.current = target

    def wrench(self):
        touching = self.contact_z_mm is not None and self.current.z <= self.contact_z_mm
        return np.array([0, 0, 6.0 if touching else 0.0, 0, 0, 0])


class FrameworkExtensionTests(unittest.TestCase):
    def test_logger_does_not_add_pose_reads_or_change_sdk_results_and_faults(self):
        with tempfile.TemporaryDirectory() as directory:
            pick, _ = build_demo()
            robot = pick.robot
            pose_reads = []
            reply = object()
            pose_fault = RuntimeError("pose read fault")
            sdk_fault = RuntimeError("gripper failure")

            def pose():
                pose_reads.append(1)
                raise pose_fault

            def gripper(closed):
                if closed:
                    raise sdk_fault
                return reply

            robot.pose = pose
            robot.move = lambda target, speed_mm_s, wait=True: reply
            robot.gripper = gripper
            with RolloutLog(pick, "pick", data_root=directory) as log:
                with log.stage("begin"):
                    self.assertIs(pick.robot.move(pick.begin_pose, speed_mm_s=1.0), reply)
                    self.assertIs(pick.robot.gripper(closed=False), reply)
                    with self.assertRaises(RuntimeError) as caught:
                        pick.robot.gripper(closed=True)
                    self.assertIs(caught.exception, sdk_fault)

                self.assertEqual(pose_reads, [])
                with self.assertRaises(RuntimeError) as caught:
                    pick.robot.pose()
                self.assertIs(caught.exception, pose_fault)
            self.assertEqual(pose_reads, [1])
            rows = events(directory, "pick", log.rollout_id)
            self.assertEqual(sum(row["event"] == "move_returned" for row in rows), 1)
            self.assertEqual(sum(row["event"] == "gripper_returned" for row in rows), 1)

    def test_logger_forwards_adapter_extensions_and_restores_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            pick, _ = build_demo()
            robot = pick.robot
            robot.joint_torques = lambda: np.arange(6.0)
            robot.payload_kg = 0.0
            with RolloutLog(pick, "pick", data_root=directory) as log:
                with log.stage("begin"):
                    np.testing.assert_array_equal(pick.robot.joint_torques(), np.arange(6.0))
                    pick.robot.payload_kg = 0.4      # Writes reach the SDK adapter.
                    self.assertEqual(pick.robot.payload_kg, 0.4)
                    self.assertIs(pick.perception.target, pick.perception.target)
                    with self.assertRaises(AttributeError):
                        pick.robot.not_an_sdk_method
                    pick.begin()
            self.assertIs(pick.robot, robot)
            self.assertEqual(robot.payload_kg, 0.4)
            rows = events(directory, "pick", log.rollout_id)
            calls = [row for row in rows if row["event"] == "call_returned"]
            self.assertEqual([(row["target"], row["name"], row["result"]) for row in calls],
                             [("DemoRobot", "joint_torques", [0, 1, 2, 3, 4, 5])])
            self.assertTrue(any(row["event"] == "move_requested" for row in rows))

    def test_replacement_perception_gets_frames_and_detections_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            pick, _ = build_demo()
            camera = Camera()
            perception = ThreeViewPerception(camera, lambda image: np.full(3, image.mean()))
            pick.perception = perception
            with RolloutLog(pick, "pick", data_root=directory) as log:
                with log.stage("observe"):
                    features = pick.perception.observe()
            self.assertEqual(features.tolist(), [0, 0, 0, 1, 1, 1])
            self.assertIs(perception.camera, camera)
            self.assertEqual(camera.exposure_ms, 4.0)
            rows = events(directory, "pick", log.rollout_id)
            self.assertEqual([row["view"] for row in rows if row["event"] == "detection"],
                             ["left", "right", "view2"])
            self.assertEqual({row["frame_id"] for row in rows if row["event"] == "detection"}, {1})
            self.assertTrue((log.directory / "frames" / "000001.npz").exists())
            self.assertTrue(rows[0]["images_available"])

    def test_declared_workspace_stops_force_descent_with_a_dead_force_signal(self):
        for index in (0, 1):
            with self.subTest(phase=("pick", "place")[index]), patch("time.sleep"):
                motion = build_demo()[index]
                robot = motion.robot = JumpRobot(Pose(100, 0, 85))   # Force never reports contact.
                motion.perception.observe = lambda: None
                motion.workspace_mm = ((-np.inf, -np.inf, 78.0), (np.inf, np.inf, np.inf))
                with self.assertRaises(WorkspaceLimit):
                    motion.contact(mode="force")
                self.assertTrue(all(target.z >= 78.0 for target, _, _ in robot.commands))
                # The last command is the hold at the measured pose, not a stale target.
                target, _, wait = robot.commands[-1]
                self.assertTrue(wait)
                self.assertEqual(target, robot.pose())

    def test_undeclared_workspace_adds_no_bound(self):
        with patch("time.sleep"):
            pick, _ = build_demo()
            pick.robot = JumpRobot(Pose(100, 0, 500), contact_z_mm=80.0)
            pick.perception.observe = lambda: None
            self.assertIsNone(pick.workspace_mm)
            pick.contact(mode="force")                      # A 420 mm search reaches contact.
            self.assertLessEqual(pick.robot.pose().z, 80.0)

    def test_stage_diagnostics_report_without_stopping_the_stage(self):
        with tempfile.TemporaryDirectory() as directory, patch("time.sleep"), patch("builtins.print"):
            pick, _ = build_demo()
            observe = pick.perception.observe
            remaining_missing = [40]

            def mostly_missing():
                if remaining_missing[0]:
                    remaining_missing[0] -= 1
                    return None
                return observe()

            pick.perception.observe = mostly_missing
            with RolloutLog(pick, "pick", data_root=directory, diagnostic_interval_s=0.0) as log:
                with log.stage("begin"):
                    pick.begin()
                with log.stage("approach"):
                    pick.approach()
            rows = events(directory, "pick", log.rollout_id)
            reports = [row for row in rows if row["event"] == "stage_diagnostic"]
            self.assertGreaterEqual(reports[-1]["missing_observations"], 40)
            self.assertGreaterEqual(reports[-1]["observations"], 3)
            self.assertEqual([row["event"] for row in rows[-2:]], ["stage_completed", "rollout_finished"])
            self.assertEqual(json.loads((log.directory / "result.json").read_text())["status"], "completed")

    def test_disturbed_demo_converges_through_noise_dropout_latency_and_mismatch(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory, \
                    patch("builtins.print"):
                pick, place = build_demo(disturbed=True, seed=seed)
                targets = {"pick": pick.perception.target, "place": place.perception.target}
                prior_gap = np.abs(pick.perception.jacobian - pick.calibration.jacobian).max()
                self.assertGreater(prior_gap, 0)
                result = run(pick, place, data_root=directory)
                for phase, target in targets.items():
                    rows = events(directory, phase, result["rollout_id"])
                    missing = sum(row["event"] == "observation_unavailable"
                                  or (row["event"] == "observation" and row["features"] is None)
                                  for row in rows)
                    self.assertGreater(missing, 0)
                    contact = next(row["pose"] for row in rows if row["event"] == "stage_completed"
                                   and row["stage"] == "contact")
                    # True TCP error at visual contact, not the controller's own estimate.
                    self.assertLess(np.hypot(contact["x"] - target.x, contact["y"] - target.y), 0.5)
                    self.assertLess(abs(contact["z"] - target.z), 0.5)
                    self.assertLess(abs(contact["yaw"] - target.yaw), 0.02)


if __name__ == "__main__":
    unittest.main()

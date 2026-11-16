"""Force descent and contact stopping must remain independent of vision latency."""
import json
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from main import build_demo
from pick.motion import PickMotion, Pose
from pick.perception import BackgroundObservation, ObservationUnavailable, PickPerception
from place.motion import PlaceMotion
from place.perception import PlacePerception
from rollout import RolloutLog


class ForceRobot:
    def __init__(self):
        self.current = Pose(0, 0, 0)
        self.moves = []
        self.force_reads = 0

    def pose(self):
        return self.current

    def move(self, target, speed_mm_s, wait=True):
        self.moves.append((target, wait))
        self.current = target

    def wrench(self):
        self.force_reads += 1
        return [0, 0, 6 if self.current.z <= -4.5 else 0, 0, 0, 0]


class ForceContactTests(unittest.TestCase):
    def test_streaming_targets_are_renewed_before_simulated_robot_reaches_them(self):
        for phase in ("pick", "place"):
            with self.subTest(phase=phase):
                now = [0.]

                def advance(seconds):
                    now[0] += seconds

                with patch("time.monotonic", side_effect=lambda: now[0]), patch("time.sleep", side_effect=advance):
                    pick, place = build_demo()
                    motion = pick if phase == "pick" else place
                    robot = motion.robot
                    robot.current = Pose(100, 0, 80.8)
                    motion.perception = SimpleNamespace(observe=lambda: None)
                    force, move = robot.wrench, robot.move
                    renewed = []

                    def wrench():
                        self.assertLess(now[0], 2., "Descent stalled while vision was missing")
                        return force()

                    def streaming_move(target, speed_mm_s, wait=True):
                        current = robot.pose()
                        if not wait and robot.pending_target is not None:
                            renewed.append(current.z > robot.pending_target.z)
                        move(target, speed_mm_s, wait)

                    robot.wrench, robot.move = wrench, streaming_move
                    motion.contact(mode="force")
                    self.assertGreater(len(renewed), 20)
                    self.assertTrue(all(renewed))
                    self.assertLessEqual(robot.pose().z, 80.05)
                    self.assertIsNone(robot.pending_target)

    def test_missing_vision_does_not_pause_descent(self):
        for cls in (PickMotion, PlaceMotion):
            for unavailable in (None, ObservationUnavailable("target missing")):
                with self.subTest(phase=cls.__name__, unavailable=unavailable):
                    robot = ForceRobot()
                    attempts = []

                    def observe():
                        attempts.append(1)
                        if isinstance(unavailable, Exception):
                            raise unavailable
                        return unavailable

                    motion = cls(robot, SimpleNamespace(observe=observe),
                                 SimpleNamespace(correction=np.asarray), robot.pose())
                    motion.contact(mode="force")
                    self.assertGreaterEqual(len(attempts), 2)
                    self.assertEqual(robot.current.z, -4.5)
                    self.assertEqual(robot.force_reads, 10)
                    self.assertTrue(robot.moves[-1][1])
                    self.assertEqual([move[0].z for move in robot.moves[:-1]],
                                     [-0.5 * i for i in range(1, 10)])

    def test_blocked_capture_or_inference_cannot_delay_force_stop_or_write_late_logs(self):
        for phase in ("pick", "place"):
            for blocked_part in ("capture", "detect"):
                with self.subTest(phase=phase, blocked=blocked_part), tempfile.TemporaryDirectory() as directory:
                    entered, release = threading.Event(), threading.Event()
                    readers = []
                    detect_calls = []

                    def block():
                        entered.set()
                        if not release.wait(3):
                            raise RuntimeError("Test failed to finish force contact independently")

                    def capture():
                        if blocked_part == "capture":
                            block()
                        return np.zeros((2, 2)), np.zeros((2, 2))

                    def detect(image):
                        detect_calls.append(1)
                        if blocked_part == "detect":
                            block()
                        return {"tool": [0, 0, 0], "object": [0, 0, 0], "socket": [0, 0, 0]}

                    robot = ForceRobot()
                    force = robot.wrench

                    def wrench():
                        # Synchronize the test with a genuinely blocked vision call.
                        self.assertTrue(entered.wait(1))
                        return force()

                    robot.wrench = wrench
                    pick, place = build_demo()
                    motion = pick if phase == "pick" else place
                    cls = PickPerception if phase == "pick" else PlacePerception
                    observer = cls(SimpleNamespace(capture=capture), detect)
                    motion.robot, motion.perception = robot, observer

                    def background(perception):
                        reader = BackgroundObservation(perception)
                        readers.append(reader)
                        return reader

                    try:
                        with patch(f"{phase}.motion.BackgroundObservation", side_effect=background):
                            with RolloutLog(motion, phase, data_root=directory) as log:
                                with log.stage("contact"):
                                    motion.contact(mode="force")
                        self.assertFalse(release.is_set())
                        self.assertTrue(readers[0].thread.is_alive())
                        self.assertTrue(robot.moves[-1][1])
                        self.assertEqual(robot.current.z, -4.5)
                        path = log.directory / "events.jsonl"
                        saved = path.read_bytes()
                        release.set()
                        readers[0].thread.join(1)
                        self.assertFalse(readers[0].thread.is_alive())
                        self.assertEqual(path.read_bytes(), saved)
                        self.assertIs(observer.detect, detect)
                        self.assertEqual(len(detect_calls), 1 if blocked_part == "detect" else 0)
                        events = [json.loads(line) for line in saved.splitlines()]
                        self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
                    finally:
                        release.set()
                        for reader in readers:
                            reader.thread.join(1)

    def test_logged_background_observations_update_the_original_adapter(self):
        for phase_index, phase in enumerate(("pick", "place")):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                observed = threading.Event()

                class StatefulPerception:
                    def __init__(self):
                        self.captures = 0

                    def observe(self):
                        self.captures += 1
                        observed.set()
                        return None

                observer = StatefulPerception()
                robot = ForceRobot()
                read_force = robot.wrench

                def wrench():
                    self.assertTrue(observed.wait(1))
                    return read_force()

                robot.wrench = wrench
                motion = build_demo()[phase_index]
                motion.robot, motion.perception = robot, observer
                with RolloutLog(motion, phase, data_root=directory) as log:
                    with log.stage("contact"):
                        motion.contact(mode="force")

                self.assertIs(motion.perception, observer)
                self.assertGreater(observer.captures, 0)
                self.assertEqual(robot.pose().z, -4.5)

    def test_one_visual_result_is_not_reapplied_while_next_observation_is_blocked(self):
        for cls in (PickMotion, PlaceMotion):
            with self.subTest(phase=cls.__name__):
                release = threading.Event()
                robot = ForceRobot()
                calls, corrections, readers = [], [], []

                def observe():
                    calls.append(1)
                    if len(calls) == 1:
                        return np.array([2., -1., 999., 0.1])
                    release.wait(3)
                    return None

                def correction(features):
                    corrections.append(features)
                    return features

                def background(perception):
                    reader = BackgroundObservation(perception)
                    readers.append(reader)
                    return reader

                motion = cls(robot, SimpleNamespace(observe=observe),
                             SimpleNamespace(correction=correction), robot.pose())
                try:
                    with patch(f"{cls.__module__}.BackgroundObservation", side_effect=background):
                        motion.contact(mode="force")
                    self.assertEqual(len(corrections), 1)
                    self.assertEqual((robot.current.x, robot.current.y), (2., -1.))
                    self.assertAlmostEqual(robot.current.yaw, 0.1)
                    self.assertEqual(robot.current.z, -4.5)
                finally:
                    release.set()
                    for reader in readers:
                        reader.thread.join(1)

    def test_fatal_background_error_stops_an_active_descent_and_propagates(self):
        for cls in (PickMotion, PlaceMotion):
            with self.subTest(phase=cls.__name__):
                robot = ForceRobot()
                descending = threading.Event()
                fault = ValueError("invalid detector configuration")
                move = robot.move

                def tracked_move(target, speed_mm_s, wait=True):
                    move(target, speed_mm_s, wait)
                    if not wait:
                        descending.set()

                def observe():
                    descending.wait(2)
                    raise fault

                robot.move = tracked_move
                motion = cls(robot, SimpleNamespace(observe=observe),
                             SimpleNamespace(correction=np.asarray), robot.pose())
                with self.assertRaises(ValueError) as caught:
                    motion.contact(mode="force")
                self.assertIs(caught.exception, fault)
                self.assertTrue(descending.is_set())
                self.assertTrue(robot.moves[-1][1])
                self.assertGreater(robot.current.z, -4.5)


if __name__ == "__main__":
    unittest.main()

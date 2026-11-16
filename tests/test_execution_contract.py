"""Regression cases from hardware integration: recoverable states must not add gates."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pick.motion import PickMotion, Pose
from pick.perception import ObservationUnavailable
from place.motion import PlaceMotion


class Robot:
    """Complete commands immediately, including finite asynchronous test targets."""
    def __init__(self):
        self.current = Pose(0, 0, 0)
        self.moves = []
        self.closed = False

    def pose(self):
        return self.current

    def move(self, target, speed_mm_s, wait=True):
        self.moves.append((target, wait))
        self.current = target

    def gripper(self, closed):
        self.closed = closed

    def wrench(self):
        return np.array([0, 0, 6 if self.current.z <= -4.5 else 0, 0, 0, 0])


class ExecutionContractTests(unittest.TestCase):
    def motion(self, cls, robot, observe):
        return cls(robot, SimpleNamespace(observe=observe),
                   SimpleNamespace(correction=np.asarray), Pose(0, 0, 0))

    def test_visual_stages_reobserve_after_twenty_corrections_and_missing_frames(self):
        for cls in (PickMotion, PlaceMotion):
            for stage in ("approach", "contact"):
                with self.subTest(phase=cls.__name__, stage=stage):
                    robot = Robot()
                    # Missing frames occur both before positioning and within correction.
                    features = [None, ObservationUnavailable("frame unavailable")]
                    if stage == "approach":
                        features += [np.zeros(4)]
                    for _ in range(20):
                        features += [np.array([1., 0, 0, 0]), None,
                                     ObservationUnavailable("temporary inference failure")]
                    features += [np.zeros(4)] * 3
                    samples = iter(features)
                    move_counts_at_missing = []
                    pending_missing = False

                    def observe():
                        nonlocal pending_missing
                        if pending_missing:
                            self.assertEqual(len(robot.moves), move_counts_at_missing[-1])
                        value = next(samples)
                        pending_missing = value is None or isinstance(value, ObservationUnavailable)
                        if pending_missing:
                            move_counts_at_missing.append(len(robot.moves))
                        if isinstance(value, ObservationUnavailable):
                            raise value
                        return value

                    motion = self.motion(cls, robot, observe)
                    with patch("time.sleep"):
                        getattr(motion, stage)()
                    self.assertEqual(len(robot.moves), 23 if stage == "approach" else 20)
                    self.assertEqual(move_counts_at_missing[:2], [0, 0])
                    self.assertEqual(list(samples), [])

    def test_force_search_can_reach_contact_beyond_three_mm(self):
        for cls in (PickMotion, PlaceMotion):
            with self.subTest(phase=cls.__name__):
                robot = Robot()
                missing = True

                def observe():
                    nonlocal missing
                    if missing:
                        missing = False
                        return None
                    return np.zeros(4)

                motion = self.motion(cls, robot, observe)
                with patch("time.sleep"):
                    motion.contact(mode="force")
                self.assertEqual(robot.current.z, -4.5)
                self.assertEqual(len(robot.moves), 10)  # Nine steps, then replace with hold.
                self.assertTrue(robot.moves[-1][1])

    def test_sdk_fault_propagates_without_additional_motion(self):
        fault = RuntimeError("motor feedback unavailable")
        for cls in (PickMotion, PlaceMotion):
            with self.subTest(phase=cls.__name__):
                robot = Robot()

                def fail():
                    raise fault

                robot.wrench = fail
                motion = self.motion(cls, robot, lambda: np.zeros(4))
                with self.assertRaises(RuntimeError) as caught:
                    motion.contact(mode="force")
                self.assertIs(caught.exception, fault)
                self.assertEqual(robot.moves, [])

    def test_sdk_completed_begin_does_not_add_cartesian_residual_gate(self):
        for cls in (PickMotion, PlaceMotion):
            with self.subTest(phase=cls.__name__):
                robot = Robot()

                def move(target, speed_mm_s, wait=True):
                    robot.current = target.shifted([0.646, 0, 0, 0])

                robot.move = move
                motion = self.motion(cls, robot, lambda: np.zeros(4))
                motion.begin()
                self.assertAlmostEqual(robot.current.x, 0.646)

    def test_press_and_release_do_not_require_unspecified_visual_acceptance(self):
        robot = Robot()
        robot.closed = True

        def observe():
            self.fail("No visual acceptance was requested for pressing or release")

        motion = self.motion(PlaceMotion, robot, observe)
        motion.press(depth_mm=10)
        self.assertEqual(robot.current.z, -10)
        motion.release()
        self.assertFalse(robot.closed)
        self.assertEqual(robot.current.z, 20)



if __name__ == "__main__":
    unittest.main()

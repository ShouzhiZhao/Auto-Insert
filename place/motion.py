"""Place actions: begin, approach, contact, press with twists, and release."""
import math
import time
import numpy as np
from pick.motion import Pose, check_workspace
from pick.perception import BackgroundObservation, ObservationUnavailable


class PlaceMotion:
    """Execute this phase using stereo observations and a taught model.

    robot.pose() returns the TCP Pose in the base frame. move(..., wait=True)
    waits for arrival; wait=False returns immediately, and subsequent moves
    update the active target. gripper(closed) operates synchronously.
    Force-based contact uses wrench(): [Fx,Fy,Fz,Tx,Ty,Tz] in N/Nm at the TCP.
    perception.observe() returns six image features; calibration.correction()
    returns an xyz/yaw correction.

    workspace_mm optionally declares base-frame TCP bounds from the device or
    task contract: ((x_min, y_min, z_min), (x_max, y_max, z_max)) in mm, with
    +/-inf for open sides. A task target outside them is not sent and raises
    pick.motion.WorkspaceLimit. None (default) declares no bound.
    """

    def __init__(self, robot, perception, calibration, begin_pose: Pose, workspace_mm=None):
        self.robot = robot
        self.perception = perception
        self.calibration = calibration
        self.begin_pose = begin_pose
        self.workspace_mm = workspace_mm

    def _move(self, target, speed_mm_s, wait=True):
        """Send a task target; a declared workspace bound stops it before the SDK call."""
        if self.workspace_mm is not None:
            check_workspace(target, self.workspace_mm)
        self.robot.move(target, speed_mm_s=speed_mm_s, wait=wait)

    def _observe(self):
        """Let the control loop retry transient vision failures without issuing a move."""
        try:
            return self.perception.observe()
        except ObservationUnavailable:
            return None

    def begin(self):
        # Carry the grasped object to the insertion start pose with the gripper closed.
        target = self.begin_pose
        self._move(target, speed_mm_s=20.0, wait=True)

    def approach(self, clearance_mm=25.0):
        """Move clearance_mm above the target, then align xy/yaw while holding height."""
        features = self._observe()
        while features is None:
            time.sleep(0.02)
            features = self._observe()
        current = self.robot.pose()
        delta = self.calibration.correction(features)
        contact_pose = current.shifted(delta)
        above = contact_pose.shifted([0, 0, clearance_mm, 0])
        # Traverse at height, then descend to the approach pose.
        transit_z = max(current.z, above.z)
        speed_mm_s = 20.0
        target = current.shifted([0, 0, transit_z - current.z, 0])
        self._move(target, speed_mm_s=speed_mm_s, wait=True)
        target = Pose(above.x, above.y, transit_z, above.roll, above.pitch, above.yaw)
        self._move(target, speed_mm_s=speed_mm_s, wait=True)
        target = above
        self._move(target, speed_mm_s=speed_mm_s, wait=True)

        # Hold approach height; accept xy/yaw alignment after three consecutive frames.
        stable = 0
        while True:
            time.sleep(0.02)
            features = self._observe()
            if features is None:
                stable = 0
                continue
            error = self.calibration.correction(features)
            aligned = np.linalg.norm(error[:2]) <= 0.3 and abs(error[3]) <= 0.01
            if aligned:
                stable += 1
                if stable >= 3:
                    return self.robot.pose()
                continue
            stable = 0
            correction = np.array([error[0], error[1], 0.0, error[3]])
            target = self.robot.pose().shifted(correction)
            self._move(target, speed_mm_s=1.0, wait=True)

    def contact(self, mode="vision"):
        """Keep descending and polling force independently of background vision.
        Vision mode applies synchronous corrections of 50% of the xyz/yaw error.
        """
        if mode == "force":
            setpoint = self.robot.pose()
            moving = False
            with BackgroundObservation(self.perception) as vision:
                try:
                    while True:
                        # Force polling and target replacement never wait for vision.
                        fz = float(np.asarray(self.robot.wrench())[2])
                        if fz >= 5.0:  # Environment-on-tool reaction force, N
                            self.robot.move(self.robot.pose(), speed_mm_s=1.0, wait=True)
                            return self.robot.pose()

                        current = self.robot.pose()
                        features = vision.take()
                        if features is not None:
                            error = self.calibration.correction(features)
                            setpoint = current.shifted([error[0], error[1], 0, error[3]])
                        # Retain the last XY/yaw target through missing or slow vision.
                        # The 0.5 mm lookahead is renewed during motion, not awaited.
                        target = Pose(setpoint.x, setpoint.y, current.z - 0.5,
                                      setpoint.roll, setpoint.pitch, setpoint.yaw)
                        moving = True
                        self._move(target, speed_mm_s=1.0, wait=False)
                        time.sleep(0.02)
                except BaseException as error:
                    if moving:
                        try:
                            self.robot.move(self.robot.pose(), speed_mm_s=1.0, wait=True)
                        except Exception as stop_error:
                            raise error from stop_error
                    raise

        # Vision loop: observe, correct half the xyz/yaw error, wait, then observe again.
        consistent = getattr(self.calibration, "visual_contact_consistent", None)
        stable = 0
        while True:
            time.sleep(0.02)
            features = self._observe()
            if features is None:
                stable = 0
                continue
            error = self.calibration.correction(features)
            aligned = np.linalg.norm(error[:2]) <= 0.3 and abs(error[3]) <= 0.01
            reached = aligned and abs(error[2]) <= 0.25  # xy/z: mm; yaw: rad
            if reached:
                # An unmodeled feature error cannot establish contact at this pose.
                if consistent is not None and not consistent(features, error):
                    stable = 0
                    continue
                stable += 1
                if stable >= 3:
                    return self.robot.pose()
                continue
            stable = 0
            target = self.robot.pose().shifted(0.5 * error)
            self._move(target, speed_mm_s=1.0, wait=True)

    def press(self, depth_mm=3.0, inserted=None):
        """Press with alternating twists to the specified depth or until visual acceptance."""
        origin = self.robot.pose()
        step_mm = 0.25
        steps = math.ceil(depth_mm / step_mm)
        for index in range(steps + 1 if steps > 0 else 0):
            if inserted is not None:
                while True:
                    try:
                        accepted = inserted()
                    except ObservationUnavailable:
                        accepted = None
                    if accepted is not None:
                        break
                    time.sleep(0.02)
                if accepted:
                    break
            if index == steps:
                break
            depth = min((index + 1) * step_mm, depth_mm)
            sign = 1 if index % 2 == 0 else -1
            # Alternate twists around entry yaw; adjust roll/pitch for the insertion, in rad.
            roll = origin.roll
            pitch = origin.pitch
            yaw = origin.yaw + sign * 0.04
            target = Pose(origin.x, origin.y, origin.z - depth,
                          roll, pitch, yaw)
            self._move(target, speed_mm_s=1.0, wait=True)
        pose = self.robot.pose()
        target = Pose(pose.x, pose.y, pose.z, origin.roll, origin.pitch, origin.yaw)
        self._move(target, speed_mm_s=1.0, wait=True)

    def release(self, height_mm=30.0):
        """Open the gripper, then lift along +z."""
        self.robot.gripper(closed=False)
        target = self.robot.pose().shifted([0, 0, height_mm, 0])
        self._move(target, speed_mm_s=20.0, wait=True)

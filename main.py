"""Pick-and-place entry point; direct execution uses kinematic demo devices."""
import argparse
import os
import time
import numpy as np
from pick.calibration import PickCalibration
from pick.motion import PickMotion, Pose
from pick.perception import ObservationUnavailable
from place.calibration import PlaceCalibration
from place.motion import PlaceMotion
from rollout import RolloutLog, new_rollout_id


def run(pick: PickMotion, place: PlaceMotion, label="unspecified", *, rollout_id=None, data_root=None):
    """Execute pick/place, save both phase logs under one rollout_id, and return the result."""
    rollout_id = rollout_id or os.environ.get("CAP_ROLLOUT_ID") or new_rollout_id()
    data_root = data_root or os.environ.get("CAP_DATA_ROOT")
    with (
        RolloutLog(pick, "pick", rollout_id, label, data_root=data_root) as pick_log,
        RolloutLog(place, "place", rollout_id, label, data_root=data_root) as place_log,
    ):
        with pick_log.stage("begin"):
            pick.begin()
        with pick_log.stage("approach"):
            pick.approach(clearance_mm=25.0)
        with pick_log.stage("contact"):
            pick.contact(mode="vision")
        with pick_log.stage("close"):
            pick.close()
        with pick_log.stage("lift"):
            pick.lift(height_mm=30.0)
        pick_log.finish()

        with place_log.stage("begin"):
            place.begin()
        with place_log.stage("approach"):
            place.approach(clearance_mm=25.0)
        with place_log.stage("contact"):
            place.contact(mode="vision")
        with place_log.stage("press"):
            place.press(depth_mm=3.0)
        with place_log.stage("release"):
            place.release(height_mm=30.0)
        place_log.finish()
    return {"pick": "done", "place": "done", "rollout_id": rollout_id}


class DemoRobot:
    """Kinematic demo device: blocking moves arrive immediately; nonblocking moves advance over time."""
    def __init__(self):
        self.current = Pose(90, -5, 130)
        self.closed = False
        self.commands = []
        self.pending_target = None
        self.move_speed_mm_s = 1.0
        self.last_update = time.monotonic()

    def pose(self):
        now = time.monotonic()
        elapsed = now - self.last_update
        if self.pending_target is not None:
            # Advance a nonblocking move toward its finite target and stop on arrival.
            delta = self.pending_target.q() - self.current.q()
            delta[3] = (delta[3] + np.pi) % (2 * np.pi) - np.pi
            duration = max(np.linalg.norm(delta[:3]) / self.move_speed_mm_s,
                           abs(delta[3]) / 0.2)  # Demo yaw speed: 0.2 rad/s
            fraction = min(elapsed / duration, 1.0) if duration else 1.0
            self.current = self.current.shifted(delta * fraction)
            if fraction == 1.0:
                self.current = self.pending_target
                self.pending_target = None
        self.last_update = now
        return self.current

    def move(self, target, speed_mm_s, wait=True):
        self.pose()
        if wait:
            self.current = target
            self.pending_target = None
        else:
            self.pending_target = target
            self.move_speed_mm_s = speed_mm_s
        self.commands.append((target, speed_mm_s, wait))

    def gripper(self, closed):
        self.closed = closed

    def wrench(self):
        force = 6.0 if self.pose().z <= 80.05 else 0.0
        return np.array([0, 0, force, 0, 0, 0])


class DemoPerception:
    """Map the current TCP error to six image features.

    The defaults give an exact linear model. Optional disturbances:
    noise_px / noise_rad   Gaussian sigma on the u/v and theta features
    dropout                probability of a missing observation (alternating
                           None and ObservationUnavailable)
    latency_s              capture-to-result delay; features describe the pose at capture
    perspective_per_mm     relative image-scale change per mm of z error, so the
                           true feature map departs from any single local Jacobian
    """
    def __init__(self, robot, target, jacobian, *, noise_px=0.0, noise_rad=0.0, dropout=0.0,
                 latency_s=0.0, perspective_per_mm=0.0, seed=0):
        self.robot, self.target, self.jacobian = robot, target, jacobian
        self.noise = np.array([noise_px, noise_px, noise_rad] * 2)
        self.dropout, self.latency_s = dropout, latency_s
        self.perspective_per_mm = perspective_per_mm
        self.rng = np.random.default_rng(seed)
        self.dropped = 0

    def project(self, q):
        """Return the disturbance-free features at TCP coordinates q = [x, y, z, yaw]."""
        delta = np.asarray(q, dtype=float) - self.target.q()
        delta[3] = (delta[3] + np.pi) % (2 * np.pi) - np.pi
        return self.jacobian @ delta * (1 + self.perspective_per_mm * delta[2])

    def observe(self):
        q = self.robot.pose().q()
        if self.latency_s:
            time.sleep(self.latency_s)
        if self.rng.random() < self.dropout:
            self.dropped += 1
            if self.dropped % 2:
                return None
            raise ObservationUnavailable("demo frame dropped")
        return self.project(q) + self.rng.normal(0, 1, 6) * self.noise


def build_demo(disturbed=False, seed=0):
    """Create demo devices, synthetic teaching samples, and pick/place motion objects.

    disturbed=True teaches from and runs against a camera model the prior does
    not match: each Jacobian entry differs by up to 15%, image scale changes
    0.4% per mm of height, features carry 0.05 px / 0.001 rad noise, 15% of
    observations are missing, and results arrive 10 ms after capture.
    """
    robot = DemoRobot()
    jacobian = np.array([
        [-2, 0, 0.4, 8], [0, -2, 0.2, 3], [0, 0, 0, -1],
        [-2, 0, -0.4, 8], [0, -2, 0.1, 3], [0, 0, 0, -1],
    ])
    pick_target = Pose(100, 0, 80, yaw=0.08)
    place_target = Pose(200, 20, 80, yaw=-0.06)

    rng = np.random.default_rng(seed)
    disturbance = {}
    camera_jacobian = jacobian
    if disturbed:
        camera_jacobian = jacobian * (1 + 0.15 * rng.uniform(-1, 1, jacobian.shape))
        disturbance = dict(noise_px=0.05, noise_rad=0.001, dropout=0.15, latency_s=0.01,
                           perspective_per_mm=0.004)
    pick_perception = DemoPerception(robot, pick_target, camera_jacobian, seed=seed + 1, **disturbance)
    place_perception = DemoPerception(robot, place_target, camera_jacobian, seed=seed + 2, **disturbance)

    # Fit separate pick/place models from positive and negative x/y/z/yaw perturbations,
    # measured through each phase's camera model (with its noise when disturbed).
    displacements = np.vstack([np.eye(4), -np.eye(4)]) * [2, 2, 2, 0.05]
    calibrations = []
    for cls, perception in ((PickCalibration, pick_perception), (PlaceCalibration, place_perception)):
        sample_q = perception.target.q() + displacements
        features = np.array([perception.project(q) for q in sample_q])
        features += rng.normal(0, 1, features.shape) * perception.noise
        calibration = cls.fit(perception.target.q(), np.zeros(6), sample_q, features,
                              prior=jacobian, prior_weight=0.1)
        calibration.samples["source"] = "synthetic_demo_disturbed" if disturbed else "synthetic_demo"
        calibrations.append(calibration)
    pick = PickMotion(robot, pick_perception, calibrations[0], begin_pose=Pose(90, -5, 130))
    place = PlaceMotion(robot, place_perception, calibrations[1], begin_pose=Pose(185, 10, 130))
    return pick, place


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disturbed", action="store_true",
                        help="add camera-model mismatch, noise, latency, and dropped frames")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("Kinematic demo (no hardware connection)")
    print(run(*build_demo(args.disturbed, args.seed),
              label="demo-disturbed" if args.disturbed else "demo"))

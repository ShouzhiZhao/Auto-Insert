"""Rollout recording: stages, device calls, images, detections, and calibration snapshots."""
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo
import json
import shutil
import time
import threading
import traceback
import numpy as np
from pick.perception import ObservationUnavailable


def new_rollout_id():
    """Generate a run ID from Beijing time and a random suffix."""
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f")
    return f"{stamp}_{uuid4().hex[:8]}"


def jsonable(value):
    """Convert event fields json cannot encode; large arrays are summarized, not stored."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.ndarray) and value.size > 64:
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


class Recorded:
    """Stand in for one motion dependency (robot, perception, calibration, camera).

    The calls passed as keywords go to the log's recording methods. Every other
    attribute read or write reaches the original object, so SDK adapter
    extensions keep working inside the logging scope; their method calls are
    recorded as call/call_returned events and exceptions propagate unchanged.
    """

    def __init__(self, target, log, **recorded):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_recorded", recorded)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in self._recorded:
            return self._recorded[name]
        value = getattr(self._target, name)
        if not callable(value) or name.startswith("_"):
            return value
        owner = type(self._target).__name__

        def call(*args, **kwargs):
            self._log.write("call", target=owner, name=name, args=list(args), kwargs=kwargs)
            result = value(*args, **kwargs)
            self._log.write("call_returned", target=owner, name=name, result=result)
            return result

        return call

    def __setattr__(self, name, value):
        setattr(self._target, name, value)


class RolloutLog:
    """Record one pick or place execution under that phase's data/rollouts directory.

    The with scope records device and algorithm calls and restores the original
    motion dependencies on exit. A perception object exposing `camera` and
    `detect` attributes (the built-in classes, subclasses, or replacements) gets
    its frames and per-view detections recorded and linked through frame_id.

    While a stage runs, a stage_diagnostic event and one terminal line report
    progress every diagnostic_interval_s seconds. It is a diagnostic only: it
    never stops, delays, or fails the stage. None disables it.
    """

    def __init__(self, motion, phase, rollout_id=None, label="unspecified", *, data_root=None,
                 diagnostic_interval_s=5.0):
        self.motion = motion
        self.phase = phase
        self.rollout_id = rollout_id or new_rollout_id()
        self.label = label
        self.robot = motion.robot
        self.perception = motion.perception
        self.calibration = motion.calibration
        self.camera = None
        self.detector = None
        if (getattr(self.perception, "camera", None) is not None
                and callable(getattr(self.perception, "detect", None))):
            self.camera = self.perception.camera
            self.detector = self.perception.detect
        self.root = Path(__file__).parent
        self.data_root = Path(data_root) if data_root is not None else self.root
        self.directory = self.data_root / phase / "data" / "rollouts" / self.rollout_id
        self.current_stage = "idle"
        self.started = False
        self.finished = False
        self.sequence = 0
        self.frame_id = 0
        self._last_pose = None
        self._write_lock = threading.Lock()
        self._closed = threading.Event()
        self._observation_context = threading.local()
        self.diagnostic_interval_s = diagnostic_interval_s
        self._progress_lock = threading.Lock()
        self._progress = None

    def __enter__(self):
        self.directory.mkdir(parents=True)
        calibration_path = self.calibration.save(
            self.data_root / self.phase / "data" / "calibrations" / f"{self.rollout_id}.json"
        )
        self.calibration.save(self.directory / "calibration.json")
        source_dir = self.directory / "source"
        source_dir.mkdir()
        for path in (self.root / self.phase).rglob("*.py"):
            relative = path.relative_to(self.root / self.phase)
            if "data" in relative.parts or "__pycache__" in relative.parts:
                continue
            destination = source_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        for name in ("main.py", "rollout.py", "AGENTS.md"):
            shutil.copy2(self.root / name, source_dir / name)
        if self.phase == "place":
            shutil.copy2(self.root / "pick" / "motion.py", source_dir / "pick_motion.py")
            shutil.copy2(self.root / "pick" / "perception.py", source_dir / "pick_perception.py")
        self.write("rollout_started", rollout_id=self.rollout_id, phase=self.phase,
                   label=self.label, calibration_path=str(calibration_path),
                   begin_pose=asdict(self.motion.begin_pose),
                   perception=type(self.perception).__name__, images_available=self.camera is not None)

        # Record calls within this scope; __exit__ restores these references.
        self.motion.robot = Recorded(self.robot, self, pose=self.pose, move=self.move,
                                     gripper=self.gripper, wrench=self.wrench)
        self.motion.perception = Recorded(self.perception, self, observe=self.observe,
                                          observation_reader=self.observation_reader)
        self.motion.calibration = Recorded(self.calibration, self, correction=self.correction)
        if self.camera is not None:
            self.perception.camera = Recorded(self.camera, self, capture=self.capture)
            self.perception.detect = self.detect
            self.write("perception_setup", tool_key=getattr(self.perception, "tool_key", None),
                       target_key=getattr(self.perception, "target_key", None))
        return self

    def __exit__(self, error_type, error, tb):
        try:
            if not self.finished:
                details = None
                status = "completed"
                if error is not None:
                    details = {"type": error_type.__name__, "message": str(error),
                               "traceback": "".join(traceback.format_exception(error_type, error, tb))}
                    status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                    if not self.started:
                        status = "not_started"
                self.finish(status, details)
        finally:
            with self._write_lock:
                self._closed.set()
            self.motion.robot = self.robot
            self.motion.perception = self.perception
            self.motion.calibration = self.calibration
            if self.camera is not None:
                self.perception.camera = self.camera
                self.perception.detect = self.detector
        return False  # Propagate the exception to the caller.

    def write(self, event, **data):
        with self._write_lock:
            if self._observation_stopped():
                return
            self.sequence += 1
            row = {"sequence": self.sequence, "time": datetime.now(timezone.utc).isoformat(),
                   "monotonic_s": time.monotonic(), "stage": self.current_stage,
                   "event": event, **data}
            with (self.directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=jsonable) + "\n")

    @contextmanager
    def stage(self, name):
        self.current_stage = name
        self.started = True
        print(f"{self.phase}.{name}")
        self.write("stage_started", pose=self._last_pose)
        now = time.monotonic()
        with self._progress_lock:
            self._progress = {"started_s": now, "reported_s": now, "observations": 0,
                              "missing_observations": 0, "corrections": 0,
                              "last_correction_xyz_yaw": None, "last_wrench": None}
        try:
            yield
        except BaseException as error:
            self.write("stage_failed", error_type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
            raise
        finally:
            with self._progress_lock:
                self._progress = None
        self.write("stage_completed", pose=self._last_pose)

    def _progress_update(self, count=None, **latest):
        """Count stage activity and report it periodically; never affects execution."""
        with self._progress_lock:
            progress = self._progress
            if progress is None or self._observation_stopped():
                return
            if count is not None:
                progress[count] += 1
            progress.update(latest)
            now = time.monotonic()
            if (self.diagnostic_interval_s is None
                    or now - progress["reported_s"] < self.diagnostic_interval_s):
                return
            progress["reported_s"] = now
            report = {key: value for key, value in progress.items()
                      if key not in ("started_s", "reported_s")}
            elapsed_s = now - progress["started_s"]
        self.write("stage_diagnostic", elapsed_s=elapsed_s, **report)
        print(f"{self.phase}.{self.current_stage} running {elapsed_s:.0f} s: "
              f"{report['observations']} observations "
              f"({report['missing_observations']} missing), "
              f"{report['corrections']} corrections, "
              f"last correction {report['last_correction_xyz_yaw']}")

    def finish(self, status="completed", error=None):
        self.write("rollout_finished", status=status, error=error)
        result = {"rollout_id": self.rollout_id, "phase": self.phase, "label": self.label,
                  "status": status, "last_stage": self.current_stage, "error": error}
        (self.directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        self.finished = True

    def pose(self):
        pose = self.robot.pose()
        self._last_pose = asdict(pose)
        self.write("pose", pose=self._last_pose)
        return pose

    def move(self, target, speed_mm_s, wait=True):
        """Record SDK completion without adding another device read."""
        self.write("move_requested", target=asdict(target), speed_mm_s=speed_mm_s, wait=wait)
        self._last_pose = None
        result = self.robot.move(target, speed_mm_s=speed_mm_s, wait=wait)
        self.write("move_returned", target=asdict(target), wait=wait, pose=None)
        return result

    def gripper(self, closed):
        self.write("gripper_requested", closed=closed)
        result = self.robot.gripper(closed=closed)
        self.write("gripper_returned", closed=closed, pose=None)
        return result

    def wrench(self):
        values = self.robot.wrench()
        self.write("wrench", values=np.asarray(values).tolist())
        self._progress_update(last_wrench=np.asarray(values).tolist())
        return values

    def observe(self):
        return self._observe(self.perception.observe)

    def observation_reader(self, cancelled):
        """Record the original adapter's background observations until cancellation."""
        reader = getattr(self.perception, "observation_reader", None)
        observe = reader(cancelled) if reader is not None else self.perception.observe

        def read():
            self._observation_context.cancelled = cancelled
            try:
                return self._observe(observe)
            finally:
                del self._observation_context.cancelled

        return read

    def _observation_stopped(self):
        cancelled = getattr(self._observation_context, "cancelled", None)
        return self._closed.is_set() or (cancelled is not None and cancelled.is_set())

    def _observe(self, observe):
        if self._observation_stopped():
            raise ObservationUnavailable("Observation scope closed")
        self._observation_context.frame_id = None
        self.write("observation_requested")
        try:
            features = observe()
        except ObservationUnavailable as error:
            if not self._observation_stopped():
                self.write("observation_unavailable", error_type=type(error).__name__, message=str(error))
                self._progress_update("missing_observations")
            raise
        if self._observation_stopped():
            raise ObservationUnavailable("Observation scope closed during observation")
        self.write("observation", frame_id=getattr(self._observation_context, "frame_id", None),
                   features=np.asarray(features).tolist() if features is not None else None)
        self._progress_update("observations" if features is not None else "missing_observations")
        return features

    def correction(self, features):
        correction = self.calibration.correction(features)
        residual = getattr(self.calibration, "correction_residual", None)
        data = {}
        if callable(residual):
            data["residual_px_rad"] = np.asarray(residual(features, correction)).tolist()
        self.write("correction", features=np.asarray(features).tolist(),
                   delta_xyz_yaw=np.asarray(correction).tolist(), **data)
        self._progress_update("corrections",
                              last_correction_xyz_yaw=np.asarray(correction).round(4).tolist())
        return correction

    def capture(self):
        """Save synchronized stereo images, timestamped when received by this process."""
        if self._observation_stopped():
            raise ObservationUnavailable("Observation scope closed before capture")
        self.write("capture_requested")
        images = self.camera.capture()
        if self._observation_stopped():
            raise ObservationUnavailable("Observation scope closed during capture")
        if images is None or any(image is None for image in images):
            self.write("capture_unavailable")
            return images
        left, right = images
        captured_at = datetime.now(timezone.utc).isoformat()
        captured_monotonic_s = time.monotonic()
        with self._write_lock:
            self.frame_id += 1
            frame_id = self.frame_id
        self._observation_context.frame_id = frame_id
        self._observation_context.view_index = 0
        frame_dir = self.directory / "frames"
        frame_dir.mkdir(exist_ok=True)
        frame_path = frame_dir / f"{frame_id:06d}.npz"
        np.savez_compressed(frame_path, left=left, right=right)
        self.write("capture", frame_id=frame_id, images=str(frame_path.relative_to(self.directory)),
                   captured_at=captured_at, captured_monotonic_s=captured_monotonic_s)
        return images

    def detect(self, image):
        if self._observation_stopped():
            raise ObservationUnavailable("Observation scope closed before detection")
        # Views are numbered per capture: left, right, then view2... for other layouts.
        view_index = getattr(self._observation_context, "view_index", 0)
        view = ("left", "right")[view_index] if view_index < 2 else f"view{view_index}"
        frame_id = getattr(self._observation_context, "frame_id", None)
        self.write("detection_requested", frame_id=frame_id, view=view)
        objects = self.detector(image)
        if self._observation_stopped():
            raise ObservationUnavailable("Observation scope closed during detection")
        # Built-in detectors return a mapping; other output types are recorded as returned.
        self.write("detection", frame_id=frame_id, view=view,
                   objects={str(key): np.asarray(value).tolist() for key, value in objects.items()}
                   if isinstance(objects, dict) else objects)
        self._observation_context.view_index = view_index + 1
        return objects

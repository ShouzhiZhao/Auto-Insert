"""Pick perception: detect both views and combine tool/object relative features."""
import threading
import numpy as np


class ObservationUnavailable(Exception):
    """A recoverable capture/detection failure; obtain a fresh stereo observation."""


class BackgroundObservation:
    """Acquire vision independently; consume each latest result at most once.

    Closing discards late results and prevents another capture. An in-flight
    camera/inference call may finish in its daemon thread but never issues motion.
    """

    def __init__(self, perception):
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.result = None
        # An adapter can supply a cancellation-aware reader for background use.
        reader = getattr(perception, "observation_reader", None)
        self.observe = reader(self.closed) if reader is not None else perception.observe
        self.thread = threading.Thread(target=self._run, name="force-contact-vision", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.closed.set()
        # Never wait for camera/network work in the force-control thread.
        with self.lock:
            self.result = None

    def _run(self):
        while not self.closed.is_set():
            error = None
            try:
                features = self.observe()
            except ObservationUnavailable:
                features = None
            except BaseException as failure:
                features, error = None, failure
            with self.lock:
                if self.closed.is_set():
                    return
                self.result = (features, error)
            if error is not None:
                return
            self.closed.wait(0.02)

    def take(self):
        """Return a new feature vector or None immediately; propagate real failures."""
        with self.lock:
            result, self.result = self.result, None
        if result is None:
            return None
        features, error = result
        if error is not None:
            raise error
        return features


class PickPerception:
    """Convert task detections into relative stereo features.

    camera.capture() returns synchronized (left_image, right_image) arrays.
    detect(image) returns {object name or ID: [u_px, v_px, theta_rad]}.
    Prefer robust physical features and current SOTA deep models matched to the
    available hardware; validate repeatability under occlusion and viewpoint changes.
    OpenCV-only hand-tuned detection pipelines are not allowed. See README.md.
    Matching keys identify the same physical feature in both views;
    theta is a directed angle with a 2*pi period.
    """

    def __init__(self, camera, detect, tool_key="tool", target_key="object"):
        self.camera = camera
        self.detect = detect
        self.tool_key = tool_key
        self.target_key = target_key

    def observe(self):
        """Return target-minus-tool du/dv/dtheta for both views, or None for missing features."""
        capture, detect = self.camera.capture, self.detect
        images = capture()
        if images is None or any(image is None for image in images):
            return None
        left_image, right_image = images
        left_objects = detect(left_image)
        right_objects = detect(right_image)

        features = []
        for objects in (left_objects, right_objects):
            if objects is None or objects.get(self.tool_key) is None or objects.get(self.target_key) is None:
                return None
            tool = np.asarray(objects[self.tool_key], dtype=float)
            target = np.asarray(objects[self.target_key], dtype=float)
            if not np.isfinite(tool).all() or not np.isfinite(target).all():
                return None
            relative = target - tool
            relative[2] = (relative[2] + np.pi) % (2 * np.pi) - np.pi
            features.extend(relative)
        return np.asarray(features)

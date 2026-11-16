"""Place perception: detect both views and combine held-object/socket relative features."""
import numpy as np


class PlacePerception:
    """Convert task detections into relative stereo features.

    camera.capture() returns synchronized (left_image, right_image) arrays.
    detect(image) returns {object name or ID: [u_px, v_px, theta_rad]}.
    Prefer robust physical features and current SOTA deep models matched to the
    available hardware; validate repeatability under occlusion and viewpoint changes.
    OpenCV-only hand-tuned detection pipelines are not allowed. See README.md.
    Matching keys identify the same physical feature in both views;
    theta is a directed angle with a 2*pi period.
    For an occluded placement location, target_key may identify a visible
    reference fixed relative to that location. Teach and calibrate using
    the same reference feature used during execution.
    """

    def __init__(self, camera, detect, tool_key="object", target_key="socket"):
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

"""Pick calibration: teach grasp alignment, fit the forward image Jacobian, and invert for motion."""
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
from pathlib import Path
import json
import numpy as np


def feature_delta(current, reference):
    """Subtract image features and wrap both views' orientation differences to [-pi, pi)."""
    delta = np.asarray(current, dtype=float) - np.asarray(reference, dtype=float)
    delta[..., [2, 5]] = (delta[..., [2, 5]] + np.pi) % (2 * np.pi) - np.pi
    return delta


@dataclass
class PickCalibration:
    """A taught model with saved arrays and normalization scales for motion correction."""
    reference: np.ndarray       # Six-dimensional stereo features at the taught alignment
    jacobian: np.ndarray        # df / d[x_mm, y_mm, z_mm, yaw_rad], shape (6, 4)
    reference_q: np.ndarray     # Taught TCP [x, y, z, yaw]
    feature_scale: np.ndarray   # Image feature scales in pixels/rad
    action_scale: np.ndarray    # TCP displacement scales in mm/rad
    samples: dict | None = None  # Raw poses/features, prior, and fitting provenance

    @classmethod
    def fit(cls, reference_q, reference, sample_q, sample_features,
            prior=None, prior_weight=0.0):
        """Fit the local Jacobian of six features with respect to xyz/yaw.

        reference_q/reference are length-4/6 reference arrays. sample_q contains
        Nx4 measured absolute poses; sample_features contains matching Nx6 features.
        prior is a 6x4 forward Jacobian; prior_weight sets its regularization weight.
        """
        reference_q = np.asarray(reference_q, dtype=float)
        reference = np.asarray(reference, dtype=float)
        pose_changes = np.asarray(sample_q, dtype=float) - reference_q
        pose_changes[:, 3] = (pose_changes[:, 3] + np.pi) % (2 * np.pi) - np.pi
        feature_changes = feature_delta(sample_features, reference)

        # Normalize pixels, mm, and rad to comparable scales before fitting.
        feature_scale = np.array([1, 1, 0.02, 1, 1, 0.02])
        action_scale = np.array([1, 1, 1, 0.05])
        poses = pose_changes / action_scale
        features = feature_changes / feature_scale

        # Fit poses @ mapping to features; add regularization when a prior is supplied.
        if prior is not None and prior_weight:
            scaled_prior = np.asarray(prior) * action_scale[None, :] / feature_scale[:, None]
            mapping = np.linalg.solve(
                poses.T @ poses + prior_weight * np.eye(4),
                poses.T @ features + prior_weight * scaled_prior.T,
            )
        else:
            mapping = np.linalg.lstsq(poses, features, rcond=None)[0]

        jacobian = mapping.T * feature_scale[:, None] / action_scale[None, :]
        samples = {
            "source": "measured_samples",
            "q": np.asarray(sample_q).tolist(),
            "features": np.asarray(sample_features).tolist(),
            "prior": np.asarray(prior).tolist() if prior is not None else None,
            "prior_weight": prior_weight,
        }
        result = cls(reference, jacobian, reference_q, feature_scale, action_scale, samples)
        samples["fit_rmse"] = result.fit_rmse(sample_q, sample_features)
        samples["jacobian_rank"] = result.jacobian_rank()
        return result

    @classmethod
    def from_projection(cls, reference_q, project):
        """Use central differences on project(q) along four axes.
        q contains xyz/yaw; project returns six image features.
        """
        reference_q = np.asarray(reference_q, dtype=float)
        steps = [1.0, 1.0, 1.0, 0.01]  # xyz: mm; yaw: rad
        jacobian = np.zeros((6, 4))
        sampled_poses = []
        sampled_features = []
        for axis, step in enumerate(steps):
            displacement = np.zeros(4)
            displacement[axis] = step
            positive = project(reference_q + displacement)
            negative = project(reference_q - displacement)
            jacobian[:, axis] = feature_delta(positive, negative) / (2 * step)
            sampled_poses.extend([(reference_q + displacement).tolist(), (reference_q - displacement).tolist()])
            sampled_features.extend([np.asarray(positive).tolist(), np.asarray(negative).tolist()])

        feature_scale = np.array([1, 1, 0.02, 1, 1, 0.02])
        action_scale = np.array([1, 1, 1, 0.05])
        samples = {"source": "projection", "q": sampled_poses,
                   "features": sampled_features, "steps": steps}
        result = cls(np.asarray(project(reference_q)), jacobian, reference_q,
                     feature_scale, action_scale, samples)
        samples["jacobian_rank"] = result.jacobian_rank()
        return result

    def correction(self, features):
        """Solve J dq = f_ref - f_current; return [dx_mm, dy_mm, dz_mm, dyaw_rad]."""
        matrix = self._normalized_jacobian()
        error = feature_delta(self.reference, features) / self.feature_scale
        normalized_motion = np.linalg.lstsq(matrix, error, rcond=None)[0]
        return normalized_motion * self.action_scale

    def _normalized_jacobian(self):
        return self.jacobian * self.action_scale[None, :] / self.feature_scale[:, None]

    def jacobian_rank(self):
        """Report the number of motion directions observable in the normalized model."""
        return int(np.linalg.matrix_rank(self._normalized_jacobian()))

    def correction_residual(self, features, correction):
        """Return the unexplained stereo feature error in pixels/radians."""
        return feature_delta(self.reference, features) - self.jacobian @ correction

    def visual_contact_consistent(self, features, correction):
        """A rank-deficient fit can report zero motion for an unexplained feature error."""
        matrix = self._normalized_jacobian()
        if np.linalg.matrix_rank(matrix) == matrix.shape[1]:
            return True
        residual = self.correction_residual(features, correction) / self.feature_scale
        # Allow numerical roundoff when the deficient model still explains this observation.
        error = feature_delta(self.reference, features) / self.feature_scale
        roundoff = np.finfo(matrix.dtype).eps * max(matrix.shape) * max(1.0, np.linalg.norm(error))
        return np.linalg.norm(residual) <= roundoff

    def fit_rmse(self, sample_q, sample_features):
        """Return image-feature RMSE normalized by feature_scale."""
        pose_changes = np.asarray(sample_q, dtype=float) - self.reference_q
        pose_changes[:, 3] = (pose_changes[:, 3] + np.pi) % (2 * np.pi) - np.pi
        predicted = pose_changes @ self.jacobian.T
        measured = feature_delta(sample_features, self.reference)
        return float(np.sqrt(np.mean(((predicted - measured) / self.feature_scale) ** 2)))

    def save(self, path=None):
        """Save the model and raw samples and return its Path.
        The default destination is a unique file under data/calibrations.
        """
        if path is None:
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f")
            path = Path(__file__).parent / "data" / "calibrations" / f"{stamp}_{uuid4().hex[:8]}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "reference": self.reference.tolist(),
            "jacobian": self.jacobian.tolist(),
            "reference_q": self.reference_q.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "action_scale": self.action_scale.tolist(),
            "samples": self.samples,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path):
        """Load the model, normalization scales, and saved sampling data."""
        data = json.loads(Path(path).read_text())
        return cls(
            reference=np.asarray(data["reference"]),
            jacobian=np.asarray(data["jacobian"]),
            reference_q=np.asarray(data["reference_q"]),
            feature_scale=np.asarray(data["feature_scale"]),
            action_scale=np.asarray(data["action_scale"]),
            samples=data.get("samples"),
        )

"""
Pose Normalization Module
=========================
Normalizes the detected cow pose to a canonical orientation before
extracting body dimensions.

Pipeline:
  1. Detect the cow's orientation (lateral / frontal / dorsal / oblique)
  2. Estimate the yaw angle using keypoints (shoulder→hip axis)
  3. Apply geometric corrections:
     - Rotation alignment to canonical lateral view
     - Perspective foreshortening compensation
     - Scale normalization
  4. Output the normalized keypoints + transformation matrix
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class PoseResult:
    """Result of pose normalization."""
    orientation: str                       # lateral | frontal | dorsal | oblique
    yaw_angle_deg: float                   # Estimated yaw (0 = perfect lateral)
    pitch_angle_deg: float                 # Estimated pitch
    roll_angle_deg: float                  # Estimated roll
    is_valid_pose: bool                    # Whether pose is usable for measurement
    normalized_keypoints: np.ndarray       # (N, 2) keypoints after normalization
    original_keypoints: np.ndarray         # (N, 2) original keypoints
    transform_matrix: np.ndarray           # 3×3 homography / affine
    foreshortening_factor: float           # Correction factor for depth compression
    confidence: float


class PoseNormalizer:
    """
    Normalizes detected cow keypoints to a canonical pose, compensating
    for camera angle, animal orientation, and perspective distortion.
    """

    # Keypoint indices (must match KeypointConfig order)
    NOSE = 0
    LEFT_EAR = 1
    RIGHT_EAR = 2
    POLL = 3
    WITHERS = 4
    SPINE_MID = 5
    HIP_POINT = 6
    TAIL_HEAD = 7
    LEFT_SHOULDER = 8
    RIGHT_SHOULDER = 9
    LEFT_HIP = 10
    RIGHT_HIP = 11
    LEFT_KNEE_FRONT = 12
    RIGHT_KNEE_FRONT = 13
    LEFT_HOCK = 14
    RIGHT_HOCK = 15
    BRISKET = 16

    def __init__(self, pose_config):
        self.cfg = pose_config

    def normalize(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        image_shape: Tuple[int, int],
        confidence_threshold: float = 0.3,
    ) -> PoseResult:
        """
        Normalize the cow's pose to a canonical lateral view.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape (N, 2) – detected keypoint coordinates in pixel space.
        confidences : np.ndarray
            Shape (N,) – per-keypoint confidence scores.
        image_shape : tuple
            (height, width) of the image.
        confidence_threshold : float
            Minimum confidence to consider a keypoint valid.

        Returns
        -------
        PoseResult
        """
        # Filter low-confidence keypoints
        valid_mask = confidences >= confidence_threshold
        kp = keypoints.copy()

        # ── Step 1: Determine orientation ────────────────────────────────
        orientation, yaw = self._estimate_orientation(kp, valid_mask)

        # ── Step 2: Estimate roll from spine line ────────────────────────
        roll = self._estimate_roll(kp, valid_mask)

        # ── Step 3: Estimate pitch ───────────────────────────────────────
        pitch = self._estimate_pitch(kp, valid_mask)

        # ── Step 4: Compute foreshortening correction ────────────────────
        foreshortening = self._compute_foreshortening(yaw)

        # ── Step 5: Build normalization transform ────────────────────────
        M, normalized_kp = self._build_transform(
            kp, valid_mask, roll, image_shape
        )

        # ── Step 6: Validate pose ────────────────────────────────────────
        is_valid = self._validate_pose(orientation, yaw, valid_mask)

        confidence = self._compute_confidence(
            valid_mask, yaw, roll, orientation
        )

        return PoseResult(
            orientation=orientation,
            yaw_angle_deg=yaw,
            pitch_angle_deg=pitch,
            roll_angle_deg=roll,
            is_valid_pose=is_valid,
            normalized_keypoints=normalized_kp,
            original_keypoints=keypoints,
            transform_matrix=M,
            foreshortening_factor=foreshortening,
            confidence=confidence,
        )

    # ── Orientation Detection ────────────────────────────────────────────

    def _estimate_orientation(
        self, kp: np.ndarray, valid: np.ndarray
    ) -> Tuple[str, float]:
        """
        Determine if the cow is in lateral, frontal, dorsal, or oblique view.
        Uses the shoulder-to-shoulder vs shoulder-to-hip ratio.
        """
        has_shoulders = valid[self.LEFT_SHOULDER] and valid[self.RIGHT_SHOULDER]
        has_hips = valid[self.LEFT_HIP] and valid[self.RIGHT_HIP]
        has_spine = valid[self.WITHERS] and valid[self.TAIL_HEAD]

        if has_shoulders and has_spine:
            shoulder_width = np.linalg.norm(
                kp[self.LEFT_SHOULDER] - kp[self.RIGHT_SHOULDER]
            )
            body_length = np.linalg.norm(
                kp[self.WITHERS] - kp[self.TAIL_HEAD]
            )

            if body_length < 1e-6:
                return "unknown", 90.0

            ratio = shoulder_width / body_length

            # Perfect lateral: shoulders overlap → ratio ≈ 0
            # Perfect frontal: shoulders spread, body short → ratio >> 1
            if ratio < 0.2:
                yaw = ratio * 90.0 / 0.2  # map [0, 0.2] → [0°, 90°]
                yaw = min(yaw, 15.0)
                return "lateral", yaw
            elif ratio < 0.5:
                yaw = 15.0 + (ratio - 0.2) * 30.0 / 0.3
                return "oblique", yaw
            elif ratio < 1.0:
                yaw = 45.0 + (ratio - 0.5) * 45.0 / 0.5
                return "frontal", yaw
            else:
                return "frontal", 90.0

        elif has_spine:
            # If we have withers→tail but no shoulder pair, assume lateral
            return "lateral", 5.0

        return "unknown", 45.0

    def _estimate_roll(self, kp: np.ndarray, valid: np.ndarray) -> float:
        """
        Estimate roll angle from the spine line (withers → tail_head).
        A horizontal spine = 0° roll.
        """
        if valid[self.WITHERS] and valid[self.TAIL_HEAD]:
            dx = kp[self.TAIL_HEAD][0] - kp[self.WITHERS][0]
            dy = kp[self.TAIL_HEAD][1] - kp[self.WITHERS][1]
            angle = np.degrees(np.arctan2(dy, dx))
            return angle
        return 0.0

    def _estimate_pitch(self, kp: np.ndarray, valid: np.ndarray) -> float:
        """
        Estimate pitch using leg-length asymmetry (front vs back legs).
        If front legs appear longer, camera is pitched up, etc.
        """
        front_leg_len = 0.0
        back_leg_len = 0.0
        count_front = 0
        count_back = 0

        if valid[self.LEFT_SHOULDER] and valid[self.LEFT_KNEE_FRONT]:
            front_leg_len += np.linalg.norm(
                kp[self.LEFT_SHOULDER] - kp[self.LEFT_KNEE_FRONT]
            )
            count_front += 1
        if valid[self.RIGHT_SHOULDER] and valid[self.RIGHT_KNEE_FRONT]:
            front_leg_len += np.linalg.norm(
                kp[self.RIGHT_SHOULDER] - kp[self.RIGHT_KNEE_FRONT]
            )
            count_front += 1
        if valid[self.LEFT_HIP] and valid[self.LEFT_HOCK]:
            back_leg_len += np.linalg.norm(
                kp[self.LEFT_HIP] - kp[self.LEFT_HOCK]
            )
            count_back += 1
        if valid[self.RIGHT_HIP] and valid[self.RIGHT_HOCK]:
            back_leg_len += np.linalg.norm(
                kp[self.RIGHT_HIP] - kp[self.RIGHT_HOCK]
            )
            count_back += 1

        if count_front > 0 and count_back > 0:
            avg_front = front_leg_len / count_front
            avg_back = back_leg_len / count_back
            if avg_back > 1e-6:
                ratio = avg_front / avg_back
                # ratio > 1 → front legs look longer → camera pitched down
                pitch = (ratio - 1.0) * 30.0  # rough linear mapping
                return np.clip(pitch, -30.0, 30.0)

        return 0.0

    # ── Foreshortening Correction ────────────────────────────────────────

    def _compute_foreshortening(self, yaw_deg: float) -> float:
        """
        Compute foreshortening correction factor.
        When viewing at an angle, the apparent length is shortened by cos(θ).
        We divide by cos(yaw) to recover the true length.
        """
        yaw_rad = np.radians(min(abs(yaw_deg), 80.0))  # cap at 80°
        cos_yaw = np.cos(yaw_rad)
        if cos_yaw < 0.15:
            cos_yaw = 0.15  # prevent extreme correction
        return 1.0 / cos_yaw

    # ── Transform Construction ───────────────────────────────────────────

    def _build_transform(
        self,
        kp: np.ndarray,
        valid: np.ndarray,
        roll_deg: float,
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build an affine transform that:
          1. Rotates the image to level the spine (correct roll).
          2. Centers the cow in the normalized canvas.

        Returns (3×3 matrix, normalized keypoints).
        """
        h, w = image_shape
        center = np.array([w / 2.0, h / 2.0])

        # Use cow centroid as rotation center
        valid_kp = kp[valid]
        if len(valid_kp) > 0:
            centroid = valid_kp.mean(axis=0)
        else:
            centroid = center

        # Rotation matrix to correct roll
        M = cv2.getRotationMatrix2D(
            (float(centroid[0]), float(centroid[1])),
            -roll_deg,  # Negate to de-rotate
            1.0,
        )

        # Apply to keypoints
        ones = np.ones((kp.shape[0], 1))
        kp_h = np.hstack([kp, ones])  # (N, 3)
        normalized_kp = (M @ kp_h.T).T  # (N, 2)

        # Extend M to 3×3
        M_3x3 = np.eye(3)
        M_3x3[:2, :] = M

        return M_3x3, normalized_kp

    # ── Validation ───────────────────────────────────────────────────────

    def _validate_pose(
        self, orientation: str, yaw: float, valid: np.ndarray
    ) -> bool:
        """Check if the pose is suitable for dimension measurement."""
        # Need at least 8 valid keypoints
        if valid.sum() < 8:
            return False

        # Must have core keypoints
        core_kps = [self.WITHERS, self.TAIL_HEAD, self.BRISKET]
        if not all(valid[k] for k in core_kps):
            return False

        # Prefer lateral view (yaw < tolerance)
        if orientation == "lateral" and abs(yaw) < self.cfg.angle_tolerance:
            return True

        # Accept oblique if yaw is moderate
        if orientation == "oblique" and abs(yaw) < 45.0:
            return True

        return False

    def _compute_confidence(
        self,
        valid: np.ndarray,
        yaw: float,
        roll: float,
        orientation: str,
    ) -> float:
        """Compute an overall pose-normalization confidence score."""
        # Keypoint coverage score
        kp_score = valid.sum() / len(valid)

        # Orientation score (lateral is best)
        orientation_scores = {
            "lateral": 1.0,
            "oblique": 0.6,
            "frontal": 0.3,
            "dorsal": 0.4,
            "unknown": 0.1,
        }
        orient_score = orientation_scores.get(orientation, 0.1)

        # Angle penalty
        yaw_penalty = max(0, 1.0 - abs(yaw) / 90.0)
        roll_penalty = max(0, 1.0 - abs(roll) / 45.0)

        confidence = (
            0.3 * kp_score
            + 0.3 * orient_score
            + 0.2 * yaw_penalty
            + 0.2 * roll_penalty
        )
        return round(np.clip(confidence, 0.0, 1.0), 3)

    # ── Utility ──────────────────────────────────────────────────────────

    def apply_transform_to_image(
        self, image: np.ndarray, transform: np.ndarray
    ) -> np.ndarray:
        """Apply the normalization transform to the full image."""
        h, w = image.shape[:2]
        M_2x3 = transform[:2, :]
        return cv2.warpAffine(image, M_2x3, (w, h))

    def apply_transform_to_mask(
        self, mask: np.ndarray, transform: np.ndarray
    ) -> np.ndarray:
        """Apply the normalization transform to a segmentation mask."""
        h, w = mask.shape[:2]
        M_2x3 = transform[:2, :]
        return cv2.warpAffine(mask, M_2x3, (w, h), flags=cv2.INTER_NEAREST)

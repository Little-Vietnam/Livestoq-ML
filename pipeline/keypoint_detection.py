"""
Keypoint Detection Module
=========================
Detects anatomical keypoints on the cow's body for dimension measurement.

Uses a pre-trained animal pose estimation model (MMPose / ViTPose) or
a heuristic skeleton estimation from the segmentation contour.

17 Keypoints:
    0: nose, 1: left_ear, 2: right_ear, 3: poll, 4: withers,
    5: spine_mid, 6: hip_point, 7: tail_head, 8: left_shoulder,
    9: right_shoulder, 10: left_hip, 11: right_hip,
    12: left_knee_front, 13: right_knee_front, 14: left_hock,
    15: right_hock, 16: brisket
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class KeypointResult:
    """Result of keypoint detection."""
    keypoints: np.ndarray        # (N, 2) pixel coordinates
    confidences: np.ndarray      # (N,) confidence scores per keypoint
    num_detected: int            # Number of keypoints with confidence > threshold
    method_used: str             # model | heuristic
    skeleton_edges: List[Tuple[int, int]]  # Edges connecting keypoints for visualization


# Skeleton connectivity for visualization
COW_SKELETON = [
    (0, 3),   # nose → poll
    (3, 1),   # poll → left_ear
    (3, 2),   # poll → right_ear
    (3, 4),   # poll → withers
    (4, 5),   # withers → spine_mid
    (5, 6),   # spine_mid → hip_point
    (6, 7),   # hip_point → tail_head
    (4, 8),   # withers → left_shoulder
    (4, 9),   # withers → right_shoulder
    (6, 10),  # hip_point → left_hip
    (6, 11),  # hip_point → right_hip
    (8, 12),  # left_shoulder → left_knee_front
    (9, 13),  # right_shoulder → right_knee_front
    (10, 14), # left_hip → left_hock
    (11, 15), # right_hip → right_hock
    (4, 16),  # withers → brisket
    (8, 16),  # left_shoulder → brisket
]


class KeypointDetector:
    """
    Detects anatomical keypoints on a cow using either:
      - A deep-learning pose model (preferred)
      - Heuristic contour analysis (fallback)
    """

    # Keypoint name-to-index mapping
    KP = {
        'nose': 0, 'left_ear': 1, 'right_ear': 2, 'poll': 3,
        'withers': 4, 'spine_mid': 5, 'hip_point': 6, 'tail_head': 7,
        'left_shoulder': 8, 'right_shoulder': 9, 'left_hip': 10,
        'right_hip': 11, 'left_knee_front': 12, 'right_knee_front': 13,
        'left_hock': 14, 'right_hock': 15, 'brisket': 16,
    }

    def __init__(self, kp_config, device: str = "cpu"):
        self.cfg = kp_config
        self.device = device
        self._model = None

    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> KeypointResult:
        """
        Detect cow anatomical keypoints.

        Parameters
        ----------
        image : np.ndarray  – BGR input image.
        mask  : np.ndarray  – Binary cow mask (H, W).
        bbox  : tuple       – Bounding box (x1, y1, x2, y2).

        Returns
        -------
        KeypointResult
        """
        # Try DL model first
        result = self._detect_with_model(image, bbox)
        if result is not None and result.num_detected >= 8:
            return result

        # Fallback: heuristic from contour
        return self._detect_heuristic(image, mask, bbox)

    # ── DL-based Detection ───────────────────────────────────────────────

    def _load_model(self):
        """Attempt to load an animal pose model."""
        try:
            from ultralytics import YOLO
            # YOLOv8 pose model (if trained on animal keypoints)
            # In practice you'd use a custom-trained model;
            # we try the standard pose model as a starting point
            self._model = YOLO("yolov8x-pose.pt")
            print("[Keypoints] Loaded YOLOv8 pose model")
        except Exception as e:
            print(f"[Keypoints] Could not load pose model: {e}")
            self._model = None

    def _detect_with_model(
        self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]
    ) -> Optional[KeypointResult]:
        """Detect keypoints using a DL model."""
        if self._model is None:
            self._load_model()
        if self._model is None:
            return None

        try:
            results = self._model(image, verbose=False)
            for result in results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    kps = result.keypoints.xy[0].cpu().numpy()        # (K, 2)
                    confs = result.keypoints.conf[0].cpu().numpy()    # (K,)

                    # Map COCO human keypoints → cow keypoints heuristically
                    cow_kps, cow_confs = self._map_to_cow_keypoints(kps, confs)

                    num_detected = int((cow_confs >= self.cfg.confidence_threshold).sum())
                    return KeypointResult(
                        keypoints=cow_kps,
                        confidences=cow_confs,
                        num_detected=num_detected,
                        method_used="model",
                        skeleton_edges=COW_SKELETON,
                    )
        except Exception as e:
            print(f"[Keypoints] Model inference failed: {e}")

        return None

    def _map_to_cow_keypoints(
        self, kps: np.ndarray, confs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map generic pose keypoints to cow anatomical points.
        In practice, a custom model trained on cow keypoints is preferred.
        This provides a rough mapping from COCO human pose.
        """
        n = self.cfg.num_keypoints
        cow_kps = np.zeros((n, 2), dtype=np.float32)
        cow_confs = np.zeros(n, dtype=np.float32)

        # Direct copy with heuristic mapping (limited accuracy)
        if len(kps) >= 5:
            cow_kps[0] = kps[0]   # nose
            cow_confs[0] = confs[0]

        return cow_kps, cow_confs

    # ── Heuristic Contour-Based Detection ────────────────────────────────

    def _detect_heuristic(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> KeypointResult:
        """
        Estimate keypoints from the cow's contour shape using geometric
        heuristics. Works well for lateral views.

        Strategy:
          - Find the bounding box and contour extremes
          - Use topological analysis of the contour shape
          - Place keypoints at anatomically plausible positions
        """
        n = self.cfg.num_keypoints
        keypoints = np.zeros((n, 2), dtype=np.float32)
        confidences = np.zeros(n, dtype=np.float32)

        if mask is None:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
            else:
                # No mask, no bbox – create a rough mask
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[h // 6 : 5 * h // 6, w // 8 : 7 * w // 8] = 255

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return KeypointResult(
                keypoints=keypoints,
                confidences=confidences,
                num_detected=0,
                method_used="heuristic",
                skeleton_edges=COW_SKELETON,
            )

        contour = max(contours, key=cv2.contourArea)
        pts = contour.reshape(-1, 2).astype(np.float32)

        # Bounding box of contour
        x, y, bw, bh = cv2.boundingRect(contour)
        cx, cy = x + bw / 2, y + bh / 2

        # ── Extreme points ───────────────────────────────────────────
        leftmost = pts[pts[:, 0].argmin()]
        rightmost = pts[pts[:, 0].argmax()]
        topmost = pts[pts[:, 1].argmin()]
        bottommost = pts[pts[:, 1].argmax()]

        # Determine head side: the narrower extreme is likely the head
        # Split contour vertically into left-half and right-half
        left_pts = pts[pts[:, 0] < cx]
        right_pts = pts[pts[:, 0] >= cx]

        left_height_range = (left_pts[:, 1].max() - left_pts[:, 1].min()) if len(left_pts) > 2 else bh
        right_height_range = (right_pts[:, 1].max() - right_pts[:, 1].min()) if len(right_pts) > 2 else bh

        head_is_left = left_height_range < right_height_range

        if head_is_left:
            nose_x, tail_x = leftmost[0], rightmost[0]
        else:
            nose_x, tail_x = rightmost[0], leftmost[0]

        # ── Place keypoints ──────────────────────────────────────────
        spine_y = y + bh * 0.25       # Spine is roughly in the upper quarter
        belly_y = y + bh * 0.75       # Belly line lower quarter
        mid_y = y + bh * 0.5

        # Nose (0)
        nose_region = pts[np.abs(pts[:, 0] - nose_x) < bw * 0.1]
        if len(nose_region) > 0:
            keypoints[0] = [nose_x, nose_region[:, 1].mean()]
        else:
            keypoints[0] = [nose_x, mid_y]
        confidences[0] = 0.7

        # Ears (1, 2) – above and to the side of the nose
        ear_x = nose_x + (0.08 * bw if head_is_left else -0.08 * bw)
        keypoints[1] = [ear_x, spine_y - bh * 0.05]  # left ear (upper)
        keypoints[2] = [ear_x, spine_y + bh * 0.05]  # right ear (lower)
        confidences[1] = 0.5
        confidences[2] = 0.5

        # Poll (3) – top of head
        keypoints[3] = [ear_x, spine_y - bh * 0.08]
        confidences[3] = 0.5

        # Withers (4) – top of the shoulder area
        withers_x = nose_x + (0.25 * bw if head_is_left else -0.25 * bw)
        # Find topmost point near this x
        near_withers = pts[np.abs(pts[:, 0] - withers_x) < bw * 0.1]
        if len(near_withers) > 0:
            keypoints[4] = [withers_x, near_withers[:, 1].min()]
        else:
            keypoints[4] = [withers_x, y]
        confidences[4] = 0.7

        # Spine mid (5)
        keypoints[5] = [cx, spine_y]
        confidences[5] = 0.6

        # Hip point (6) – rear top
        hip_x = tail_x + (-0.2 * bw if head_is_left else 0.2 * bw)
        near_hip = pts[np.abs(pts[:, 0] - hip_x) < bw * 0.1]
        if len(near_hip) > 0:
            keypoints[6] = [hip_x, near_hip[:, 1].min()]
        else:
            keypoints[6] = [hip_x, y + bh * 0.15]
        confidences[6] = 0.6

        # Tail head (7)
        keypoints[7] = [tail_x, y + bh * 0.3]
        confidences[7] = 0.7

        # Shoulders (8, 9) – left/right of withers
        keypoints[8] = [withers_x, keypoints[4][1] + bh * 0.15]  # left shoulder
        keypoints[9] = [withers_x, keypoints[4][1] + bh * 0.20]  # right shoulder
        confidences[8] = 0.5
        confidences[9] = 0.5

        # Hips (10, 11)
        keypoints[10] = [hip_x, keypoints[6][1] + bh * 0.15]
        keypoints[11] = [hip_x, keypoints[6][1] + bh * 0.20]
        confidences[10] = 0.5
        confidences[11] = 0.5

        # Front knees (12, 13) – below shoulders
        knee_y = y + bh * 0.70
        keypoints[12] = [withers_x - bw * 0.02, knee_y]
        keypoints[13] = [withers_x + bw * 0.02, knee_y]
        confidences[12] = 0.5
        confidences[13] = 0.5

        # Hocks (14, 15) – below hips
        hock_y = y + bh * 0.72
        keypoints[14] = [hip_x - bw * 0.02, hock_y]
        keypoints[15] = [hip_x + bw * 0.02, hock_y]
        confidences[14] = 0.5
        confidences[15] = 0.5

        # Brisket (16) – lowest point of chest
        brisket_x = withers_x + (0.05 * bw if head_is_left else -0.05 * bw)
        near_brisket = pts[np.abs(pts[:, 0] - brisket_x) < bw * 0.15]
        if len(near_brisket) > 0:
            keypoints[16] = [brisket_x, near_brisket[:, 1].max()]
        else:
            keypoints[16] = [brisket_x, y + bh * 0.85]
        confidences[16] = 0.6

        num_detected = int((confidences >= self.cfg.confidence_threshold).sum())

        return KeypointResult(
            keypoints=keypoints,
            confidences=confidences,
            num_detected=num_detected,
            method_used="heuristic",
            skeleton_edges=COW_SKELETON,
        )

    # ── Visualization ────────────────────────────────────────────────────

    @staticmethod
    def draw_keypoints(
        image: np.ndarray,
        result: KeypointResult,
        kp_names: List[str],
        min_conf: float = 0.3,
    ) -> np.ndarray:
        """Draw detected keypoints and skeleton on the image."""
        vis = image.copy()

        # Draw skeleton edges
        for i, j in result.skeleton_edges:
            if (result.confidences[i] >= min_conf and
                    result.confidences[j] >= min_conf):
                pt1 = tuple(result.keypoints[i].astype(int))
                pt2 = tuple(result.keypoints[j].astype(int))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for idx, (kp, conf) in enumerate(
            zip(result.keypoints, result.confidences)
        ):
            if conf >= min_conf:
                pt = tuple(kp.astype(int))
                color = (0, 0, 255) if conf >= 0.6 else (0, 165, 255)
                cv2.circle(vis, pt, 6, color, -1)
                cv2.circle(vis, pt, 8, (255, 255, 255), 1)
                label = kp_names[idx] if idx < len(kp_names) else str(idx)
                cv2.putText(
                    vis, f"{label}", (pt[0] + 10, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                )

        return vis

"""
Dimension Extraction Module
============================
Extracts the 8 standard body measurements from normalized keypoints
and segmentation contour, converting pixel distances to real-world
centimeters using the distance estimation scale factor.

Standard Measurements:
  1. Body Length        – Withers → Tail head (along spine)
  2. Body Width         – Maximum lateral width (dorsal view or contour)
  3. Tube Girth         – Circumference of the barrel area
  4. Body Height        – Ground → Withers (vertical)
  5. Chest Width        – Width across the chest (frontal view)
  6. Abdominal Girth    – Circumference of the abdomen
  7. Chest Depth        – Withers → Brisket (vertical)
  8. Chest Girth        – Circumference around the chest (heart girth)
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DimensionResult:
    """All 8 body measurements plus metadata."""
    # Measurements in centimeters
    body_length_cm: float = 0.0
    body_width_cm: float = 0.0
    tube_girth_cm: float = 0.0
    body_height_cm: float = 0.0
    chest_width_cm: float = 0.0
    abdominal_girth_cm: float = 0.0
    chest_depth_cm: float = 0.0
    chest_girth_cm: float = 0.0

    # Pixel measurements (before conversion)
    body_length_px: float = 0.0
    body_width_px: float = 0.0
    body_height_px: float = 0.0
    chest_depth_px: float = 0.0

    # Conversion factor used
    pixels_per_meter: float = 0.0
    foreshortening_factor: float = 1.0

    # Confidence per measurement
    measurement_confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Return measurements as a flat dictionary."""
        return {
            "body_length_cm": round(self.body_length_cm, 1),
            "body_width_cm": round(self.body_width_cm, 1),
            "tube_girth_cm": round(self.tube_girth_cm, 1),
            "body_height_cm": round(self.body_height_cm, 1),
            "chest_width_cm": round(self.chest_width_cm, 1),
            "abdominal_girth_cm": round(self.abdominal_girth_cm, 1),
            "chest_depth_cm": round(self.chest_depth_cm, 1),
            "chest_girth_cm": round(self.chest_girth_cm, 1),
        }


class DimensionExtractor:
    """
    Extracts real-world body dimensions from keypoints, contour, and
    calibration data.
    """

    # Keypoint indices
    NOSE = 0
    WITHERS = 4
    SPINE_MID = 5
    HIP_POINT = 6
    TAIL_HEAD = 7
    LEFT_SHOULDER = 8
    RIGHT_SHOULDER = 9
    LEFT_HIP = 10
    RIGHT_HIP = 11
    BRISKET = 16

    def __init__(self):
        pass

    def extract(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        contour: np.ndarray,
        mask: np.ndarray,
        pixels_per_meter: float,
        foreshortening_factor: float = 1.0,
        confidence_threshold: float = 0.3,
    ) -> DimensionResult:
        """
        Extract all 8 standard body measurements.

        Parameters
        ----------
        keypoints : np.ndarray          – (N, 2) normalized keypoints.
        confidences : np.ndarray        – (N,) keypoint confidences.
        contour : np.ndarray            – Cow contour from segmentation.
        mask : np.ndarray               – Binary cow mask (H, W).
        pixels_per_meter : float        – Scale from distance estimation.
        foreshortening_factor : float   – Correction for oblique view.
        confidence_threshold : float    – Min keypoint confidence.

        Returns
        -------
        DimensionResult
        """
        result = DimensionResult(
            pixels_per_meter=pixels_per_meter,
            foreshortening_factor=foreshortening_factor,
        )

        def kp_valid(idx):
            return confidences[idx] >= confidence_threshold

        def px_to_cm(px_val):
            """Convert pixel distance to centimeters."""
            if pixels_per_meter <= 0:
                return 0.0
            meters = (px_val / pixels_per_meter) * foreshortening_factor
            return meters * 100.0  # → cm

        # ── 1. Body Length ───────────────────────────────────────────────
        if kp_valid(self.WITHERS) and kp_valid(self.TAIL_HEAD):
            body_len_px = self._spine_length(keypoints)
            result.body_length_px = body_len_px
            result.body_length_cm = px_to_cm(body_len_px)
            result.measurement_confidence["body_length"] = min(
                confidences[self.WITHERS], confidences[self.TAIL_HEAD]
            )

        # ── 2. Body Width ───────────────────────────────────────────────
        body_width_px = self._measure_body_width(keypoints, confidences, contour, mask)
        result.body_width_px = body_width_px
        result.body_width_cm = px_to_cm(body_width_px)
        result.measurement_confidence["body_width"] = 0.5

        # ── 3. Tube Girth ───────────────────────────────────────────────
        # Approximated from body width using elliptical cross-section model
        # Tube girth ≈ π × √(2 × (a² + b²)) where a = width/2, b ≈ depth/2
        tube_girth_px = self._estimate_girth_at_position(
            contour, mask, keypoints, self.SPINE_MID, kp_valid(self.SPINE_MID)
        )
        result.tube_girth_cm = px_to_cm(tube_girth_px)
        result.measurement_confidence["tube_girth"] = 0.4

        # ── 4. Body Height ──────────────────────────────────────────────
        if kp_valid(self.WITHERS):
            body_height_px = self._measure_body_height(
                keypoints, contour, mask
            )
            result.body_height_px = body_height_px
            result.body_height_cm = px_to_cm(body_height_px)
            result.measurement_confidence["body_height"] = confidences[self.WITHERS]

        # ── 5. Chest Width ──────────────────────────────────────────────
        if kp_valid(self.LEFT_SHOULDER) and kp_valid(self.RIGHT_SHOULDER):
            chest_w_px = np.linalg.norm(
                keypoints[self.LEFT_SHOULDER] - keypoints[self.RIGHT_SHOULDER]
            )
            result.chest_width_cm = px_to_cm(chest_w_px)
        else:
            # Estimate from contour at shoulder x-position
            chest_w_px = self._measure_width_at_x(
                contour, mask, keypoints[self.WITHERS][0] if kp_valid(self.WITHERS) else None
            )
            result.chest_width_cm = px_to_cm(chest_w_px)
        result.measurement_confidence["chest_width"] = 0.4

        # ── 6. Abdominal Girth ──────────────────────────────────────────
        abd_girth_px = self._estimate_girth_at_position(
            contour, mask, keypoints, self.SPINE_MID, kp_valid(self.SPINE_MID),
            position_fraction=0.6,  # Slightly behind mid-body
        )
        result.abdominal_girth_cm = px_to_cm(abd_girth_px)
        result.measurement_confidence["abdominal_girth"] = 0.4

        # ── 7. Chest Depth ──────────────────────────────────────────────
        if kp_valid(self.WITHERS) and kp_valid(self.BRISKET):
            chest_depth_px = abs(
                keypoints[self.BRISKET][1] - keypoints[self.WITHERS][1]
            )
            result.chest_depth_px = chest_depth_px
            result.chest_depth_cm = px_to_cm(chest_depth_px)
            result.measurement_confidence["chest_depth"] = min(
                confidences[self.WITHERS], confidences[self.BRISKET]
            )
        else:
            # Estimate from contour vertical span at shoulder position
            chest_depth_px = self._measure_vertical_span(
                contour, mask, keypoints[self.WITHERS][0] if kp_valid(self.WITHERS) else None
            )
            result.chest_depth_px = chest_depth_px
            result.chest_depth_cm = px_to_cm(chest_depth_px)
            result.measurement_confidence["chest_depth"] = 0.35

        # ── 8. Chest Girth (Heart Girth) ────────────────────────────────
        # Most important measurement for weight estimation
        # Approximated using elliptical model: C = π × √(2(a²+b²))
        # where a = chest_width/2, b = chest_depth/2
        if result.chest_depth_cm > 0:
            a = result.chest_width_cm / 2.0 if result.chest_width_cm > 0 else result.chest_depth_cm * 0.4
            b = result.chest_depth_cm / 2.0
            chest_girth_cm = np.pi * np.sqrt(2 * (a ** 2 + b ** 2))
            result.chest_girth_cm = chest_girth_cm
        else:
            cg_px = self._estimate_girth_at_position(
                contour, mask, keypoints, self.WITHERS, kp_valid(self.WITHERS),
                position_fraction=0.25,
            )
            result.chest_girth_cm = px_to_cm(cg_px)
        result.measurement_confidence["chest_girth"] = 0.45

        return result

    # ── Measurement Helpers ──────────────────────────────────────────────

    def _spine_length(self, kp: np.ndarray) -> float:
        """
        Measure body length along the spine: withers → spine_mid → hip → tail.
        Uses polyline distance for a more accurate path.
        """
        spine_points = [
            kp[self.WITHERS],
            kp[self.SPINE_MID],
            kp[self.HIP_POINT],
            kp[self.TAIL_HEAD],
        ]
        total = 0.0
        for i in range(len(spine_points) - 1):
            total += np.linalg.norm(spine_points[i + 1] - spine_points[i])
        return total

    def _measure_body_height(
        self,
        kp: np.ndarray,
        contour: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        Body height = withers to ground.
        'Ground' is estimated as the lowest point of the mask near the legs.
        """
        withers_y = kp[self.WITHERS][1]

        # Find the lowest visible point of the cow (feet / ground contact)
        if mask is not None and mask.any():
            ys = np.where(mask > 0)[0]
            ground_y = ys.max()
        elif len(contour) > 0:
            ground_y = contour.reshape(-1, 2)[:, 1].max()
        else:
            ground_y = withers_y + 200  # fallback

        return abs(ground_y - withers_y)

    def _measure_body_width(
        self,
        kp: np.ndarray,
        confidences: np.ndarray,
        contour: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        Body width from the contour at the widest horizontal cross-section.
        In lateral view, this is approximated from the contour's max
        vertical extent at the barrel region.
        """
        if len(contour) == 0:
            return 0.0

        pts = contour.reshape(-1, 2)

        # Find the widest vertical cross-section (top-to-bottom span)
        # Sample multiple x positions along the body
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        max_span = 0.0
        n_samples = 50

        for i in range(n_samples):
            x = x_min + (x_max - x_min) * (i / n_samples)
            col_pts = pts[np.abs(pts[:, 0] - x) < 5]
            if len(col_pts) >= 2:
                span = col_pts[:, 1].max() - col_pts[:, 1].min()
                max_span = max(max_span, span)

        # In lateral view, maximum contour height at barrel ≈ body width
        # This is a rough approximation; true width needs dorsal view
        return max_span * 0.45  # Heuristic: width ≈ 45% of lateral height

    def _measure_width_at_x(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        x_pos: Optional[float],
    ) -> float:
        """Measure the vertical span of the contour at a given x position."""
        if len(contour) == 0 or x_pos is None:
            return 0.0

        pts = contour.reshape(-1, 2)
        near = pts[np.abs(pts[:, 0] - x_pos) < 10]
        if len(near) >= 2:
            return near[:, 1].max() - near[:, 1].min()
        return 0.0

    def _measure_vertical_span(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        x_pos: Optional[float],
    ) -> float:
        """Measure vertical extent at a given x position."""
        return self._measure_width_at_x(contour, mask, x_pos)

    def _estimate_girth_at_position(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        kp: np.ndarray,
        kp_idx: int,
        kp_is_valid: bool,
        position_fraction: float = 0.5,
    ) -> float:
        """
        Estimate circumference (girth) at a body cross-section.

        Uses an elliptical model: measure the height (depth) and estimated
        width at a given body position, then compute the ellipse perimeter.

        C ≈ π × √(2(a² + b²))   (Ramanujan approximation)
        """
        if len(contour) == 0:
            return 0.0

        pts = contour.reshape(-1, 2)

        # Determine x position for the cross-section
        if kp_is_valid:
            x_pos = kp[kp_idx][0]
        else:
            x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
            x_pos = x_min + (x_max - x_min) * position_fraction

        # Measure cross-section height at this x
        near = pts[np.abs(pts[:, 0] - x_pos) < 15]
        if len(near) < 2:
            return 0.0

        height = near[:, 1].max() - near[:, 1].min()

        # Estimate width as a fraction of height (typical cattle cross-section)
        # Cattle barrel cross-section is roughly 60-70% as wide as it is tall
        width = height * 0.65

        # Ellipse perimeter (Ramanujan)
        a = height / 2.0
        b = width / 2.0
        girth = np.pi * np.sqrt(2 * (a ** 2 + b ** 2))

        return girth

    # ── Visualization ────────────────────────────────────────────────────

    @staticmethod
    def draw_measurements(
        image: np.ndarray,
        keypoints: np.ndarray,
        result: DimensionResult,
        kp_confidences: np.ndarray,
        min_conf: float = 0.3,
    ) -> np.ndarray:
        """Draw measurement lines on the image."""
        vis = image.copy()
        WITHERS, TAIL_HEAD, BRISKET = 4, 7, 16

        def pt(idx):
            return tuple(keypoints[idx].astype(int))

        def draw_line(p1, p2, label, color=(0, 255, 255)):
            cv2.line(vis, p1, p2, color, 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(vis, label, (mid[0] + 5, mid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Body length
        if kp_confidences[WITHERS] >= min_conf and kp_confidences[TAIL_HEAD] >= min_conf:
            draw_line(pt(WITHERS), pt(TAIL_HEAD),
                      f"BL: {result.body_length_cm:.0f}cm", (0, 255, 255))

        # Chest depth
        if kp_confidences[WITHERS] >= min_conf and kp_confidences[BRISKET] >= min_conf:
            draw_line(pt(WITHERS), pt(BRISKET),
                      f"CD: {result.chest_depth_cm:.0f}cm", (255, 0, 255))

        # Body height (withers to ground)
        if kp_confidences[WITHERS] >= min_conf:
            w = pt(WITHERS)
            ground = (w[0], vis.shape[0] - 20)
            draw_line(w, ground,
                      f"BH: {result.body_height_cm:.0f}cm", (255, 255, 0))

        # Add text overlay with all measurements
        y_offset = 30
        measurements = result.to_dict()
        for name, value in measurements.items():
            text = f"{name}: {value} cm"
            cv2.putText(vis, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 22

        return vis

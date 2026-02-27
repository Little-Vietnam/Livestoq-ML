"""
Age Prediction Module
======================
Predicts the age of cattle from dental (teeth) images using
morphological analysis of incisors.

Cattle Dentition – Age Estimation Reference
---------------------------------------------
  - **<12 months**: All deciduous (milk) incisors present — small, white,
    smooth-edged, evenly spaced.
  - **12–18 months**: First pair of permanent central incisors (I1) erupting
    — larger, wider, slightly yellowed compared to remaining milk teeth.
  - **18–24 months**: Central incisors (I1) fully erupted, second pair (I2)
    beginning to appear.
  - **24–30 months**: I1 and I2 permanent, third pair (I3) erupting.
  - **30–36 months**: Three pairs of permanent incisors; fourth pair (corner)
    still deciduous.
  - **36–42 months**: "Full mouth" — all 4 pairs of permanent incisors erupted.
  - **>42 months**: Wear patterns begin — scoring (levelling) of incisors.
  - **>60 months**: Significant wear, teeth may be "broken mouth" or missing.

Pipeline approach
-----------------
1. Pre-process teeth image (crop, enhance contrast, normalise).
2. Detect jaw region using segmentation / colour-thresholding.
3. Count visible teeth and classify deciduous vs permanent incisors by
   size, colour (whiteness), and edge sharpness.
4. Grade wear level (0 = unworn, 1 = slight, 2 = moderate, 3 = heavy).
5. Map the combination to an age bracket.

Since we do not ship a heavy dental segmentation model in this MVP, we
use a **rule-based heuristic** that analyses colour histograms, edge
density and aspect ratios of detected tooth-like blobs to approximate
the count and wear.  The model is designed to be replaced with a
fine-tuned CNN (EfficientNet / ResNet-50) in production.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AgeResult:
    """Result of teeth-based age prediction."""
    predicted_age_months: int          # point estimate
    age_range_months: Tuple[int, int]  # (min, max)
    dentition_stage: str               # e.g. "2-tooth", "4-tooth", etc.
    wear_grade: int                    # 0–3
    tooth_count: int                   # detected permanent incisors (0–8)
    confidence: float                  # 0–1
    details: Dict = field(default_factory=dict)


# ── Dentition stage rules ────────────────────────────────────────────────
_DENTITION_TABLE = [
    # (permanent_incisors, wear_grade) → (age_months_mid, age_range, stage_name)
    # Baby teeth only
    ((0, 0), (6, (3, 12), "milk-teeth")),
    # First pair erupting / erupted
    ((2, 0), (18, (12, 24), "2-tooth")),
    # Two pairs
    ((4, 0), (27, (24, 30), "4-tooth")),
    # Three pairs
    ((6, 0), (33, (30, 36), "6-tooth")),
    # Full mouth, minimal wear
    ((8, 0), (42, (36, 48), "full-mouth")),
    # Full mouth, moderate wear
    ((8, 1), (54, (48, 60), "full-mouth-worn")),
    # Full mouth, heavy wear
    ((8, 2), (72, (60, 84), "aged")),
    ((8, 3), (96, (84, 120), "old")),
]


class AgePredictor:
    """
    Predicts cattle age from a teeth / mouth image.
    """

    def __init__(self):
        pass

    def predict(self, teeth_image_path: str) -> AgeResult:
        """
        Predict the age from a teeth image.

        Parameters
        ----------
        teeth_image_path : str
            Path to the teeth / mouth image.

        Returns
        -------
        AgeResult
        """
        img = cv2.imread(teeth_image_path)
        if img is None:
            return self._default_result("Image could not be loaded")

        h, w = img.shape[:2]

        # ── Step 1: Pre-process ──────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # ── Step 2: Detect bright tooth-like regions ─────────────────
        # Teeth are typically the brightest objects in a mouth image.
        _, bright_mask = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ── Step 3: Find tooth blobs ─────────────────────────────────
        contours, _ = cv2.findContours(
            bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_tooth_area = (h * w) * 0.002  # at least 0.2 % of image area
        max_tooth_area = (h * w) * 0.15

        tooth_contours = [
            c for c in contours
            if min_tooth_area < cv2.contourArea(c) < max_tooth_area
        ]

        tooth_count = len(tooth_contours)

        # ── Step 4: Classify deciduous vs permanent ──────────────────
        # Permanent incisors are wider and slightly yellowed.
        permanent_count = 0
        avg_brightness = float(np.mean(enhanced))

        for cnt in tooth_contours:
            x, y, tw, th = cv2.boundingRect(cnt)
            aspect = tw / max(th, 1)
            roi = enhanced[y:y + th, x:x + tw]
            mean_val = float(np.mean(roi))

            # Permanent teeth tend to be wider (aspect > 0.6) and
            # slightly less white than deciduous (still > 170 though).
            if aspect >= 0.5 and mean_val > 160:
                permanent_count += 1

        # Clamp to even number (incisors come in pairs)
        permanent_count = min(permanent_count, 8)
        permanent_count = (permanent_count // 2) * 2  # round to pair

        # ── Step 5: Wear grade ───────────────────────────────────────
        # Worn teeth have smoother edges → lower edge density.
        edges = cv2.Canny(enhanced, 80, 200)
        tooth_edge_mask = cv2.bitwise_and(edges, edges, mask=bright_mask)
        edge_density = float(np.sum(tooth_edge_mask > 0)) / max(float(np.sum(bright_mask > 0)), 1)

        if edge_density > 0.20:
            wear_grade = 0  # sharp edges → young teeth
        elif edge_density > 0.12:
            wear_grade = 1
        elif edge_density > 0.06:
            wear_grade = 2
        else:
            wear_grade = 3

        # ── Step 6: Map to age bracket ───────────────────────────────
        best_match = None
        best_dist = float("inf")
        for (perm, wear), (age_mid, age_range, stage) in _DENTITION_TABLE:
            dist = abs(perm - permanent_count) * 5 + abs(wear - wear_grade) * 3
            if dist < best_dist:
                best_dist = dist
                best_match = (age_mid, age_range, stage)

        if best_match is None:
            return self._default_result("Could not match dentition")

        age_mid, age_range, stage = best_match

        # Confidence heuristic: based on how many teeth we detected and
        # how good the image quality is.
        quality_score = min(1.0, avg_brightness / 200)
        count_score = min(1.0, tooth_count / 4)  # at least 4 visible teeth
        confidence = round(0.4 * quality_score + 0.6 * count_score, 2)
        confidence = max(0.3, min(0.95, confidence))

        return AgeResult(
            predicted_age_months=age_mid,
            age_range_months=age_range,
            dentition_stage=stage,
            wear_grade=wear_grade,
            tooth_count=tooth_count,
            confidence=confidence,
            details={
                "permanent_incisors_detected": permanent_count,
                "edge_density": round(edge_density, 4),
                "avg_brightness": round(avg_brightness, 1),
                "tooth_blobs_found": tooth_count,
            },
        )

    @staticmethod
    def _default_result(reason: str) -> AgeResult:
        return AgeResult(
            predicted_age_months=0,
            age_range_months=(0, 0),
            dentition_stage="unknown",
            wear_grade=0,
            tooth_count=0,
            confidence=0.0,
            details={"error": reason},
        )

"""
Skin Disease Detection Module
===============================
Detects common cattle skin diseases and lesions from a lateral (side)
body image.

Targeted conditions
-------------------
1. **Lumpy Skin Disease (LSD)** – firm, round nodules 2–5 cm diameter,
   scattered across the body; affects hide texture significantly.
2. **Dermatophilosis** ("rain scald") – matted, crusty scabs forming
   paint-brush-like tufts, mostly on back and sides.
3. **Ringworm** (Dermatophytosis) – circular, grey, scaly patches
   typically 1–5 cm in diameter.
4. **Mange** (Sarcoptic / Demodectic) – hair loss, thickened/wrinkled
   skin, crusting, often around the neck, shoulders, and legs.
5. **Tick infestation** – visible dark spots / engorged ticks, hair loss
   in attachment areas.
6. **Healthy** – no detectable lesions.

Pipeline approach (MVP – image-analysis heuristic)
----------------------------------------------------
1. Segment the animal body (reuse existing segmentation mask).
2. Within the body mask, analyse texture anomalies:
   a. Detect circular/blob-like dark or light patches → lumps / ringworm.
   b. Measure texture roughness (Laplacian variance) → crusty / scaly.
   c. Detect regions of abnormal redness (BGR colour analysis).
   d. Count anomalous blobs and classify by morphology.
3. Score each condition and report findings.

Designed to be replaced by a fine-tuned classification model
(e.g., EfficientNet-B3 trained on cattle skin disease datasets)
in production.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DiseaseDetection:
    """A single detected condition."""
    name: str                    # disease / condition name
    confidence: float            # 0–1
    severity: str                # "none" | "mild" | "moderate" | "severe"
    affected_area_pct: float     # % of body area affected
    description: str             # human-readable description


@dataclass
class SkinDiseaseResult:
    """Full result of skin disease screening."""
    overall_status: str               # "healthy" | "suspect" | "diseased"
    overall_confidence: float         # 0–1
    conditions: List[DiseaseDetection]
    skin_quality_score: float         # 0–100 (100 = perfect skin)
    details: Dict = field(default_factory=dict)


# ── Disease profiles & thresholds ────────────────────────────────────────

_DISEASE_PROFILES = {
    "lumpy_skin_disease": {
        "display": "Lumpy Skin Disease",
        "blob_size_range": (15, 80),   # px diameter at 1080p
        "min_blobs": 3,
        "colour": "dark",
        "description": "Firm, round nodules scattered across the body, characteristic of Lumpy Skin Disease (LSD).",
    },
    "ringworm": {
        "display": "Ringworm (Dermatophytosis)",
        "blob_size_range": (20, 100),
        "min_blobs": 1,
        "colour": "grey",
        "description": "Circular, grey scaly patches indicating possible ringworm infection.",
    },
    "dermatophilosis": {
        "display": "Dermatophilosis",
        "blob_size_range": (10, 60),
        "min_blobs": 5,
        "colour": "crusty",
        "description": "Matted, crusty scabs forming tufts, indicative of dermatophilosis (rain scald).",
    },
    "mange": {
        "display": "Mange",
        "blob_size_range": (30, 200),
        "min_blobs": 1,
        "colour": "hairless",
        "description": "Hair loss and thickened/wrinkled skin patches suggesting mange infestation.",
    },
}


class SkinDiseaseDetector:
    """
    Detects skin diseases and lesions from a lateral body image.
    """

    def __init__(self):
        pass

    def detect(
        self,
        image_path: str,
        body_mask: Optional[np.ndarray] = None,
    ) -> SkinDiseaseResult:
        """
        Analyse a side-view image for skin diseases.

        Parameters
        ----------
        image_path : str
            Path to the lateral (side) image of the animal.
        body_mask : np.ndarray, optional
            Binary mask of the animal body (from segmentation).
            If None, uses the full image.

        Returns
        -------
        SkinDiseaseResult
        """
        img = cv2.imread(image_path)
        if img is None:
            return self._default_result("Image could not be loaded")

        h, w = img.shape[:2]

        # Apply body mask if provided
        if body_mask is not None:
            # Resize mask to match image if needed
            if body_mask.shape[:2] != (h, w):
                body_mask = cv2.resize(body_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masked = cv2.bitwise_and(img, img, mask=body_mask)
            analysis_mask = body_mask
        else:
            masked = img.copy()
            analysis_mask = np.ones((h, w), dtype=np.uint8) * 255

        body_area = float(np.sum(analysis_mask > 0))
        if body_area < 100:
            return self._default_result("Insufficient body area detected")

        # ── Analysis channels ────────────────────────────────────────
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)

        conditions: List[DiseaseDetection] = []

        # ── 1. Blob / nodule detection (LSD, ringworm) ──────────────
        blob_detections = self._detect_anomalous_blobs(
            gray, analysis_mask, h, w, body_area
        )
        conditions.extend(blob_detections)

        # ── 2. Texture roughness analysis (dermatophilosis, mange) ───
        texture_detections = self._analyse_texture(
            gray, analysis_mask, h, w, body_area
        )
        conditions.extend(texture_detections)

        # ── 3. Colour anomaly detection ──────────────────────────────
        colour_detections = self._analyse_colour_anomalies(
            hsv, lab, analysis_mask, body_area
        )
        conditions.extend(colour_detections)

        # ── Aggregate results ────────────────────────────────────────
        # Remove low-confidence detections
        # Scale all confidences down by 2/3 to reduce false positives
        conditions = [DiseaseDetection(
            name=c.name,
            confidence=round(c.confidence * (2 / 3), 3),
            severity=c.severity,
            affected_area_pct=c.affected_area_pct,
            description=c.description,
        ) for c in conditions]
        conditions = [c for c in conditions if c.confidence >= 0.25]

        # Calculate skin quality score
        if conditions:
            worst_conf = max(c.confidence for c in conditions)
            total_affected = sum(c.affected_area_pct for c in conditions)
            raw_quality = max(0, 100 - worst_conf * 50 - total_affected * 2)
        else:
            raw_quality = 95.0  # high quality, no issues
            worst_conf = 0.0

        # Scale score into 80–100 range so healthy animals consistently score well
        skin_quality = 80.0 + (raw_quality / 100.0) * 20.0

        # Overall status
        if worst_conf >= 0.6:
            overall_status = "diseased"
        elif worst_conf >= 0.35:
            overall_status = "suspect"
        else:
            overall_status = "healthy"

        overall_confidence = round(max(0.4, 1.0 - worst_conf * 0.5), 2) if not conditions else round(max(c.confidence for c in conditions), 2)
        if not conditions:
            overall_confidence = 0.85  # confident it's healthy

        return SkinDiseaseResult(
            overall_status=overall_status,
            overall_confidence=overall_confidence,
            conditions=conditions,
            skin_quality_score=round(skin_quality, 1),
            details={
                "body_area_px": int(body_area),
                "image_size": {"width": w, "height": h},
                "num_conditions_detected": len(conditions),
            },
        )

    # ── Detection methods ────────────────────────────────────────────────

    def _detect_anomalous_blobs(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        h: int,
        w: int,
        body_area: float,
    ) -> List[DiseaseDetection]:
        """Detect circular / blob-like anomalies (nodules, ringworm)."""
        detections = []

        # Adaptive threshold to find local anomalies
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray, blurred)

        # Threshold the difference
        _, anomaly_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        anomaly_mask = cv2.bitwise_and(anomaly_mask, mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by size
        min_blob = max(10, int(np.sqrt(body_area) * 0.01))
        max_blob = int(np.sqrt(body_area) * 0.15)

        valid_blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_blob * min_blob:
                continue
            if area > max_blob * max_blob:
                continue
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            if circularity > 0.3:  # reasonably circular
                valid_blobs.append((cnt, area, circularity))

        blob_count = len(valid_blobs)
        total_blob_area = sum(a for _, a, _ in valid_blobs)
        blob_area_pct = (total_blob_area / body_area) * 100

        # LSD: many small circular blobs
        if blob_count >= 3:
            conf = min(0.9, 0.3 + blob_count * 0.08 + blob_area_pct * 0.05)
            severity = "severe" if blob_count > 8 else "moderate" if blob_count > 5 else "mild"
            detections.append(DiseaseDetection(
                name="Lumpy Skin Disease",
                confidence=round(conf, 2),
                severity=severity,
                affected_area_pct=round(blob_area_pct, 1),
                description=f"Detected {blob_count} nodule-like formations across the body surface.",
            ))

        # Ringworm: fewer but larger circular patches
        large_blobs = [b for b in valid_blobs if b[1] > (min_blob * min_blob * 4)]
        if 1 <= len(large_blobs) <= 5:
            avg_circ = np.mean([c for _, _, c in large_blobs])
            if avg_circ > 0.5:
                conf = min(0.85, 0.25 + len(large_blobs) * 0.12 + avg_circ * 0.2)
                severity = "moderate" if len(large_blobs) > 2 else "mild"
                detections.append(DiseaseDetection(
                    name="Ringworm",
                    confidence=round(conf, 2),
                    severity=severity,
                    affected_area_pct=round(blob_area_pct, 1),
                    description=f"Found {len(large_blobs)} circular patch(es) with high circularity ({avg_circ:.2f}).",
                ))

        return detections

    def _analyse_texture(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        h: int,
        w: int,
        body_area: float,
    ) -> List[DiseaseDetection]:
        """Detect rough / crusty texture regions (dermatophilosis, mange)."""
        detections = []

        # Compute Laplacian (texture roughness)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = np.abs(lap).astype(np.uint8)

        # Apply body mask
        lap_masked = cv2.bitwise_and(lap_abs, lap_abs, mask=mask)

        # Calculate regional roughness by dividing into a grid
        grid_rows, grid_cols = 4, 6
        cell_h, cell_w = h // grid_rows, w // grid_cols
        rough_cells = 0
        total_cells = 0

        for r in range(grid_rows):
            for c in range(grid_cols):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                cell_mask = mask[y1:y2, x1:x2]
                if np.sum(cell_mask > 0) < (cell_h * cell_w * 0.3):
                    continue  # skip cells with little body coverage
                total_cells += 1
                cell_lap = lap_masked[y1:y2, x1:x2]
                roughness = float(np.mean(cell_lap[cell_mask > 0]))
                if roughness > 40:  # high texture roughness
                    rough_cells += 1

        if total_cells > 0:
            rough_ratio = rough_cells / total_cells

            # Dermatophilosis: widespread roughness
            if rough_ratio > 0.3:
                conf = min(0.8, 0.2 + rough_ratio * 0.6)
                severity = "severe" if rough_ratio > 0.6 else "moderate" if rough_ratio > 0.4 else "mild"
                detections.append(DiseaseDetection(
                    name="Dermatophilosis",
                    confidence=round(conf, 2),
                    severity=severity,
                    affected_area_pct=round(rough_ratio * 100, 1),
                    description=f"Rough/crusty texture detected in {rough_cells}/{total_cells} body regions.",
                ))

            # Mange: localised hair loss + roughness (detected as very high
            # local variance combined with low mean intensity)
            hairless_cells = 0
            for r in range(grid_rows):
                for c in range(grid_cols):
                    y1, y2 = r * cell_h, (r + 1) * cell_h
                    x1, x2 = c * cell_w, (c + 1) * cell_w
                    cell_mask = mask[y1:y2, x1:x2]
                    if np.sum(cell_mask > 0) < (cell_h * cell_w * 0.3):
                        continue
                    cell_gray = gray[y1:y2, x1:x2]
                    local_std = float(np.std(cell_gray[cell_mask > 0]))
                    local_mean = float(np.mean(cell_gray[cell_mask > 0]))
                    # Hairless regions often have higher std and different mean
                    if local_std > 35 and local_mean < 100:
                        hairless_cells += 1

            if hairless_cells >= 2:
                hair_ratio = hairless_cells / max(total_cells, 1)
                conf = min(0.75, 0.2 + hairless_cells * 0.1)
                severity = "moderate" if hairless_cells > 3 else "mild"
                detections.append(DiseaseDetection(
                    name="Mange",
                    confidence=round(conf, 2),
                    severity=severity,
                    affected_area_pct=round(hair_ratio * 100, 1),
                    description=f"Hair loss and skin texture changes in {hairless_cells} body regions.",
                ))

        return detections

    def _analyse_colour_anomalies(
        self,
        hsv: np.ndarray,
        lab: np.ndarray,
        mask: np.ndarray,
        body_area: float,
    ) -> List[DiseaseDetection]:
        """Detect colour anomalies — redness, unusual patches."""
        detections = []

        # Extract redness channel from LAB (a* channel, positive = red)
        a_channel = lab[:, :, 1].astype(np.float32)
        a_masked = cv2.bitwise_and(
            a_channel.astype(np.uint8),
            a_channel.astype(np.uint8),
            mask=mask,
        )

        body_pixels = mask > 0
        if np.sum(body_pixels) == 0:
            return detections

        a_values = a_channel[body_pixels]
        mean_a = float(np.mean(a_values))
        std_a = float(np.std(a_values))

        # Inflammation / redness: high a* values
        if mean_a > 140:  # LAB a* centre is 128; high values = reddish
            red_excess = (mean_a - 128) / max(std_a, 1)
            if red_excess > 1.5:
                red_area = float(np.sum(a_values > (mean_a + std_a))) / len(a_values) * 100
                conf = min(0.7, 0.2 + red_excess * 0.1)
                detections.append(DiseaseDetection(
                    name="Skin Inflammation",
                    confidence=round(conf, 2),
                    severity="mild" if red_excess < 2.5 else "moderate",
                    affected_area_pct=round(red_area, 1),
                    description="Abnormal redness detected, possibly indicating inflammation or irritation.",
                ))

        return detections

    @staticmethod
    def _default_result(reason: str) -> SkinDiseaseResult:
        return SkinDiseaseResult(
            overall_status="unknown",
            overall_confidence=0.0,
            conditions=[],
            skin_quality_score=0.0,
            details={"error": reason},
        )

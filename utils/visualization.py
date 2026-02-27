"""
Visualization Utilities
========================
Drawing helpers for the livestock measurement pipeline.
"""

import numpy as np
import cv2
from typing import Dict, Optional


def draw_pipeline_result(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    keypoints: Optional[np.ndarray],
    kp_confidences: Optional[np.ndarray],
    kp_names: list,
    dimensions: Optional[Dict],
    weight_kg: float,
    weight_range: tuple,
    distance_m: float,
    orientation: str,
    method_weights: Optional[Dict] = None,
    min_conf: float = 0.3,
) -> np.ndarray:
    """
    Draw a comprehensive visualization of all pipeline results on the image.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # ── 1. Overlay segmentation mask ────────────────────────────────────
    if mask is not None:
        overlay = vis.copy()
        color_mask = np.zeros_like(vis)
        color_mask[mask > 0] = (255, 180, 0)  # Cyan-ish overlay
        cv2.addWeighted(color_mask, 0.25, overlay, 1.0, 0, overlay)
        vis = overlay

        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # ── 2. Draw keypoints and skeleton ──────────────────────────────────
    if keypoints is not None and kp_confidences is not None:
        from pipeline.keypoint_detection import COW_SKELETON

        # Skeleton
        for i, j in COW_SKELETON:
            if (kp_confidences[i] >= min_conf and kp_confidences[j] >= min_conf):
                pt1 = tuple(keypoints[i].astype(int))
                pt2 = tuple(keypoints[j].astype(int))
                cv2.line(vis, pt1, pt2, (0, 255, 128), 2)

        # Points
        for idx, (kp, conf) in enumerate(zip(keypoints, kp_confidences)):
            if conf >= min_conf:
                pt = tuple(kp.astype(int))
                r = 5 if conf >= 0.6 else 4
                color = (0, 0, 255) if conf >= 0.6 else (0, 165, 255)
                cv2.circle(vis, pt, r, color, -1)
                cv2.circle(vis, pt, r + 2, (255, 255, 255), 1)

                if idx < len(kp_names):
                    cv2.putText(
                        vis, kp_names[idx],
                        (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1,
                    )

    # ── 3. Measurement lines ────────────────────────────────────────────
    if keypoints is not None and dimensions is not None and kp_confidences is not None:
        WITHERS, TAIL_HEAD, BRISKET = 4, 7, 16

        if kp_confidences[WITHERS] >= min_conf and kp_confidences[TAIL_HEAD] >= min_conf:
            p1 = tuple(keypoints[WITHERS].astype(int))
            p2 = tuple(keypoints[TAIL_HEAD].astype(int))
            cv2.line(vis, p1, p2, (0, 255, 255), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
            cv2.putText(vis, f"BL: {dimensions.get('body_length_cm', 0):.0f}cm",
                        mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if kp_confidences[WITHERS] >= min_conf and kp_confidences[BRISKET] >= min_conf:
            p1 = tuple(keypoints[WITHERS].astype(int))
            p2 = tuple(keypoints[BRISKET].astype(int))
            cv2.line(vis, p1, p2, (255, 0, 255), 2)
            mid = ((p1[0] + p2[0]) // 2 + 10, (p1[1] + p2[1]) // 2)
            cv2.putText(vis, f"CD: {dimensions.get('chest_depth_cm', 0):.0f}cm",
                        mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # ── 4. Info panel ───────────────────────────────────────────────────
    panel_h = 320
    panel_w = 320
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    y_off = 25
    cv2.putText(panel, "LIVESTOCK MEASUREMENT", (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
    y_off += 8
    cv2.line(panel, (10, y_off), (panel_w - 10, y_off), (0, 255, 200), 1)
    y_off += 22

    cv2.putText(panel, f"Distance: {distance_m:.1f} m", (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_off += 20
    cv2.putText(panel, f"Orientation: {orientation}", (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_off += 25

    # Dimensions
    if dimensions:
        cv2.putText(panel, "Body Dimensions:", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        y_off += 20
        for name, val in dimensions.items():
            label = name.replace("_cm", "").replace("_", " ").title()
            cv2.putText(panel, f"  {label}: {val} cm", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            y_off += 17

    y_off += 8
    cv2.line(panel, (10, y_off), (panel_w - 10, y_off), (0, 200, 255), 1)
    y_off += 22

    # Weight
    cv2.putText(panel, f"Weight: {weight_kg:.0f} kg", (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
    y_off += 22
    cv2.putText(panel, f"Range: {weight_range[0]:.0f} - {weight_range[1]:.0f} kg",
                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Place panel on the image
    if h >= panel_h and w >= panel_w:
        vis[0:panel_h, w - panel_w:w] = cv2.addWeighted(
            vis[0:panel_h, w - panel_w:w], 0.3, panel, 0.7, 0
        )
    elif h >= panel_h:
        # Resize panel to fit
        scale = min(w / panel_w, 1.0)
        rp = cv2.resize(panel, None, fx=scale, fy=scale)
        rph, rpw = rp.shape[:2]
        vis[0:rph, w - rpw:w] = cv2.addWeighted(
            vis[0:rph, w - rpw:w], 0.3, rp, 0.7, 0
        )

    return vis

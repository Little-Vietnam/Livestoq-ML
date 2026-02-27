"""
Distance Estimation Module
===========================
Estimates the real-world distance from the camera to the livestock animal.

Methods supported:
  1. Monocular depth estimation (MiDaS)
  2. Reference-object based (known-size object in frame)
  3. Known-height heuristic (uses average cow withers height)

The distance is critical for converting pixel measurements into real-world
metric dimensions (cm / m).
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class DistanceResult:
    """Result of distance estimation."""
    distance_m: float                     # Estimated distance in meters
    method_used: str                      # Which method was applied
    confidence: float                     # 0-1 confidence score
    depth_map: Optional[np.ndarray] = None  # Full depth map if available
    pixels_per_meter: float = 0.0         # Scale factor: pixels / real-meter at object plane


class DistanceEstimator:
    """
    Estimates the distance from the camera to the cow and computes
    the pixel-to-meter conversion factor.
    """

    def __init__(self, camera_config, distance_config):
        self.cam = camera_config
        self.cfg = distance_config
        self._midas_model = None
        self._midas_transform = None

    # ── Public API ────────────────────────────────────────────────────────

    def estimate(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        reference_px_length: Optional[float] = None,
    ) -> DistanceResult:
        """
        Estimate distance from the camera to the cow.

        Parameters
        ----------
        image : np.ndarray
            BGR input image.
        mask : np.ndarray, optional
            Binary segmentation mask of the cow (H, W).
        bbox : tuple, optional
            Bounding box (x1, y1, x2, y2) around the cow.
        reference_px_length : float, optional
            Length in pixels of a known reference object in the image.

        Returns
        -------
        DistanceResult
        """
        method = self.cfg.method

        if method == "reference_object" and reference_px_length is not None:
            return self._estimate_from_reference(reference_px_length)
        elif method == "monocular_depth":
            return self._estimate_monocular_depth(image, mask, bbox)
        else:
            # Fallback: use known average cow height vs bounding-box height
            return self._estimate_from_known_height(image, mask, bbox)

    # ── Monocular Depth (MiDaS) ──────────────────────────────────────────

    def _load_midas(self):
        """Lazy-load MiDaS depth estimation model."""
        import torch
        model_type = self.cfg.midas_model_type

        self._midas_model = torch.hub.load("intel-isl/MiDaS", model_type)
        self._midas_model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self._midas_transform = midas_transforms.dpt_transform
        else:
            self._midas_transform = midas_transforms.small_transform

    def _estimate_monocular_depth(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> DistanceResult:
        """
        Use MiDaS monocular depth estimation to get a relative depth map,
        then calibrate it using the known average cow height.
        """
        import torch

        if self._midas_model is None:
            self._load_midas()

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self._midas_transform(img_rgb)

        with torch.no_grad():
            prediction = self._midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Get the median depth in the cow region
        if mask is not None:
            cow_depths = depth_map[mask > 0]
        elif bbox is not None:
            x1, y1, x2, y2 = bbox
            cow_depths = depth_map[y1:y2, x1:x2].flatten()
        else:
            # Center crop as fallback
            h, w = depth_map.shape
            cow_depths = depth_map[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].flatten()

        median_relative_depth = np.median(cow_depths)

        # Calibrate: use known cow height to derive absolute scale
        # relative_depth is inversely proportional to real distance
        # We use the bounding box height + known withers height to calibrate
        if bbox is not None:
            _, y1, _, y2 = bbox
            cow_px_height = y2 - y1
        elif mask is not None:
            ys = np.where(mask > 0)[0]
            cow_px_height = ys.max() - ys.min() if len(ys) > 0 else image.shape[0] // 2
        else:
            cow_px_height = image.shape[0] // 2

        # distance = (focal_length × real_height) / pixel_height
        distance_m = (
            self.cam.focal_length_px * self.cam.avg_cow_height_m
        ) / cow_px_height

        distance_m *= self.cfg.depth_scale_factor

        # Compute pixels-per-meter at this distance
        ppm = self.cam.focal_length_px / distance_m

        return DistanceResult(
            distance_m=distance_m,
            method_used="monocular_depth",
            confidence=0.65,
            depth_map=depth_map,
            pixels_per_meter=ppm,
        )

    # ── Reference Object Method ──────────────────────────────────────────

    def _estimate_from_reference(self, ref_px_length: float) -> DistanceResult:
        """
        Estimate distance using a known-size reference object visible
        in the frame (e.g., a 1-meter stick placed near the cow).

        distance = (focal_length_px × real_length) / pixel_length
        """
        real_length = self.cam.reference_object_length_m
        distance_m = (self.cam.focal_length_px * real_length) / ref_px_length
        ppm = self.cam.focal_length_px / distance_m

        return DistanceResult(
            distance_m=distance_m,
            method_used="reference_object",
            confidence=0.90,
            pixels_per_meter=ppm,
        )

    # ── Known-Height Heuristic ───────────────────────────────────────────

    def _estimate_from_known_height(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> DistanceResult:
        """
        Use the known average cow withers height and the bounding-box /
        mask height to estimate camera distance.

        Z = (f × H_real) / h_pixels
        """
        if bbox is not None:
            _, y1, _, y2 = bbox
            cow_px_height = float(y2 - y1)
        elif mask is not None:
            ys = np.where(mask > 0)[0]
            cow_px_height = float(ys.max() - ys.min()) if len(ys) > 0 else 300.0
        else:
            cow_px_height = float(image.shape[0]) * 0.6

        distance_m = (
            self.cam.focal_length_px * self.cam.avg_cow_height_m
        ) / cow_px_height

        ppm = self.cam.focal_length_px / distance_m

        return DistanceResult(
            distance_m=distance_m,
            method_used="known_height_heuristic",
            confidence=0.50,
            pixels_per_meter=ppm,
        )

    # ── Utility ──────────────────────────────────────────────────────────

    def pixels_to_meters(self, pixel_length: float, distance_result: DistanceResult) -> float:
        """Convert a pixel measurement to real-world meters."""
        if distance_result.pixels_per_meter > 0:
            return pixel_length / distance_result.pixels_per_meter
        return 0.0

    def get_scale_at_depth(self, distance_m: float) -> float:
        """Return pixels-per-meter at a given depth."""
        return self.cam.focal_length_px / distance_m

"""
Livestock Weight & Dimension Prediction Pipeline
=================================================

Full end-to-end pipeline:

  Image Input
      │
      ▼
  ┌──────────────────────────┐
  │  1. Cow Segmentation     │  → binary mask + bounding box
  └──────────┬───────────────┘
             ▼
  ┌──────────────────────────┐
  │  2. Distance Estimation  │  → camera-to-cow distance (meters)
  │     (pixel → real scale) │    + pixels_per_meter conversion
  └──────────┬───────────────┘
             ▼
  ┌──────────────────────────┐
  │  3. Keypoint Detection   │  → 17 anatomical keypoints
  └──────────┬───────────────┘
             ▼
  ┌──────────────────────────┐
  │  4. Pose Normalization   │  → canonical lateral view
  │     (roll/yaw/pitch fix) │    + foreshortening correction
  └──────────┬───────────────┘
             ▼
  ┌──────────────────────────┐
  │  5. Dimension Extraction │  → 8 body measurements (cm)
  └──────────┬───────────────┘
             ▼
  ┌──────────────────────────┐
  │  6. Weight Prediction    │  → live weight (kg) + range
  └──────────────────────────┘
"""

import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime
from typing import Optional, Dict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import PipelineConfig, MEASUREMENT_NAMES
from pipeline.segmentation import CowSegmentor
from pipeline.distance_estimation import DistanceEstimator
from pipeline.keypoint_detection import KeypointDetector
from pipeline.pose_normalization import PoseNormalizer
from pipeline.dimension_extraction import DimensionExtractor
from models.weight_predictor import WeightPredictor
from utils.visualization import draw_pipeline_result


class LivestockPipeline:
    """
    End-to-end pipeline for livestock weight and dimension prediction
    from a single image.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize pipeline stages
        self.segmentor = CowSegmentor(
            self.config.segmentation, self.config.device
        )
        self.distance_estimator = DistanceEstimator(
            self.config.camera, self.config.distance
        )
        self.keypoint_detector = KeypointDetector(
            self.config.keypoint, self.config.device
        )
        self.pose_normalizer = PoseNormalizer(self.config.pose)
        self.dimension_extractor = DimensionExtractor()
        self.weight_predictor = WeightPredictor(self.config.weight)

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        print("=" * 60)
        print("  LIVESTOCK WEIGHT & DIMENSION PREDICTION PIPELINE")
        print("=" * 60)
        print(f"  Device:  {self.config.device}")
        print(f"  Output:  {self.config.output_dir}")
        print(f"  Debug:   {self.config.debug}")
        print("=" * 60)

    def run(
        self,
        image_path: str,
        breed: str = "generic",
        save_visualization: bool = True,
    ) -> Dict:
        """
        Run the full pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        breed : str
            Cow breed for weight correction.
        save_visualization : bool
            Whether to save annotated output image.

        Returns
        -------
        dict – Complete results with measurements and weight.
        """
        print(f"\n{'-' * 50}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'-' * 50}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        h, w = image.shape[:2]
        print(f"[Input] Image size: {w}x{h}")

        # Update camera config with actual image size
        self.config.camera.image_width_px = w
        self.config.camera.image_height_px = h

        results = {
            "image_path": image_path,
            "image_size": {"width": w, "height": h},
            "timestamp": datetime.now().isoformat(),
            "breed": breed,
        }

        # ── Stage 1: Segmentation ───────────────────────────────────────
        print("\n[Stage 1/6] Cow Segmentation...")
        seg_result = self.segmentor.segment(image)
        print(f"  -> Mask area: {seg_result.area_pixels:.0f} px "
              f"({100 * seg_result.area_pixels / (h * w):.1f}% of image)")
        print(f"  -> BBox: {seg_result.bbox}")
        print(f"  -> Confidence: {seg_result.confidence:.2f}")

        results["segmentation"] = {
            "bbox": list(seg_result.bbox),
            "area_pixels": seg_result.area_pixels,
            "confidence": seg_result.confidence,
        }

        if self.config.debug:
            self._save_debug(image, seg_result.mask, "01_segmentation")

        # ── Stage 2: Distance Estimation ────────────────────────────────
        print("\n[Stage 2/6] Distance Estimation...")
        dist_result = self.distance_estimator.estimate(
            image, mask=seg_result.mask, bbox=seg_result.bbox
        )
        print(f"  -> Distance: {dist_result.distance_m:.2f} m")
        print(f"  -> Method: {dist_result.method_used}")
        print(f"  -> Pixels/meter: {dist_result.pixels_per_meter:.1f}")
        print(f"  -> Confidence: {dist_result.confidence:.2f}")

        results["distance"] = {
            "distance_m": round(dist_result.distance_m, 2),
            "method": dist_result.method_used,
            "pixels_per_meter": round(dist_result.pixels_per_meter, 1),
            "confidence": dist_result.confidence,
        }

        # ── Stage 3: Keypoint Detection ─────────────────────────────────
        print("\n[Stage 3/6] Keypoint Detection...")
        kp_result = self.keypoint_detector.detect(
            image, mask=seg_result.mask, bbox=seg_result.bbox
        )
        print(f"  -> Detected: {kp_result.num_detected}/{self.config.keypoint.num_keypoints} keypoints")
        print(f"  -> Method: {kp_result.method_used}")

        results["keypoints"] = {
            "num_detected": kp_result.num_detected,
            "method": kp_result.method_used,
        }

        if self.config.debug:
            kp_vis = KeypointDetector.draw_keypoints(
                image, kp_result,
                self.config.keypoint.keypoint_names,
            )
            self._save_image(kp_vis, "02_keypoints")

        # ── Stage 4: Pose Normalization ─────────────────────────────────
        print("\n[Stage 4/6] Pose Normalization...")
        pose_result = self.pose_normalizer.normalize(
            kp_result.keypoints,
            kp_result.confidences,
            (h, w),
        )
        print(f"  -> Orientation: {pose_result.orientation}")
        print(f"  -> Yaw: {pose_result.yaw_angle_deg:.1f} deg  "
              f"Roll: {pose_result.roll_angle_deg:.1f} deg  "
              f"Pitch: {pose_result.pitch_angle_deg:.1f} deg")
        print(f"  -> Foreshortening factor: {pose_result.foreshortening_factor:.3f}")
        print(f"  -> Valid pose: {pose_result.is_valid_pose}")
        print(f"  -> Confidence: {pose_result.confidence:.3f}")

        results["pose"] = {
            "orientation": pose_result.orientation,
            "yaw_deg": round(pose_result.yaw_angle_deg, 1),
            "roll_deg": round(pose_result.roll_angle_deg, 1),
            "pitch_deg": round(pose_result.pitch_angle_deg, 1),
            "foreshortening_factor": round(pose_result.foreshortening_factor, 3),
            "is_valid": pose_result.is_valid_pose,
            "confidence": pose_result.confidence,
        }

        if self.config.debug:
            norm_img = self.pose_normalizer.apply_transform_to_image(
                image, pose_result.transform_matrix
            )
            self._save_image(norm_img, "03_normalized_pose")

        # Use normalized keypoints for dimension extraction
        norm_kps = pose_result.normalized_keypoints
        norm_mask = self.pose_normalizer.apply_transform_to_mask(
            seg_result.mask, pose_result.transform_matrix
        )

        # Re-find contour on normalized mask
        contours_norm, _ = cv2.findContours(
            norm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        norm_contour = (
            max(contours_norm, key=cv2.contourArea) if contours_norm
            else seg_result.contour
        )

        # ── Stage 5: Dimension Extraction ───────────────────────────────
        print("\n[Stage 5/6] Dimension Extraction...")
        dim_result = self.dimension_extractor.extract(
            keypoints=norm_kps,
            confidences=kp_result.confidences,
            contour=norm_contour,
            mask=norm_mask,
            pixels_per_meter=dist_result.pixels_per_meter,
            foreshortening_factor=pose_result.foreshortening_factor,
        )

        dims = dim_result.to_dict()
        print(f"  Measurements:")
        for name, val in dims.items():
            label = name.replace("_cm", "").replace("_", " ").title()
            print(f"    {label:20s}: {val:7.1f} cm")

        results["dimensions"] = dims
        results["dimension_confidence"] = dim_result.measurement_confidence

        # ── Stage 6: Weight Prediction ──────────────────────────────────
        print("\n[Stage 6/6] Weight Prediction...")
        weight_result = self.weight_predictor.predict(dims, breed=breed)

        print(f"  -> Predicted weight: {weight_result.predicted_weight_kg:.1f} kg")
        print(f"  -> Range: {weight_result.weight_range_kg[0]:.0f} - "
              f"{weight_result.weight_range_kg[1]:.0f} kg")
        print(f"  -> BCS estimate: {weight_result.bcs_estimate}/9")
        print(f"  -> Methods used:")
        for method, w in weight_result.method_weights.items():
            conf = weight_result.method_confidences.get(method, 0)
            print(f"      {method:20s}: {w:7.1f} kg  (conf: {conf:.2f})")

        results["weight"] = {
            "predicted_kg": weight_result.predicted_weight_kg,
            "range_kg": list(weight_result.weight_range_kg),
            "bcs": weight_result.bcs_estimate,
            "breed": weight_result.breed,
            "breed_factor": weight_result.breed_factor,
            "method_weights": weight_result.method_weights,
        }

        # ── Save Results ────────────────────────────────────────────────
        if save_visualization:
            vis = draw_pipeline_result(
                image=image,
                mask=seg_result.mask,
                keypoints=kp_result.keypoints,
                kp_confidences=kp_result.confidences,
                kp_names=self.config.keypoint.keypoint_names,
                dimensions=dims,
                weight_kg=weight_result.predicted_weight_kg,
                weight_range=weight_result.weight_range_kg,
                distance_m=dist_result.distance_m,
                orientation=pose_result.orientation,
                method_weights=weight_result.method_weights,
            )
            self._save_image(vis, "result")

        # Save JSON results
        json_path = os.path.join(
            self.config.output_dir,
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[Output] Results saved to: {json_path}")

        # Summary
        print(f"\n{'=' * 50}")
        print(f"  RESULT SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Image:    {os.path.basename(image_path)}")
        print(f"  Breed:    {breed}")
        print(f"  Distance: {dist_result.distance_m:.1f} m")
        print(f"  Body Length:   {dims['body_length_cm']:>7.1f} cm")
        print(f"  Body Height:   {dims['body_height_cm']:>7.1f} cm")
        print(f"  Chest Girth:   {dims['chest_girth_cm']:>7.1f} cm")
        print(f"  Chest Depth:   {dims['chest_depth_cm']:>7.1f} cm")
        print(f"  {'=' * 30}")
        print(f"  WEIGHT:   {weight_result.predicted_weight_kg:.0f} kg "
              f"({weight_result.weight_range_kg[0]:.0f}-{weight_result.weight_range_kg[1]:.0f} kg)")
        print(f"  BCS:      {weight_result.bcs_estimate}/9")
        print(f"{'=' * 50}\n")

        return results

    # ── Debug / Save Helpers ─────────────────────────────────────────────

    def _save_debug(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        name: str,
    ):
        """Save segmentation overlay for debugging."""
        vis = image.copy()
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        self._save_image(vis, name)

    def _save_image(self, image: np.ndarray, name: str):
        """Save an image to the output directory."""
        path = os.path.join(self.config.output_dir, f"{name}.jpg")
        cv2.imwrite(path, image)
        print(f"  [Saved] {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Livestock Weight & Dimension Prediction Pipeline",
    )
    parser.add_argument(
        "image", type=str,
        help="Path to the input cow image",
    )
    parser.add_argument(
        "--breed", type=str, default="generic",
        help="Cow breed (holstein, angus, jersey, brahman, etc.)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--distance-method", type=str, default="known_height",
        choices=["monocular_depth", "reference_object", "known_height"],
        help="Distance estimation method",
    )
    parser.add_argument(
        "--cow-height", type=float, default=1.4,
        help="Known/assumed cow withers height in meters",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Disable debug outputs",
    )

    args = parser.parse_args()

    # Build config
    config = PipelineConfig()
    config.device = args.device
    config.distance.method = args.distance_method
    config.camera.avg_cow_height_m = args.cow_height
    config.debug = not args.no_debug

    if args.output_dir:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)

    # Run pipeline
    pipeline = LivestockPipeline(config)
    results = pipeline.run(args.image, breed=args.breed)

    return results


if __name__ == "__main__":
    main()

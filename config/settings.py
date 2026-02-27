"""
Livestock ML - Configuration Settings
=====================================
Central configuration for the livestock weight & dimension prediction pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ─── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "weights")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ─── Livestock Body Measurement Standards ─────────────────────────────────────
# These are the 8 standard measurements from the body-size measurement chart.
MEASUREMENT_NAMES = [
    "body_length",       # Nose-to-tail length along the spine
    "body_width",        # Maximum lateral width viewed from above
    "tube_girth",        # Circumference around the barrel/tube area
    "body_height",       # Ground to top of the withers
    "chest_width",       # Width of the chest (front view)
    "abdominal_girth",   # Circumference around the abdomen
    "chest_depth",       # Top of the withers to bottom of the chest
    "chest_girth",       # Circumference around the chest (heart girth)
]


@dataclass
class CameraConfig:
    """Camera intrinsic / calibration parameters."""
    focal_length_mm: float = 4.25          # Typical smartphone focal length
    sensor_width_mm: float = 6.17          # Typical smartphone sensor width
    sensor_height_mm: float = 4.55
    image_width_px: int = 1920
    image_height_px: int = 1080
    # If a known reference object is in frame (e.g., a stick of known length)
    reference_object_length_m: float = 1.0  # meters
    # Average cow withers height used as reference when no calibration object
    avg_cow_height_m: float = 1.40          # meters (typical dairy cow)

    @property
    def focal_length_px(self) -> float:
        """Focal length in pixels."""
        return (self.focal_length_mm / self.sensor_width_mm) * self.image_width_px

    @property
    def cx(self) -> float:
        return self.image_width_px / 2.0

    @property
    def cy(self) -> float:
        return self.image_height_px / 2.0


@dataclass
class SegmentationConfig:
    """Configuration for cow segmentation."""
    model_name: str = "yolov8x-seg"        # YOLOv8 segmentation model
    confidence_threshold: float = 0.5
    cow_class_id: int = 19                  # COCO class ID for 'cow'
    mask_dilation_kernel: int = 5


@dataclass
class KeypointConfig:
    """Anatomical keypoint detection configuration."""
    # Number of anatomical keypoints to detect on a cow
    num_keypoints: int = 17
    keypoint_names: List[str] = field(default_factory=lambda: [
        "nose",               # 0
        "left_ear",           # 1
        "right_ear",          # 2
        "poll",               # 3  (top of head)
        "withers",            # 4
        "spine_mid",          # 5
        "hip_point",          # 6
        "tail_head",          # 7
        "left_shoulder",      # 8
        "right_shoulder",     # 9
        "left_hip",           # 10
        "right_hip",          # 11
        "left_knee_front",    # 12
        "right_knee_front",   # 13
        "left_hock",          # 14
        "right_hock",         # 15
        "brisket",            # 16 (lowest point of the chest)
    ])
    confidence_threshold: float = 0.3


@dataclass
class PoseNormalizationConfig:
    """Pose normalization parameters."""
    target_orientation: str = "lateral"  # lateral | dorsal | frontal
    target_image_size: Tuple[int, int] = (640, 640)
    # Acceptable pose angle tolerance (degrees)
    angle_tolerance: float = 15.0
    # Minimum keypoint confidence for pose analysis
    confidence_threshold: float = 0.3


@dataclass
class DistanceEstimationConfig:
    """Distance estimation configuration."""
    method: str = "monocular_depth"  # monocular_depth | reference_object | stereo
    # MiDaS model type for monocular depth estimation
    midas_model_type: str = "DPT_Large"
    # Depth map scaling factor (tuned per camera)
    depth_scale_factor: float = 1.0


@dataclass
class WeightPredictionConfig:
    """Weight prediction model configuration."""
    # Regression model type
    model_type: str = "gradient_boosting"  # gradient_boosting | mlp | linear
    # Known allometric relationships (empirical coefficients)
    # Weight ≈ a × chest_girth^b × body_length^c  (Schaeffer's formula variant)
    schaeffer_a: float = 0.000214
    schaeffer_b: float = 2.028
    schaeffer_c: float = 0.811
    # Breed correction factors
    breed_factors: dict = field(default_factory=lambda: {
        "holstein": 1.0,
        "angus": 1.05,
        "hereford": 1.03,
        "jersey": 0.85,
        "brahman": 1.08,
        "generic": 1.0,
    })


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    pose: PoseNormalizationConfig = field(default_factory=PoseNormalizationConfig)
    distance: DistanceEstimationConfig = field(default_factory=DistanceEstimationConfig)
    weight: WeightPredictionConfig = field(default_factory=WeightPredictionConfig)
    # General
    device: str = "cpu"  # cpu | cuda
    debug: bool = True
    output_dir: str = OUTPUT_DIR

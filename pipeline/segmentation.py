"""
Cow Segmentation Module
========================
Segments the cow from the background using YOLOv8 instance segmentation
or SAM (Segment Anything Model) as fallback.

Outputs:
  - Binary mask of the cow
  - Bounding box
  - Contour for shape analysis
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SegmentationResult:
    """Result of cow segmentation."""
    mask: np.ndarray                        # Binary mask (H, W), uint8 0/255
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2)
    contour: np.ndarray                     # Largest contour points
    area_pixels: float                      # Mask area in pixels
    confidence: float                       # Detection confidence
    cropped_image: Optional[np.ndarray] = None
    cropped_mask: Optional[np.ndarray] = None


class CowSegmentor:
    """
    Segments the cow from the image using YOLOv8-seg or
    fallback color/GrabCut-based segmentation.
    """

    def __init__(self, seg_config, device: str = "cpu"):
        self.cfg = seg_config
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the YOLOv8 segmentation model."""
        try:
            from ultralytics import YOLO
            model_name = self.cfg.model_name
            if not model_name.endswith(".pt"):
                model_name += ".pt"
            self._model = YOLO(model_name)
            print(f"[Segmentation] Loaded {model_name}")
        except ImportError:
            print("[Segmentation] ultralytics not found, using fallback segmentation")
            self._model = None

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Segment the cow from the image.

        Parameters
        ----------
        image : np.ndarray
            BGR input image.

        Returns
        -------
        SegmentationResult
        """
        if self._model is None:
            self._load_model()

        if self._model is not None:
            return self._segment_yolo(image)
        else:
            return self._segment_fallback(image)

    # ── YOLOv8 Segmentation ─────────────────────────────────────────────

    def _segment_yolo(self, image: np.ndarray) -> SegmentationResult:
        """Segment using YOLOv8 instance segmentation."""
        results = self._model(image, conf=self.cfg.confidence_threshold, verbose=False)

        best_mask = None
        best_conf = 0.0
        best_bbox = None

        for result in results:
            if result.masks is None:
                continue

            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Filter for cow class
                if cls_id == self.cfg.cow_class_id and conf > best_conf:
                    best_conf = conf
                    best_mask = mask.data[0].cpu().numpy()
                    best_bbox = box.xyxy[0].cpu().numpy().astype(int)

        if best_mask is not None:
            # Resize mask to image dimensions
            h, w = image.shape[:2]
            mask_resized = cv2.resize(
                best_mask.astype(np.float32), (w, h),
                interpolation=cv2.INTER_LINEAR
            )
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

            # Optionally dilate to capture edges
            if self.cfg.mask_dilation_kernel > 0:
                kernel = np.ones(
                    (self.cfg.mask_dilation_kernel, self.cfg.mask_dilation_kernel),
                    np.uint8,
                )
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            return self._build_result(image, binary_mask, best_conf)

        # No cow detected by YOLO – fallback
        print("[Segmentation] No cow detected by YOLO, using fallback")
        return self._segment_fallback(image)

    # ── Fallback Segmentation (GrabCut + Color) ──────────────────────────

    def _segment_fallback(self, image: np.ndarray) -> SegmentationResult:
        """
        Fallback segmentation using GrabCut when no DL model is available.
        Assumes the cow occupies a significant portion of the image.
        Resizes to max 480px for speed, then scales mask back.
        """
        h, w = image.shape[:2]

        # Resize for faster GrabCut
        max_dim = 480
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = image

        sh, sw = small.shape[:2]

        # Initialize with a centered rectangle (assume cow is roughly centered)
        margin_x = int(sw * 0.05)
        margin_y = int(sh * 0.05)
        rect = (margin_x, margin_y, sw - 2 * margin_x, sh - 2 * margin_y)

        mask = np.zeros((sh, sw), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            small, mask, rect, bgd_model, fgd_model,
            iterCount=5, mode=cv2.GC_INIT_WITH_RECT
        )

        # Convert GrabCut mask to binary
        binary_mask_small = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Scale mask back to original size
        if scale < 1.0:
            binary_mask = cv2.resize(
                binary_mask_small, (w, h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            binary_mask = binary_mask_small

        # Morphological cleanup
        kernel = np.ones((7, 7), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        return self._build_result(image, binary_mask, confidence=0.40)

    # ── Result Construction ──────────────────────────────────────────────

    def _build_result(
        self,
        image: np.ndarray,
        binary_mask: np.ndarray,
        confidence: float,
    ) -> SegmentationResult:
        """Build a SegmentationResult from a binary mask."""
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            h, w = image.shape[:2]
            return SegmentationResult(
                mask=binary_mask,
                bbox=(0, 0, w, h),
                contour=np.array([]),
                area_pixels=0,
                confidence=0.0,
            )

        # Largest contour = the cow
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(largest)
        bbox = (x, y, x + bw, y + bh)

        # Crop
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + bw + pad)
        y2 = min(image.shape[0], y + bh + pad)

        cropped_img = image[y1:y2, x1:x2].copy()
        cropped_mask = binary_mask[y1:y2, x1:x2].copy()

        return SegmentationResult(
            mask=binary_mask,
            bbox=bbox,
            contour=largest,
            area_pixels=area,
            confidence=confidence,
            cropped_image=cropped_img,
            cropped_mask=cropped_mask,
        )

# Livestock Weight & Dimension Prediction Pipeline

Computer-vision pipeline that estimates **live weight** and **8 standard body dimensions** of cattle from a single photograph.

## Pipeline Architecture

```
Image Input
    │
    ▼
┌──────────────────────────┐
│  1. Cow Segmentation     │  YOLOv8-seg / GrabCut fallback
│     → binary mask + bbox │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  2. Distance Estimation  │  Monocular depth / known-height heuristic
│     → pixels_per_meter   │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  3. Keypoint Detection   │  DL pose model / contour heuristics
│     → 17 anatomical pts  │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  4. Pose Normalization   │  Roll/yaw/pitch correction
│     → canonical lateral  │  + foreshortening compensation
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  5. Dimension Extraction │  8 body measurements → cm
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  6. Weight Prediction    │  Multi-formula ensemble:
│     → kg + range         │  Schaeffer / Heart-Girth / Crevat-Lagneau
│                          │  + ML regression + BSA method
└──────────────────────────┘
```

## 8 Standard Body Measurements

| # | Measurement      | Description                                |
|---|------------------|--------------------------------------------|
| 1 | Body Length       | Withers → Tail head (along spine)          |
| 2 | Body Width        | Maximum lateral width                      |
| 3 | Tube Girth        | Circumference of the barrel area           |
| 4 | Body Height       | Ground → Withers                           |
| 5 | Chest Width       | Width across the chest                     |
| 6 | Abdominal Girth   | Circumference of the abdomen               |
| 7 | Chest Depth       | Withers → Brisket (vertical)               |
| 8 | Chest Girth       | Heart girth – circumference around chest   |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on your image
python pipeline/main.py "path/to/cow_image.jpg" --breed generic

# With options
python pipeline/main.py "image.jpg" \
    --breed holstein \
    --device cpu \
    --distance-method known_height \
    --cow-height 1.4
```

## Project Structure

```
Livestoq-ML/
├── config/
│   ├── __init__.py
│   └── settings.py              # All configuration dataclasses
├── pipeline/
│   ├── __init__.py
│   ├── main.py                  # Pipeline orchestrator + CLI
│   ├── segmentation.py          # Stage 1: Cow segmentation
│   ├── distance_estimation.py   # Stage 2: Camera distance
│   ├── keypoint_detection.py    # Stage 3: Anatomical keypoints
│   ├── pose_normalization.py    # Stage 4: Pose correction
│   └── dimension_extraction.py  # Stage 5: Body measurements
├── models/
│   ├── __init__.py
│   ├── weight_predictor.py      # Stage 6: Weight prediction
│   └── weights/                 # Pre-trained model weights
├── utils/
│   ├── __init__.py
│   └── visualization.py         # Drawing utilities
├── output/                      # Generated results
├── requirements.txt
└── README.md
```

## Weight Prediction Methods

The pipeline uses an **ensemble of 5 methods** with confidence-weighted averaging:

1. **Schaeffer's Formula**: `W = a × CG^b × BL^c` (allometric)
2. **Heart-Girth Formula**: `W = (CG² × BL) / 300` (converted from imperial)
3. **Crevat-Lagneau**: `W = 80 × CG³` (metric)
4. **Multi-measurement Regression**: Linear model with literature coefficients (or trained GBR)
5. **Body Surface Area**: Cylinder approximation → Brody's equation

## Breed Support

| Breed    | Correction Factor |
|----------|------------------|
| Holstein | 1.00             |
| Angus    | 1.05             |
| Hereford | 1.03             |
| Jersey   | 0.85             |
| Brahman  | 1.08             |
| Generic  | 1.00             |

## Notes

- Best accuracy is achieved with a **lateral (side) view** of the cow
- The pipeline includes a **pose validation** step that warns about suboptimal viewing angles
- A calibration reference object (known-length stick) in the frame significantly improves distance accuracy
- For production use, train the Gradient Boosting model on your herd's actual weight data
